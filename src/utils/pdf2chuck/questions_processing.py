import json
from typing import Union, Dict, List, Optional
import re
from pathlib import Path
from src.utils.pdf2chuck.retrieval import VectorRetriever, MilvusVectorRetriever
from src.utils.pdf2chuck.api_requests import APIProcessor
from src.rerank.reranker import ModelReranker, VLLMReranker
from tqdm import tqdm
import pandas as pd
import threading
import concurrent.futures
import time
from src.utils.pdf2chuck.hybrid_weighted_retrieval import HybridWeightedRetriever
import logging

_log = logging.getLogger(__name__)


class QuestionsProcessor:
    def __init__(
        self,
        vector_db_dir: Union[str, Path] = "./vector_dbs",
        documents_dir: Union[str, Path] = "./documents",
        questions_file_path: Optional[Union[str, Path]] = None,
        subset_path: Optional[Union[str, Path]] = None,
        parent_document_retrieval: bool = False,
        reranker: Optional[Union[VLLMReranker, ModelReranker]] = None,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "ollama",
        answering_model: str = "qwen2.5:14b",
        full_context: bool = False,
        # 新增查询重写相关参数
        use_query_rewrite: bool = False,
        query_rewrite_model: str = "qwen2.5:72b",
        max_query_variations: int = 3,
        use_hybrid_retriever: bool = False,  # 新增
        hybrid_es_index: str = "knowledge_test",  # 新增
        hybrid_alpha: float = 0.5,  # 新增
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.vector_db_dir = Path(vector_db_dir)
        self.subset_path = Path(subset_path) if subset_path else None

        self.return_parent_pages = parent_document_retrieval
        self.reranker = reranker
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        # 初始化查询重写相关属性
        self.use_query_rewrite = use_query_rewrite
        self.query_rewrite_model = query_rewrite_model
        self.max_query_variations = max_query_variations
        if self.use_query_rewrite:
            from src.query_rewrite.query_rewriter import QueryRewriter

            self.query_rewriter = QueryRewriter(
                model=None,
                system_prompt="Generate query variations for improved document retrieval.",
                api_provider=self.api_provider,
            )

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()
        if use_hybrid_retriever:
            self.retriever = HybridWeightedRetriever(
                collection_name="challenge_data",  # 使用默认的collection_name
                documents_dir=documents_dir,
                es_index=hybrid_es_index,
                alpha=hybrid_alpha,
                top_k=top_n_retrieval,
                return_parent_pages=parent_document_retrieval,
            )
        else:
            self.retriever = None

    # 打印QuestionsProcessor的所有的信息
    def info_print(self):
        """Print out all information about the processor's state"""
        print("\nQuestionsProcessor Configuration:")
        print(f"questions: {self.questions}")
        print(f"documents_dir: {self.documents_dir}")
        print(f"vector_db_dir: {self.vector_db_dir}")
        print(f"subset_path: {self.subset_path}")
        print(f"return_parent_pages: {self.return_parent_pages}")
        print(f"llm_reranking: {self.llm_reranking}")
        print(f"llm_reranking_sample_size: {self.llm_reranking_sample_size}")
        print(f"top_n_retrieval: {self.top_n_retrieval}")
        print(f"answering_model: {self.answering_model}")
        print(f"parallel_requests: {self.parallel_requests}")
        print(f"api_provider: {self.api_provider}")
        print(f"full_context: {self.full_context}")

    def _load_questions(
        self, questions_file_path: Optional[Union[str, Path]]
    ) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""

        context_parts = []
        for result in retrieval_results:
            page_number = result["page"]
            text = result["text"]
            context_parts.append(
                f'Text retrieved from page {page_number}: \n"""\n{text}\n"""'
            )

        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        # Load companies data
        if self.subset_path is None:
            raise ValueError(
                "subset_path is required for new challenge pipeline when processing references."
            )
        self.companies_df = pd.read_csv(self.subset_path)

        # Find the company's SHA1 from the subset CSV
        matching_rows = self.companies_df[
            self.companies_df["company_name"] == company_name
        ]
        if matching_rows.empty:
            company_sha1 = ""
        else:
            company_sha1 = matching_rows.iloc[0]["sha1"]

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": company_sha1, "page_index": page})
        return refs

    def _validate_page_references(
        self,
        claimed_pages: list,
        retrieval_results: list,
        min_pages: int = 2,
        max_pages: int = 8,
    ) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if claimed_pages is None:
            claimed_pages = []

        retrieved_pages = [result["page"] for result in retrieval_results]

        validated_pages = [page for page in claimed_pages if page in retrieved_pages]

        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(
                f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}"
            )

        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)

            for result in retrieval_results:
                page = result["page"]
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)

                    if len(validated_pages) >= min_pages:
                        break

        if len(validated_pages) > max_pages:
            print(
                f"Trimming references from {len(validated_pages)} to {max_pages} pages"
            )
            validated_pages = validated_pages[:max_pages]

        return validated_pages

    def _merge_retrieval_results(
        self, results_list: List[List[dict]], top_n: int
    ) -> List[dict]:
        """
        合并多个检索结果，并按页面去重，保留top_n个结果

        Args:
            results_list: 多个检索结果列表
            top_n: 需要保留的结果数量

        Returns:
            合并并去重后的检索结果列表
        """
        # 使用字典来存储唯一的结果，以(pdf_sha1, page)为键
        unique_results = {}

        # 遍历所有检索结果
        for results in results_list:
            for result in results:
                key = (result.get("pdf_sha1", ""), result.get("page", 0))
                if key not in unique_results:
                    unique_results[key] = result

        # 转换为列表并截取top_n
        merged_results = list(unique_results.values())
        return merged_results[:top_n]

    def _get_retrieval_results(self, company_name: str, query: str) -> List[dict]:
        """
        获取单个查询的检索结果

        Args:
            company_name: 公司名称
            query: 查询文本

        Returns:
            检索结果列表
        """
        if self.retriever is not None:
            results = self.retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=(
                    self.reranker.sample_size if self.reranker else self.top_n_retrieval
                ),
            )
            if self.reranker:
                return self.reranker.send_message_for_reranking(
                    query=query, documents=results, top_n=self.top_n_retrieval
                )
            return results

        retriever = VectorRetriever(
            vector_db_dir=self.vector_db_dir, documents_dir=self.documents_dir
        )

        if self.full_context:
            return retriever.retrieve_all(company_name)

        if self.reranker:
            results = retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=self.reranker.sample_size,
                return_parent_pages=self.return_parent_pages,
            )
            _log.debug(
                f"Retrieval results for query '{query}' (company: {company_name}): {len(results)} pages"
            )
            return self.reranker.send_message_for_reranking(
                query=query, documents=results, top_n=self.top_n_retrieval
            )

        return retriever.retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=self.top_n_retrieval,
            return_parent_pages=self.return_parent_pages,
        )

    def get_answer_for_company(
        self, company_name: str, question: str, schema: str
    ) -> dict:
        """
        获取公司相关问题的答案

        Args:
            company_name: 公司名称
            question: 原始问题
            schema: 答案模式

        Returns:
            包含答案的字典
        """
        # 生成查询变体
        queries = [question]  # 始终包含原始查询
        if self.use_query_rewrite:
            rewritten_queries = self.query_rewriter.rewrite(
                query=question, max_variations=self.max_query_variations
            )
            queries.extend(rewritten_queries)

        # _log.debug(f"Processing queries for company '{company_name}':")
        # _log.debug(f"Original query: {question}")
        if self.use_query_rewrite:
            _log.debug(f"Rewritten queries: {rewritten_queries}")

        # 获取所有查询的检索结果
        all_results = []
        for query in queries:
            results = self._get_retrieval_results(company_name, query)
            all_results.append(results)

        # 合并检索结果
        retrieval_results = self._merge_retrieval_results(
            all_results, self.top_n_retrieval
        )

        # _log.debug(f"Total retrieval results after merging: {len(retrieval_results)}")

        if not retrieval_results:
            _log.error(
                f"No relevant context found for company '{company_name}' and query '{question}'"
            )
            raise ValueError("No relevant context found")

        # 格式化检索结果并获取答案
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        # 处理页面引用
        pages = answer_dict.get("relevant_pages", [])
        validated_pages = self._validate_page_references(pages, retrieval_results)
        answer_dict["relevant_pages"] = validated_pages
        answer_dict["references"] = self._extract_references(
            validated_pages, company_name
        )

        return answer_dict

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """Extract company names from a question by matching against companies in the subset file."""
        if not hasattr(self, "companies_df"):
            if self.subset_path is None:
                raise ValueError(
                    "subset_path must be provided to use subset extraction"
                )
            self.companies_df = pd.read_csv(self.subset_path)

        found_companies = []
        company_names = sorted(
            self.companies_df["company_name"].unique(), key=len, reverse=True
        )

        for company in company_names:
            escaped_company = re.escape(company)

            pattern = rf"{escaped_company}(?:\W|$)"

            if re.search(pattern, question_text, re.IGNORECASE):
                found_companies.append(company)
                question_text = re.sub(pattern, "", question_text, flags=re.IGNORECASE)

        # _log.debug(f"Matched company from question '{question_text}': {found_companies}")
        return found_companies

    def process_question(self, question: str, schema: str):
        extracted_companies = self._extract_companies_from_subset(question)
        # print("question", question)
        # print("extracted_companies", extracted_companies)
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")

        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(
                company_name=company_name, question=question, schema=schema
            )
            return answer_dict
        else:
            if self.parallel_requests <= 1:
                return self.process_comparative_question(
                    question, extracted_companies, schema
                )
            else:
                return self.process_comparative_question_parallel(
                    question, extracted_companies, schema
                )

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict["step_by_step_analysis"],
                "reasoning_summary": answer_dict["reasoning_summary"],
                "relevant_pages": answer_dict["relevant_pages"],
                "response_data": self.response_data,
                "self": ref_id,
            }
        return ref_id

    def _calculate_statistics(
        self, processed_questions: List[dict], print_stats: bool = False
    ) -> dict:
        """Calculate statistics about processed questions."""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(
            1
            for q in processed_questions
            if (q.get("value") if "value" in q else q.get("answer")) == "N/A"
        )
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(
                f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n"
            )

        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count,
        }

    def process_questions_list(
        self,
        questions_list: List[dict],
        output_path: str = None,
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ) -> dict:
        # print("questions_list", questions_list)
        total_questions = len(questions_list)
        # Add index to each question so we know where to write the answer details
        questions_with_index = [
            {**q, "_question_index": i} for i, q in enumerate(questions_list)
        ]
        self.answer_details = [
            None
        ] * total_questions  # Preallocate list for answer details
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(
                questions_with_index, desc="Processing questions"
            ):
                print("single question", question_data)
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(
                        processed_questions,
                        output_path,
                        submission_file=submission_file,
                        team_email=team_email,
                        submission_name=submission_name,
                        pipeline_details=pipeline_details,
                    )
        else:
            print("parallel processing")
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=parallel_threads
                    ) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(
                            executor.map(self._process_single_question, batch)
                        )
                    processed_questions.extend(batch_results)

                    if output_path:
                        self._save_progress(
                            processed_questions,
                            output_path,
                            submission_file=submission_file,
                            team_email=team_email,
                            submission_name=submission_name,
                            pipeline_details=pipeline_details,
                        )
                    pbar.update(len(batch_results))

        statistics = self._calculate_statistics(processed_questions, print_stats=True)

        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics,
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        # print("question_data", question_data)
        # print("question_index", question_index)
        # print("--------------------------------")
        question_text = question_data.get("text")
        schema = question_data.get("kind")
        try:
            answer_dict = self.process_question(question_text, schema)

            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref(
                    {
                        "step_by_step_analysis": None,
                        "reasoning_summary": None,
                        "relevant_pages": None,
                    },
                    question_index,
                )
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": None,
                    "references": [],
                    "error": answer_dict["error"],
                    "answer_details": {"$ref": detail_ref},
                }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            return {
                "question_text": question_text,
                "kind": schema,
                "value": answer_dict.get("final_answer"),
                "references": answer_dict.get("references", []),
                "answer_details": {"$ref": detail_ref},
            }
        except Exception as err:
            return self._handle_processing_error(
                question_text, schema, err, question_index
            )

    def _handle_processing_error(
        self, question_text: str, schema: str, err: Exception, question_index: int
    ) -> dict:
        """
        Handle errors during question processing.
        Log error details and return a dictionary containing error information.
        """
        import traceback

        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {"error_traceback": tb, "self": error_ref}

        with self._lock:
            self.answer_details[question_index] = error_detail

        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")

        return {
            "question_text": question_text,
            "kind": schema,
            "value": None,
            "references": [],
            "error": f"{type(err).__name__}: {error_message}",
            "answer_details": {"$ref": error_ref},
        }

    def _post_process_submission_answers(
        self, processed_questions: List[dict]
    ) -> List[dict]:
        """
        Post-process answers for submission format:
        1. Convert page indices from one-based to zero-based
        2. Clear references for N/A answers
        3. Format answers according to submission schema
        4. Include step_by_step_analysis from answer details
        """
        submission_answers = []

        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = (
                "N/A"
                if "error" in q
                else (q.get("value") if "value" in q else q.get("answer"))
            )
            references = q.get("references", [])

            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith(
                "#/answer_details/"
            ):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if (
                        0 <= index < len(self.answer_details)
                        and self.answer_details[index]
                    ):
                        step_by_step_analysis = self.answer_details[index].get(
                            "step_by_step_analysis"
                        )
                except (ValueError, IndexError):
                    pass

            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": (
                            (ref["page_index"] - 1)
                            if ref["page_index"] is not None
                            else 0
                        ),
                    }
                    for ref in references
                    if ref.get("page_index") is not None  # 过滤掉page_index为None的引用
                ]

            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }

            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis

            submission_answers.append(submission_answer)

        return submission_answers

    def _save_progress(
        self,
        processed_questions: List[dict],
        output_path: Optional[str],
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)

            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics,
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(
                output_file.stem + "_debug" + output_file.suffix
            )
            with open(debug_file, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)

            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(
                    processed_questions
                )
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details,
                    "time": time.strftime(
                        "%Y-%m-%d, %H:%M:%S", time.localtime(time.time())
                    ),
                    "signature": hex(int(time.time())),
                    "tsp_signature": "",
                    "submission_digest": "",
                }
                with open(output_file, "w", encoding="utf-8") as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(
        self,
        output_path: str = "questions_with_answers.json",
        team_email: str = "79250515615@yandex.com",
        submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval",
        submission_file: bool = False,
        pipeline_details: str = "",
    ):
        # print("self.questions", self.questions)
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details,
        )
        return result

    def process_comparative_question_parallel(
        self, question: str, companies: List[str], schema: str
    ) -> dict:
        """
        Process a question involving multiple companies in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question, companies=companies
        )

        individual_answers = {}
        aggregated_references = []

        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(
                    f"Could not generate sub-question for company: {company}"
                )

            answer_dict = self.get_answer_for_company(
                company_name=company, question=sub_question, schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company
                for company in companies
            }

            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict

                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise

        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())

        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        comparative_answer["references"] = aggregated_references
        return comparative_answer

    def process_comparative_question(
        self, question: str, companies: List[str], schema: str
    ) -> dict:
        """
        Process a question involving multiple companies in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question, companies=companies
        )

        individual_answers = {}
        aggregated_references = []

        for company in companies:
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(
                    f"Could not generate sub-question for company: {company}"
                )

            answer_dict = self.get_answer_for_company(
                company_name=company, question=sub_question, schema="number"
            )

            individual_answers[company] = answer_dict

            company_references = answer_dict.get("references", [])
            aggregated_references.extend(company_references)

        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())

        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        comparative_answer["references"] = aggregated_references

        return comparative_answer


from src.database.db_connection import milvus_connection
from pymilvus import Collection


class QuestionsProcessorMilvus:
    def __init__(
        self,
        documents_dir: Union[str, Path] = "./documents",
        questions_file_path: Optional[Union[str, Path]] = None,
        subset_path: Optional[Union[str, Path]] = None,
        pdf_path: Union[str, Path] = "src/resources/data/pdf_reports",
        parent_document_retrieval: bool = False,
        reranker: Optional[Union[VLLMReranker, ModelReranker]] = None,
        top_n_retrieval: int = 10,
        parallel_requests: int = 10,
        api_provider: str = "ollama",
        answering_model: str = "qwen2.5:14b",
        full_context: bool = False,
        collection_name: str = "challenge_data",
        # 新增查询重写相关参数
        use_query_rewrite: bool = False,
        query_rewrite_model: str = "qwen2.5:72b",
        max_query_variations: int = 3,
        use_hybrid_retriever: bool = False,  # 新增
        hybrid_es_index: str = "knowledge_test",  # 新增
        hybrid_alpha: float = 0.5,  # 新增
    ):
        self.questions = self._load_questions(questions_file_path)
        self.documents_dir = Path(documents_dir)
        self.subset_path = Path(subset_path) if subset_path else None

        self.return_parent_pages = parent_document_retrieval
        self.reranker = reranker
        self.top_n_retrieval = top_n_retrieval
        self.answering_model = answering_model
        self.parallel_requests = parallel_requests
        self.api_provider = api_provider
        self.openai_processor = APIProcessor(provider=api_provider)
        self.full_context = full_context

        # 初始化查询重写相关属性
        self.use_query_rewrite = use_query_rewrite
        self.query_rewrite_model = query_rewrite_model
        self.max_query_variations = max_query_variations
        if self.use_query_rewrite:
            from src.query_rewrite.query_rewriter import QueryRewriter

            self.query_rewriter = QueryRewriter(
                model=None,
                system_prompt="Generate query variations for improved document retrieval.",
                api_provider=self.api_provider,
            )

        self.answer_details = []
        self.detail_counter = 0
        self._lock = threading.Lock()
        self.pdf_path = pdf_path
        self.collection_name = collection_name  # 保存collection_name
        self.collection = self._load_milvus_collection(collection_name)
        if use_hybrid_retriever:
            self.retriever = HybridWeightedRetriever(
                collection_name=collection_name,
                documents_dir=documents_dir,
                es_index=hybrid_es_index,
                alpha=hybrid_alpha,
                top_k=top_n_retrieval,
                return_parent_pages=parent_document_retrieval,
            )
        else:
            self.retriever = None

    # 打印QuestionsProcessor的所有的信息
    def info_print(self):
        """Print out all information about the processor's state"""
        print("\nQuestionsProcessor Configuration:")
        print(f"questions: {self.questions}")
        print(f"documents_dir: {self.documents_dir}")
        print(f"subset_path: {self.subset_path}")
        print(f"return_parent_pages: {self.return_parent_pages}")
        print(f"llm_reranking: {self.llm_reranking}")
        print(f"llm_reranking_sample_size: {self.llm_reranking_sample_size}")
        print(f"top_n_retrieval: {self.top_n_retrieval}")
        print(f"answering_model: {self.answering_model}")
        print(f"parallel_requests: {self.parallel_requests}")
        print(f"api_provider: {self.api_provider}")
        print(f"full_context: {self.full_context}")
        print(f"pdf_path: {self.pdf_path}")
        print(f"collection_name: {self.collection_name}")

    def _load_milvus_collection(self, collection_name: str):
        from src.database.db_connection import get_milvus_collection

        return get_milvus_collection(collection_name)

    def _load_questions(
        self, questions_file_path: Optional[Union[str, Path]]
    ) -> List[Dict[str, str]]:
        if questions_file_path is None:
            return []
        with open(questions_file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _format_retrieval_results(self, retrieval_results) -> str:
        """Format vector retrieval results into RAG context string"""
        if not retrieval_results:
            return ""

        context_parts = []
        for result in retrieval_results:
            page_number = result["page"]
            text = result["text"]
            context_parts.append(
                f'Text retrieved from page {page_number}: \n"""\n{text}\n"""'
            )

        return "\n\n---\n\n".join(context_parts)

    def _extract_references(self, pages_list: list, company_name: str) -> list:
        # Load companies data
        if self.subset_path is None:
            raise ValueError(
                "subset_path is required for new challenge pipeline when processing references."
            )
        self.companies_df = pd.read_csv(self.subset_path)

        # Find the company's SHA1 from the subset CSV
        matching_rows = self.companies_df[
            self.companies_df["company_name"] == company_name
        ]
        if matching_rows.empty:
            company_sha1 = ""
        else:
            company_sha1 = matching_rows.iloc[0]["sha1"]

        refs = []
        for page in pages_list:
            refs.append({"pdf_sha1": company_sha1, "page_index": page})
        return refs

    def _validate_page_references(
        self,
        claimed_pages: list,
        retrieval_results: list,
        min_pages: int = 2,
        max_pages: int = 8,
    ) -> list:
        """
        Validate that all page numbers mentioned in the LLM's answer are actually from the retrieval results.
        If fewer than min_pages valid references remain, add top pages from retrieval results.
        """
        if claimed_pages is None:
            claimed_pages = []

        retrieved_pages = [result["page"] for result in retrieval_results]

        validated_pages = [page for page in claimed_pages if page in retrieved_pages]

        if len(validated_pages) < len(claimed_pages):
            removed_pages = set(claimed_pages) - set(validated_pages)
            print(
                f"Warning: Removed {len(removed_pages)} hallucinated page references: {removed_pages}"
            )

        if len(validated_pages) < min_pages and retrieval_results:
            existing_pages = set(validated_pages)

            for result in retrieval_results:
                page = result["page"]
                if page not in existing_pages:
                    validated_pages.append(page)
                    existing_pages.add(page)

                    if len(validated_pages) >= min_pages:
                        break

        if len(validated_pages) > max_pages:
            print(
                f"Trimming references from {len(validated_pages)} to {max_pages} pages"
            )
            validated_pages = validated_pages[:max_pages]

        return validated_pages

    def _merge_retrieval_results(
        self, results_list: List[List[dict]], top_n: int
    ) -> List[dict]:
        """
        合并多个检索结果，并按页面去重，保留top_n个结果

        Args:
            results_list: 多个检索结果列表
            top_n: 需要保留的结果数量

        Returns:
            合并并去重后的检索结果列表
        """
        # 使用字典来存储唯一的结果，以(pdf_sha1, page)为键
        unique_results = {}

        # 遍历所有检索结果
        for results in results_list:
            for result in results:
                key = (result.get("pdf_sha1", ""), result.get("page", 0))
                if key not in unique_results:
                    unique_results[key] = result

        # 转换为列表并截取top_n
        merged_results = list(unique_results.values())
        return merged_results[:top_n]

    def _get_retrieval_results(self, company_name: str, query: str) -> List[dict]:
        """
        获取单个查询的检索结果

        Args:
            company_name: 公司名称
            query: 查询文本

        Returns:
            检索结果列表
        """
        if self.retriever is not None:
            results = self.retriever.retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=(
                    self.reranker.sample_size if self.reranker else self.top_n_retrieval
                ),
            )
            if self.reranker:
                return self.reranker.send_message_for_reranking(
                    query=query, documents=results, top_n=self.top_n_retrieval
                )
            return results

        retriever = MilvusVectorRetriever(collection=self.collection)

        if self.full_context:
            return retriever.retrieve_all(company_name)

        if self.reranker:
            results = retriever.milvus_retrieve_by_company_name(
                company_name=company_name,
                query=query,
                top_n=self.reranker.sample_size,
                return_parent_pages=self.return_parent_pages,
                documents_dir=self.documents_dir,
                subset_path=self.subset_path,
                pdf_path=self.pdf_path,
            )
            _log.debug(
                f"Retrieval results for query '{query}' (company: {company_name}): {len(results)} pages"
            )
            return self.reranker.send_message_for_reranking(
                query=query, documents=results, top_n=self.top_n_retrieval
            )

        results = retriever.milvus_retrieve_by_company_name(
            company_name=company_name,
            query=query,
            top_n=self.top_n_retrieval,
            return_parent_pages=self.return_parent_pages,
            documents_dir=self.documents_dir,
            subset_path=self.subset_path,
            pdf_path=self.pdf_path,
        )
        # print(f"[DEBUG] Retrieval results for query '{query}' (company: {company_name}): {len(results)} pages")
        return results

    def get_answer_for_company(
        self, company_name: str, question: str, schema: str
    ) -> dict:
        """
        获取公司相关问题的答案

        Args:
            company_name: 公司名称
            question: 原始问题
            schema: 答案模式

        Returns:
            包含答案的字典
        """
        # 生成查询变体
        queries = [question]  # 始终包含原始查询
        if self.use_query_rewrite:
            rewritten_queries = self.query_rewriter.rewrite(
                query=question, max_variations=self.max_query_variations
            )
            queries.extend(rewritten_queries)

        if self.use_query_rewrite:
            _log.debug(f"Rewritten queries: {rewritten_queries}")

        # 获取所有查询的检索结果
        all_results = []
        for query in queries:
            results = self._get_retrieval_results(company_name, query)
            all_results.append(results)

        # 合并检索结果
        retrieval_results = self._merge_retrieval_results(
            all_results, self.top_n_retrieval
        )

        # print(f"[DEBUG] Total retrieval results after merging: {len(retrieval_results)}")

        if not retrieval_results:
            _log.error(
                f"No relevant context found for company '{company_name}' and query '{question}'"
            )
            raise ValueError("No relevant context found")

        # 格式化检索结果并获取答案
        rag_context = self._format_retrieval_results(retrieval_results)
        answer_dict = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=rag_context,
            schema=schema,
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        # 处理页面引用
        pages = answer_dict.get("relevant_pages", [])
        validated_pages = self._validate_page_references(pages, retrieval_results)
        answer_dict["relevant_pages"] = validated_pages
        answer_dict["references"] = self._extract_references(
            validated_pages, company_name
        )

        return answer_dict

    def _extract_companies_from_subset(self, question_text: str) -> list[str]:
        """Extract company names from a question by matching against companies in the subset file."""
        if not hasattr(self, "companies_df"):
            if self.subset_path is None:
                raise ValueError(
                    "subset_path must be provided to use subset extraction"
                )
            self.companies_df = pd.read_csv(self.subset_path)

        found_companies = []
        company_names = sorted(
            self.companies_df["company_name"].unique(), key=len, reverse=True
        )

        for company in company_names:
            escaped_company = re.escape(company)

            pattern = rf"{escaped_company}(?:\W|$)"

            if re.search(pattern, question_text, re.IGNORECASE):
                found_companies.append(company)
                question_text = re.sub(pattern, "", question_text, flags=re.IGNORECASE)

        # print(f"[DEBUG] Matched company from question '{question_text}': {found_companies}")
        return found_companies

    def process_question(self, question: str, schema: str):

        extracted_companies = self._extract_companies_from_subset(question)

        # print("question", question)
        # print("extracted_companies", extracted_companies)
        if len(extracted_companies) == 0:
            raise ValueError("No company name found in the question.")

        if len(extracted_companies) == 1:
            company_name = extracted_companies[0]
            answer_dict = self.get_answer_for_company(
                company_name=company_name, question=question, schema=schema
            )
            return answer_dict
        else:
            if self.parallel_requests <= 1:
                return self.process_comparative_question(
                    question, extracted_companies, schema
                )
            else:
                return self.process_comparative_question_parallel(
                    question, extracted_companies, schema
                )

    def _create_answer_detail_ref(self, answer_dict: dict, question_index: int) -> str:
        """Create a reference ID for answer details and store the details"""
        ref_id = f"#/answer_details/{question_index}"
        with self._lock:
            self.answer_details[question_index] = {
                "step_by_step_analysis": answer_dict["step_by_step_analysis"],
                "reasoning_summary": answer_dict["reasoning_summary"],
                "relevant_pages": answer_dict["relevant_pages"],
                "response_data": self.response_data,
                "self": ref_id,
            }
        return ref_id

    def _calculate_statistics(
        self, processed_questions: List[dict], print_stats: bool = False
    ) -> dict:
        """Calculate statistics about processed questions."""
        total_questions = len(processed_questions)
        error_count = sum(1 for q in processed_questions if "error" in q)
        na_count = sum(
            1
            for q in processed_questions
            if (q.get("value") if "value" in q else q.get("answer")) == "N/A"
        )
        success_count = total_questions - error_count - na_count
        if print_stats:
            print(f"\nFinal Processing Statistics:")
            print(f"Total questions: {total_questions}")
            print(f"Errors: {error_count} ({(error_count/total_questions)*100:.1f}%)")
            print(f"N/A answers: {na_count} ({(na_count/total_questions)*100:.1f}%)")
            print(
                f"Successfully answered: {success_count} ({(success_count/total_questions)*100:.1f}%)\n"
            )

        return {
            "total_questions": total_questions,
            "error_count": error_count,
            "na_count": na_count,
            "success_count": success_count,
        }

    def process_questions_list(
        self,
        questions_list: List[dict],
        output_path: str = None,
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ) -> dict:
        # print("questions_list", questions_list)
        total_questions = len(questions_list)
        # Add index to each question so we know where to write the answer details
        questions_with_index = [
            {**q, "_question_index": i} for i, q in enumerate(questions_list)
        ]
        self.answer_details = [
            None
        ] * total_questions  # Preallocate list for answer details
        processed_questions = []
        parallel_threads = self.parallel_requests

        if parallel_threads <= 1:
            for question_data in tqdm(
                questions_with_index, desc="Processing questions"
            ):
                print("single question", question_data)
                processed_question = self._process_single_question(question_data)
                processed_questions.append(processed_question)
                if output_path:
                    self._save_progress(
                        processed_questions,
                        output_path,
                        submission_file=submission_file,
                        team_email=team_email,
                        submission_name=submission_name,
                        pipeline_details=pipeline_details,
                    )
        else:
            print("parallel processing")
            with tqdm(total=total_questions, desc="Processing questions") as pbar:
                for i in range(0, total_questions, parallel_threads):
                    batch = questions_with_index[i : i + parallel_threads]
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=parallel_threads
                    ) as executor:
                        # executor.map will return results in the same order as the input list.
                        batch_results = list(
                            executor.map(self._process_single_question, batch)
                        )
                    processed_questions.extend(batch_results)

                    if output_path:
                        self._save_progress(
                            processed_questions,
                            output_path,
                            submission_file=submission_file,
                            team_email=team_email,
                            submission_name=submission_name,
                            pipeline_details=pipeline_details,
                        )
                    pbar.update(len(batch_results))

        statistics = self._calculate_statistics(processed_questions, print_stats=True)

        return {
            "questions": processed_questions,
            "answer_details": self.answer_details,
            "statistics": statistics,
        }

    def _process_single_question(self, question_data: dict) -> dict:
        question_index = question_data.get("_question_index", 0)
        # print("question_data", question_data)
        # print("question_index", question_index)
        # print("--------------------------------")
        question_text = question_data.get("text")
        schema = question_data.get("kind")

        try:
            answer_dict = self.process_question(question_text, schema)

            if "error" in answer_dict:
                detail_ref = self._create_answer_detail_ref(
                    {
                        "step_by_step_analysis": None,
                        "reasoning_summary": None,
                        "relevant_pages": None,
                    },
                    question_index,
                )
                return {
                    "question_text": question_text,
                    "kind": schema,
                    "value": None,
                    "references": [],
                    "error": answer_dict["error"],
                    "answer_details": {"$ref": detail_ref},
                }
            detail_ref = self._create_answer_detail_ref(answer_dict, question_index)
            return {
                "question_text": question_text,
                "kind": schema,
                "value": answer_dict.get("final_answer"),
                "references": answer_dict.get("references", []),
                "answer_details": {"$ref": detail_ref},
            }
        except Exception as err:
            return self._handle_processing_error(
                question_text, schema, err, question_index
            )

    def _handle_processing_error(
        self, question_text: str, schema: str, err: Exception, question_index: int
    ) -> dict:
        """
        Handle errors during question processing.
        Log error details and return a dictionary containing error information.
        """
        import traceback

        error_message = str(err)
        tb = traceback.format_exc()
        error_ref = f"#/answer_details/{question_index}"
        error_detail = {"error_traceback": tb, "self": error_ref}

        with self._lock:
            self.answer_details[question_index] = error_detail

        print(f"Error encountered processing question: {question_text}")
        print(f"Error type: {type(err).__name__}")
        print(f"Error message: {error_message}")
        print(f"Full traceback:\n{tb}\n")

        return {
            "question_text": question_text,
            "kind": schema,
            "value": None,
            "references": [],
            "error": f"{type(err).__name__}: {error_message}",
            "answer_details": {"$ref": error_ref},
        }

    def _post_process_submission_answers(
        self, processed_questions: List[dict]
    ) -> List[dict]:
        """
        Post-process answers for submission format:
        1. Convert page indices from one-based to zero-based
        2. Clear references for N/A answers
        3. Format answers according to submission schema
        4. Include step_by_step_analysis from answer details
        """
        submission_answers = []

        for q in processed_questions:
            question_text = q.get("question_text") or q.get("question")
            kind = q.get("kind") or q.get("schema")
            value = (
                "N/A"
                if "error" in q
                else (q.get("value") if "value" in q else q.get("answer"))
            )
            references = q.get("references", [])

            answer_details_ref = q.get("answer_details", {}).get("$ref", "")
            step_by_step_analysis = None
            if answer_details_ref and answer_details_ref.startswith(
                "#/answer_details/"
            ):
                try:
                    index = int(answer_details_ref.split("/")[-1])
                    if (
                        0 <= index < len(self.answer_details)
                        and self.answer_details[index]
                    ):
                        step_by_step_analysis = self.answer_details[index].get(
                            "step_by_step_analysis"
                        )
                except (ValueError, IndexError):
                    pass

            # Clear references if value is N/A
            if value == "N/A":
                references = []
            else:
                # Convert page indices from one-based to zero-based (competition requires 0-based page indices, but for debugging it is easier to use 1-based)
                references = [
                    {
                        "pdf_sha1": ref["pdf_sha1"],
                        "page_index": (
                            (ref["page_index"] - 1)
                            if ref["page_index"] is not None
                            else 0
                        ),
                    }
                    for ref in references
                    if ref.get("page_index") is not None  # 过滤掉page_index为None的引用
                ]

            submission_answer = {
                "question_text": question_text,
                "kind": kind,
                "value": value,
                "references": references,
            }

            if step_by_step_analysis:
                submission_answer["reasoning_process"] = step_by_step_analysis

            submission_answers.append(submission_answer)

        return submission_answers

    def _save_progress(
        self,
        processed_questions: List[dict],
        output_path: Optional[str],
        submission_file: bool = False,
        team_email: str = "",
        submission_name: str = "",
        pipeline_details: str = "",
    ):
        if output_path:
            statistics = self._calculate_statistics(processed_questions)

            # Prepare debug content
            result = {
                "questions": processed_questions,
                "answer_details": self.answer_details,
                "statistics": statistics,
            }
            output_file = Path(output_path)
            debug_file = output_file.with_name(
                output_file.stem + "_debug" + output_file.suffix
            )
            with open(debug_file, "w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)

            if submission_file:
                # Post-process answers for submission
                submission_answers = self._post_process_submission_answers(
                    processed_questions
                )
                submission = {
                    "answers": submission_answers,
                    "team_email": team_email,
                    "submission_name": submission_name,
                    "details": pipeline_details,
                    "time": time.strftime(
                        "%Y-%m-%d, %H:%M:%S", time.localtime(time.time())
                    ),
                    "signature": hex(int(time.time())),
                    "tsp_signature": "",
                    "submission_digest": "",
                }
                with open(output_file, "w", encoding="utf-8") as file:
                    json.dump(submission, file, ensure_ascii=False, indent=2)

    def process_all_questions(
        self,
        output_path: str = "questions_with_answers.json",
        team_email: str = "79250515615@yandex.com",
        submission_name: str = "Ilia_Ris SO CoT + Parent Document Retrieval",
        submission_file: bool = False,
        pipeline_details: str = "",
    ):
        # print("self.questions", self.questions)
        result = self.process_questions_list(
            self.questions,
            output_path,
            submission_file=submission_file,
            team_email=team_email,
            submission_name=submission_name,
            pipeline_details=pipeline_details,
        )
        return result

    def process_comparative_question_parallel(
        self, question: str, companies: List[str], schema: str
    ) -> dict:
        """
        Process a question involving multiple companies in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            original_question=question, companies=companies
        )

        individual_answers = {}
        aggregated_references = []

        # Step 2: Process each individual question in parallel
        def process_company_question(company: str) -> tuple[str, dict]:
            """Helper function to process one company's question and return (company, answer)"""
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(
                    f"Could not generate sub-question for company: {company}"
                )

            answer_dict = self.get_answer_for_company(
                company_name=company, question=sub_question, schema="number"
            )
            return company, answer_dict

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_company = {
                executor.submit(process_company_question, company): company
                for company in companies
            }

            for future in concurrent.futures.as_completed(future_to_company):
                try:
                    company, answer_dict = future.result()
                    individual_answers[company] = answer_dict

                    company_references = answer_dict.get("references", [])
                    aggregated_references.extend(company_references)
                except Exception as e:
                    company = future_to_company[future]
                    print(f"Error processing company {company}: {str(e)}")
                    raise

        # Remove duplicate references
        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())

        # Step 3: Get the comparative answer using all individual answers
        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        comparative_answer["references"] = aggregated_references
        return comparative_answer

    def process_comparative_question(
        self, question: str, companies: List[str], schema: str
    ) -> dict:
        """
        Process a question involving multiple companies in parallel:
        1. Rephrase the comparative question into individual questions
        2. Process each individual question using parallel threads
        3. Combine results into final comparative answer
        """
        # Step 1: Rephrase the comparative question
        rephrased_questions = self.openai_processor.get_rephrased_questions(
            model=self.answering_model, original_question=question, companies=companies
        )

        individual_answers = {}
        aggregated_references = []

        for company in companies:
            sub_question = rephrased_questions.get(company)
            if not sub_question:
                raise ValueError(
                    f"Could not generate sub-question for company: {company}"
                )

            answer_dict = self.get_answer_for_company(
                company_name=company, question=sub_question, schema="number"
            )

            individual_answers[company] = answer_dict

            company_references = answer_dict.get("references", [])
            aggregated_references.extend(company_references)

        unique_refs = {}
        for ref in aggregated_references:
            key = (ref.get("pdf_sha1"), ref.get("page_index"))
            unique_refs[key] = ref
        aggregated_references = list(unique_refs.values())

        comparative_answer = self.openai_processor.get_answer_from_rag_context(
            question=question,
            rag_context=individual_answers,
            schema="comparative",
            model=self.answering_model,
        )
        self.response_data = self.openai_processor.response_data

        comparative_answer["references"] = aggregated_references

        return comparative_answer
