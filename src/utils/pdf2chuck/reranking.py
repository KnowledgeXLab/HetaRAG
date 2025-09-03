import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.utils.pdf2chuck.prompts as prompts
from concurrent.futures import ThreadPoolExecutor
import json


class JinaReranker:
    def __init__(self):
        self.url = "https://api.jina.ai/v1/rerank"
        self.headers = self.get_headers()

    def get_headers(self):
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jina_api_key}",
        }
        return headers

    def rerank(self, query, documents, top_n=10):
        data = {
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "top_n": top_n,
            "documents": documents,
        }

        response = requests.post(url=self.url, headers=self.headers, json=data)

        return response.json()


class LLMReranker:
    def __init__(self):
        self.llm = self.set_up_llm()
        self.system_prompt_rerank_single_block = (
            prompts.RerankingPrompt.system_prompt_rerank_single_block
        )
        self.system_prompt_rerank_multiple_blocks = (
            prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
        )
        self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
        self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks

    def set_up_llm(self):
        import ollama

        load_dotenv()
        # ## 使用智谱的
        # llm = OpenAI(api_key="431199a4f20d7c47ad30b3b34d09a650.4qr55Y6UgqFsAult",
        #              base_url="https://open.bigmodel.cn/api/paas/v4"
        # )
        ## 使用ollama的
        llm = ollama.Client(host="http://10.1.48.15:8009/")
        return llm

    def get_rank_for_single_block(self, query, retrieved_document):
        user_prompt = f'/nHere is the query:/n"{query}"/n/nHere is the retrieved text block:/n"""/n{retrieved_document}/n"""/n'

        completion = self.llm.chat(
            model="qwen2.5:72b",
            messages=[
                {"role": "system", "content": self.system_prompt_rerank_single_block},
                {"role": "user", "content": user_prompt},
            ],
            format=self.schema_for_single_block.model_json_schema(),
            options={"temperature": 0},
        )

        response = completion["message"]["content"]
        response_dict = json.loads(response)

        return response_dict

    def get_rank_for_multiple_blocks(self, query, retrieved_documents):
        formatted_blocks = "\n\n---\n\n".join(
            [
                f'Block {i+1}:\n\n"""\n{text}\n"""'
                for i, text in enumerate(retrieved_documents)
            ]
        )
        user_prompt = (
            f'Here is the query: "{query}"\n\n'
            "Here are the retrieved text blocks:\n"
            f"{formatted_blocks}\n\n"
            f"You should provide exactly {len(retrieved_documents)} rankings, in order.\n"
            "Your response must be a valid JSON object with the following structure:\n"
            "{\n"
            '  "block_rankings": [\n'
            "    {\n"
            '      "reasoning": "Your analysis of the block",\n'
            '      "relevance_score": 0.0\n'
            "    },\n"
            "    ...\n"
            "  ]\n"
            "}"
        )

        completion = self.llm.chat(
            model="qwen2.5:72b",
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt_rerank_multiple_blocks,
                },
                {"role": "user", "content": user_prompt},
            ],
            format=self.schema_for_multiple_blocks.model_json_schema(),
            options={"temperature": 0},
        )

        response = completion["message"]["content"]
        response_dict = json.loads(response)

        return response_dict

    def rerank_documents(
        self,
        query: str,
        documents: list,
        documents_batch_size: int = 4,
        llm_weight: float = 0.7,
    ):
        """
        Rerank multiple documents using parallel processing with threading.
        Combines vector similarity and LLM relevance scores using weighted average.
        """
        # Create batches of documents
        doc_batches = [
            documents[i : i + documents_batch_size]
            for i in range(0, len(documents), documents_batch_size)
        ]
        vector_weight = 1 - llm_weight

        if documents_batch_size == 1:

            def process_single_doc(doc):
                # Get ranking for single document
                ranking = self.get_rank_for_single_block(query, doc["text"])

                doc_with_score = doc.copy()
                doc_with_score["relevance_score"] = ranking["relevance_score"]
                # Calculate combined score - note that distance is inverted since lower is better
                doc_with_score["combined_score"] = round(
                    llm_weight * ranking["relevance_score"]
                    + vector_weight * doc["distance"],
                    4,
                )
                return doc_with_score

            # Process all documents in parallel using single-block method
            with ThreadPoolExecutor() as executor:
                all_results = list(executor.map(process_single_doc, documents))

        else:

            def process_batch(batch):
                texts = [doc["text"] for doc in batch]
                rankings = self.get_rank_for_multiple_blocks(query, texts)
                results = []
                block_rankings = rankings.get("block_rankings", [])

                if len(block_rankings) < len(batch):
                    print(
                        f"\nWarning: Expected {len(batch)} rankings but got {len(block_rankings)}"
                    )
                    for i in range(len(block_rankings), len(batch)):
                        doc = batch[i]
                        print(
                            f"Missing ranking for document on page {doc.get('page', 'unknown')}:"
                        )
                        print(f"Text preview: {doc['text'][:100]}...\n")

                    for _ in range(len(batch) - len(block_rankings)):
                        block_rankings.append(
                            {
                                "relevance_score": 0.0,
                                "reasoning": "Default ranking due to missing LLM response",
                            }
                        )

                for doc, rank in zip(batch, block_rankings):
                    doc_with_score = doc.copy()
                    doc_with_score["relevance_score"] = rank["relevance_score"]
                    doc_with_score["combined_score"] = round(
                        llm_weight * rank["relevance_score"]
                        + vector_weight * doc["distance"],
                        4,
                    )
                    results.append(doc_with_score)
                return results

            # Process batches in parallel using threads
            with ThreadPoolExecutor() as executor:
                batch_results = list(executor.map(process_batch, doc_batches))

            # Flatten results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)

        # Sort results by combined score in descending order
        all_results.sort(key=lambda x: x["combined_score"], reverse=True)
        return all_results
