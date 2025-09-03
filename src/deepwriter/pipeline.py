import json
import os
from typing import Any, Dict, List, Union

import pandas as pd
from loguru import logger

from src.deepwriter.config.prompts import (
    CLUSTER_PROMPT,
    CONTENT_PROMPT,
    QUERY_DECOMPOSITION_PROMPT,
    REWRITE_PROMPT,
    SECTION_DRAFT_PROMPT,
    SECTION_TITLE_GENERATION_PROMPT,
    SUMMARIZE_PROMPT,
)
from src.deepwriter.database.document import FlattenBlock
from src.deepwriter.finders.doc_finder import DocFinder
from src.deepwriter.llms.litellm_wrapper import LitellmWrapper

# GME embedding model, add this near the top of the file
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class DeepWriter:
    def __init__(self, llm: str, base_url: str, doc_finder: DocFinder) -> None:
        self.llm = LitellmWrapper(model=llm, base_url=base_url)
        self.doc_finder = doc_finder

    def rewrite_query(self, query: str) -> str:
        message = REWRITE_PROMPT.format(query=query)
        return self.llm.generate_response(query=message)

    def decompose_query(self, rewritten_query: str) -> List[str]:
        message = QUERY_DECOMPOSITION_PROMPT.format(query=rewritten_query)
        response = self.llm.generate_response(query=message)
        logger.debug(f"Decomposed query: {response}")
        decomposed_queries = []
        for line in response.strip().split("\n"):
            # Check if line starts with a number followed by a period or parenthesis
            if line and (
                line[0].isdigit()
                or (len(line) > 1 and line[0].isdigit() and line[1] in [".", ")"])
            ):
                # Extract the query part (remove the numbering)
                query_text = (
                    line.split(".", 1)[-1].strip()
                    if "." in line[:3]
                    else line.split(")", 1)[-1].strip()
                )
                decomposed_queries.append(query_text)
        return decomposed_queries

    def find_relevant_docs(
        self, query: Union[str, List[str]], **kwargs
    ) -> List[FlattenBlock]:
        if isinstance(query, List):
            relevant_docs = []
            for q in query:
                relevant_docs.extend(self.doc_finder.find_relevant_docs(q, **kwargs))
            return relevant_docs
        else:
            return self.doc_finder.find_relevant_docs(query, **kwargs)

    def rerank_and_aggregate(self, docs: pd.DataFrame) -> pd.DataFrame:
        return docs

    def generate_image_docs_summaries(
        self, rewritten_query: str, matching_results: List[FlattenBlock]
    ) -> None:
        for result in matching_results:
            result.block_summary = result.image_caption

    def generate_text_docs_summaries(
        self, rewritten_query: str, matching_results: List[FlattenBlock]
    ) -> None:
        messages = [
            SUMMARIZE_PROMPT.format(query=rewritten_query, doc=result.block_content)
            for result in matching_results
        ]
        per_doc_summaries = []
        for message in messages:
            per_doc_summaries.append(self.llm.generate_response(query=message))
        for result, summary in zip(matching_results, per_doc_summaries, strict=False):
            result.block_summary = summary

    def filter_docs_with_summaries(
        self, matching_results: List[FlattenBlock]
    ) -> List[FlattenBlock]:
        """Filter out documents with None or empty summaries.

        Args:
            matching_results: List of FlattenBlock objects

        Returns:
            List of filtered FlattenBlock objects
        """
        filtered_results = [
            chunk for chunk in matching_results if chunk.block_summary is not None
        ]
        return filtered_results

    def generate_per_doc_summaries(
        self, rewritten_query: str, matching_results: List[FlattenBlock]
    ) -> None:
        self.generate_image_docs_summaries(rewritten_query, matching_results)
        self.generate_text_docs_summaries(rewritten_query, matching_results)

    def generate_section_titles(self, rewritten_query: str) -> List[str]:
        message = SECTION_TITLE_GENERATION_PROMPT.format(query=rewritten_query)
        section_titles = self.llm.generate_response(query=message)
        return section_titles.split("\n")

    def cluster_docs(
        self,
        rewritten_query: str,
        matching_results: List[FlattenBlock],
        section_titles: List[str],
    ) -> Dict[str, List[FlattenBlock]]:
        # TODO: let the model decide the clusters themselves
        clustered_docs: Dict[str, List[FlattenBlock]] = {
            section_title.lower().capitalize(): [] for section_title in section_titles
        }

        messages = [
            CLUSTER_PROMPT.format(
                query=rewritten_query,
                doc=doc.block_summary,
                sections="; ".join(list(clustered_docs.keys())).upper(),
            )
            for doc in matching_results
        ]
        for doc, message in zip(matching_results, messages, strict=False):
            response = self.llm.generate_response(query=message)
            if response not in clustered_docs:
                response = response.lower().capitalize()
                if response not in clustered_docs:
                    continue
                else:
                    clustered_docs[response].append(doc)
            else:
                clustered_docs[response].append(doc)

        return clustered_docs

    def generate_sections_drafts(
        self, rewritten_query: str, clustered_docs: Dict[str, List[FlattenBlock]]
    ) -> List[Any]:
        sections_drafts = []
        for section_title, relevant_docs in clustered_docs.items():
            message = SECTION_DRAFT_PROMPT.format(
                query=rewritten_query,
                section_title=section_title,
                relevant_docs=relevant_docs,
            )
            section_draft = self.llm.generate_response(query=message)
            sections_drafts.append(
                {
                    "section_title": section_title,
                    "section_draft": section_draft,
                }
            )
            logger.debug(f"Section {section_title}")
            logger.debug(f"  - Draft: {section_draft}")
        return sections_drafts

    def generate_content(
        self,
        query: str,
        citations: List[FlattenBlock],
        section_draft: str,
        section_title: str,
        already_written: str,
    ) -> str:
        message = CONTENT_PROMPT.format(
            query=query,
            relevant_docs=citations,
            section_draft=section_draft,
            section_title=section_title,
            already_written=already_written,
        )

        return self.llm.generate_response(query=message)

    def summarize_content(self, query: str, content: str):
        message = SUMMARIZE_PROMPT.format(query=query, doc=content)
        return self.llm.generate_response(query=message)

    def generate_sections_content(
        self,
        rewritten_query: str,
        clustered_docs: Dict[str, List[FlattenBlock]],
        sections_drafts: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        existed_sections: List[Dict[str, Any]] = []
        for section in sections_drafts:
            section_title = section["section_title"]
            section_draft = section["section_draft"]
            citations = clustered_docs[section_title]
            already_written = (
                existed_sections[-1]["already_written"]
                if len(existed_sections) > 0
                else ""
            )

            logger.debug(f"Generating section content for: {section_title}")
            section_content = self.generate_content(
                query=rewritten_query,
                citations=citations,
                section_draft=section_draft,
                section_title=section_title,
                already_written=already_written,
            )
            logger.debug(f"Section content: {section_content}")

            # add related images to the section
            for doc in citations:
                if doc.block_type == "image" or doc.block_type == "table":
                    image_ref = f"\n\n![{doc.image_caption}]({doc.image_path})\n\n"
                else:
                    continue
                logger.debug(f"Adding image reference: {image_ref}")
                section_content += image_ref

            logger.debug(f"Summarizing section content for: {section_title}")
            already_written = self.summarize_content(rewritten_query, section_content)
            logger.debug(f"Already written: {already_written}")
            existed_sections.append(
                {
                    "section_title": section_title,
                    "content": section_content,
                    "section_draft": section_draft,
                    "already_written": already_written,
                    "citations": citations,
                }
            )
        return existed_sections

    def generate_formatted_report(self, sections: List[Dict[str, Any]]) -> str:
        report = ""
        for section in sections:
            report += f"# {section['section_title']}\n\n"
            report += f"{section['content']}\n\n"
            report += "## References\n\n"
            for citation in section["citations"]:
                json_data = {
                    "block_bbox": citation.block_bbox,
                    "block_type": citation.block_type,
                    "pdf_path": citation.pdf_path,
                    "page_number": citation.page_number,
                }
                if citation.block_type in ["image", "table"]:
                    json_data["image_path"] = citation.image_path
                report += f"```json\n{json.dumps(json_data)}\n```\n\n"
        return report

    def generate_report(self, query: str, **kwargs) -> str:
        logger.info(f"Starting: generate report for query {query}")
        # 1. rewrite query
        logger.info("State: rewriting query.")
        rewritten_query = self.rewrite_query(query=query)
        logger.info(f"rewritten query: {rewritten_query}")

        # 2. decompose query
        logger.info("State: decomposing query.")
        decomposed_queries = self.decompose_query(rewritten_query=rewritten_query)
        logger.info(f"Decomposed queries: {decomposed_queries}")

        # 2. find relevant docs
        logger.info("State: finding relevant documents.")
        matching_results = self.find_relevant_docs(query=decomposed_queries, **kwargs)
        logger.info(f"Found {len(matching_results)} relevant documents.")

        # 3. generate per doc summaries
        logger.info("State: generating per document summaries.")
        self.generate_per_doc_summaries(
            rewritten_query=rewritten_query, matching_results=matching_results
        )
        logger.info(f"Generated {len(matching_results)} document summaries.")

        # 4. filter out docs with empty summaries
        logger.info("State: filtering out documents with empty summaries.")
        filtered_matching_results = self.filter_docs_with_summaries(
            matching_results=matching_results
        )

        # 5. generate section titles
        logger.info("State: generating section titles.")
        section_titles = self.generate_section_titles(rewritten_query=rewritten_query)
        logger.info(f"Generated {len(section_titles)} section titles.")
        for section_title in section_titles:
            logger.info(f"Section title: {section_title}")

        # 6. cluster docs according to per doc summaries and section titles
        logger.info("State: clustering documents based on summaries.")
        clustered_docs = self.cluster_docs(
            rewritten_query=rewritten_query,
            matching_results=filtered_matching_results,
            section_titles=section_titles,
        )
        logger.info(f"Created {len(clustered_docs)} document clusters.")

        # 6. generate sections drafts
        logger.info("State: generating sections drafts.")
        sections_drafts = self.generate_sections_drafts(
            rewritten_query=rewritten_query, clustered_docs=clustered_docs
        )
        logger.info(f"Generated plan with {len(sections_drafts)} sections.")

        # 7. generate sections
        logger.info("State: generating section content.")
        sections = self.generate_sections_content(
            rewritten_query=rewritten_query,
            clustered_docs=clustered_docs,
            sections_drafts=sections_drafts,
        )
        logger.info(f"Generated {len(sections)} sections of content.")

        # 8. generate formatted report
        logger.info("State: formatting final report.")
        report = self.generate_formatted_report(sections=sections)

        logger.info("Report generation completed successfully.")

        return report
