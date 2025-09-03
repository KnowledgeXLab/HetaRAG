from typing import List
from loguru import logger

from src.deepwriter.database.base_db import BaseDB
from src.deepwriter.database.document import FlattenBlock


class DocFinder:

    def __init__(
        self,
        database: BaseDB,
        context_threshold: float,
        n_rerank: int,
    ) -> None:
        self.database = database

        self.context_threshold = context_threshold

        self.context_threshold = context_threshold
        self.n_rerank = n_rerank

    def retrieve_image_passages(self, query: str, **kwargs) -> List[FlattenBlock]:
        if "param" in kwargs:  # for Milvus
            param = kwargs["param"]
            del kwargs["param"]

            matching_results = self.database.retrieve_image_blocks(
                query=query, top_n=self.n_rerank, param=param, **kwargs
            )
            return matching_results

        matching_results = self.database.retrieve_image_blocks(
            query=query, top_n=self.n_rerank, **kwargs
        )
        return matching_results

    def retrieve_text_passages(self, query: str, **kwargs) -> List[FlattenBlock]:
        if "param" in kwargs:  # for Milvus
            param = kwargs["param"]
            del kwargs["param"]

            matching_texts = self.database.retrieve_text_blocks(
                query=query, top_n=self.n_rerank, param=param, **kwargs
            )
            return matching_texts

        matching_texts = self.database.retrieve_text_blocks(
            query=query, top_n=self.n_rerank, **kwargs
        )
        return matching_texts

    def rerank(
        self,
        query: str,
        matching_image_results: List[FlattenBlock],
        matching_text_results: List[FlattenBlock],
    ) -> List[FlattenBlock]:
        # TODO: add reranking
        return matching_image_results + matching_text_results

    def find_relevant_docs(self, query: str, **kwargs) -> List[FlattenBlock]:
        # TODO: pass it with config
        matching_image_results = self.retrieve_image_passages(query, **kwargs)
        matching_text_results = self.retrieve_text_passages(query, **kwargs)
        reranked_results = self.rerank(
            query, matching_image_results, matching_text_results
        )

        logger.info(f"Reranked results: {len(reranked_results)}")
        for value in reranked_results:
            logger.debug(
                f"Score: {value.cosine_similarity}\nReranked text: {value.block_content}"
            )

        return reranked_results
