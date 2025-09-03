from typing import Any, List, Tuple
import numpy as np

from pymilvus import connections, Collection

from src.deepwriter.models.gme_embeddings import GMEmbeddingModel
from src.deepwriter.database.base_db import BaseDB
from src.deepwriter.database.document import Document, FlattenBlock


class MilvusDB(BaseDB):
    OUTPUT_FIELDS = [
        "block_id",
        "block_bbox",
        "pdf_path",
        "num_pages",
        "page_number",
        "page_height",
        "page_width",
        "num_blocks",
        "block_type",
        "block_content",
        "block_summary",
        "image_path",
        "image_caption",
        "image_footer",
        "document_title",
        # "block_embedding",
    ]

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        image_collection_name: str = "deepwriter_image",
        text_collection_name: str = "deepwriter_text",
        text_embedding_model: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        multimodal_embedding_model: str = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    ) -> None:
        connections.connect(host=host, port=port, user=user, password=password)
        self.image_collection = Collection(name=image_collection_name)
        self.text_collection = Collection(name=text_collection_name)
        self.image_collection.load()
        self.text_collection.load()
        self.text_embedding_model = GMEmbeddingModel(text_embedding_model)
        if multimodal_embedding_model == text_embedding_model:
            self.multimodal_embedding_model = self.text_embedding_model
        else:
            self.multimodal_embedding_model = GMEmbeddingModel(
                multimodal_embedding_model
            )

    def retrieve_image_blocks(
        self, query: str, top_n: int = 10, param: dict = None, **kwargs
    ) -> List[FlattenBlock]:
        query_embedding = self.text_embedding_model.encode_query(query).numpy()
        query_embedding = np.float64(query_embedding)
        image_results = self.image_collection.search(
            data=[query_embedding],
            anns_field="block_embedding",
            param=param,
            limit=top_n,
            output_fields=self.OUTPUT_FIELDS,
            **kwargs,
        )
        result = [
            FlattenBlock.from_milvus_search_result(hit)
            for hits in iter(image_results)
            for hit in hits
        ]
        return result

    def retrieve_text_blocks(
        self, query: str, top_n: int = 10, param: dict = None, **kwargs
    ) -> List[FlattenBlock]:
        query_embedding = self.text_embedding_model.encode_query(query).numpy()
        query_embedding = np.float64(query_embedding)
        text_results = self.text_collection.search(
            data=[query_embedding],
            anns_field="block_embedding",
            param=param,
            limit=top_n,
            output_fields=self.OUTPUT_FIELDS,
            **kwargs,
        )
        result = [
            FlattenBlock.from_milvus_search_result(hit)
            for hits in iter(text_results)
            for hit in hits
        ]
        return result

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector database.

        Args:
            documents: List of Document objects to add
        """
        raise NotImplementedError("MilvusDB does not support adding documents For now.")
