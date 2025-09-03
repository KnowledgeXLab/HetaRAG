"""Base vector database class for document storage, retrieval, and reranking."""

from typing import Any, Dict, List, Optional
import numpy as np
from abc import ABC, abstractmethod

from src.deepwriter.database.document import Document, FlattenBlock


class BaseDB(ABC):
    """Base vector database class with store, retrieve and rerank capabilities."""

    def __init__(self, vector_db: Any, embedding_model: Any) -> None:
        """Initialize the base database.

        Args:
            vector_db: The underlying vector database
            embedding_model: Model for generating embeddings
        """
        self.vector_db = vector_db
        self.embedding_model = embedding_model

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector database.

        Args:
            documents: List of Document objects to add
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_image_blocks(
        self, query: str, top_n: int = 5, **kwargs
    ) -> List[FlattenBlock]:
        """Retrieve documents similar to the query.

        Args:
            query: The search query
            top_n: Number of documents to retrieve

        Returns:
            List of retrieved FlattenBlock objects
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def retrieve_text_blocks(
        self, query: str, top_n: int = 5, **kwargs
    ) -> List[FlattenBlock]:
        """Retrieve documents similar to the query.

        Args:
            query: The search query
            top_n: Number of documents to retrieve

        Returns:
            List of retrieved FlattenBlock objects
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def rerank(
        self, query: str, documents: List[FlattenBlock], top_n: int = 5
    ) -> List[FlattenBlock]:
        """Rerank retrieved documents based on relevance to query.

        Args:
            query: The search query
            documents: List of documents to rerank
            top_n: Number of documents to return after reranking

        Returns:
            List of reranked FlattenBlock objects
        """
        # Default implementation uses cosine similarity
        query_embedding = self._get_embedding(query)

        scores = []
        for doc in documents:
            if doc.embedding is None:
                doc.embedding = self._get_embedding(doc.text)

            score = self._calculate_similarity(query_embedding, doc.embedding)
            scores.append((score, doc))

        # Sort by score in descending order
        reranked_docs = [
            doc for _, doc in sorted(scores, key=lambda x: x[0], reverse=True)
        ]
        return reranked_docs[:top_n]

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embedding_model.get_text_embeddings([text]).tolist()

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        return dot_product / (norm1 * norm2)
