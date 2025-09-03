from typing import List
import torch


class BaseEmbeddingModel:
    """Base class for embedding models."""

    def encode_query(self, query: str) -> torch.Tensor:
        """Encode a query string into an embedding vector."""
        raise NotImplementedError

    def encode_documents(self, documents: List[str], **kwargs) -> torch.Tensor:
        """Encode a list of document texts into embedding vectors."""
        raise NotImplementedError

    def similarity_score(
        self, query_embedding: torch.Tensor, doc_embedding: torch.Tensor
    ) -> float:
        """Calculate similarity score between query and document embeddings."""
        raise NotImplementedError
