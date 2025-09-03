"""Vector database operations for document retrieval."""

from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
import torch
import pandas as pd

from src.deepwriter.database.base_db import BaseDB
from src.deepwriter.database.document import (
    Document,
    FlattenBlock,
    BlockType,
)
from src.deepwriter.models.gme_embeddings import GMEmbeddingModel


class VectorDB(BaseDB):
    """Vector database implementation for document retrieval."""

    def __init__(
        self,
        text_embedding_model: Any,
        multimodal_embedding_model: Any,
        batch_size: int = 16,
    ) -> None:
        """Initialize the vector database.

        Args:
            text_embedding_model: Model for generating text embeddings
            multimodal_embedding_model: Model for generating image and table embeddings
            batch_size: Batch size for processing
        """
        super().__init__(None, text_embedding_model)
        self.text_embedding_model = GMEmbeddingModel(text_embedding_model)
        self.multimodal_embedding_model = (
            self.text_embedding_model
            if multimodal_embedding_model == text_embedding_model
            else GMEmbeddingModel(multimodal_embedding_model)
        )
        self.batch_size = batch_size
        self.all_metadata_df = pd.DataFrame()

    def generate_table_embeddings(
        self, flatten_blocks: List[FlattenBlock]
    ) -> List[torch.Tensor]:
        """Generate embeddings for table blocks.

        Args:
            flatten_blocks: List of table blocks

        Returns:
            List of table embeddings
        """
        images = [block.image_path for block in flatten_blocks]
        multimodal_docs = [{"text": "", "image": image} for image in images]

        embeddings = self.multimodal_embedding_model.encode_multimodal_documents(
            multimodal_docs, batch_size=self.batch_size
        )

        # Convert tensor with shape (n, dim) to list of tensors with shape (dim,)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    def generate_image_embeddings(
        self, flatten_blocks: List[FlattenBlock]
    ) -> List[torch.Tensor]:
        """Generate embeddings for image blocks.

        Args:
            flatten_blocks: List of image blocks

        Returns:
            List of image embeddings
        """
        images = [block.image_path for block in flatten_blocks]
        multimodal_docs = [{"text": "", "image": image} for image in images]

        embeddings = self.multimodal_embedding_model.encode_multimodal_documents(
            multimodal_docs, batch_size=self.batch_size
        )

        # Convert tensor with shape (n, dim) to list of tensors with shape (dim,)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    def generate_text_embeddings(
        self, flatten_blocks: List[FlattenBlock]
    ) -> List[torch.Tensor]:
        """Generate embeddings for text blocks.

        Args:
            flatten_blocks: List of text blocks

        Returns:
            List of text embeddings
        """
        texts = [block.block_content for block in flatten_blocks]
        embeddings = self.text_embedding_model.encode_documents(
            texts, batch_size=self.batch_size
        )

        # Convert tensor with shape (n, dim) to list of tensors with shape (dim,)
        return [embeddings[i] for i in range(embeddings.shape[0])]

    def generate_embeddings(self, metadata_df: pd.DataFrame) -> None:
        """Generate embeddings for all block types.

        Args:
            metadata_df: DataFrame containing metadata for blocks
        """
        if metadata_df.empty:
            return

        flatten_blocks = FlattenBlock.from_dataframe(metadata_df)
        logger.info(f"len of flatten_blocks: {len(flatten_blocks)}")
        # Group blocks by type for batch processing
        text_blocks: List[FlattenBlock] = []
        image_blocks: List[FlattenBlock] = []
        table_blocks: List[FlattenBlock] = []
        block_indices: Dict[BlockType, List[int]] = {
            BlockType.TEXT: [],
            BlockType.IMAGE: [],
            BlockType.TABLE: [],
        }

        # Initialize embeddings list with None placeholders
        embeddings_list: List[Optional[torch.Tensor]] = [None] * len(flatten_blocks)

        # First pass: categorize blocks by type and track their original indices
        for i, block in enumerate(flatten_blocks):
            block_type = block.block_type

            if block_type == BlockType.TEXT:
                text_blocks.append(block)
                block_indices[BlockType.TEXT].append(i)
            elif block_type == BlockType.IMAGE:
                image_blocks.append(block)
                block_indices[BlockType.IMAGE].append(i)
            elif block_type == BlockType.TABLE:
                table_blocks.append(block)
                block_indices[BlockType.TABLE].append(i)
            else:
                logger.warning(f"Unsupported block type: {block_type}")

        # Process each block type in batches
        if text_blocks:
            text_embeddings = self.generate_text_embeddings(text_blocks)
            for idx, embedding in zip(block_indices[BlockType.TEXT], text_embeddings):
                embeddings_list[idx] = embedding

        if image_blocks:
            image_embeddings = self.generate_image_embeddings(image_blocks)
            for idx, embedding in zip(block_indices[BlockType.IMAGE], image_embeddings):
                embeddings_list[idx] = embedding

        if table_blocks:
            table_embeddings = self.generate_table_embeddings(table_blocks)
            for idx, embedding in zip(block_indices[BlockType.TABLE], table_embeddings):
                embeddings_list[idx] = embedding

        metadata_df["embedding"] = embeddings_list

    def get_cosine_score(self, row: pd.Series, query_vector: torch.Tensor) -> float:
        """Calculate cosine similarity between row embedding and query vector.

        Args:
            row: DataFrame row containing embedding
            query_vector: Query embedding tensor

        Returns:
            Cosine similarity score

        Raises:
            ValueError: If embedding is missing or None
        """
        if "embedding" not in row:
            raise ValueError("Embedding not found in row")

        embedding = row["embedding"]
        if embedding is None:
            raise ValueError("Embedding is None")

        # Convert to tensor if needed
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding)

        # Ensure both are 1D
        embedding = embedding.squeeze()
        query_vector = query_vector.squeeze()

        # Calculate cosine similarity
        return torch.nn.functional.cosine_similarity(
            embedding.unsqueeze(0), query_vector.unsqueeze(0)
        ).item()

    def add_document(self, document: Document) -> None:
        """Add a document to the vector database.

        Args:
            document: Document object to add
        """
        logger.debug(f"Adding document: {document.metadata['filename']}")

        metadata_df = document.to_dataframe()
        # Add embedding column if it doesn't exist
        if "embedding" not in metadata_df.columns:
            metadata_df["embedding"] = None
            logger.debug("Added embedding column to metadata DataFrame")

        # Generate embeddings for all block types
        self.generate_embeddings(metadata_df)

        # Append to the main dataframe
        self.all_metadata_df = pd.concat(
            [self.all_metadata_df, metadata_df], ignore_index=True
        )

        logger.info(f"Added document with {len(metadata_df)} blocks to vector database")

    def add_documents(self, documents: List[Document]) -> None:
        """Add multiple documents to the vector database.

        Args:
            documents: List of Document objects to add
        """
        for document in documents:
            self.add_document(document)

    def save_to_disk(self, path: Path) -> None:
        """Save the vector database to disk.

        Args:
            path: File path to save the database
        """
        self.all_metadata_df.to_pickle(path)

    def load_from_disk(self, path: Path) -> None:
        """Load the vector database from disk.

        Args:
            path: File path to load the database from
        """
        self.all_metadata_df = pd.read_pickle(path)

    def retrieve_text_blocks(
        self, query: str, top_n: int = 5, **kwargs
    ) -> List[FlattenBlock]:
        """Retrieve text blocks similar to the query.

        Args:
            query: The search query
            top_n: Number of blocks to retrieve

        Returns:
            List of retrieved text blocks
        """
        return self.get_similar_text_from_query(query, top_n=top_n, **kwargs)

    def retrieve_image_blocks(
        self, query: str, top_n: int = 5, **kwargs
    ) -> List[FlattenBlock]:
        """Retrieve both text and images relevant to the query.

        Args:
            query: The search query
            top_n: Number of blocks to retrieve

        Returns:
            List of FlattenBlock objects containing matched text and images
        """
        # Get text and image matches
        matching_images = self.get_similar_image_from_query(
            query, top_n=top_n, **kwargs
        )

        return matching_images

    def get_user_query_text_embeddings(self, user_query: str) -> torch.Tensor:
        """Generates embeddings for a text query.

        Args:
            user_query: The query text

        Returns:
            Tensor containing query embeddings
        """
        return self.text_embedding_model.encode_query(user_query)

    def get_user_query_image_embeddings(self, image_query_path: str) -> torch.Tensor:
        """Generates embeddings for an image query.

        Args:
            image_query_path: Path to the query image

        Returns:
            Tensor containing query embeddings
        """
        return self.multimodal_embedding_model.encode_query(image_query_path)

    def get_similar_image_from_query(
        self,
        query: str = "",
        image_query_path: str = "",
        image_emb: bool = False,
        top_n: int = 3,
        **kwargs,
    ) -> List[FlattenBlock]:
        """Finds most similar images based on text or image query.

        Args:
            query: Text query (if image_emb is False)
            image_query_path: Path to query image (if image_emb is True)
            image_emb: Whether to use image embeddings
            top_n: Number of results to return

        Returns:
            List of FlattenBlock objects containing matched image information
        """
        # Get appropriate embeddings based on query type
        query_embedding = (
            self.get_user_query_image_embeddings(image_query_path)
            if image_emb
            else self.get_user_query_text_embeddings(query)
        )

        # Filter to only image blocks
        image_df = self.all_metadata_df[self.all_metadata_df["block_type"] == "image"]
        logger.debug(f"Length of image dataframe: {len(image_df)}")

        if image_df.empty:
            return []

        # Apply cosine similarity calculation to each row
        image_df["cosine_similarity"] = image_df.apply(
            lambda x: self.get_cosine_score(x, query_embedding),
            axis=1,
        )

        # Remove exact matches
        image_df = image_df[image_df["cosine_similarity"] < 1.0]
        if image_df.empty:
            return []

        # Get top N matches
        top_n = min(top_n, len(image_df))
        top_matches = image_df.nlargest(top_n, "cosine_similarity")

        return FlattenBlock.from_dataframe(top_matches)

    def get_similar_text_from_query(
        self, query: str, top_n: int = 3, **kwargs
    ) -> List[FlattenBlock]:
        """Finds most similar text passages based on query.

        Args:
            query: Text query
            top_n: Number of results to return

        Returns:
            List of FlattenBlock objects containing matched text information
        """
        # Filter to only text blocks
        text_df = self.all_metadata_df[self.all_metadata_df["block_type"] == "text"]
        if text_df.empty:
            return []

        # Get query embeddings
        query_embedding = self.get_user_query_text_embeddings(query)

        # Apply cosine similarity calculation to each row
        text_df["cosine_similarity"] = text_df.apply(
            lambda x: self.get_cosine_score(x, query_embedding),
            axis=1,
        )

        # Remove exact matches
        text_df = text_df[text_df["cosine_similarity"] < 1.0]
        if text_df.empty:
            return []

        # Get top N matches
        top_n = min(top_n, len(text_df))
        top_matches = text_df.nlargest(top_n, "cosine_similarity")

        return FlattenBlock.from_dataframe(top_matches)
