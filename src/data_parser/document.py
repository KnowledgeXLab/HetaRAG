from typing import Any, Dict, List, Optional
import pandas as pd
from enum import Enum


class ChunkType(str, Enum):
    """Enum representing different types of document chunks."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FLATTEN = "flatten"


class Chunk:
    """Base chunk class to store common chunk properties."""

    def __init__(
        self,
        chunk_id: int,
        bbox: List[int],
        chunk_type: ChunkType,
    ) -> None:
        """Initialize a base chunk.

        Args:
            chunk_id: Unique identifier for the chunk
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            chunk_type: Type of the chunk (text, image, table, etc.)
        """
        self.chunk_id = chunk_id
        self.bbox = bbox
        self.chunk_type = chunk_type

    def type_str(self) -> ChunkType:
        """Return the type of the chunk.

        Returns:
            ChunkType: The type of this chunk

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement this method")


class TextChunk(Chunk):
    """TextChunk class to store text content and its metadata."""

    def __init__(
        self,
        text: str,
        embedding: list,
        bbox: List[int],
        chunk_id: int = 0,
        chunk_type: ChunkType = ChunkType.TEXT,
    ) -> None:
        """Initialize a text chunk.

        Args:
            text: The text content
            bbox: Bounding box coordinates [x1, y1, x2, y2]
            chunk_id: Unique identifier for the chunk
            chunk_type: Type of the chunk (defaults to TEXT)
        """
        super().__init__(chunk_id, bbox, chunk_type)
        self.text = text
        self.embedding = embedding

    def type_str(self) -> ChunkType:
        return ChunkType.TEXT


class ImageChunk(Chunk):
    """ImageChunk class to store image data and its metadata."""

    def __init__(
        self,
        image_path: str,
        embedding: list,
        image_caption: str,
        image_bbox: List[int],
        image_footer: Optional[str] = None,
        chunk_id: int = 0,
        chunk_type: ChunkType = ChunkType.IMAGE,
    ) -> None:
        """Initialize an image chunk.

        Args:
            image_path: Path to the image file
            image_caption: Caption describing the image
            image_bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_footer: Optional footer text for the image
            chunk_id: Unique identifier for the chunk
            chunk_type: Type of the chunk (defaults to IMAGE)
        """
        super().__init__(chunk_id, image_bbox, chunk_type)
        self.image_path = image_path
        self.image_caption = image_caption
        self.image_footer = image_footer
        self.embedding = embedding

    def type_str(self) -> ChunkType:
        return ChunkType.IMAGE


class TableChunk(Chunk):
    """TableChunk class to store table data and its metadata."""

    def __init__(
        self,
        table_path: str,
        embedding: list,
        table_caption: str,
        table_footer: str,
        table_bbox: List[int],
        chunk_id: int = 0,
        chunk_type: ChunkType = ChunkType.TABLE,
    ) -> None:
        """Initialize a table chunk.

        Args:
            table_path: Path to the table data
            table_caption: Caption describing the table
            table_footer: Footer text for the table
            table_bbox: Bounding box coordinates [x1, y1, x2, y2]
            chunk_id: Unique identifier for the chunk
            chunk_type: Type of the chunk (defaults to TABLE)
        """
        super().__init__(chunk_id, table_bbox, chunk_type)
        self.table_path = table_path
        self.table_caption = table_caption
        self.table_footer = table_footer
        self.embedding = embedding

    def type_str(self) -> ChunkType:
        return ChunkType.TABLE


class Page:
    """Page class to store chunks, text, and metadata for a document page."""

    def __init__(
        self,
        text: str,
        page_num: int,
        chunks: Optional[List[Chunk]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a page.

        Args:
            text: The page text content
            page_num: The page number
            chunks: List of Chunk objects in the page
            metadata: Optional metadata associated with the page
        """
        self.text = text
        self.page_num = page_num
        self.chunks = chunks or []
        self.metadata = metadata or {}


class Document:
    """Document class to store pages and metadata for a complete document."""

    def __init__(
        self,
        pages: Optional[List[Page]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a document.

        Args:
            pages: List of Page objects in the document
            metadata: Optional metadata associated with the document
        """
        self.pages = pages or []
        self.metadata = metadata or {}

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the document to a pandas DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all chunks with their page and document metadata.
                Each row represents one chunk with columns for chunk content, type, and metadata
                from both the chunk, its parent page, and the document.
        """
        flatten_chunks = self.to_flatten_chunks()
        if not flatten_chunks:
            return pd.DataFrame()
        return pd.concat([chunk.to_dataframe() for chunk in flatten_chunks])

    def to_flatten_chunks(self) -> List["FlattenChunk"]:
        """Convert the document to a list of FlattenChunk objects.

        Returns:
            List[FlattenChunk]: A list of FlattenChunk objects
        """
        flatten_chunks = []
        file_path = self.metadata["pdf_path"]
        num_pages = self.metadata["num_pages"]
        for page in self.pages:
            page_size = page.metadata["page_size"]
            page_width, page_height = page_size
            for chunk in page.chunks:
                flatten_chunk = FlattenChunk(
                    pdf_path=file_path,
                    num_pages=num_pages,
                    page_number=page.page_num,
                    page_height=page_height,
                    page_width=page_width,
                    num_blocks=len(page.chunks),
                    block_type=chunk.chunk_type,
                    block_content="",
                    block_embedding=[],
                    block_summary="",
                    image_path="",
                    image_caption="",
                    image_footer="",
                    block_bbox=chunk.bbox,
                    block_id=chunk.chunk_id,
                    document_title=self.metadata["document_title"],
                    section_title=self.metadata["section_title"],
                )
                if chunk.chunk_type == ChunkType.TEXT:
                    flatten_chunk.block_content = chunk.text
                    flatten_chunk.block_embedding = chunk.embedding
                    flatten_chunk.image_path = ""
                    flatten_chunk.image_caption = ""
                    flatten_chunk.image_footer = ""
                elif chunk.chunk_type in ChunkType.IMAGE:
                    flatten_chunk.block_content = ""
                    flatten_chunk.block_embedding = chunk.embedding
                    flatten_chunk.image_path = chunk.image_path
                    flatten_chunk.image_caption = chunk.image_caption
                    flatten_chunk.image_footer = chunk.image_footer
                elif chunk.chunk_type in ChunkType.TABLE:
                    flatten_chunk.block_content = ""
                    flatten_chunk.block_embedding = chunk.embedding
                    flatten_chunk.image_path = chunk.table_path
                    flatten_chunk.image_caption = chunk.table_caption
                    flatten_chunk.image_footer = chunk.table_footer
                else:
                    raise ValueError(f"Unknown chunk type: {chunk.chunk_type}")

                flatten_chunks.append(flatten_chunk)
        return flatten_chunks


class FlattenChunk(Chunk):
    """Flattened representation of various chunk types with all possible attributes."""

    def __init__(
        self,
        pdf_path: str,
        num_pages: int,
        page_number: int,
        page_height: int,
        page_width: int,
        num_blocks: int,
        block_type: ChunkType,
        block_content: str,
        block_embedding: list,
        block_summary: str,
        image_path: str,
        image_caption,
        image_footer,
        block_bbox: List[int],
        block_id: int,
        document_title: str,
        section_title: str,
    ) -> None:
        """Initialize a flattened chunk.

        Args:
            pdf_path: Path to the PDF file
            num_pages: Number of pages in the document
            page_number: Page number in the document
            page_height: Height of the page
            page_width: Width of the page
            num_blocks: Number of blocks in the page
            block_type: Type of the block
            block_content: Content of the block
            block_summary: Summary of the block
            image_path: Path to the image file
            image_caption: Caption of the image
            image_footer: Footer of the image
            block_bbox: Bounding box coordinates [x1, y1, x2, y2]
            block_id: Unique identifier for the block
            document_title: Title of the document
            section_title: Title of the section
        """
        super().__init__(block_id, block_bbox, block_type)
        self.pdf_path = pdf_path
        self.num_pages = num_pages
        self.page_number = page_number
        self.page_height = page_height
        self.page_width = page_width
        self.num_blocks = num_blocks
        self.block_type = block_type
        self.block_content = block_content
        self.block_summary = block_summary
        self.block_embedding = block_embedding
        self.image_path = image_path
        self.image_caption = image_caption
        self.image_footer = image_footer
        self.document_title = document_title
        self.section_title = section_title

    def type_str(self) -> ChunkType:
        return ChunkType.FLATTEN

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the flattened chunk to a pandas DataFrame row.

        Returns:
            pd.DataFrame: A DataFrame with a single row representing this chunk
        """
        if self.block_type == ChunkType.TEXT:
            block_type = "text"
        elif self.block_type in ChunkType.IMAGE:
            block_type = "image"
        elif self.block_type in ChunkType.TABLE:
            block_type = "table"
        elif self.block_type in ChunkType.FLATTEN:
            block_type = "flatten"
        else:
            block_type = "None"
        return pd.DataFrame(
            {
                "pdf_path": [self.pdf_path],
                "num_pages": [self.num_pages],
                "page_number": [self.page_number],
                "page_height": [self.page_height],
                "page_width": [self.page_width],
                "num_blocks": [self.num_blocks],
                "block_type": [block_type],
                "block_content": [self.block_content],
                "block_summary": [self.block_summary],
                "block_embedding": [self.block_embedding],
                "image_path": [self.image_path],
                "image_caption": [self.image_caption],
                "image_footer": [self.image_footer],
                "block_bbox": [self.bbox],
                "block_id": [self.chunk_id],
                "document_title": [self.document_title],
                "section_title": [self.section_title],
            }
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["FlattenChunk"]:
        """Create a list of FlattenChunk objects from a DataFrame.

        Args:
            df: DataFrame containing chunk data

        Returns:
            List[FlattenChunk]: List of FlattenChunk objects
        """
        return [
            cls(
                pdf_path=row["pdf_path"],
                num_pages=row["num_pages"],
                page_number=row["page_number"],
                page_height=row["page_height"],
                page_width=row["page_width"],
                num_blocks=row["num_blocks"],
                block_type=row["block_type"],
                block_content=row["block_content"],
                block_summary=row["block_summary"],
                block_embedding=row["block_embedding"],
                image_path=row["image_path"],
                image_caption=row["image_caption"],
                image_footer=row["image_footer"],
                block_bbox=row["block_bbox"],
                block_id=row["block_id"],
                document_title=row["document_title"],
                section_title=row["section_title"],
            )
            for _, row in df.iterrows()
        ]

    def __str__(self) -> str:
        """Return a string representation of the FlattenChunk.

        This method creates a formatted string representation suitable for debugging
        and logging purposes. It includes all relevant chunk information in a
        structured format.

        Returns:
            str: Formatted string representation of the chunk
        """
        # Create a visually appealing string representation with formatting
        chunk_text_preview = (
            f'"{self.block_content[:50]}..."' if self.block_content else "None"
        )

        # For logger.debug compatibility, use newlines without box drawing characters
        return (
            f"FLATTEN CHUNK ID: {self.chunk_id}\n"
            f"Location:\n"
            f"  • File: {self.pdf_path or 'N/A'}\n"
            f"  • Page: {self.page_number or 'N/A'}\n"
            f"  • Page Height: {self.page_height or 'N/A'}\n"
            f"  • Page Width: {self.page_width or 'N/A'}\n"
            f"  • Bbox: {str(self.bbox)}\n"
            f"Content:\n"
            f"  • Type: {self.block_type or 'N/A'}\n"
            f"  • Text: {chunk_text_preview}\n"
            f"  • Caption: {self.image_caption or 'None'}\n"
            f"  • Footer: {self.image_footer or 'None'}\n"
            f"Resources:\n"
            f"  • Image: {self.image_path or 'None'}\n"
            f"Analysis:\n"
            f"  • Summary: {(self.block_summary[:40] + '...') if self.block_summary else 'None'}"
        )
