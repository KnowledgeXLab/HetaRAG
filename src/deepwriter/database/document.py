from typing import Any, Dict, List, Optional, Union
import pandas as pd
from enum import Enum
import torch


class BlockType(str, Enum):
    """Enum representing different types of document blocks."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FLATTEN = "flatten"


class Block:
    """Base block class to store common block properties."""

    def __init__(
        self,
        block_id: int,
        block_bbox: List[int],
        block_type: BlockType,
    ) -> None:
        """Initialize a base block.

        Args:
            block_id: Unique identifier for the block
            block_bbox: Bounding box coordinates [x1, y1, x2, y2]
            block_type: Type of the block (text, image, table, etc.)
        """
        self.block_id = block_id
        self.block_bbox = block_bbox
        self.block_type = block_type

    def type_str(self) -> BlockType:
        """Return the type of the block.

        Returns:
            BlockType: The type of this block

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement this method")


class TextBlock(Block):
    """TextBlock class to store text content and its metadata."""

    def __init__(
        self,
        content: str,
        block_bbox: List[int],
        block_id: int = 0,
        block_type: BlockType = BlockType.TEXT,
    ) -> None:
        """Initialize a text block.

        Args:
            content: The text content
            block_bbox: Bounding box coordinates [x1, y1, x2, y2]
            block_id: Unique identifier for the block
            block_type: Type of the block (defaults to TEXT)
        """
        super().__init__(block_id, block_bbox, block_type)
        self.content = content

    def type_str(self) -> BlockType:
        """Return the type of the block.

        Returns:
            BlockType: Always returns BlockType.TEXT
        """
        return BlockType.TEXT


class ImageBlock(Block):
    """ImageBlock class to store image data and its metadata."""

    def __init__(
        self,
        image_path: str,
        image_caption: str,
        image_bbox: List[int],
        image_footer: Optional[str] = None,
        block_id: int = 0,
        block_type: BlockType = BlockType.IMAGE,
    ) -> None:
        """Initialize an image block.

        Args:
            image_path: Path to the image file
            image_caption: Caption describing the image
            image_bbox: Bounding box coordinates [x1, y1, x2, y2]
            image_footer: Optional footer text for the image
            block_id: Unique identifier for the block
            block_type: Type of the block (defaults to IMAGE)
        """
        super().__init__(block_id, image_bbox, block_type)
        self.image_path = image_path
        self.image_caption = image_caption
        self.image_footer = image_footer

    def type_str(self) -> BlockType:
        """Return the type of the block.

        Returns:
            BlockType: Always returns BlockType.IMAGE
        """
        return BlockType.IMAGE


class TableBlock(Block):
    """TableBlock class to store table data and its metadata."""

    def __init__(
        self,
        table_path: str,
        table_caption: str,
        table_footer: str,
        table_bbox: List[int],
        block_id: int = 0,
        block_type: BlockType = BlockType.TABLE,
    ) -> None:
        """Initialize a table block.

        Args:
            table_path: Path to the table data
            table_caption: Caption describing the table
            table_footer: Footer text for the table
            table_bbox: Bounding box coordinates [x1, y1, x2, y2]
            block_id: Unique identifier for the block
            block_type: Type of the block (defaults to TABLE)
        """
        super().__init__(block_id, table_bbox, block_type)
        self.table_path = table_path
        self.table_caption = table_caption
        self.table_footer = table_footer

    def type_str(self) -> BlockType:
        """Return the type of the block.

        Returns:
            BlockType: Always returns BlockType.TABLE
        """
        return BlockType.TABLE


class Page:
    """Page class to store blocks, text, and metadata for a document page."""

    def __init__(
        self,
        text: str,
        page_num: int,
        blocks: Optional[List[Block]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize a page.

        Args:
            text: The page text content
            page_num: The page number
            blocks: List of Block objects in the page
            metadata: Optional metadata associated with the page
        """
        self.text = text
        self.page_num = page_num
        self.blocks = blocks or []
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
            pd.DataFrame: A DataFrame containing all blocks with their page and document metadata.
                Each row represents one block with columns for block content, type, and metadata
                from both the block, its parent page, and the document.
        """
        flatten_blocks = self.to_flatten_blocks()
        if not flatten_blocks:
            return pd.DataFrame()
        return pd.concat([block.to_dataframe() for block in flatten_blocks])

    def to_flatten_blocks(self) -> List["FlattenBlock"]:
        """Convert the document to a list of FlattenBlock objects.

        Returns:
            List[FlattenBlock]: A list of FlattenBlock objects
        """
        flatten_blocks = []
        file_name = self.metadata.get("filename", "")
        file_path = self.metadata.get("path", "")
        num_pages = len(self.pages)

        for page in self.pages:
            page_size = page.metadata.get("page_size", (0, 0))
            page_width, page_height = page_size
            num_blocks = len(page.blocks)

            for block in page.blocks:
                # Create base flatten block with common properties
                flatten_block = FlattenBlock(
                    block_id=block.block_id,
                    block_bbox=block.block_bbox,
                    file_name=file_name,
                    pdf_path=file_path,
                    num_pages=num_pages,
                    page_number=page.page_num,
                    page_height=page_height,
                    page_width=page_width,
                    num_blocks=num_blocks,
                    block_type=block.block_type,
                )

                # Add type-specific properties
                if block.block_type == BlockType.TEXT:
                    if isinstance(block, TextBlock):
                        flatten_block.block_content = block.content
                elif block.block_type == BlockType.IMAGE:
                    if isinstance(block, ImageBlock):
                        flatten_block.image_path = block.image_path
                        flatten_block.image_caption = block.image_caption
                        flatten_block.image_footer = block.image_footer
                elif block.block_type == BlockType.TABLE:
                    if isinstance(block, TableBlock):
                        flatten_block.image_path = block.table_path
                        flatten_block.image_caption = block.table_caption
                        flatten_block.image_footer = block.table_footer
                else:
                    raise ValueError(f"Unknown block type: {block.block_type}")

                flatten_blocks.append(flatten_block)

        return flatten_blocks


class FlattenBlock(Block):
    """Flattened representation of various block types with all possible attributes."""

    def __init__(
        self,
        block_id: int,
        block_bbox: List[int],
        pdf_path: str,
        num_pages: int,
        page_number: int,
        page_height: int,
        page_width: int,
        num_blocks: int,
        block_type: BlockType,
        block_content: Optional[str] = None,
        block_summary: Optional[str] = None,
        image_caption: Optional[str] = None,
        image_footer: Optional[str] = None,
        image_path: Optional[str] = None,
        block_embedding: Optional[torch.Tensor] = None,
        cosine_similarity: Optional[float] = None,
        document_title: Optional[str] = None,
        section_title: Optional[str] = None,
    ) -> None:
        """Initialize a flattened block.

        Args:
            block_id: Unique identifier for the block
            block_bbox: Bounding box coordinates [x1, y1, x2, y2]
            pdf_path: Path to the PDF file
            num_pages: Number of pages in the document
            page_number: Page number in the document
            page_height: Height of the page
            page_width: Width of the page
            num_blocks: Number of blocks in the document
            block_type: Type of the block
            block_content: Text content (for text blocks)
            block_summary: Optional summary of the block content
            image_caption: Caption for image or table
            image_footer: Footer for image or table
            image_path: Path to image file (for image blocks)
            block_embedding: Optional vector embedding of the block
            cosine_similarity: Optional similarity score
            document_title: Optional title of the document
            section_title: Optional title of the section containing this block
        """
        super().__init__(block_id, block_bbox, block_type)
        self.pdf_path = pdf_path
        self.num_pages = num_pages
        self.page_number = page_number
        self.page_height = page_height
        self.page_width = page_width
        self.num_blocks = num_blocks
        self.block_content = block_content
        self.block_summary = block_summary
        self.image_caption = image_caption
        self.image_footer = image_footer
        self.image_path = image_path
        self.document_title = document_title
        self.section_title = section_title
        self.block_embedding = block_embedding
        self.cosine_similarity = cosine_similarity

    def type_str(self) -> BlockType:
        """Return the type of the block.

        Returns:
            BlockType: Always returns BlockType.FLATTEN
        """
        return BlockType.FLATTEN

    def to_dict(
        self,
    ) -> Dict[str, Union[str, int, List[int], float, torch.Tensor, None]]:
        """Convert the flattened block to a dictionary.

        Returns:
            Dict[str, Union[str, int, float, torch.Tensor, None]]: Dictionary representation
                of this flattened block
        """
        return {
            "block_id": self.block_id,
            "block_bbox": self.block_bbox,
            "pdf_path": self.pdf_path,
            "num_pages": self.num_pages,
            "page_number": self.page_number,
            "page_height": self.page_height,
            "page_width": self.page_width,
            "num_blocks": self.num_blocks,
            "block_type": self.block_type,
            "block_content": self.block_content,
            "block_summary": self.block_summary,
            "image_caption": self.image_caption,
            "image_footer": self.image_footer,
            "image_path": self.image_path,
            "document_title": self.document_title,
            "section_title": self.section_title,
            "block_embedding": self.block_embedding,
            "cosine_similarity": self.cosine_similarity,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the flattened block to a pandas DataFrame row.

        Returns:
            pd.DataFrame: A DataFrame with a single row representing this block
        """
        data = self.to_dict()
        df = pd.DataFrame.from_dict(data, orient="index").T
        df.columns = list(data.keys())
        return df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> List["FlattenBlock"]:
        """Create a list of FlattenBlock objects from a DataFrame.

        Args:
            df: DataFrame containing block data

        Returns:
            List[FlattenBlock]: List of FlattenBlock objects
        """
        return [cls(**row.to_dict()) for _, row in df.iterrows()]

    @classmethod
    def from_milvus_search_result(cls, hit: Any) -> "FlattenBlock":
        """
        Create a FlattenBlock object from a Milvus search result.

        Args:
            hit (pymilvus.client.abstract.Hit): Milvus search result

        Returns:
            FlattenBlock: FlattenBlock object
        """

        data = hit.to_dict()
        entity_data = data["entity"]
        entity_data["cosine_similarity"] = data["distance"]
        del data["id"]
        del data["distance"]
        return cls(**data["entity"])
