import os
from pathlib import Path
from typing import Dict, List, Union, Any
from loguru import logger
from src.deepwriter.database.document import (
    Document,
    Page,
    TextBlock,
    ImageBlock,
    TableBlock,
    Block,
)
from .base_processor import BaseProcessor
from src.deepwriter.utils import file_interface


class MinerUProcessor(BaseProcessor):
    """Processor for MinerU document format.

    This processor converts MinerU intermediate JSON format into Document objects.
    The MinerU format is a structured representation of PDF documents with text,
    images, and tables extracted and organized by page.
    """

    def __init__(self, output_path: Path):
        """Initialize the MinerU processor.

        Args:
            output_path: Directory path where processed documents will be saved
        """
        super().__init__(output_path)

    def process_documents(self, input_path: Union[Path, List[Path]]) -> List[Document]:
        """Process all PDF documents in the input path.

        Args:
            input_path: Path or list of paths to process

        Returns:
            List of Document objects
        """
        pdf_files = self._find_pdf_files(input_path)
        return [self.process_single_document(pdf_file) for pdf_file in pdf_files]

    def process_single_document(self, document_path: Path) -> Document:
        """Process a single document.

        Args:
            document_path: Path to the document file

        Returns:
            Processed Document object

        Raises:
            FileNotFoundError: If required files are missing
        """
        # Output dir of mineru process, contains middle.json and images
        # If the pdf file is path/to/file.pdf, then the output dir will be path/to/file/
        base_dir = document_path.parent / document_path.stem
        logger.debug(f"Processing document: {document_path}")

        # Build related file paths
        middle_json = base_dir / f"{document_path.stem}_middle.json"
        image_dir = base_dir / "images"

        # Validate file existence
        if not middle_json.exists():
            raise FileNotFoundError(f"Missing required file: {middle_json}")

        # Load JSON data
        middle_data = file_interface.load_json(middle_json)

        # Perform conversion
        return self.convert_to_document(
            middle_data=middle_data, pdf_path=document_path, image_dir=image_dir
        )

    def convert_to_document(
        self, middle_data: Dict[str, Any], pdf_path: Path, image_dir: Path
    ) -> Document:
        """Convert intermediate data to Document object.

        Args:
            middle_data: Parsed JSON data from intermediate file
            pdf_path: Path to the original PDF file
            image_dir: Directory containing extracted images

        Returns:
            Document object with processed pages
        """
        document_metadata = {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "num_pages": len(middle_data["pdf_info"]),
        }

        pages = [
            self.process_page(page_info, image_dir)
            for page_info in middle_data["pdf_info"]
        ]

        return Document(pages=pages, metadata=document_metadata)

    def process_page(self, page_info: Dict[str, Any], image_dir: Path) -> Page:
        """Process a single page from the document.

        Args:
            page_info: Page information from intermediate JSON
            image_dir: Directory containing extracted images

        Returns:
            Processed Page object with blocks
        """
        page_no = page_info["page_idx"]
        page_size = page_info["page_size"]

        blocks: List[Block] = []

        # Process image blocks
        for image_block in page_info["images"]:
            image_block = self.process_image_block(image_block, image_dir)
            if not os.path.exists(image_block.image_path):
                logger.warning(f"Image not found: {image_block.image_path}")
                continue
            blocks.append(image_block)

        # Process table blocks
        for table_block in page_info["tables"]:
            table_block = self.process_table_block(table_block, image_dir)
            if not os.path.exists(table_block.table_path):
                logger.warning(f"Table not found: {table_block.table_path}")
                continue
            blocks.append(table_block)

        # Process text blocks
        for block in page_info["para_blocks"]:
            if block["type"] in ["image", "table"]:
                continue
            text_block = self.process_text_block(block)
            blocks.append(text_block)

        # Assign unique IDs to each block
        for index, block in enumerate(blocks):
            block.block_id = index

        return Page(
            page_num=page_no + 1,  # page number is 1-indexed
            metadata={
                "page_size": page_size,
                "num_blocks": len(blocks),
            },
            text="",  # currently, page text is not used
            blocks=blocks,
        )

    def process_text_block(self, block: Dict[str, Any]) -> TextBlock:
        """Process a text block into a TextBlock.

        Args:
            block: Block information from intermediate JSON

        Returns:
            Processed TextBlock object
        """
        return TextBlock(
            content=self.extract_block_text(block),
            block_bbox=self.format_bbox(block["bbox"]),
            block_id=0,  # Will be reassigned later
        )

    def process_table_block(
        self, table_block: Dict[str, Any], image_dir: Path
    ) -> TableBlock:
        """Process a table block into a TableBlock.

        Args:
            table_block: Table block information from intermediate JSON
            image_dir: Directory containing extracted images

        Returns:
            Processed TableBlock object
        """
        result = TableBlock(
            table_path="",
            table_caption="",
            table_footer="",
            block_id=0,  # Will be reassigned later
            block_bbox=self.format_bbox(table_block["bbox"]),
        )

        for block in table_block["blocks"]:
            if block["type"] == "table_body":
                table_path = self.get_content_path(block, image_dir)
                result.table_path = str(table_path)
            elif block["type"] == "table_caption":
                result.table_caption = self.extract_block_text(block)
            elif block["type"] == "table_footenote":
                result.table_footer = self.extract_block_text(block)

        return result

    def process_image_block(
        self, image_block: Dict[str, Any], image_dir: Path
    ) -> ImageBlock:
        """Process an image block into an ImageBlock.

        Args:
            image_block: Image block information from intermediate JSON
            image_dir: Directory containing extracted images

        Returns:
            Processed ImageBlock object
        """
        result = ImageBlock(
            image_path="",
            image_caption="",
            block_bbox=self.format_bbox(image_block["bbox"]),
            image_footer="",
            block_id=0,  # Will be reassigned later
        )

        for block in image_block["blocks"]:
            if block["type"] == "image_body":
                image_path = self.get_content_path(block, image_dir)
                result.image_path = str(image_path)
            elif block["type"] == "image_caption":
                result.image_caption = self.extract_block_text(block)
            elif block["type"] == "image_footer":
                result.image_footer = self.extract_block_text(block)

        return result

    def get_content_path(self, body_block: Dict[str, Any], image_dir: Path) -> Path:
        """Extract content file path from a body block.

        Args:
            body_block: Body block information from intermediate JSON
            image_dir: Directory containing extracted images

        Returns:
            Path to the content file or empty Path if not found
        """
        for line in body_block.get("lines", []):
            for span in line.get("spans", []):
                if "image_path" in span:
                    return image_dir / span["image_path"]
        return Path("")

    def extract_block_text(self, block: Dict[str, Any]) -> str:
        """Extract text content from a block by joining all text spans.

        Args:
            block: Block information from intermediate JSON

        Returns:
            Extracted text content
        """
        text_spans = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if span.get("type") == "text" and "content" in span:
                    text_spans.append(span["content"])

        return " ".join(text_spans).strip()

    def format_bbox(self, raw_bbox: List[float]) -> List[int]:
        """Format bounding box coordinates.

        The coordinates are in the format of [x0, y0, x1, y1],
        where (x0, y0) is the top-left corner and (x1, y1) is the bottom-right corner.
        The original coordinates may contain float values, so we round them to the nearest integer.

        Args:
            raw_bbox: Raw bounding box coordinates

        Returns:
            Formatted bounding box as a list of integers
        """
        return [int(round(x)) for x in raw_bbox]
