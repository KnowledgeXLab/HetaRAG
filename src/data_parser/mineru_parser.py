import os
from pathlib import Path
from typing import Dict, List, Union
from loguru import logger

from src.data_parser.document import (
    Document,
    Page,
    TextChunk,
    ImageChunk,
    TableChunk,
    Chunk,
)
from .base_parser import BaseProcessor
from src.utils.file_utils import read, write
from src.utils.query2vec import (
    GmeQwen2VL,
    # get_embedding_ollama,
    get_embedding_text,
    get_embedding_Qwen2VL,
)


class MinerUProcessor(BaseProcessor):
    """Processor for MinerU document format.

    This processor converts MinerU intermediate JSON format into Document objects.
    The MinerU format is a structured representation of PDF documents with text,
    images, and tables extracted and organized by page.
    """

    def __init__(self, output_path: Path, image_embedding: bool):
        """Initialize the MinerU processor.

        Args:
            output_path: Directory path where processed documents will be saved
        """
        super().__init__(output_path)

        self.image_embedding = image_embedding
        if self.image_embedding:
            self.gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")

    def process_documents(self, input_path: Union[Path, List[Path]]) -> List[Document]:
        """Process all PDF documents in the input path.

        Returns:
            List of Document objects
        """
        pdf_files = self._find_pdf_files(input_path)

        documents = []

        for pdf_file in pdf_files:
            document = self.process_single_document(pdf_file)
            if document is not None:
                documents.append(document)

        return documents

    def process_single_document(self, document_path: Path) -> Document:
        """Process a single document.

        Args:
            document_path: Path to the document file

        Returns:
            Processed Document object

        Raises:
            FileNotFoundError: If required files are missing
        """
        # output dir of mineru process, contains middle.json and images
        # if the pdf file is path/to/file.pdf, then the output dir will be path/to/file/
        base_dir = document_path.parent / document_path.stem
        logger.debug(f"Processing document: {document_path}")

        # if "wtr" in str(base_dir).lower():
        #     middle_json = base_dir / f"{document_path.stem}_middle.json"
        #     image_dir = base_dir / "images"
        # else:
        #     # Build related file paths
        #     middle_json = base_dir / f"auto/{document_path.stem}_middle.json"
        #     image_dir = base_dir / "auto/images"

        # Build related file paths
        middle_json = base_dir / f"{document_path.stem}_middle.json"
        image_dir = base_dir / "images"

        # Validate file existence
        if middle_json.exists() and middle_json.is_file():
            # Load JSON data
            middle_data = read(middle_json)
            # Perform conversion
            return self.convert_to_document(
                middle_data=middle_data, pdf_path=document_path, image_dir=image_dir
            )
        else:
            logger.error(f"Missing required file: {middle_json}")
            return None
            # raise FileNotFoundError(f"Missing required file: {middle_json}")

    def convert_to_document(
        self, middle_data: Dict, pdf_path: Path, image_dir: Path
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
            "pdf_path": str(pdf_path),
            "num_pages": len(middle_data["pdf_info"]),
            "document_title": "",
            "section_title": "",
        }

        pages = []
        for page_info in middle_data["pdf_info"]:
            pages.append(self.process_page(page_info, image_dir))

        return Document(pages=pages, metadata=document_metadata)

    def process_page(self, page_info: Dict, image_dir: Path) -> Page:
        """Process a single page from the document.

        Args:
            page_info: Page information from intermediate JSON
            image_dir: Directory containing extracted images

        Returns:
            Processed Page object with chunks
        """
        page_no = page_info["page_idx"]
        page_size = page_info["page_size"]

        chunks: List[Chunk] = []

        # Process image blocks
        for image_block in page_info["images"]:
            image_chunk = self.process_image_chunk(image_block, image_dir)
            if image_chunk.image_path is None or not image_chunk.image_path.strip():
                continue
            if not os.path.exists(image_chunk.image_path):
                logger.warning(f"Image not found: {image_chunk.image_path}")
                continue
            chunks.append(image_chunk)

        # Process table blocks
        for table_block in page_info["tables"]:
            table_chunk = self.process_table_chunk(table_block, image_dir)
            if not os.path.exists(table_chunk.table_path):
                logger.warning(f"Table not found: {table_chunk.table_path}")
                continue
            chunks.append(table_chunk)

        # Process text blocks
        for block in page_info["para_blocks"]:
            if block["type"] in ["image", "table"]:
                continue
            text_chunk = self.process_text_chunk(block)
            chunks.append(text_chunk)

        # Assign unique IDs to each chunk
        for index, chunk in enumerate(chunks):
            chunk.chunk_id = index

        return Page(
            page_num=page_no + 1,  # page number is 1-indexed
            metadata={
                "page_size": page_size,
                "num_chunks": len(chunks),
            },
            text="",  # currently, page text is not used
            chunks=chunks,
        )

    def process_text_chunk(self, block: Dict) -> TextChunk:
        """Process a text block into a TextChunk.

        Args:
            block: Block information from intermediate JSON

        Returns:
            Processed TextChunk object
        """
        text = self.extract_block_text(block)
        if self.image_embedding:
            embedding = get_embedding_Qwen2VL(self.gme, "text", text)
        else:
            embedding = get_embedding_text(text)

        return TextChunk(
            text=text,
            embedding=embedding,
            bbox=self.format_bbox(block["bbox"]),
            chunk_id=0,  # Will be reassigned later
        )

    def process_table_chunk(self, table_block: Dict, image_dir: Path) -> TableChunk:
        """Process a table block into a TableChunk.

        Args:
            table_block: Table block information from intermediate JSON
            image_dir: Directory containing extracted images

        Returns:
            Processed TableChunk object
        """
        table_chunk = TableChunk(
            table_path="",
            embedding=[],
            table_caption="",
            table_footer="",
            chunk_id=0,  # Will be reassigned later
            table_bbox=[],
        )

        for block in table_block["blocks"]:
            if block["type"] == "table_body":
                table_path = self.get_content_path(block, image_dir)
                table_chunk.table_path = str(table_path)
            elif block["type"] == "table_caption":
                table_chunk.table_caption = self.extract_block_text(block)
            elif block["type"] == "table_footenote":
                table_chunk.table_footer = self.extract_block_text(block)

        if table_chunk.table_path is None or not table_chunk.table_path.strip():
            logger.warning(f"Table path is None")
            table_chunk.bbox = self.format_bbox(table_block["bbox"])
            return table_chunk

        if self.image_embedding:
            try:
                embedding = get_embedding_Qwen2VL(
                    self.gme, "image", table_chunk.table_path
                )
            except Exception as e:
                logger.error(
                    f"Failed to get embedding for table image: {table_chunk.table_path}"
                )
                logger.error(f"Error details: {str(e)}")
                embedding = None
                table_chunk.embedding = embedding
                table_chunk.bbox = self.format_bbox(table_block["bbox"])
                return table_chunk
        else:
            embedding = None

        table_chunk.embedding = embedding
        table_chunk.bbox = self.format_bbox(table_block["bbox"])
        return table_chunk

    def process_image_chunk(self, image_block: Dict, image_dir: Path) -> ImageChunk:
        """Process an image block into an ImageChunk.

        Args:
            image_block: Image block information from intermediate JSON
            image_dir: Directory containing extracted images

        Returns:
            Processed ImageChunk object
        """
        image_chunk = ImageChunk(
            image_path="",
            embedding=[],
            image_caption="",
            image_bbox=[],
            image_footer="",
            chunk_id=0,  # Will be reassigned later
        )

        for block in image_block["blocks"]:
            if block["type"] == "image_body":
                image_path = self.get_content_path(block, image_dir)
                image_chunk.image_path = str(image_path)
            elif block["type"] == "image_caption":
                image_chunk.image_caption = self.extract_block_text(block)
            elif block["type"] == "image_footer":
                image_chunk.image_footer = self.extract_block_text(block)

        if image_chunk.image_path is None or not image_chunk.image_path.strip():
            logger.warning(f"Image path is None")
            image_chunk.bbox = self.format_bbox(image_block["bbox"])
            return image_chunk

        if self.image_embedding:
            try:
                embedding = get_embedding_Qwen2VL(
                    self.gme, "image", image_chunk.image_path
                )
            except Exception as e:
                logger.error(
                    f"Failed to get embedding for image: {image_chunk.image_path}"
                )
                logger.error(f"Error details: {str(e)}")
                embedding = None
                image_chunk.embedding = embedding
                image_chunk.bbox = self.format_bbox(image_block["bbox"])
                return image_chunk
        else:
            embedding = None

        image_chunk.embedding = embedding

        image_chunk.bbox = self.format_bbox(image_block["bbox"])
        return image_chunk

    def get_content_path(self, body_block: Dict, image_dir: Path) -> Path:
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

    def extract_block_text(self, block: Dict) -> str:
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
