"""Document processing utilities for extracting text and images from PDFs."""

import io
import os
from typing import List

import fitz
from PIL import Image
from pathlib import Path
from loguru import logger

from .document import Document, Page, TextChunk, ImageChunk
from .base_parser import BaseProcessor


class FitzProcessor(BaseProcessor):
    """Class for processing PDF documents and extracting structured content."""

    def __init__(
        self,
        output_path: Path,
        chunk_size: int = 1000,
        overlap: int = 100,
    ) -> None:
        """Initialize the PDF processor.

        Args:
            output_path: Path to output directory for extracted images
            chunk_size: Maximum characters per text chunk
            overlap: Overlap between text chunks
        """
        super().__init__(output_path)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def process_documents(self, input_path: Path | List[Path]) -> List[Document]:
        """Process all PDF documents in the input path.

        Returns:
            List of Document objects
        """
        pdf_files = self._find_pdf_files(input_path)
        logger.info(f"Found {len(pdf_files)} PDF files")

        documents = []

        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file}")
            document = self.process_single_document(pdf_file)
            logger.info(f"Processed {pdf_file}")
            documents.append(document)

        return documents

    def process_single_document(self, pdf_file: Path) -> Document:
        """Process a single PDF document.

        Args:
            pdf_file: Path to the PDF file

        Returns:
            Document object containing structured content
        """
        # Open the PDF document
        fitz_doc = fitz.open(pdf_file)
        pdf_file_name = os.path.basename(pdf_file)
        pdf_file_name_wo_ext = os.path.splitext(pdf_file_name)[0]

        # Create document metadata
        metadata = {
            "filename": pdf_file_name,
            "pdf_path": str(pdf_file.absolute()),
            "num_pages": len(fitz_doc),
            "document_title": "",
            "section_title": "",
        }

        # Create Document object
        document = Document(metadata=metadata)

        # Process each page
        for page_num, page in enumerate(fitz_doc):
            # Extract text
            text = page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")

            # Create Page object
            page_obj = Page(
                text=text,
                page_num=page_num,
                metadata={
                    "page_number": page_num + 1,
                    "page_size": [page.rect.width, page.rect.height],
                },
            )

            # Process text chunks
            text_chunks = self._process_text_chunks(page)

            # Process images
            image_chunks = self._process_images(
                fitz_doc, page, page_num, pdf_file_name_wo_ext
            )

            # Add chunks to page
            page_obj.chunks.extend(text_chunks)
            page_obj.chunks.extend(image_chunks)

            # Add page to document
            document.pages.append(page_obj)

        return document

    def _process_text_chunks(self, page: fitz.Page) -> List[TextChunk]:
        """Process text into chunks.

        Args:
            text: Page text content
            page_num: Page number

        Returns:
            List of TextChunk objects
        """
        chunks = []

        # flags=11 gives the most accurate results
        blocks = page.get_text("dict", flags=11)["blocks"]
        for block in blocks:
            bbox = block["bbox"]
            text = ""
            # check if the block contains text lines
            if not ("lines" in block):
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text += span["text"]

            chunk = TextChunk(text=text, chunk_id=block["id"], bbox=bbox)
            chunks.append(chunk)

        return chunks

    def _process_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_num: int,
        pdf_file_name_wo_ext: str,
    ) -> List[ImageChunk]:
        """Process images on a page.

        Args:
            doc: PDF document
            page: PDF page
            page_num: Page number
            pdf_file_name_wo_ext: PDF filename without extension

        Returns:
            List of ImageChunk objects
        """
        chunks = []
        page_images = page.get_images()

        for image_no, image in enumerate(page_images):
            xref = image[0]
            pix = fitz.Pixmap(doc, xref)

            # Save image
            image_path = f"{self.output_path.absolute()}/{pdf_file_name_wo_ext}_image_{page_num}_{image_no}_{xref}.png"
            pil_image = Image.open(io.BytesIO(pix.tobytes("png")))
            pil_image.save(image_path)

            # For simplicity, using a placeholder bounding box and caption
            # In a real implementation, you would extract actual bounding boxes and captions
            # WARNING: extract bounding boxes and captions
            bbox = [0, 0, 0, 0]
            caption = f"Image {image_no + 1} on page {page_num + 1}"

            chunk = ImageChunk(
                image_path=image_path,
                image_caption=caption,
                image_bbox=bbox,
                chunk_id=image_no + 1,
            )
            chunks.append(chunk)

            pix = None  # Free up memory

        return chunks
