"""Document processing utilities for extracting text and images from PDFs."""

import io
import os
from pathlib import Path
from typing import Dict, List, Union

import fitz
from PIL import Image

from src.deepwriter.database.document import Document, Page, TextBlock, ImageBlock
from src.deepwriter.processors.base_processor import BaseProcessor


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

    def process_documents(self, input_path: Union[Path, List[Path]]) -> List[Document]:
        """Process all PDF documents in the input path.

        Args:
            input_path: Path to directory containing PDFs or list of PDF file paths

        Returns:
            List of Document objects containing structured content
        """
        pdf_files = self._find_pdf_files(input_path)
        return [self.process_single_document(pdf_file) for pdf_file in pdf_files]

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
            "path": str(pdf_file.absolute()),
            "num_pages": len(fitz_doc),
        }

        # Create Document object
        document = Document(metadata=metadata)

        # Process each page
        for page_num, fitz_page in enumerate(fitz_doc):
            # Extract text with encoding fallbacks
            text = (
                fitz_page.get_text().encode("ascii", "ignore").decode("utf-8", "ignore")
            )

            # Create Page object
            page = Page(
                text=text,
                page_num=page_num,
                metadata={
                    "page_number": page_num + 1,
                    "page_size": [fitz_page.rect.width, fitz_page.rect.height],
                },
                blocks=[],
            )

            # Process text and image chunks
            text_chunks = self._process_text_chunks(text, page_num)
            image_chunks = self._process_images(
                fitz_doc, fitz_page, page_num, pdf_file_name_wo_ext
            )

            # Add chunks to page
            page.blocks.extend(text_chunks)
            page.blocks.extend(image_chunks)

            # Add page to document
            document.pages.append(page)

        return document

    def _process_text_chunks(self, text: str, page_num: int) -> list[TextBlock]:
        """Process text into chunks.

        Args:
            text: Page text content
            page_num: Page number

        Returns:
            List of TextBlock objects
        """
        chunks = []
        chunked_text_dict = self._get_text_overlapping_chunk(
            text, self.chunk_size, self.overlap
        )

        for chunk_id, chunk_text in chunked_text_dict.items():
            # For simplicity, using a placeholder bounding box
            # In a real implementation, you would extract actual bounding boxes
            # TODO: extract bounding boxes and captions
            bbox = [0, 0, 0, 0]

            chunk = TextBlock(content=chunk_text, block_id=chunk_id, block_bbox=bbox)
            chunks.append(chunk)

        return chunks

    def _process_images(
        self,
        doc: fitz.Document,
        page: fitz.Page,
        page_num: int,
        pdf_file_name_wo_ext: str,
    ) -> List[ImageBlock]:
        """Process images on a page.

        Args:
            doc: PDF document
            page: PDF page
            page_num: Page number
            pdf_file_name_wo_ext: PDF filename without extension

        Returns:
            List of ImageBlock objects
        """
        chunks = []
        page_images = page.get_images()

        for image_no, image in enumerate(page_images):
            xref = image[0]
            pix = fitz.Pixmap(doc, xref)

            # Save image
            image_filename = (
                f"{pdf_file_name_wo_ext}_image_{page_num}_{image_no}_{xref}.png"
            )
            image_path = self.output_path.absolute() / image_filename

            pil_image = Image.open(io.BytesIO(pix.tobytes("png")))
            pil_image.save(image_path)

            # For simplicity, using a placeholder bounding box and caption
            # TODO: extract bounding boxes and captions
            bbox = [0, 0, 0, 0]
            caption = f"Image {image_no + 1} on page {page_num + 1}"

            chunk = ImageBlock(
                image_path=str(image_path),
                image_caption=caption,
                image_bbox=bbox,
                block_id=image_no + 1,
            )
            chunks.append(chunk)

            # Free up memory
            pix = None

        return chunks

    def _get_text_overlapping_chunk(
        self,
        text: str,
        character_limit: int = 1000,
        overlap: int = 100,
    ) -> Dict[int, str]:
        """Breaks text into overlapping chunks of specified size.

        Args:
            text: The text document to be chunked
            character_limit: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks

        Returns:
            Dictionary mapping chunk numbers to text chunks

        Raises:
            ValueError: If overlap is greater than character limit
        """
        if overlap > character_limit:
            raise ValueError("Overlap cannot be larger than character limit.")

        chunk_number = 1
        chunked_text_dict = {}

        for i in range(0, len(text), character_limit - overlap):
            end_index = min(i + character_limit, len(text))
            chunk = text[i:end_index]

            chunked_text_dict[chunk_number] = chunk.encode("ascii", "ignore").decode(
                "utf-8", "ignore"
            )
            chunk_number += 1

        return chunked_text_dict
