"""Base processor module for document processing."""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union

from loguru import logger

from src.data_parser.document import Document


class BaseProcessor(ABC):
    """Abstract base class for document processors.

    This class defines the interface that all document processors must implement.
    It provides common functionality for processing documents and extracting structured content.
    """

    def __init__(self, output_path: Path) -> None:
        """Initialize the base processor.

        Args:
            output_path: Path to output directory for extracted content
            chunk_size: Maximum characters per text chunk
            overlap: Overlap between text chunks
        """
        self.output_path = output_path

        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Output directory created at {output_path}")

    def _find_pdf_files(self, input_path: Path | List[Path]) -> List[Path]:
        """Find PDF files in the input path.

        Args:
            input_path: Optional path to override self.input_path

        Returns:
            List of PDF file paths
        """
        pdf_files = []

        if isinstance(input_path, list):
            # Handle list of paths
            for p in input_path:
                if isinstance(p, Path) and p.suffix.lower() == ".pdf":
                    pdf_files.append(p)
                elif isinstance(p, str) and p.lower().endswith(".pdf"):
                    pdf_files.append(p)
        elif isinstance(input_path, Path):
            # Handle single Path object
            if input_path.is_file() and input_path.suffix.lower() == ".pdf":
                # Single PDF file
                pdf_files = [input_path]
            elif input_path.is_dir():
                # Directory containing PDFs
                pdf_files = [p for p in input_path.glob("*.pdf")]

        if not pdf_files:
            logger.error(f"No PDF files found in {input_path}")
        else:
            logger.info(f"Found {len(pdf_files)} PDF files")

        return pdf_files

    @abstractmethod
    def process_documents(self, input_path: Union[Path, List[Path]]) -> List[Document]:
        """Process all documents in the input path.

        Args:
            input_path: Path or list of paths to documents

        Returns:
            List of Document objects
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def process_single_document(self, document_path: Path) -> Document:
        """Process a single document.

        Args:
            document_path: Path to the document

        Returns:
            Document object containing structured content
        """
        raise NotImplementedError("Subclasses must implement this method.")
