from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import os
import time
import json
import pandas as pd
import logging

from src.data_parser.docling_pdf_parser import PDFParser
from src.data_parser.docling_parsed_reports_merging import PageTextPreparation
from src.utils.pdf2chuck.text_splitter import TextSplitter
from src.utils.pdf2chuck.ingestion import VectorDBIngestor, BM25Ingestor, VectorIngestor
from src.utils.pdf2chuck.questions_processing import (
    QuestionsProcessor,
    QuestionsProcessorMilvus,
)


@dataclass
class PipelineConfig:
    def __init__(
        self,
        root_path: Path,
        pdf_reports_dir_name: str = "pdf_reports",
        serialized: bool = False,
        config_suffix: str = "",
    ):
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        self.pdf_reports_dir = root_path / pdf_reports_dir_name

        self.answers_file_path = root_path / f"answers{config_suffix}.json"
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"

        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        self.parsed_reports_dirname = "01_parsed_reports"
        self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = (
            self.debug_data_path / self.parsed_reports_debug_dirname
        )
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = (
            self.debug_data_path / self.reports_markdown_dirname
        )


@dataclass
class RunConfig:
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 10
    team_email: str = "79250515615@yandex.com"
    submission_name: str = "Ilia_Ris vDB + SO CoT"
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = "ollama"
    answering_model: str = "qwen2.5:72b"
    config_suffix: str = ""
    use_vector_dbs: str = "milvus"


class Pipeline:
    def __init__(
        self,
        root_path: Path,
        pdf_reports_dir_name: str = "pdf_reports",
        run_config: RunConfig = RunConfig(),
    ):
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(
        self, root_path: Path, pdf_reports_dir_name: str
    ) -> PipelineConfig:
        """Initialize paths configuration based on run config settings"""
        return PipelineConfig(
            root_path=root_path,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix,
        )

    def _convert_json_to_csv_if_needed(self):
        """
        Checks if subset.json exists in root dir and subset.csv is absent.
        If so, converts the JSON to CSV format.
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"

        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                df.to_csv(csv_path, index=False)

            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    # Docling automatically downloads some models from huggingface when first used
    # I wanted to download them prior to running the pipeline and created this crutch
    @staticmethod
    def download_docling_models():
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self):
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """Parse PDF reports in parallel using multiple processes.

        Args:
            chunk_size: Number of PDFs to process in each worker
            num_workers: Number of parallel worker processes to use
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path,
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))

        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size,
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def merge_reports(self):
        """Merge complex JSON reports into a simpler structure with a list of pages, where all text blocks are combined into a single string."""
        ptp = PageTextPreparation(
            use_serialized_tables=self.run_config.use_serialized_tables
        )
        _ = ptp.process_reports(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.merged_reports_path,
        )
        print(f"Reports saved to {self.paths.merged_reports_path}")

    def export_reports_to_markdown(self):
        """Export processed reports to markdown format for review."""
        ptp = PageTextPreparation(
            use_serialized_tables=self.run_config.use_serialized_tables
        )
        ptp.export_to_markdown(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.reports_markdown_path,
        )
        print(f"Reports saved to {self.paths.reports_markdown_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """Split processed reports into smaller chunks for better processing."""
        text_splitter = TextSplitter()

        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path

        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            serialized_tables_dir,
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")

    def chunk_to_pkl(self, output_dir):
        """Create pkl from chunked reports."""
        input_dir = self.paths.documents_dir
        pdf_path = self.paths.pdf_reports_dir
        vdb_ingestor = VectorIngestor()
        vdb_ingestor.process_reports_to_pkl(input_dir, output_dir, pdf_path)
        print(f"Remaining data saved to {output_dir}")

    def create_vector_dbs(self):
        """Create vector databases from chunked reports."""
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")

    def create_bm25_db(self):
        """Create BM25 database from chunked reports."""
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def parse_pdf_reports(
        self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10
    ):
        if parallel:
            self.parse_pdf_reports_parallel(
                chunk_size=chunk_size, max_workers=max_workers
            )
        else:
            self.parse_pdf_reports_sequential()

    def process_parsed_reports(self):
        """Process already parsed PDF reports through the pipeline:
        1. Merge to simpler JSON structure
        2. Export to markdown
        3. Chunk the reports
        4. Create vector databases
        """
        print("Starting reports processing pipeline...")

        print("Step 1: Merging reports...")
        self.merge_reports()

        print("Step 2: Exporting reports to markdown...")
        self.export_reports_to_markdown()

        print("Step 3: Chunking reports...")
        self.chunk_reports()

        print("Step 4: Creating vector databases...")
        self.create_vector_dbs()

        print("Reports processing pipeline completed successfully!")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        Returns the next available filename by adding a numbered suffix if the file exists.
        Example: If answers.json exists, returns answers_01.json, etc.
        """
        if not base_path.exists():
            return base_path

        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename

            if not new_path.exists():
                return new_path
            counter += 1


ollama_qwen_config2 = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ollama Qwen v.2",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = qwen2.5:72b; embedding = bge-m3:latest",
    api_provider="ollama",
    answering_model="qwen2.5:72b",
    config_suffix="_ollama_qwen",
    use_vector_dbs="faiss",
)

ollama_qwen_config_now = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ollama Qwen v.2",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = qwen2.5:72b; embedding = bge-m3:latest; milvus",
    api_provider="ollama",
    answering_model="qwen2.5:72b",
    config_suffix="_ollama_qwen_milvus",
    use_vector_dbs="milvus",
)
