import argparse
from pathlib import Path
from loguru import logger
import logging

# 文档经过docling处理后，进行分块与向量化，并针对问题进行检索


from src.utils.logging_utils import setup_logger
from src.data_processor.converters.challenge_pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the report generation demo.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="docling_parser")
    parser.add_argument(
        "--root_path", type=Path, required=True, help="data root path for the pipeline"
    )
    parser.add_argument(
        "--pdf_reports_dir_name",
        type=str,
        required=False,
        default="pdf_reports",
        help="data root path for the pipeline",
    )

    parser.add_argument(
        "--team_email",
        type=str,
        required=False,
        default="",
        help="Record your email address",
    )
    return parser.parse_args()


def get_pdf_docling_info(input_path: Path = "src/resources/data/pdf_reports"):
    logging.basicConfig(level=logging.INFO)
    pdf_reports = Path(input_path).name
    root_path = Path(input_path).parent
    pipeline = Pipeline(root_path, pdf_reports_dir_name=pdf_reports)
    pipeline.parse_pdf_reports_sequential()
    pipeline.merge_reports()
    pipeline.export_reports_to_markdown()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    print(args)
    root_path = args.root_path
    pipeline = Pipeline(root_path, pdf_reports_dir_name=args.pdf_reports_dir_name)
    pipeline.parse_pdf_reports_sequential()
    pipeline.merge_reports()
    pipeline.export_reports_to_markdown()


if __name__ == "__main__":
    main()

"""
python tests/data_parser/test_docling_pdf_parser.py --root_path src/resources/test_data
"""
