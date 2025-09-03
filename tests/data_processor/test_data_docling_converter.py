import argparse
from pathlib import Path
from loguru import logger
import logging

# docling处理后的pdf向量化后存入pkl


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
        "--output_path",
        type=Path,
        required=False,
        default="src/pkl_files/challenge_docling.pkl",
        help="data root path for the pipeline",
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


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    print(args)
    root_path = args.root_path
    pipeline = Pipeline(root_path, pdf_reports_dir_name=args.pdf_reports_dir_name)
    print("\n=== 数据向量化后存入pkl ===")
    pipeline.chunk_to_pkl(args.output_path)


if __name__ == "__main__":
    main()

"""
python tests/data_processor/test_data_docling_converter.py \
    --root_path src/resources/challenge_data \
    --output_path src/pkl_files/challenge_docling.pkl
"""
