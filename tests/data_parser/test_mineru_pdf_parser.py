from src.data_parser.mineru_pdf_parser import get_pdf_mineru_info
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the report generation demo.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="mineru_parser")
    parser.add_argument(
        "--input_path", type=Path, required=True, help="PDF files in 'input_path' "
    )
    return parser.parse_args()


if __name__ == "__main__":
    # input_path = "src/resources/pdf"
    args = parse_args()
    input_path = args.input_path
    get_pdf_mineru_info(input_path)
