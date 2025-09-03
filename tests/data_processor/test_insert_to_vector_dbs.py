import argparse
from pathlib import Path
from loguru import logger
import logging
import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

# docling处理后的pdf向量化后存入pkl


from src.utils.logging_utils import setup_logger
from src.data_processor.converters.challenge_pipeline import Pipeline, RunConfig


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
        "--pkl_path",
        type=Path,
        required=False,
        default="src/pkl_files/challenge_docling.pkl",
        help="data root path for the pipeline",
    )
    parser.add_argument(
        "--vector_db",
        type=str,
        required=False,
        default="milvus",
        help="Choose milvus or faiss, default: milvus",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    print(args)
    root_path = args.root_path

    run_config = RunConfig(
        use_vector_dbs=args.vector_db,
    )

    pipeline = Pipeline(root_path, run_config=run_config)
    print("\n=== 创建向量数据库 ===")
    pipeline.create_vector_dbs(pkl_path=args.pkl_path)


if __name__ == "__main__":
    main()

"""
python tests/data_processor/test_insert_to_vector_dbs.py \
    --root_path src/resources/challenge_data_mineru \
    --pkl_path src/pkl_files/challenge_docling.pkl \
    --vector_db milvus
"""
