import argparse
from pathlib import Path
import pandas as pd
from loguru import logger

from typing import Union, Dict, List, Optional

import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)
from src.utils.logging_utils import setup_logger
from src.data_processor.converters.challenge_pipeline import Pipeline, RunConfig
from src.rerank.reranker import ModelReranker, setup_models, VLLMReranker
import logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the report generation demo.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="challenge pipeline")
    parser.add_argument(
        "--root_path",
        type=Path,
        required=False,
        default=Path("src/resources/data"),
        help="data root path for the pipeline",
    )
    parser.add_argument(
        "--parent_document_retrieval",
        action="store_true",
    )
    parser.add_argument(
        "--top_n_retrieval",
        type=int,
        required=False,
        default=14,
        help="The number of documents to retrieve for each question, default: 14",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        required=False,
        default="ollama",
        help="Choose api provider, default: ollama",
        choices=["ollama", "openai", "ibm", "vllm"],
    )
    parser.add_argument(
        "--answering_model",
        type=str,
        required=False,
        default="qwen2.5:72b",
        help="LLM to use for report generation, default: qwen2.5:72b",
    )
    parser.add_argument(
        "--vector_db",
        type=str,
        required=False,
        default="milvus",
        help="Choose milvus or faiss, default: milvus",
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
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    root_path = args.root_path
    run_config = RunConfig(
        parent_document_retrieval=args.parent_document_retrieval,
        parallel_requests=1,
        top_n_retrieval=args.top_n_retrieval,
        submission_name=f"{args.answering_model}",
        pipeline_details=f"{args.vector_db} + {args.parent_document_retrieval} + SO CoT; llm = {args.answering_model}; embedding = bge-m3:latest",
        api_provider=args.api_provider,
        answering_model=args.answering_model,
        config_suffix=f"_{args.api_provider}_{args.answering_model}",
        use_vector_dbs=args.vector_db,
        rerank_method=None,
    )

    pipeline = Pipeline(root_path, run_config=run_config)

    print("\n=== 处理问题 ===")
    pipeline.process_questions()

    print("\n=== 结果测评 ===")
    pipeline.get_rank()


if __name__ == "__main__":
    main()


"""
# without rerank
python tests/rerank/test_without_rerank.py \
    --root_path src/resources/data \
    --parent_document_retrieval \
    --top_n_retrieval 14 \
    --answering_model Qwen3-32B \
    --api_provider vllm
"""
