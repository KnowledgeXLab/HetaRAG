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
import logging
from pyprojroot import here


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the query rewrite test.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Test query rewrite functionality in the pipeline"
    )
    parser.add_argument(
        "--root_path", type=Path, required=True, help="data root path for the pipeline"
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
        "--answering_model",
        type=str,
        required=False,
        default="qwen2.5:72b",
        help="LLM to use for answer generation, default: qwen2.5:72b",
    )
    parser.add_argument(
        "--vector_db",
        type=str,
        required=False,
        default="milvus",
        help="Choose milvus or faiss, default: milvus",
    )
    parser.add_argument(
        "--rerank_model",
        type=str,
        required=False,
        default=None,
        help="Choose the model for reranker_model",
    )
    parser.add_argument(
        "--rerank_model_path",
        type=Path,
        required=False,
        default="src/rerank/model_path",
        help="rerank model path",
    )
    parser.add_argument(
        "--rerank_sample_size",
        type=int,
        required=False,
        default=36,
        help="rerank sample_size > top_n_retrieval",
    )
    parser.add_argument(
        "--team_email",
        type=str,
        required=False,
        default="",
        help="Record your email address",
    )
    parser.add_argument(
        "--api_provider",
        type=str,
        required=False,
        default="ollama",
        help="Choose api provider, default: ollama",
        choices=["ollama", "openai", "ibm", "vllm"],
    )
    # Query rewrite specific arguments
    parser.add_argument(
        "--query_rewrite_model",
        type=str,
        required=False,
        default="qwen2.5:72b",
        help="Model to use for query rewrite, default: qwen2.5:72b",
    )
    parser.add_argument(
        "--max_query_variations",
        type=int,
        required=False,
        default=3,
        help="Maximum number of query variations to generate, default: 3",
    )
    return parser.parse_args()


def main():
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    # 配置运行参数
    run_config = RunConfig(
        parent_document_retrieval=args.parent_document_retrieval,
        parallel_requests=1,
        top_n_retrieval=args.top_n_retrieval,
        submission_name=f"QueryRewrite_{args.api_provider}_{args.answering_model}_{args.vector_db}_{args.rerank_model}",
        pipeline_details=f"QueryRewrite + {args.vector_db} + {args.parent_document_retrieval} + reranking + SO CoT; llm = {args.answering_model}; embedding = bge-m3:latest",
        api_provider=args.api_provider,
        answering_model=args.answering_model,
        config_suffix=f"_query_rewrite_{args.api_provider}_{args.answering_model}_{args.vector_db}_{args.rerank_model}",
        use_vector_dbs=args.vector_db,
        # Rerank 配置
        rerank_method="huggingface",
        rerank_model=args.rerank_model,
        rerank_model_path=args.rerank_model_path,
        sample_size=args.rerank_sample_size,
        # Query rewrite 配置
        use_query_rewrite=True,  # 始终启用 query rewrite
        query_rewrite_model=args.query_rewrite_model,
        max_query_variations=args.max_query_variations,
    )

    # 初始化 pipeline
    # 使用相对路径而不是绝对路径
    root_path = Path("src/resources/data")
    # 修正PDF路径，指向实际的PDF文件位置
    pipeline = Pipeline(
        root_path, pdf_reports_dir_name="pdf_reports", run_config=run_config
    )

    print("\n=== 使用 Query Rewrite 处理问题 ===")
    pipeline.process_questions()

    print("\n=== 结果测评 ===")
    pipeline.get_rank()


if __name__ == "__main__":
    main()

"""
# 使用查询改写功能 + 重排序 (Ollama)
python tests/query_rewrite/test_query_rewrite.py \
    --root_path src/resources/data \
    --parent_document_retrieval \
    --top_n_retrieval 14 \
    --query_rewrite_model qwen2.5:72b \
    --max_query_variations 3 \
    --vector_db milvus \
    --rerank_model bge-reranker-large \
    --api_provider ollama

# 使用查询改写功能 + 重排序 (VLLM)
python tests/query_rewrite/test_query_rewrite.py \
    --root_path src/resources/data \
    --parent_document_retrieval \
    --top_n_retrieval 14 \
    --query_rewrite_model Qwen2.5-72B-Instruct \
    --max_query_variations 3 \
    --vector_db milvus \
    --rerank_model bge-reranker-large \
    --api_provider vllm

# 仅使用查询改写功能 (VLLM)
python tests/query_rewrite/test_query_rewrite.py \
    --root_path src/resources/data \
    --parent_document_retrieval \
    --top_n_retrieval 14 \
    --query_rewrite_model Qwen2.5-72B-Instruct \
    --max_query_variations 3 \
    --vector_db milvus \
    --api_provider vllm

# 使用faiss (VLLM)
python tests/query_rewrite/test_query_rewrite.py \
    --root_path src/resources/data \
    --parent_document_retrieval \
    --top_n_retrieval 14 \
    --query_rewrite_model Qwen2.5-72B-Instruct \
    --max_query_variations 3 \
    --vector_db faiss
"""
