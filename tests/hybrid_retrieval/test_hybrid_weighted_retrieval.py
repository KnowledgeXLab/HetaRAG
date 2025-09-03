import os
import sys
import argparse
import json
from pathlib import Path
from pyprojroot import here

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor.converters.challenge_pipeline import Pipeline, RunConfig
from src.utils.pdf2chuck.hybrid_weighted_retrieval import HybridWeightedRetriever
from src.utils.pdf2chuck.ingestion import VectorDBIngestor


def check_and_build_es(documents_dir, es_index):
    try:
        from src.database.db_connection import es_connection
        from elasticsearch_dsl import Index
        from src.database.operations.elastic_operations import delete

        client = es_connection()
        index = Index(es_index)

        # 检查ES索引是否存在
        if not index.exists():
            print(f"[INFO] ES索引 {es_index} 不存在，正在创建...")
            from src.database.operations.elastic_operations import (
                upload_es,
                TextDocument,
            )

            TextDocument.init()  # 初始化索引
            upload_es(str(documents_dir), client)
            print("[INFO] ES索引构建完成。")
        else:
            print(f"[INFO] ES索引 {es_index} 已存在，跳过构建。")
    except Exception as e:
        print(f"[WARN] ES检查失败: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hybrid Weighted Retrieval Pipeline Test"
    )
    parser.add_argument(
        "--root_path",
        type=Path,
        default=here() / "src/resources/data",
        help="数据根路径",
    )
    parser.add_argument(
        "--es_index", type=str, default="knowledge_test", help="Elasticsearch索引名"
    )
    parser.add_argument(
        "--collection_name", type=str, default="challenge_data", help="Milvus集合名"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="加权融合参数alpha，越大越偏向向量分数"
    )
    parser.add_argument("--top_k", type=int, default=5, help="返回top_k条结果")
    parser.add_argument(
        "--submission_name", type=str, help="提交名称（可选，默认自动生成）"
    )
    parser.add_argument(
        "--parent_document_retrieval",
        action="store_true",
        help="使用页面级别检索（默认使用chunk级别）",
    )
    return parser.parse_args()


def generate_es_pkl_from_jsons(json_dir: Path, output_pkl: Path):
    """
    批量将json目录下所有分块json转为适配ES的pkl文件
    """
    import json, pickle
    from tqdm import tqdm

    all_chunks = []
    for json_file in tqdm(list(json_dir.glob("*.json")), desc="处理json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            filename = data.get("metainfo", {}).get("sha1_name", json_file.stem)
            for chunk in data.get("content", {}).get("chunks", []):
                all_chunks.append(
                    {
                        "filename": filename,
                        "chunk_id": chunk.get("id"),
                        "text": chunk.get("text"),
                        "page": chunk.get("page", None),  # 添加页面信息
                    }
                )
    with open(output_pkl, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"[INFO] 已生成ES专用pkl: {output_pkl}，共{len(all_chunks)}条")


def main():
    args = parse_args()
    root_path = args.root_path
    vector_db_dir = root_path / "databases/vector_dbs"
    documents_dir = root_path / "databases/chunked_reports"
    questions_file = root_path / "questions.json"

    # === 自动检测并生成适配ES的pkl文件（严格在databases/chunked_reports目录下） ===
    es_pkl = documents_dir / "es_chunks.pkl"
    if not es_pkl.exists():
        print(f"[INFO] 未检测到ES专用pkl，正在生成...")
        generate_es_pkl_from_jsons(documents_dir, es_pkl)
    else:
        print(f"[INFO] 已检测到ES专用pkl: {es_pkl}")

    # 只检查并自动建ES索引
    check_and_build_es(documents_dir, args.es_index)

    # 配置Pipeline，批量处理所有问题
    # 自动生成提交名称，包含alpha和top_k信息
    if args.submission_name:
        submission_name = args.submission_name
    else:
        submission_name = f"HybridWeightedRetrieval_alpha{args.alpha}_top{args.top_k}"

    run_config = RunConfig(
        parent_document_retrieval=args.parent_document_retrieval,
        parallel_requests=1,
        top_n_retrieval=args.top_k,
        submission_name=submission_name,
        pipeline_details=f"HybridWeightedRetrieval + alpha={args.alpha} + top_k={args.top_k} + {'page_level' if args.parent_document_retrieval else 'chunk_level'}",
        api_provider="vllm",
        answering_model="Qwen2.5-72B-Instruct",
        config_suffix=f"_hybrid_weighted",
        use_vector_dbs="milvus",
        collection_name=args.collection_name,
        use_hybrid_retriever=True,
        hybrid_es_index=args.es_index,
        hybrid_alpha=args.alpha,
    )
    pipeline = Pipeline(root_path, run_config=run_config)

    print(f"\n=== 使用 HybridWeightedRetriever 处理所有问题 ===")
    print(f"检索模式: {'页面级别' if args.parent_document_retrieval else 'Chunk级别'}")
    # 处理所有问题，直接用RunConfig控制是否混合检索
    pipeline.process_questions()

    print("\n=== 结果测评 ===")
    pipeline.get_rank()


if __name__ == "__main__":
    main()

"""
使用示例:
# 批量评测 - Chunk级别
python tests/hybrid_retrieval/test_hybrid_weighted_retrieval.py \
  --alpha 0.5 \
  --top_k 14 \
  --collection_name "challenge_data"

# 批量评测 - 页面级别
python tests/hybrid_retrieval/test_hybrid_weighted_retrieval.py \
  --alpha 0.5 \
  --top_k 14 \
  --collection_name "challenge_data" \
  --parent_document_retrieval
"""
