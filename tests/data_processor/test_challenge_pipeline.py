import argparse
from pathlib import Path
import pandas as pd
from loguru import logger
import pickle

# 文档经过docling处理后，进行分块与向量化，并针对问题进行检索


from src.data_processor.converters.challenge_pipeline import Pipeline, RunConfig
from src.config.api_config import get_llm_model
import logging
from pyprojroot import here

# 相关设置
ollama_qwen_config2 = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=5,  # 问题回答的并发请求数
    top_n_retrieval=14,  # 不进行重排序时，检索结果的数量
    submission_name="Ollama Qwen v.2",  # 提交名称
    pipeline_details="vDB + Parent Document Retrieval + reranking + SO CoT; llm = qwen2.5:72b; embedding = bge-m3:latest",  # 提交详情
    api_provider="ollama",
    answering_model="qwen2.5:72b",  # 使用的模型
    config_suffix="_ollama_qwen",  # 答案文件名后缀
    use_vector_dbs="faiss",  # 使用的向量数据库
)

ollama_qwen_config_now = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=5,
    top_n_retrieval=14,
    submission_name="Ollama Qwen v.2",
    pipeline_details="milvus + Parent Document Retrieval + reranking + SO CoT; llm = qwen2.5:72b; embedding = bge-m3:latest",
    api_provider="ollama",
    answering_model="qwen2.5:72b",
    config_suffix="_ollama_qwen_milvus",
    use_vector_dbs="milvus",
)


ollama_qwen_config_rerank = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=1,
    top_n_retrieval=10,
    submission_name="milvus_VLLMReranker",
    pipeline_details="milvus + VLLMReranker; llm = qwen2.5:72b; embedding = bge-m3:latest",
    api_provider="ollama",
    answering_model="qwen2.5:72b",
    config_suffix="_ollama_qwen_milvus_VLLMReranker",
    use_vector_dbs="milvus",
)


vllm_qwen_config_now = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=1,
    top_n_retrieval=14,
    submission_name="vllm Qwen v.2",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval; llm = qwen2.5:72b; embedding = bge-m3:latest; milvus",
    api_provider="openai",
    answering_model="Qwen2.5-72B-Instruct",
    config_suffix="_vllm_qwen_milvus",
    use_vector_dbs="milvus",
    collection_name="challenge_data_vllm",
)
vllm_qwen_config_now.answering_model = get_llm_model()


def test_full_pipeline(root_path):
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 初始化pipeline
    # pipeline = Pipeline(root_path, run_config=ollama_qwen_config2)
    pipeline = Pipeline(root_path, run_config=vllm_qwen_config_now)
    # pipeline = Pipeline(root_path, questions_file_name="problem_qa.json", run_config=ollama_qwen_config2)

    print("\n=== 步骤1: 解析PDF报告 ===")
    pipeline.parse_pdf_reports_sequential()

    print("\n=== 步骤2: 合并报告 ===")
    pipeline.merge_reports()

    print("\n=== 步骤3: 导出报告为markdown ===")
    pipeline.export_reports_to_markdown()

    print("\n=== 步骤4: 分块报告 ===")
    pipeline.chunk_reports()

    print("\n=== 数据向量化后存入pkl ===")
    output_dir = here() / "src" / "pkl_files"
    pipeline.chunk_to_pkl(output_dir)

    print("\n=== 步骤5: 创建向量数据库 ===")
    pipeline.create_vector_dbs()

    print("\n=== 步骤6: 处理问题 ===")
    pipeline.process_questions()

    print("\n=== 步骤7: 结果测评 ===")
    pipeline.get_rank()

    print("\n=== Pipeline测试完成! ===")


if __name__ == "__main__":

    root_path = here() / "src" / "resources" / "data"
    print(root_path)
    test_full_pipeline(root_path)
