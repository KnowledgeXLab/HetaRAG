import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import snapshot_download
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

MODEL_MAPPING = {
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
    "bge-reranker-v2-gemma": "BAAI/bge-reranker-v2-gemma",
    "ms-marco-TinyBERT-L-2-v2": "cross-encoder/ms-marco-TinyBERT-L-2-v2",
    "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
}


def setup_model(
    model_name: str,
    local_model_path: Optional[str] = None,
    force_download: bool = False,
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    设置模型和分词器，如果本地没有则自动下载

    Args:
        model_name: 模型名称
        local_model_path: 本地模型路径
        force_download: 是否强制重新下载

    Returns:
        model, tokenizer
    """
    # 获取实际的模型ID
    model_id = MODEL_MAPPING.get(model_name, model_name)

    # 设置默认本地路径
    if local_model_path is None:
        local_model_path = os.path.join("src/rerank/model_path", model_name)

    # 确保目录存在
    os.makedirs(local_model_path, exist_ok=True)

    try:
        # 检查本地是否已有模型
        if not force_download and os.path.exists(
            os.path.join(local_model_path, "config.json")
        ):
            logger.info(f"Loading model from local path: {local_model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        else:
            # 下载模型到本地
            logger.info(f"Downloading model {model_id} to {local_model_path}")
            snapshot_download(
                repo_id=model_id,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
            )

            # 加载模型和分词器
            model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
            tokenizer = AutoTokenizer.from_pretrained(local_model_path)

            logger.info(f"Model downloaded and loaded successfully")

        # 如果有GPU则移到GPU
        if torch.cuda.is_available():
            model = model.cuda()

        return model, tokenizer

    except Exception as e:
        logger.error(f"Error setting up model: {str(e)}")
        raise


def list_available_models():
    """列出所有可用的模型"""
    print("\nAvailable models:")
    for name, model_id in MODEL_MAPPING.items():
        print(f"- {name} ({model_id})")


if __name__ == "__main__":
    import torch
    import argparse

    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 命令行参数
    parser = argparse.ArgumentParser(description="Download and setup reranker models")
    parser.add_argument(
        "--model_name",
        type=str,
        default="bge-reranker-large",
        help="Name of the model to download",
    )
    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Force re-download even if model exists locally",
    )
    parser.add_argument(
        "--list_models", action="store_true", help="List all available models"
    )

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
    else:
        # 下载并设置模型
        model, tokenizer = setup_model(
            args.model_name, force_download=args.force_download
        )
        logger.info(f"Model {args.model_name} setup completed successfully")
