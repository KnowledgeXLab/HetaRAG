import argparse
from pathlib import Path
import shutil
import sys, os

# 获取项目根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)
# print(project_root)

from src.data_parser.pure_pdf_docling_parser import (
    Pipeline,
    ollama_qwen_config2,
    ollama_qwen_config_now,
)
import logging
from pyprojroot import here


def organize_pdf_files(data_path: Path):
    """
    递归遍历指定路径下的所有 PDF 文件，将它们移动到 pdf_reports 文件夹中。
    如果 pdf_reports 文件夹不存在，则创建它。
    移动完成后，删除路径目录下的其他文件和文件夹，只保留 pdf_reports 文件夹。
    """
    # 创建 pdf_reports 文件夹（如果不存在）
    pdf_reports_dir = data_path / "pdf_reports"
    pdf_reports_dir.mkdir(exist_ok=True)

    # 递归查找所有 PDF 文件
    pdf_files = list(data_path.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in the directory: {data_path}")

    # 移动 PDF 文件到 pdf_reports 文件夹
    for pdf_file in pdf_files:
        new_path = pdf_reports_dir / pdf_file.name
        shutil.move(str(pdf_file), str(new_path))
        print(f"Moved {pdf_file} to {new_path}")

    # 删除路径目录下的其他文件和文件夹，只保留 pdf_reports 文件夹
    for item in data_path.iterdir():
        if item != pdf_reports_dir:
            if item.is_file():
                item.unlink()
                print(f"Deleted file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Deleted directory: {item.name}")

    print(f"All PDF files have been moved to {pdf_reports_dir}")
    print(f"Original directory now contains only the 'pdf_reports' folder")


def test_full_pipeline(root_path: Path):
    # 设置日志级别
    logging.basicConfig(level=logging.INFO)

    # 初始化pipeline
    pipeline = Pipeline(root_path, run_config=ollama_qwen_config_now)

    print("\n=== 步骤1: 解析PDF报告 ===")
    pipeline.parse_pdf_reports_sequential()

    print("\n=== 步骤2: 合并报告 ===")
    pipeline.merge_reports()

    print("\n=== 步骤3: 导出报告为markdown ===")
    pipeline.export_reports_to_markdown()

    print("\n=== 步骤4: 分块报告 ===")
    pipeline.chunk_reports()

    print("\n=== 数据向量化后存入pkl ===")
    # output_dir = here() / "src" / "pkl_files"
    pipeline.chunk_to_pkl(root_path)

    print("\n=== Pipeline测试完成! ===")


if __name__ == "__main__":
    # 使用 argparse 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Run the full pipeline with a specified data directory."
    )
    parser.add_argument(
        "data_path", type=str, help="Path to the data directory (e.g., data/pdf)"
    )

    args = parser.parse_args()

    # 将输入路径转换为绝对路径
    data_path = Path(args.data_path).resolve()
    print(f"Using data directory: {data_path}")

    # 组织 PDF 文件
    organize_pdf_files(data_path)

    # 调用 test_full_pipeline 函数
    test_full_pipeline(data_path)
