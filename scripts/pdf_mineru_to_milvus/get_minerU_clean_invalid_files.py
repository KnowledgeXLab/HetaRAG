import os
import argparse
from pathlib import Path
from src.data_parser.mineru_pdf_parser import MineruPDFParser


def get_pdf_mineru_info(folder_path):
    """
    统计指定文件夹中所有 PDF 能够成功进行MinerU解析的 PDF 文件数量和无法解析的文件数量。

    :param folder_path: 源文件夹路径
    :return: 成功进行MinerU解析的 PDF 文件数量和无法解析的文件数量
    """

    pdf_count = 0
    valid_pdf_count = 0
    invalid_pdf_count = 0
    parser = MineruPDFParser()

    # 遍历文件夹及其所有子目录
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = Path(os.path.join(root, file))
                # print(type(file_path),file_path)
                pdf_count += 1
                try:
                    output_dir = file_path.parent / file_path.stem
                    parser.process_pdf(str(file_path), str(output_dir))
                    valid_pdf_count += 1
                except Exception as e:
                    print(f"无法处理文件 {file_path}: {e}")
                    output_dir = file_path.parent / file_path.stem
                    if os.path.exists(output_dir):
                        os.rmdir(output_dir)

                    # 增加无法处理的文件计数
                    invalid_pdf_count += 1

    print(f"有效 PDF 文件数量: {valid_pdf_count}")
    return pdf_count, invalid_pdf_count


import time
from datetime import timedelta


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="统计指定文件夹中所有 PDF 文件的总页数、总大小、有效 PDF 文件数量和无法处理的文件数量"
    )
    parser.add_argument("folder_path", type=str, help="目标文件夹路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.folder_path):
        print(f"路径 {args.folder_path} 不存在，请检查后重新输入。")
        return

    # 记录开始时间
    start_time = time.time()

    # 调用函数统计 PDF 信息
    print("开始处理PDF文件...")
    pdf_count, invalid_pdf_count = get_pdf_mineru_info(args.folder_path)

    # 记录结束时间
    end_time = time.time()

    # 计算运行时长（秒）
    duration_seconds = end_time - start_time

    # 转换为可读格式（时:分:秒）
    duration_str = str(timedelta(seconds=duration_seconds))

    # 打印总统计结果
    print(f"PDF 文件总数: {pdf_count}")
    print(f"无法解析的文件数量: {invalid_pdf_count}")
    print(f"总耗时: {duration_str}")


if __name__ == "__main__":
    main()


# CUDA_VISIBLE_DEVICES=3 nohup python -u scripts/get_minerU_clean_invalid_files.py /data/H-RAG/gov_decision/en-database-dash-clean > en-database-dash.out 2>&1 &
