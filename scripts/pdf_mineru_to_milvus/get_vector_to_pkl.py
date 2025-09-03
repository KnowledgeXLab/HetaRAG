import argparse

import time
from datetime import timedelta
from src.data_processor.converters.pdf_mineru_to_chunk_converter import (
    process_directory,
)


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="统计指定文件夹中所有 PDF 文件的总页数、总大小、有效 PDF 文件数量和无法处理的文件数量"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="src/resources/data/pdf_reports",
        help="目标文件夹路径",
    )
    parser.add_argument(
        "--output_path", type=str, default="src/pkl_files", help="pkl输出路径"
    )
    parser.add_argument(
        "--image_embedding", action="store_true", help="是否进行图片embedding"
    )
    # 解析命令行参数
    args = parser.parse_args()

    # mineru处理后的pdf转pkl
    input = args.input_path
    output = args.output_path
    processor_type = "mineru"
    # 记录开始时间
    start_time = time.time()
    print(args.image_embedding)
    process_directory(
        input, output, processor_type, image_embedding=args.image_embedding
    )
    # 记录结束时间
    end_time = time.time()
    # 计算运行时长（秒）
    duration_seconds = end_time - start_time
    # 转换为可读格式（时:分:秒）
    duration_str = str(timedelta(seconds=duration_seconds))
    print(f"总耗时: {duration_str}")


if __name__ == "__main__":
    main()
