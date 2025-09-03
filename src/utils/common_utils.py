from pathlib import Path
from loguru import logger
import os
import heapq


def find_pdf_files(input_path):
    """Find PDF files in the input path.

    Args:
        input_path: Optional path to override self.input_path

    Returns:
        List of PDF file paths
    """
    pdf_files = []

    if isinstance(input_path, list):
        # Handle list of paths
        for p in input_path:
            if isinstance(p, Path) and p.suffix.lower() == ".pdf":
                pdf_files.append(p)
            elif isinstance(p, str) and p.lower().endswith(".pdf"):
                pdf_files.append(p)
    elif isinstance(input_path, Path):
        # Handle single Path object
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            # Single PDF file
            pdf_files = [input_path]
        elif input_path.is_dir():
            # Directory containing PDFs
            pdf_files = [p for p in input_path.glob("*.pdf")]

    if not pdf_files:
        logger.error(f"No PDF files found in {input_path}")
    else:
        logger.info(f"Found {len(pdf_files)} PDF files")

    return pdf_files


def sort_large_file(
    input_file, output_file, chunk_size=100000
):  # chunk_size 可根据内存调整
    chunks = []
    try:
        with open(input_file, "r", encoding="utf-8") as infile:
            while True:
                lines = infile.readlines(chunk_size)
                if not lines:
                    break
                lines.sort()  # 对每一块数据进行排序

                chunk_filename = f"temp_chunk_{len(chunks)}.txt"
                chunks.append(chunk_filename)
                with open(chunk_filename, "w", encoding="utf-8") as chunk_file:
                    chunk_file.writelines(lines)

        # 合并排序后的块
        cnt = 1
        with open(output_file, "w", encoding="utf-8") as outfile:
            files = [open(f, "r", encoding="utf-8") for f in chunks]

            for line in heapq.merge(*files):
                cnt += 1
                outfile.write(line)
                if cnt % 10000 == 0:
                    print(f"merging process: {cnt}/{len(chunks)*chunk_size}")
            for f in files:
                f.close()

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 清理临时文件
        for chunk in chunks:
            try:
                os.remove(chunk)
            except FileNotFoundError:
                pass


def remove_duplicates(input_file, output_file):
    """
    从已排序的文本文件中去除重复行。

    Args:
        input_file: 已排序的输入文件路径。
        output_file: 输出无重复行的文件路径。
    """
    try:
        duplicate_cnt = 0
        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(output_file, "w", encoding="utf-8") as outfile,
        ):
            # 初始化前一行的内容
            previous_line = None
            index = 0
            for current_line in infile:
                index += 1
                # 如果当前行与前一行不同，则写入输出文件
                if current_line != previous_line:
                    outfile.write(current_line)
                    duplicate_cnt += 1
                    previous_line = current_line
                progress = index + 1
                if progress % 10000 == 0:
                    print(f"处理进度: {progress}")

        print(f"去除重复 {duplicate_cnt} 行完成")

    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 不存在。")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")


import re
import jieba


def custom_lower_fast(s):
    """中英文兼容的小写转换"""
    return s.lower() if s.isascii() else s  # 中文保持原样


def is_word_boundary(text, start, end):
    """自适应中英文词边界检测"""
    # 判断文本是否包含中文（包括扩展CJK字符）
    has_chinese = re.search(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df]", text)

    if has_chinese:
        # 中文模式：使用jieba分词检测词边界
        words = list(jieba.cut(text))
        current_pos = 0
        boundaries = set()

        # 构建词边界集合
        for word in words:
            boundaries.add(current_pos)  # 词开始位置
            boundaries.add(current_pos + len(word))  # 词结束位置
            current_pos += len(word)

        # 检查输入位置是否在分词边界上
        return start in boundaries or end in boundaries
    else:
        # 英文模式：使用正则表达式检测单词边界
        word_chars = r"\w"  # 仅字母、数字、下划线

        # 前字符检查
        prev_is_word = False
        if start > 0:
            prev_char = text[start - 1]
            prev_is_word = re.match(f"[{word_chars}]", prev_char, re.UNICODE)

        # 后字符检查
        next_is_word = False
        if end < len(text):
            next_char = text[end]
            next_is_word = re.match(f"[{word_chars}]", next_char, re.UNICODE)

        return not prev_is_word and not next_is_word


import json
from src.utils.file_utils import write


def deal_source(input_dir, output_path=""):
    root = input_dir
    res = []
    for dir in os.listdir(root):
        with open(os.path.join(root, dir), "r") as f:
            for uline in f:
                line = json.loads(uline)
                line["doc_name"] = Path(dir).stem
                res.append(line)
    # 将每个文档的chunk汇集起来，用于生成实体的描述
    # 当输入语料库路径为空时，根据输入路径生成语料库
    if output_path is None or output_path == "":
        # 数据预处理：根据 MinerU 处理结果，生成包含"id"、"page_idx"、"paragraph_idx"、"text"的 jsonl 文件
        output = os.path.join(os.path.dirname(root), "all_data.jsonl")
        write(output, res)
    elif os.path.isfile(output_path):
        output = output_path
        write(output, res)
    elif os.path.isdir(output_path):
        output = os.path.join(output_path, "all_data.jsonl")
        write(output, res)

    return output
