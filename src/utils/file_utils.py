import json
import os
import json
import csv
import yaml
import pandas as pd
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor
import os
from src.utils.logging_utils import setup_logger

# 初始化日志记录器
logger = setup_logger("file_utils")


# 如果目录不存在，递归创建该目录
def create_if_not_exist(path):
    if not os.path.exists(path):  # 如果目录不存在，递归创建该目录
        os.makedirs(path, exist_ok=True)


def read(
    file_path, chunk_size=1024 * 1024 * 10000, line_threshold=1000000, encoding="utf-8"
):
    """
    根据文件扩展名读取文件内容，并自动判断是否使用多线程。
    TODO: 修正多线程读取功能，暂不可用，所以阈值设置的很大

    参数:
    - file_path (str): 文件路径。
    - chunk_size (int): 当文件大于此大小时，使用多线程读取，单位为字节，默认为 10000MB。
    - line_threshold (int): 当文件行数大于此值时，使用多线程读取，默认为 1000000 行。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - list 或 str: 读取的文件内容。
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == ".txt":
            return read_txt(file_path, chunk_size, encoding)
        elif file_ext == ".csv":
            return read_csv(file_path, chunk_size, encoding)
        elif file_ext == ".json":
            return read_json(file_path, encoding)
        elif file_ext == ".yaml" or file_ext == ".yml":
            return read_yaml(file_path, encoding)
        elif file_ext == ".xlsx":
            return read_xlsx(file_path, encoding)
        elif file_ext == ".md":
            return read_markdown(file_path, encoding)
        elif file_ext == ".jsonl":
            return read_jsonl(file_path, chunk_size, line_threshold, encoding)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        raise


def write(file_path, data, mode="a", chunk_size=1024 * 1024 * 100000, encoding="utf-8"):
    """
    根据文件扩展名写入数据到文件，并自动判断是否使用多线程。
    TODO: 修正多线程保存功能，暂不可用，所以阈值设置的很大

    参数:
    - file_path (str): 文件路径。
    - data (any): 要写入的数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），也可以选择 'w'（覆盖模式）。
    - chunk_size (int): 当数据量较大时，使用多线程写入，单位为字节，默认为 10MB。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """

    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)  # 创建目录

    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == ".txt":
            write_txt(file_path, data, mode, chunk_size, encoding)
        elif file_ext == ".csv":
            write_csv(file_path, data, mode, chunk_size, encoding)
        elif file_ext == ".json":
            write_json(file_path, data, mode, encoding)
        elif file_ext == ".yaml" or file_ext == ".yml":
            write_yaml(file_path, data, mode, encoding)
        elif file_ext == ".xlsx":
            write_xlsx(file_path, data, encoding)
        elif file_ext == ".md":
            write_markdown(file_path, data, mode, encoding)
        elif file_ext == ".jsonl":
            write_jsonl(file_path, data, mode, chunk_size, encoding)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")
    except Exception as e:
        logger.error(f"写入文件 {file_path} 时出错: {e}")
        raise


# 读取文件函数
def read_txt(file_path, chunk_size, encoding="utf-8"):
    """
    读取文本文件。

    参数:
    - file_path (str): 文件路径。
    - chunk_size (int): 当文件大于此大小时，使用多线程读取。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - str: 文件内容。
    """
    file_size = os.path.getsize(file_path)
    if file_size > chunk_size:
        logger.info(f"文件 {file_path} 较大，共{file_size} 字节，使用多线程读取")
        return read_txt_multithread(file_path, chunk_size, encoding)
    else:
        logger.info(f"读取文本文件: {file_path}")
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()


def read_txt_multithread(file_path, chunk_size, encoding="utf-8"):
    """
    使用多线程读取大文本文件。

    参数:
    - file_path (str): 文件路径。
    - chunk_size (int): 文件大小阈值，单位为字节。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - str: 拼接后的文件内容。
    """
    file_size = os.path.getsize(file_path)
    num_chunks = (file_size // chunk_size) + 1
    logger.info(f"分割文件为 {num_chunks} 个块进行多线程读取")

    # 获取合适的线程数
    cpu_count = os.cpu_count() // 2  # 限制使用一半的CPU核心数
    num_threads = min(cpu_count, num_chunks)

    chunks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 将文件分块读取并并行处理
        futures = [
            executor.submit(
                read_chunk, file_path, i * chunk_size, (i + 1) * chunk_size, encoding
            )
            for i in range(num_chunks)
        ]
        for future in tqdm(futures, desc="读取文件进度"):
            chunks.append(future.result())

    return "".join(chunks)


def read_chunk(file_path, start, end, encoding="utf-8"):
    """
    读取文件的某个块。

    参数:
    - file_path (str): 文件路径。
    - start (int): 起始字节位置。
    - end (int): 结束字节位置。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - str: 读取的块数据。
    """
    with open(file_path, "r", encoding=encoding) as f:
        f.seek(start)
        return f.read(end - start)


def read_csv(file_path, chunk_size, encoding="utf-8"):
    """
    读取CSV文件。

    参数:
    - file_path (str): 文件路径。
    - chunk_size (int): 当文件大于此大小时，使用多线程读取。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - DataFrame: 读取的CSV文件内容。
    """
    file_size = os.path.getsize(file_path)
    if file_size > chunk_size:
        logger.info(f"文件 {file_path} 较大，启用多线程读取")
        return read_csv_multithread(file_path, chunk_size, encoding)
    else:
        logger.info(f"读取CSV文件: {file_path}")
        return pd.read_csv(file_path, encoding=encoding)


def read_csv_multithread(file_path, chunk_size, encoding="utf-8"):
    """
    使用多线程读取大CSV文件。

    参数:
    - file_path (str): 文件路径。
    - chunk_size (int): 文件大小阈值，单位为字节。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - DataFrame: 拼接后的CSV文件内容。
    """
    file_size = os.path.getsize(file_path)
    num_chunks = (file_size // chunk_size) + 1
    logger.info(f"分割文件为 {num_chunks} 个块进行多线程读取")

    cpu_count = os.cpu_count() // 2  # 限制使用一半的CPU核心数
    num_threads = min(cpu_count, num_chunks)

    data_chunks = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                read_csv_chunk,
                file_path,
                i * chunk_size,
                (i + 1) * chunk_size,
                encoding,
            )
            for i in range(num_chunks)
        ]
        for future in tqdm(futures, desc="读取CSV进度"):
            data_chunks.append(future.result())

    return pd.concat(data_chunks, ignore_index=True)


def read_csv_chunk(file_path, start, end, encoding="utf-8"):
    """
    读取CSV文件的某一块。

    参数:
    - file_path (str): 文件路径。
    - start (int): 起始字节位置。
    - end (int): 结束字节位置。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - DataFrame: 该块读取的内容。
    """
    return pd.read_csv(
        file_path, skiprows=range(1, start), nrows=end - start, encoding=encoding
    )


def read_json(file_path, encoding="utf-8"):
    """
    读取JSON文件。

    参数:
    - file_path (str): 文件路径。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - dict: 读取的JSON文件内容。
    """
    with open(file_path, "r", encoding=encoding) as f:
        logger.info(f"读取JSON文件: {file_path}")
        return json.load(f)


def read_yaml(file_path, encoding="utf-8"):
    """
    读取YAML文件。

    参数:
    - file_path (str): 文件路径。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - dict: 读取的YAML文件内容。
    """
    with open(file_path, "r", encoding=encoding) as f:
        logger.info(f"读取YAML文件: {file_path}")
        return yaml.safe_load(f)


def read_xlsx(file_path, encoding="utf-8"):
    """
    读取Excel文件。

    参数:
    - file_path (str): 文件路径。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - DataFrame: 读取的Excel文件内容。
    """
    logger.info(f"读取Excel文件: {file_path}")
    return pd.read_excel(file_path)


def read_markdown(file_path, encoding="utf-8"):
    """
    读取Markdown文件。

    参数:
    - file_path (str): 文件路径。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - str: 读取的Markdown文件内容。
    """
    with open(file_path, "r", encoding=encoding) as f:
        logger.info(f"读取Markdown文件: {file_path}")
        return f.read()


def read_jsonl(file_path, chunk_size, line_threshold, encoding="utf-8"):
    """
    读取JSONL文件，支持多线程处理。

    参数:
    - file_path (str): 文件路径。
    - chunk_size (int): 文件大小阈值，单位为字节。
    - line_threshold (int): 行数阈值，超过此值使用多线程。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - list: 读取的JSONL文件内容。
    """
    file_size = os.path.getsize(file_path)
    num_lines = sum(1 for _ in open(file_path, "r", encoding=encoding))

    if file_size > chunk_size or num_lines > line_threshold:
        logger.info(f"文件 {file_path} 较大，启用多线程读取")
        return read_jsonl_multithread(file_path, chunk_size, num_lines, encoding)
    else:
        logger.info(f"读取JSONL文件: {file_path}")
        with open(file_path, "r", encoding=encoding) as f:
            return [json.loads(line) for line in f]


def read_jsonl_multithread(file_path, chunk_size, num_lines, encoding="utf-8"):
    """
    使用多线程读取大JSONL文件。

    参数:
    - file_path (str): 文件路径。
    - chunk_size (int): 文件大小阈值，单位为字节。
    - num_lines (int): 文件行数。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - list: 读取的JSONL文件内容。
    """
    chunks = []
    lines_per_chunk = num_lines // (os.cpu_count() // 2)  # 使用一半CPU核心数

    cpu_count = os.cpu_count() // 2  # 限制使用一半的CPU核心数
    num_threads = min(cpu_count, (num_lines // lines_per_chunk) + 1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                read_jsonl_chunk,
                file_path,
                i * lines_per_chunk,
                (i + 1) * lines_per_chunk,
                encoding,
            )
            for i in range(num_threads)
        ]
        for future in tqdm(futures, desc="读取JSONL文件进度"):
            chunks.append(future.result())

    return [item for sublist in chunks for item in sublist]


def read_jsonl_chunk(file_path, start_line, end_line, encoding="utf-8"):
    """
    读取JSONL文件的某一块。

    参数:
    - file_path (str): 文件路径。
    - start_line (int): 起始行。
    - end_line (int): 结束行。
    - encoding (str): 文件编码格式，默认为 'utf-8'。

    返回:
    - list: 该块读取的内容。
    """
    with open(file_path, "r", encoding=encoding) as f:
        return [json.loads(line) for _, line in zip(range(start_line), f)][
            start_line:end_line
        ]


# 写入文件函数
def write_txt(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    写入文本文件。

    参数:
    - file_path (str): 文件路径。
    - data (str): 要写入的数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 当数据量较大时，使用多线程写入。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    data_size = len(data)
    if data_size > chunk_size:
        logger.info(f"数据较大，启用多线程写入")
        write_txt_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        logger.info(f"写入文本文件: {file_path}")
        with open(file_path, mode, encoding=encoding) as f:
            f.write(data)


def write_txt_multithread(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    使用多线程写入文本文件。

    参数:
    - file_path (str): 文件路径。
    - data (str): 要写入的数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 数据大小阈值，单位为字节。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2  # 限制使用一半的CPU核心数
    num_threads = min(cpu_count, len(chunks))

    logger.info(f"分割数据为 {len(chunks)} 个块进行多线程写入")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(write_chunk, file_path, chunk, mode, encoding)
            for chunk in chunks
        ]
        for future in tqdm(futures, desc="写入文件进度"):
            future.result()


def write_chunk(file_path, chunk, mode, encoding="utf-8"):
    """
    写入文本文件的某一块。

    参数:
    - file_path (str): 文件路径。
    - chunk (str): 要写入的文本块。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    with open(file_path, mode, encoding=encoding) as f:
        f.write(chunk)


def write_jsonl(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    写入JSONL文件，支持多线程处理。

    参数:
    - file_path (str): 文件路径。
    - data (list): 要写入的JSON数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 数据大小阈值，单位为字节。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    data_size = len(data)
    if data_size > chunk_size:
        logger.info(f"数据较大，启用多线程写入")
        write_jsonl_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        # if os.path.isdir(file_path):
        #     os.rmdir(file_path)  # 如果是空目录

        logger.info(f"写入JSONL文件: {file_path}")
        with open(file_path, mode, encoding=encoding) as f:
            for entry in data:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")


def write_jsonl_multithread(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    使用多线程写入JSONL文件。

    参数:
    - file_path (str): 文件路径。
    - data (list): 要写入的JSON数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 数据大小阈值，单位为字节。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2  # 限制使用一半的CPU核心数
    num_threads = min(cpu_count, len(chunks))

    logger.info(f"分割数据为 {len(chunks)} 个块进行多线程写入")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(write_jsonl_chunk, file_path, chunk, mode, encoding)
            for chunk in chunks
        ]
        for future in tqdm(futures, desc="写入JSONL文件进度"):
            future.result()


def write_jsonl_chunk(file_path, chunk, mode, encoding="utf-8"):
    """
    写入JSONL文件的某一块。

    参数:
    - file_path (str): 文件路径。
    - chunk (list): 要写入的数据块。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    with open(file_path, mode, encoding=encoding) as f:
        for entry in chunk:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")


def write_csv(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    写入CSV文件，支持多线程处理。

    参数:
    - file_path (str): 文件路径。
    - data (DataFrame): 要写入的CSV数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 当数据量较大时，使用多线程写入，单位为字节。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    data_size = data.memory_usage(index=True).sum()  # 计算DataFrame的大小
    if data_size > chunk_size:
        logger.info(f"数据较大，启用多线程写入")
        write_csv_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        logger.info(f"写入CSV文件: {file_path}")
        data.to_csv(file_path, mode=mode, encoding=encoding, index=False)


def write_csv_multithread(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    使用多线程写入大CSV文件。

    参数:
    - file_path (str): 文件路径。
    - data (DataFrame): 要写入的CSV数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 数据大小阈值，单位为字节。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    chunks = [data.iloc[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2  # 限制使用一半的CPU核心数
    num_threads = min(cpu_count, len(chunks))

    logger.info(f"分割数据为 {len(chunks)} 个块进行多线程写入")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(write_csv_chunk, file_path, chunk, mode, encoding)
            for chunk in chunks
        ]
        for future in tqdm(futures, desc="写入CSV文件进度"):
            future.result()


def write_csv_chunk(file_path, chunk, mode, encoding="utf-8"):
    """
    写入CSV文件的某一块。

    参数:
    - file_path (str): 文件路径。
    - chunk (DataFrame): 要写入的CSV数据块。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    chunk.to_csv(
        file_path, mode=mode, encoding=encoding, index=False, header=(mode == "w")
    )


def write_json(file_path, data, mode, encoding="utf-8"):
    """
    写入JSON文件，支持多线程处理。

    参数:
    - file_path (str): 文件路径。
    - data (list): 要写入的JSON数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    if mode == "a" and os.path.exists(file_path):
        with open(file_path, "r+", encoding=encoding) as f:
            existing_data = json.load(f)
            existing_data.extend(data)
            f.seek(0)
            json.dump(existing_data, f, ensure_ascii=False, indent=4)
    else:
        logger.info(f"写入JSON文件: {file_path}")
        with open(file_path, mode, encoding=encoding) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


def write_yaml(file_path, data, mode, encoding="utf-8"):
    """
    写入YAML文件。

    参数:
    - file_path (str): 文件路径。
    - data (dict): 要写入的YAML数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    logger.info(f"写入YAML文件: {file_path}")
    with open(file_path, mode, encoding=encoding) as f:
        yaml.dump(data, f, allow_unicode=True)


def write_csv(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    写入CSV文件，支持多线程处理。

    参数:
    - file_path (str): 文件路径。
    - data (DataFrame): 要写入的CSV数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 当数据量较大时，使用多线程写入，单位为字节。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    data_size = data.memory_usage(index=True).sum()  # 计算DataFrame的大小
    if data_size > chunk_size:
        logger.info(f"数据较大，启用多线程写入")
        write_csv_multithread(file_path, data, mode, chunk_size, encoding)
    else:
        logger.info(f"写入CSV文件: {file_path}")
        data.to_csv(file_path, mode=mode, encoding=encoding, index=False)


def write_csv_multithread(file_path, data, mode, chunk_size, encoding="utf-8"):
    """
    使用多线程写入大CSV文件。

    参数:
    - file_path (str): 文件路径。
    - data (DataFrame): 要写入的CSV数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - chunk_size (int): 数据大小阈值，单位为字节。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    chunks = [data.iloc[i : i + chunk_size] for i in range(0, len(data), chunk_size)]
    cpu_count = os.cpu_count() // 2  # 限制使用一半的CPU核心数
    num_threads = min(cpu_count, len(chunks))

    logger.info(f"分割数据为 {len(chunks)} 个块进行多线程写入")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(write_csv_chunk, file_path, chunk, mode, encoding)
            for chunk in chunks
        ]
        for future in tqdm(futures, desc="写入CSV文件进度"):
            future.result()


def write_csv_chunk(file_path, chunk, mode, encoding="utf-8"):
    """
    写入CSV文件的某一块。

    参数:
    - file_path (str): 文件路径。
    - chunk (DataFrame): 要写入的CSV数据块。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    chunk.to_csv(
        file_path, mode=mode, encoding=encoding, index=False, header=(mode == "w")
    )


def write_yaml(file_path, data, mode, encoding="utf-8"):
    """
    写入YAML文件。

    参数:
    - file_path (str): 文件路径。
    - data (dict): 要写入的YAML数据。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    logger.info(f"写入YAML文件: {file_path}")
    with open(file_path, mode, encoding=encoding) as f:
        yaml.dump(data, f, allow_unicode=True)


def write_xlsx(file_path, data, encoding="utf-8"):
    """
    写入Excel文件。

    参数:
    - file_path (str): 文件路径。
    - data (DataFrame): 要写入的Excel数据。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    logger.info(f"写入Excel文件: {file_path}")
    data.to_excel(file_path, index=False)


def write_markdown(file_path, data, mode, encoding="utf-8"):
    """
    写入Markdown文件。

    参数:
    - file_path (str): 文件路径。
    - data (str): 要写入的Markdown内容。
    - mode (str): 写入模式，默认为 'a'（追加模式），可以选择 'w'（覆盖模式）。
    - encoding (str): 写入文件的编码格式，默认为 'utf-8'。
    """
    logger.info(f"写入Markdown文件: {file_path}")
    with open(file_path, mode, encoding=encoding) as f:
        f.write(data)


# jsonl
def read_jsonl_list(file_path):
    """
    读取jsonl文件，并返回包含每行JSON对象的列表。

    :param file_path: .jsonl文件的路径
    :return: 包含每行JSON对象的列表
    """
    data = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # logger.info(f"读取JSONL文件: {file_path},数据存入List")
            for line in f:
                # 去除空行
                if line.strip():
                    json_obj = json.loads(line.strip())  # 解析每一行的JSON对象
                    data.append(json_obj)
        return data
    except Exception as e:
        logger.info(f"Error reading {file_path}: {e}")
        return None


def write_jsonl_list(data, path, mode="a", encoding="utf-8"):
    parent_dir = os.path.dirname(path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)  # 创建目录
    with open(path, mode, encoding=encoding) as f:
        # logger.info(f"将List数据写入JSONL文件: {path}")
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


# txt
def write_txt_line(path: str, data, mode="a"):
    with open(path, mode=mode, encoding="utf-8") as f:
        if isinstance(data, str):
            f.write(data)
        elif isinstance(data, list) or isinstance(data, set):
            for line in data:
                f.write(line + "\n")


def read_txt_line(path: str):
    # logger.info(f"按行读取文本文件: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.readlines()
