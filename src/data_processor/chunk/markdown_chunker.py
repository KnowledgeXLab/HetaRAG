"""
#### 使用说明：
该模块提供了处理Markdown文本的功能，包括语言检测、按标题分块和按长度合并分块。

#### 主要功能：
1. `detect_language`: 检测文本中的主语言，判断是否以中文或英文为主。
2. `split_markdown_by_headers`: 根据标题级别将Markdown文本分块。
3. `merge_markdown_chunks`: 将已分块的Markdown文本合并，支持按固定长度分割，同时考虑中文和英文标点。
4. `_split_text_by_size`: 按照给定长度对文本进行分割，支持中英文智能分割。

#### 参数说明：
1. detect_language 函数：
    - text (str): 输入的文本。
    - 返回值：'zh' 表示中文占主导，'en' 表示英文占主导。

2. split_markdown_by_headers 函数：
    - text (str): 输入的Markdown文本。
    - min_level (int): 处理的最小标题级别，默认为1。
    - max_level (int): 处理的最大标题级别，默认为6。
    - 返回值：一个字典列表，包含标题、内容和标题级别。

3. merge_markdown_chunks 函数：
    - chunks (List[Dict[str, any]]): 已分块的Markdown文本。
    - chunk_size (int): 每个块的最大长度，默认为1000。
    - chunk_overlap (int): 块之间的重叠长度，默认为200。
    - separator (str): 块之间的分隔符，默认为"\n\n"。
    - language (str): 语言选项，默认为'auto'，自动检测语言，'zh' 表示中文，'en' 表示英文。
    - 返回值：合并后的Markdown分块。

4. _split_text_by_size 函数：
    - text (str): 要分割的文本。
    - chunk_size (int): 每个分块的最大长度。
    - chunk_overlap (int): 分块之间的重叠部分。
    - header (str): 当前分块的标题。
    - level (int): 当前分块的标题级别。
    - language (str): 语言选项，默认为'auto'，可以手动指定'zh'或'en'。
    - 返回值：一个字典列表，每个字典包含一个分块的标题、内容和级别。

#### 注意事项：
1. 确保传入的文本符合Markdown格式。
2. 对于合并后的文本，可能需要手动调整块之间的分隔符和内容。
3. `chunk_size` 和 `chunk_overlap` 需要合理设置，以避免产生过小或过大的文本块。

"""

from typing import List, Dict, Any


def split_text_detect_language(text: str) -> str:
    """
    检测文本中的主语言，判断是否以中文或英文为主。

    参数:
        text (str): 输入的文本。

    返回值:
        str: 'zh' 表示中文占主导，'en' 表示英文占主导。
    """
    if not text:
        raise ValueError("输入文本不能为空")

    # 初始化计数器
    zh_count = 0
    en_count = 0

    # 遍历文本中的每个字符
    for char in text:
        # 检测中文字符（基于 Unicode 范围）
        if "\u4e00" <= char <= "\u9fff":  # 常见的汉字范围
            zh_count += 1
        # 检测英文字符（基于 ASCII 范围）
        elif char.isascii() and char.isalpha():
            en_count += 1

    # 判断主语言
    if zh_count > en_count:
        return "zh"  # 中文占主导
    else:
        return "en"  # 英文占主导


def split_markdown_by_headers(
    text: str, min_level: int = 1, max_level: int = 6
) -> List[Dict[str, Any]]:
    """
    功能描述: 将Markdown文本按标题级别分块。

    参数:
        text (str): 输入的Markdown文本。
        min_level (int): 处理的最小标题级别，默认为1。
        max_level (int): 处理的最大标题级别，默认为6。

    返回值:
        List[Dict[str, Any]]: 一个字典列表，包含标题、内容和标题级别。
    """
    if not isinstance(text, str):
        raise TypeError("text参数必须是字符串")
    if not isinstance(min_level, int) or not isinstance(max_level, int):
        raise TypeError("min_level和max_level必须是整数")
    if not text.strip():
        return []

    # 确保标题级别在1到6之间
    min_header_level = max(1, min(6, min_level))
    max_header_level = max(min_header_level, min(6, max_level))

    # 将文本按行分割
    lines = text.split("\n")
    chunks = []
    current_chunk = {"header": "", "content": [], "level": 0}

    # 遍历每一行
    for line in lines:
        is_header = False
        # 检查当前行是否为标题
        for level in range(min_header_level, max_header_level + 1):
            header_prefix = "#" * level + " "
            if line.strip().startswith(header_prefix):
                # 如果当前块有内容或标题，将其添加到块列表中
                if current_chunk["content"] or current_chunk["header"]:
                    chunks.append(
                        {
                            "header": current_chunk["header"],
                            "content": "\n".join(current_chunk["content"]).strip(),
                            "level": current_chunk["level"],
                        }
                    )
                # 初始化新的块
                current_chunk = {
                    "header": line.strip()[level + 1 :].strip(),
                    "content": [],
                    "level": level,
                }
                is_header = True
                break
        # 如果当前行不是标题，将其添加到当前块的内容中
        if not is_header:
            current_chunk["content"].append(line)

    # 将最后一个块添加到块列表中
    if current_chunk["content"] or current_chunk["header"]:
        chunks.append(
            {
                "header": current_chunk["header"],
                "content": "\n".join(current_chunk["content"]).strip(),
                "level": current_chunk["level"],
            }
        )
    return chunks


def merge_markdown_chunks(
    chunks: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
    language: str = "auto",
) -> List[Dict[str, Any]]:
    """
    功能描述: 合并Markdown分块，按固定长度分割，支持中英文标点。

    参数:
        chunks (List[Dict[str, Any]]): 已分块的Markdown文本。
        chunk_size (int): 每个块的最大长度，默认为1000。
        chunk_overlap (int): 块之间的重叠长度，默认为200。
        separator (str): 块之间的分隔符，默认为"\n\n"。
        language (str): 语言选项，默认为'auto'，自动检测语言，'zh' 表示中文，'en' 表示英文。

    返回值:
        List[Dict[str, Any]]: 合并后的Markdown分块。
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks参数必须是列表")
    if not isinstance(chunk_size, int) or not isinstance(chunk_overlap, int):
        raise TypeError("chunk_size和chunk_overlap必须是整数")
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size必须大于chunk_overlap")
    if chunk_size <= 0 or chunk_overlap < 0:
        raise ValueError("chunk_size必须大于0且chunk_overlap不能为负数")

    merged_chunks = []
    current_text = ""
    current_level = 0
    current_header = ""

    for chunk in chunks:
        chunk_text = (
            f"{chunk['header']}\n{chunk['content']}"
            if chunk["header"]
            else chunk["content"]
        )

        if chunk["level"] > 0 and current_text:
            if current_text.strip():
                merged_chunks.extend(
                    _split_text_by_size(
                        current_text.strip(),
                        chunk_size,
                        chunk_overlap,
                        current_header,
                        current_level,
                        language,
                    )
                )

            current_text = ""
            current_header = chunk["header"]
            current_level = chunk["level"]

        if current_text.strip():
            current_text += separator + chunk_text
        else:
            current_text = chunk_text

    if current_text.strip():
        merged_chunks.extend(
            _split_text_by_size(
                current_text.strip(),
                chunk_size,
                chunk_overlap,
                current_header,
                current_level,
                language,
            )
        )

    return merged_chunks


def _split_text_by_size(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    header: str,
    level: int,
    language: str = "auto",
) -> List[Dict[str, Any]]:
    """
    功能描述: 内部函数：按固定长度分割文本，支持中英文智能分割

    参数:
        text (str): 要分割的文本。
        chunk_size (int): 每个分块的最大长度。
        chunk_overlap (int): 分块之间的重叠部分。
        header (str): 当前分块的标题。
        level (int): 当前分块的标题级别。
        language (str): 语言选项，默认为'auto'，可以手动指定'zh'或'en'。

    返回值:
        List[Dict[str, Any]]: 按要求分割后的文本块。
    """
    EN_PUNCTUATION = " \n\t.,!?"
    ZH_PUNCTUATION = " \n\t。！？，、；"

    if language == "auto":
        lang = split_text_detect_language(text)
    else:
        lang = language.lower()

    if lang == "zh":
        split_chars = ZH_PUNCTUATION
    elif lang == "en":
        split_chars = EN_PUNCTUATION
    else:
        raise ValueError("Unsupported language type: {}".format(language))

    chunks = []
    text_length = len(text)
    start_pos = 0

    while start_pos < text_length:
        end_pos = min(start_pos + chunk_size, text_length)

        if end_pos < text_length:
            while end_pos > start_pos and text[end_pos - 1] not in split_chars:
                end_pos -= 1
            if end_pos == start_pos:
                end_pos = start_pos + chunk_size

        chunk_content = text[start_pos:end_pos].strip()
        if chunk_content:
            chunks.append(
                {
                    "header": header if not chunks else "",
                    "content": chunk_content,
                    "level": level if not chunks else 0,
                }
            )

        start_pos = end_pos - chunk_overlap if end_pos < text_length else end_pos

    return chunks


from typing import List, Dict

# 使用示例
if __name__ == "__main__":
    # 中英文混合示例
    markdown_text = """
# 主标题
这是一个混合内容的示例。It has both Chinese and English。这里是中文部分。

## 二级标题
更多内容在这里。这是中文句子！This is an English sentence with more words to test splitting.

没有标题的段落。Another paragraph without header。这里继续中文。
    """

    initial_chunks = split_markdown_by_headers(markdown_text)
    # 测试自动检测语言
    merged_chunks_auto = merge_markdown_chunks(
        initial_chunks, chunk_size=50, chunk_overlap=10
    )

    # 测试指定中文
    merged_chunks_zh = merge_markdown_chunks(
        initial_chunks, chunk_size=50, chunk_overlap=10, language="zh"
    )
    # 测试指定英文
    merged_chunks_en = merge_markdown_chunks(
        initial_chunks, chunk_size=50, chunk_overlap=10, language="en"
    )

    print("自动检测语言结果:")
    for i, chunk in enumerate(merged_chunks_auto):
        print(f"Chunk {i}:")
        print(f"Header: '{chunk['header']}'")
        print(f"Level: {chunk['level']}")
        print(f"Content: '{chunk['content']}'")
        print("-" * 50)

    print("\n指定中文分割结果:")
    for i, chunk in enumerate(merged_chunks_zh):
        print(f"Chunk {i}:")
        print(f"Header: '{chunk['header']}'")
        print(f"Level: {chunk['level']}")
        print(f"Content: '{chunk['content']}'")
        print("-" * 50)
