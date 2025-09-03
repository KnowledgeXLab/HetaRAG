from src.utils.logging_utils import setup_logger
from src.utils.file_utils import read, write
from src.utils.common_utils import find_pdf_files
from pathlib import Path
import os
import tiktoken
from hashlib import md5

logger = setup_logger("get_chunk_data")


def merge_files(input_dir):
    input_path = Path(input_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_path}")
    pdf_files = find_pdf_files(input_path)
    data = []
    for pdf_file in pdf_files:
        base_dir = pdf_file.parent / pdf_file.stem
        logger.debug(f"Processing document: {pdf_file}")

        # Build related file paths
        content_json = base_dir / f"{pdf_file.stem}_content_list.json"
        raw_data = read(content_json)
        content = ""
        for item in raw_data:
            if item["type"] != "text":
                continue
            text = item["text"].strip()
            text_length = len(text)

            if text_length > 0:
                content += text
        data.append(content)
    return data


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def chunk_documents(
    docs,
    model_name="cl100k_base",
    max_token_size=512,
    overlap_token_size=64,
):
    ENCODER = tiktoken.get_encoding(model_name)
    tokens_list = ENCODER.encode_batch(docs, num_threads=16)

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token_ids = []
        lengths = []

        for start in range(0, len(tokens), max_token_size - overlap_token_size):
            chunk = tokens[start : start + max_token_size]
            chunk_token_ids.append(chunk)
            lengths.append(len(chunk))

        # 解码所有 chunk
        chunk_texts = ENCODER.decode_batch(chunk_token_ids)

        for i, text in enumerate(chunk_texts):
            results.append(
                {
                    # "tokens": lengths[i],
                    "hash_code": compute_mdhash_id(text),  ##使用hash进行编码
                    "text": text.strip().replace("\n", ""),
                    # "chunk_order_index": i,
                }
            )

    return results


def reports2jsonl(input_dir, output_dir):

    output_path = os.path.join(output_dir, "data_chunk.jsonl")
    max_token_size = 1024
    overlap_token_size = 128
    if not os.path.exists(output_path):
        logger.info(f"Generate document chunks in {output_path}")
        all_data = merge_files(input_dir)
        results = chunk_documents(
            all_data,
            max_token_size=max_token_size,
            overlap_token_size=overlap_token_size,
        )
        write(output_path, results, mode="w")
    else:
        logger.info(f"Document chunks are already in {output_path}")
    return output_path


def main():
    input_path = "src/resources/pdf"
    output_path = "src/resources/temp/knowledge_graph/data"
    reports2jsonl(input_path, output_path)


if __name__ == "__main__":
    main()
