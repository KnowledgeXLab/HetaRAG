# 输入：1.input：存放PDF目录（此路径下包含文档解析生成文件）"src/resources/pdf"
#      2.output：pkl输出文件存放的目录 "src/pkl_files"
#      3.processor_type文档解析类型（fitz or mineru）
# 功能：生成规定的数据类型的 pkl 文件，存入"src/pkl_files/vector_db.pkl"


import argparse
from pathlib import Path
import pandas as pd
from loguru import logger
import pickle

from src.data_parser import fitz_parser, mineru_parser


def process_directory(
    input_dir: str,
    output_dir: str,
    processor_type: str = "fitz",
    image_embedding: bool = False,
) -> None:
    """
    Process all documents in the specified directory.

    Args:
        input_dir: Path to the directory containing documents to process
        output_dir: Path to directory where processed documents will be saved
        processor_type: Type of processor to use ('fitz' or 'mineru')

    Returns:
        int: Number of documents successfully processed

    Raises:
        FileNotFoundError: If the specified directory does not exist
        PermissionError: If there are permission issues accessing files
        ValueError: If an invalid processor type is specified
    """
    input_path = Path(input_dir)

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {input_path}")

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    if output_path.suffix == ".pkl":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pkl_file_path = output_dir
        output_path = output_path.parent
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        pkl_file_path = output_path / "vector_db.pkl"

    # Initialize the appropriate processor
    if processor_type == "fitz":
        processor = fitz_parser.FitzProcessor(output_path)
    elif processor_type == "mineru":
        processor = mineru_parser.MinerUProcessor(output_path, image_embedding)
    else:
        raise ValueError(
            f"Invalid processor type: {processor_type}. Choose 'fitz' or 'mineru'."
        )

    logger.info(
        f"Processing documents in {input_path} using {processor_type} processor"
    )
    documents = processor.process_documents(input_path)

    # Count processed documents
    doc_count = len(documents)
    logger.info(f"Successfully processed {doc_count} documents")

    # Flatten document chunks and convert to DataFrame
    flattened_chunks = []
    for document in documents:
        flattened_chunks.extend(document.to_flatten_chunks())

    data_pkl = []
    for chunk in flattened_chunks:
        data_pkl.append(chunk.to_dataframe().iloc[0].to_dict())

    # Save the processed data
    try:
        with open(pkl_file_path, "wb") as f:
            pickle.dump(data_pkl, f)
        print(f"Remaining data saved to {pkl_file_path}")
        logger.info(f"Saved processed data to {pkl_file_path}")
    except Exception as e:
        logger.error(f"Save to PKL Error: {e}")


if __name__ == "__main__":
    input = "src/resources/pdf"
    output = "src/pkl_files"
    processor_type = "mineru"

    process_directory(input, output, processor_type, image_embedding=False)
