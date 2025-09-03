from src.data_processor.converters.pdf_mineru_to_chunk_converter import (
    process_directory,
)

if __name__ == "__main__":

    # mineru 处理后的 pdf 向量化后存入pkl
    input = "src/resources/pdf"
    output = "src/pkl_files"
    processor_type = "mineru"
    process_directory(input, output, processor_type, image_embedding=False)
