from pathlib import Path
import logging
from src.data_processor.converters.pdf_mineru_to_chunk_converter import (
    process_directory,
)
from src.data_processor.converters.challenge_pipeline import Pipeline


class PDFToChunkConverter:

    def mineru_convert(self, input_path, output_path, image_embedding=False):
        process_directory(
            input_path, output_path, "mineru", image_embedding=image_embedding
        )

    def docling_convert(self, input_path, output_path):
        logging.basicConfig(level=logging.INFO)
        pdf_reports = Path(input_path).name
        root_path = Path(input_path).parent
        pipeline = Pipeline(root_path, pdf_reports_dir_name=pdf_reports)
        print("\n=== 数据向量化后存入pkl ===")
        pipeline.chunk_to_pkl(output_path)


if __name__ == "__main__":

    converter = PDFToChunkConverter()

    # 执行转换
    converter.mineru_convert(
        input_path="src/resources/pdf",
        output_path="src/pkl_files/mineru.pkl",
        image_embedding=False,  # 是否对图片进行向量化
    )
