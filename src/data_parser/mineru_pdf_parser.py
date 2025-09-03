import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from pathlib import Path


class MineruPDFParser:
    def __init__(self):
        """
        初始化PDF解析器。
        """
        pass

    def process_pdf(self, pdf_file_name, output_dir="output"):
        """
        Process a PDF file using custom model and save results.

        Args:
        - pdf_file_name (str): The path to the PDF file.
        - output_dir (str): The directory to save output files. Defaults to "output".
        """
        # 使用 os.path.basename 获取文件名（包含扩展名）
        file_name_with_extension = os.path.basename(pdf_file_name)
        # 使用 os.path.splitext 去掉扩展名
        name_without_suff = os.path.splitext(file_name_with_extension)[0]

        # prepare env
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        image_writer = FileBasedDataWriter(image_dir)
        md_writer = FileBasedDataWriter(output_dir)

        # read bytes
        reader1 = FileBasedDataReader("")
        pdf_bytes = reader1.read(pdf_file_name)  # read the pdf content

        # proc
        ## Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        ## inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            ## pipeline
            pipe_result = infer_result.pipe_ocr_mode(image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            ## pipeline
            pipe_result = infer_result.pipe_txt_mode(image_writer)

        ### draw model result on each page
        # infer_result.draw_model(os.path.join(output_dir, f"{name_without_suff}_model.pdf"))

        ### draw layout result on each page
        # pipe_result.draw_layout(os.path.join(output_dir, f"{name_without_suff}_layout.pdf"))

        ### draw spans result on each page
        # pipe_result.draw_span(os.path.join(output_dir, f"{name_without_suff}_spans.pdf"))

        ### dump markdown
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

        ### dump content list
        pipe_result.dump_content_list(
            md_writer, f"{name_without_suff}_content_list.json", image_dir
        )

        ### dump middle json
        pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")

        print(f"Processing completed for {pdf_file_name}")


def get_pdf_mineru_info(input_path):
    path = Path(input_path)
    pdf_files = []
    if path.is_dir():
        for filename in os.listdir(input_path):  # 查看pkl文件夹中的所有文件
            if filename.endswith(".pdf"):
                file_path = os.path.join(input_path, filename)
                pdf_files.append(file_path)
    elif path.is_file():
        pdf_files.append(input_path)

    for p in pdf_files:
        file_path = Path(p)
        try:
            output_dir = file_path.parent / file_path.stem
            parser = MineruPDFParser()
            parser.process_pdf(str(file_path), str(output_dir))
        except Exception as e:
            print(f"无法处理文件 {file_path}: {e}")


if __name__ == "__main__":
    # Example usage:
    pdf_file_name = "src/resources/pdf/diffusion.pdf"
    output_dir = "output"
    parser = MineruPDFParser()
    parser.process_pdf(pdf_file_name, output_dir)
