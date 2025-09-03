import os
import time
import logging
import json
from tabulate import tabulate
from pathlib import Path
from typing import Iterable, List


from src.data_parser.docling_pdf_parser import DoclingPDFParser

if __name__ == "__main__":
    # Example usage:
    pdf_file_name = "src/resources/pdf/diffusion.pdf"
    output_dir = "output"
    parser = DoclingPDFParser(pdf_file_name, output_dir)
    parser.process_pdf()
