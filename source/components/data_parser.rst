.. _components_data_parser:

数据解析组件
============

本章节详细介绍了 HRAG 系统中的数据解析组件。

文档解析
^^^^^^^^^^^^

HRAG 支持两种文档解析方法：MinerU 和 Docling。

使用 MinerU 解析文档
~~~~~~~~~~~~~~~~~~~~
.. tip::
    初次使用 MinerU 请下载对应模型文件，操作指南请查看:安装指南的 :ref:`MinerU_installation` 部分。

单文档解析

.. code-block:: python

    from src.data_parser.mineru_parser import MinerUParser
    
    # 初始化解析器
    parser = MinerUParser()
    
    # 解析 PDF 文档
    pdf_file_name = "src/resources/pdf/XXX.pdf"
    output_dir = "src/resources/pdf/XXXoutput" # 解析后文件路径
    parser.process_pdf(pdf_file_name, output_dir)

批量文档解析

.. code-block:: python

    from src.data_parser.mineru_pdf_parser import get_pdf_mineru_info
    
    # 解析 PDF 文档
    input_path = "src/resources/pdf" # PDF 文件路径，（同解析后文件路径）
    get_pdf_mineru_info(input_path) 


使用 Docling 解析文档
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

单文档解析

.. code-block:: python
    
    from src.data_parser.docling_pdf_parser import DoclingPDFParser

    # PDF 文件路径
    pdf_file_name = "src/resources/pdf/XXX.pdf"
    # 解析后文件路径
    output_dir = "output"

    # 初始化解析器
    parser = DoclingPDFParser(pdf_file_name, output_dir)
    # 解析 PDF 文档
    parser.process_pdf()

批量文档解析

.. code-block:: python

    from src.data_processor.converters.challenge_pipeline import Pipeline
    
    # 解析 PDF 文档（默认为 RAG-Challenge 数据）
    input_path = "src/resources/data/pdf_reports" 
    logging.basicConfig(level=logging.INFO)
    pdf_reports = Path(input_path).name
    root_path = Path(input_path).parent
    pipeline = Pipeline(root_path, pdf_reports_dir_name=pdf_reports)
    pipeline.parse_pdf_reports_sequential() 
    pipeline.merge_reports() 
    pipeline.export_reports_to_markdown()

