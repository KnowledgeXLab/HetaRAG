.. _components_data_processor:

数据处理组件
============

本章节详细介绍了 HRAG 系统中的数据处理组件。


.. _data_frame:

数据格式
^^^^^^^^^^^^
从原始 PDF 文件经文档解析为文本，图片，表格文件，标准格式为:


.. list-table:: PDF 解析字段说明
   :widths: 20 40 40
   :header-rows: 1
   :align: left

   * - 字段
     - 值
     - 说明
   * - pdf_path
     - "src/resources/pdf/XXX.pdf"
     - 文件路径
   * - num_pages
     - 96
     - 页数
   * - page_number
     - 1
     - 页码
   * - page_height
     - 2339
     - 页高
   * - page_width
     - 1653
     - 页宽
   * - num_blocks
     - 7
     - 块数
   * - block_type
     - text
     - 块的类型（"text"、"image"、"table"）
   * - block_content
     - "Multimodal LLM as an Agent for Unified Image..."
     - 文本内容
   * - block_summary
     - "..."
     - 文本summary，图片description等
   * - block_embedding
     - [-1.2784948348999023,1.451412320137,...]
     - 图片或文本内容embedding，维度根据 Embedding 模型确定
   * - image_path
     - "src/resources/pdf/XXX/images/xxxxx.jpg"
     - 图片内容（图片和表格）
   * - image_caption
     - "generation and editing. For text-to-image generation..."
     - 图片caption
   * - image_footer
     - "..."
     - 图片对应的脚注
   * - block_bbox
     - [110, 97, 502, 137]
     - 块的bbox
   * - block_id
     - 1
     - 块id
   * - document_title
     - "..."
     - 文档标题（如有）
   * - section_title
     - "..."
     - 段落标题（如有）


处理后的数据将以上面的格式存入 PKL 文件中


数据转换
^^^^^^^^^^^^

将解析后的文档转换为向量数据库格式，保存为 PKL：

针对 MinerU 解析的结果的处理
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_processor.converters.pdf_to_chunk_converter import PDFToChunkConverter
    
    # 配置转换器
    converter = PDFToChunkConverter()

    # 执行转换
    converter.mineru_convert(
        input_path="src/resources/pdf",
        output_path="src/pkl_files/mineru.pkl",
        image_embedding=False # 是否对图片进行向量化
    )

仅进行文本向量化，默认使用 :ref:`config <configuration>` 中的 embedding 模型；
进行图片文本向量化，默认使用 QwenVL 模型，详见代码 src/utils/query2vec.py。

MinerU 解析数据插入向量数据库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from pymilvus import utility, Collection
    from src.database.db_connection import milvus_connection
    from src.database.operations.milvus_operations import (
        create_collection,
        pkl_insert,
        search,
        delete_collection
    )

    # 初始化连接
    milvus_connection()

    # 配置参数
    collection_name = "world_trade_report"
    pkl_path = "src/pkl_files/vector_db.pkl"  # 确保此路径存在一个有效的.pkl
    embedding_dim = 1024 # 根据 Embedding 模型确定
    image_embedding = False  # 是否包含图片向量

    # 删除旧集合（如果存在）
    if utility.has_collection(collection_name):
        delete_collection(collection_name)

    # 创建新集合
    collection = create_collection(collection_name, embedding_dim)

    # 插入数据
    pkl_insert(collection, pkl_path, image_embedding=image_embedding)

需要注意的是 embedding_dim 具体由对应的 Embedding 模型确定。本项目使用的纯文本 Embedding 模型 bge-reranker-large ，向量化后的向量为1024维；使用图片文本多模态 QwenVL 模型编码器，向量化后的向量为1536维

针对 Docling 解析的结果
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.data_processor.converters.pdf_to_chunk_converter import PDFToChunkConverter
    
    # 配置转换器
    converter = PDFToChunkConverter()

    # 执行转换
    converter.docling_convert(
        input_path="src/resources/data/pdf_reports", # 默认为 RAG-Challenge 数据
        output_path="src/pkl_files/docling.pkl"
    )

Docling 处理后的结果默认只有文本编码，不支持图片编码。


Docling 解析数据插入向量数据库
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python tests/data_processor/test_insert_to_vector_dbs.py \
        --root_path src/resources/data \
        --pkl_path src/pkl_files/challenge_docling.pkl \ 
        --vector_db milvus 

vector_db 可以选择 milvus 或者 faiss。其中 milvus 安装与配置见 :ref:`database_installation` 。

选择 milvus 时，数据格式如上，存储在启动的 milvus 服务中，支持前端页面查看数据；选择 faiss 时，仅存储索引与 block_embedding 在本地文件中，不支持前端页面查看数据。