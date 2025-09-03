.. _examples_basic_rag:

基础 RAG 示例
=============

本章节提供了一个基础的检索增强生成示例。展示了如何从 PDF 文档中提取信息、处理数据、生成向量表示，并将其存储到 Milvus 向量数据库中以便后续检索。


.. code-block:: bash

    bash scripts/pdf_mineru_to_milvus.sh


示例流程概述
-----------------------

**PDF 预处理与统计：** 清理无效 PDF 文件并统计文档信息。

**PDF 解析：** 使用 MinerU 解析 PDF 文件，提取文本和图像数据。

**向量化处理：** 将解析后的数据转换为向量表示并保存为 PKL 文件。

**数据插入 Milvus：** 将向量数据导入 Milvus 集合，支持后续检索。


详细步骤
-----------------------


定义共享参数
~~~~~~~~~~~~~~~~~~~~

在脚本 ``scripts/pdf_mineru_to_milvus.sh`` 中确定共享参数。

示例命令：

.. code-block:: bash

    # 需处理的批量 PDF 存放目录
    input_path="src/resources/pdf" 

    # PKL 文件保存路径（支持目录与pkl文件）
    pkl_path="src/pkl_files"

    # 导入 Milvus 数据库的库名
    collection_name="data_search"

    # 如果需要测试问答功能，可以设置一个问题
    question="" 


1. PDF 预处理与统计
~~~~~~~~~~~~~~~~~~~~

运行脚本 get_info_clean_invalid_files.py，清理无效 PDF 文件并统计文档的总页数、大小等信息。

示例命令：

.. code-block:: bash

    python scripts/pdf_mineru_to_milvus/get_info_clean_invalid_files.py "$input_path" 

2. PDF 解析
~~~~~~~~~~~~~~~~~~~~

使用 get_minerU_clean_invalid_files.py 调用 MinerU 解析工具，提取 PDF 中的文本和图像数据。

示例命令：

.. code-block:: bash

    python scripts/pdf_mineru_to_milvus/get_minerU_clean_invalid_files.py "$input_path"

3. 向量化处理
~~~~~~~~~~~~~~~~~~~~

运行 get_vector_to_pkl.py，将解析后的数据转换为向量表示（支持文本和可选的图像嵌入），存入 PKL 文件中。

示例命令（不包含图像嵌入,使用BGE）：

.. code-block:: bash

    python scripts/pdf_mineru_to_milvus/get_vector_to_pkl.py 
        --input_path "$input_path" \
        --output_path "$pkl_path"

示例命令（包含图像嵌入,使用QwenVL）：

.. code-block:: bash

    python scripts/get_vector_to_pkl.py \
        --input_path "$input_path" \
        --output_path "$pkl_path" \
        --image_embedding

4. 数据插入 Milvus
~~~~~~~~~~~~~~~~~~~~

使用 pkl_instert_milvus.py 将向量数据导入 Milvus 集合。

示例命令：

.. code-block:: bash

    python scripts/pdf_mineru_to_milvus/pkl_instert_milvus.py \
        --collection_name "$collection_name" \
        --pkl_path "$pkl_path" \
        --question "$question" \
        --image_embedding # 此行可选，须与步骤3统一
