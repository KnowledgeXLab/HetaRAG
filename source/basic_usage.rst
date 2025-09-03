.. _basic_usage:

基础使用
========

1. 文档解析
^^^^^^^^^^^^

HRAG 支持两种文档解析方法：MinerU 和 Docling。

使用 MinerU 解析文档
~~~~~~~~~~~~~~~~~~~~
.. tip::
    初次使用 MinerU 请下载对应模型文件，操作指南请查看:安装指南的 :ref:`MinerU_installation` 部分。

批量文档解析

.. code-block:: bash

    python tests/data_parser/test_mineru_pdf_parser.py --input_path src/resources/pdf

更详细的使用指南见 :ref:`components_data_parser` 

使用 Docling 解析文档
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

批量文档解析

.. code-block:: bash

    python tests/data_parser/test_docling_pdf_parser.py --root_path src/resources/test_data


更详细的使用指南见 :ref:`components_data_parser` 


2. 数据转换
^^^^^^^^^^^^

将解析后的文档转换为向量数据库格式，保存为 PKL：

针对 MinerU 解析的结果的处理

.. code-block:: bash

    python tests/data_processor/test_data_mineru_converter.py

针对 Docling 解析的结果

.. code-block:: bash

    python tests/data_processor/test_data_docling_converter.py \
        --root_path src/resources/data \
        --output_path src/pkl_files/challenge_docling.pkl

Docling 解析数据插入向量数据库

.. code-block:: bash

    python tests/data_processor/test_insert_to_vector_dbs.py \
        --root_path src/resources/data \
        --pkl_path src/pkl_files/challenge_docling.pkl \
        --vector_db milvus 



3. 数据库操作
^^^^^^^^^^^^^^^^

本项目提供 Elasticsearch、Milvus、MySQL、Neo4j 四种数据库操作。具体安装流程见 :ref:`database_installation`。

.. code-block:: bash


    # Elasticsearch
    python tests/database/test_elastic_operations.py

    # Milvus
    python tests/database/test_milvus_operations.py

    # MySQL
    python tests/database/test_mysql_operations.py

    # Neo4j
    python tests/database/test_neo4j_operation.py



4. 知识图谱构建
^^^^^^^^^^^^^^^^

知识图谱构建提供 HiRAG 与 LearnRAG 两种方法。两种方法均由：构建实体关系三元组、生成实体关系对应描述、构建知识图谱三部分组成，其中共用同一个构建实体关系三元组方法。


从语料中抽取实体关系：

.. code-block:: bash

    python tests/data_processor/knowledge_graph/test_get_entity_relation.py


构建知识图谱：

.. code-block:: bash

    # HiRAG 
    python tests/data_processor/knowledge_graph/test_create_hirag.py

    # TRAG 
    python tests/data_processor/knowledge_graph/test_create_learnrag.py


5. 启动服务
^^^^^^^^^^^^^^^^

启动后端服务进行问答：

后端服务包括  **数据检索（data_search）** 、 **论文生成（deepwriter）** 、 **深度搜索（deepsearch）** 。该部分提供后台启动后端服务的命令，服务启动后的输出文件在目录 ``logs/backend/`` 下；对应的端口号配置见 :ref:`backend_configuration` 。

- 数据检索（data_search）服务:

.. code-block:: bash

    nohup python src/backend/data_search_services.py > logs/backend/data_search.out 2>&1 &

- 论文生成（deepwriter）服务:

.. code-block:: bash

    nohup python src/backend/deepwriter_services.py > logs/backend/deepwriter.out 2>&1 &

- 深度搜索（deepsearch）服务:

.. code-block:: bash

    nohup python src/backend/deepsearch_services.py > logs/backend/deepsearch.out 2>&1 &

