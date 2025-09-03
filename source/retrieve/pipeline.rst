.. _retrieve_pipeline:

检索流程
===========

本章节介绍了 HRAG 项目中，针对 RAG-Challenge 数据集的完整检索流程。

基础流程
---------

PDF解析 --> 报告合并 --> Markdown导出 --> 报告分块 --> 向量化存储 --> 向量数据库 --> 问题处理 --> 结果测评

.. code-block:: bash

    python tests/data_processor/test_challenge_pipeline.py

本管道(Pipeline)包含完整的报告处理流程，从原始PDF到最终结果评估共7个步骤。详见 ``tests/data_processor/test_challenge_pipeline.py`` 。




扩展用法
---------


1. 混合检索
^^^^^^^^^^^^

.. code-block:: bash

    # 批量评测 - Chunk级别
    python tests/hybrid_retrieval/test_hybrid_weighted_retrieval.py \
        --alpha 0.5 \
        --top_k 14 \
        --collection_name "challenge_data"

    # 批量评测 - 页面级别
    python tests/hybrid_retrieval/test_hybrid_weighted_retrieval.py \
        --alpha 0.5 \
        --top_k 14 \
        --collection_name "challenge_data" \
        --parent_document_retrieval

该方法实现了向量数据库检索(Milvus or Faiss) 与关键字检索(Elastic) 的混合检索， 参数 alpha 控制向量数据库检索的占比。

2. 重排序
^^^^^^^^^^^^

下载相关模型以及详细的使用方法，请查看 :ref:`rerank` 章节。

.. code-block:: bash

    # 从 huggingface 中下载模型进行重排序
    # 使用 bge-reranker-large 模型进行重排序
    python tests/rerank/test_rerank_huggingface.py \
        --root_path src/resources/data \
        --parent_document_retrieval \
        --top_n_retrieval 14 \
        --vector_db milvus
    
    # 使用 VLLM 部署的模型进行重排序
    python tests/rerank/test_rerank_VLLM.py \
        --root_path src/resources/data \
        --parent_document_retrieval \
        --top_n_retrieval 14 \
        --vector_db milvus


3. 问题重写
^^^^^^^^^^^^

.. code-block:: bash

    # 使用查询改写功能 + 重排序
    python tests/query_rewrite/test_query_rewrite.py \
        --root_path src/resources/data \
        --parent_document_retrieval \
        --top_n_retrieval 14 \
        --query_rewrite_model qwen2.5:72b \
        --max_query_variations 3 \
        --vector_db milvus \
        --rerank_model bge-reranker-large

    # 仅使用查询改写功能
    python tests/query_rewrite/test_query_rewrite.py \
        --root_path src/resources/data \
        --parent_document_retrieval \
        --top_n_retrieval 14 \
        --query_rewrite_model qwen2.5:72b \
        --max_query_variations 3 \
        --vector_db milvus

    # 使用faiss
    python tests/query_rewrite/test_query_rewrite.py \
        --root_path src/resources/data \
        --parent_document_retrieval \
        --top_n_retrieval 14 \
        --query_rewrite_model qwen2.5:72b \
        --max_query_variations 3 \
        --vector_db faiss


4. 灵活分块
^^^^^^^^^^^^


- 固定文档长度分块（fixed_doc）

.. code-block:: bash

   python tests/chunk_test/test_chunk_reports.py \
       --chunk_mode fixed_doc \
       --chunk_size 300 \
       --chunk_overlap 50


-  固定页面长度分块（fixed_page）

.. code-block:: bash

   python tests/chunk_test/test_chunk_reports.py \
       --chunk_mode fixed_page \
       --chunk_size 400 \
       --chunk_overlap 40

- 基础分块（base）

.. code-block:: bash

   python tests/chunk_test/test_chunk_reports.py \
       --chunk_mode base \
       --chunk_size 256 \
       --chunk_overlap 32

-  语义分块（semantic）

.. code-block:: bash

   python tests/chunk_test/test_chunk_reports.py \
       --chunk_mode semantic \
       --chunk_size 300 \
       --chunk_overlap 50 \
       --breakpoint_type percentile \
       --breakpoint_amount 85

参数说明

.. code-block:: bash

   python tests/chunk_test/test_chunk_reports.py --help

测评不同 chunk_size 对应的检索结果

- 使用默认chunk_sizes (128, 256)

.. code-block:: bash

    python tests/chunk_test/test_chunk_size_performance.py \
        --root_path src/resources/data

- 自定义chunk_sizes

.. code-block:: bash
    python tests/chunk_test/test_chunk_size_performance.py \
        --root_path src/resources/data \
        --chunk_sizes 64 128 256 512

- 测试单个chunk_size

.. code-block:: bash
    python tests/chunk_test/test_chunk_size_performance.py \
        --root_path src/resources/data \
        --chunk_sizes 256
