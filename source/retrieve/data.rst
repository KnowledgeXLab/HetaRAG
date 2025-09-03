.. _retrieve_data:

检索数据
===========

本章节详细介绍了 HRAG 系统的所用到的检索公开数据集 ———— RAG-Challenge 数据集,包含问题及对应的公司报告。可用于研究示例问题、报告和系统输出。

文件内容
--------

下载后的文件位置： ``src/resources/data``，
本系统数据按照以下目录结构组织：

.. code-block:: text

   resources/
   ├── data/ 
   │   ├── answers/ 
   │   │   ├── answers_ollama_qwen_milvus.json
   │   │   ├── answers_ollama_qwen_milvus_debug.json
   │   │   └── ...
   │   ├── databases/
   │   │   ├── vector_dbs/               # faiss向量数据库
   │   │   └── chunked_reports/          # 分chunk的报告
   │   ├── debug_data/
   │   │   ├── 01_parsed_reports/        # 解析后的报告
   │   │   ├── 01_parsed_reports_debug/  # 解析调试数据
   │   │   ├── 02_merged_reports/        # 合并后的报告  
   │   │   └── 03_reports_markdown/      # Markdown格式报告
   │   ├── pdf_reports/
   │   │   ├── XXX.pdf
   │   │   └── ...
   ├── answers.json
   ├── questions.json
   ├── ranking.csv
   ├── subset.csv
   └── subset.json



.. list-table:: 数据集文件说明
   :header-rows: 1
   :widths: 30 50 20

   * - 文件名
     - 说明
     - 是否必须
   * - answers.json
     - 问题正确答案
     - 是
   * - questions.json
     - 竞赛问题集
     - 是
   * - subset.csv
     - 测试文档元数据（CSV格式）
     - 是
   * - subset.json
     - 测试文档元数据（JSON格式）
     - 是
   * - data/pdf_reports/
     - 存放原始 PDF 文件
     - 是
   * - data/answers/
     - 存放生成答案文件
     - 否
   * - data/databases/
     - 存放分块文件与 faiss 向量数据库数据
     - 否
   * - data/debug_data/
     - 存放 PDF 解析过程文件
     - 否

运行系统
--------

按照以下步骤在本数据集上运行系统：

1. 原始 PDF 文件

   - ``pdf_reports`` (`Google Drive下载 <https://drive.google.com/file/d/1MvcN_-KpI-9nS4hDFAcPxFU2lRmwMP7M/view?usp=sharing>`__)

2. 相关必须文件

   - ``answers.json`` (`GitHub <https://github.com/trustbit/enterprise-rag-challenge/blob/main/round2/answers.json>`__)

   - ``questions.json`` (`GitHub <https://github.com/trustbit/enterprise-rag-challenge/blob/main/round2/questions.json>`__)

   - ``subset.json`` (`GitHub <https://github.com/trustbit/enterprise-rag-challenge/blob/main/round2/subset.json>`__)

   - ``subset.csv`` (`GitHub <https://github.com/trustbit/enterprise-rag-challenge/blob/main/round2/subset.csv>`__)

3. 可选文件

   - ``databases`` (`Google Drive下载 <https://drive.google.com/file/d/1mp-hYhMAit4rdi7RURuIsM33zbXq1nQJ/view?usp=sharing>`__)

   - ``debug_data`` (`Google Drive下载 <https://drive.google.com/file/d/13RT456tZVTAwPIsy8OndZ1EWASNCdfe3/view?usp=sharing>`__)

     - 需要以下情况时使用：

       * 调试特定 Pipeline 阶段

       * 运行单独的预处理步骤

       * 研究系统中间输出
