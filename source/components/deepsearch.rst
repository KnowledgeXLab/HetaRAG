.. _components_deepsearch:

深度搜索
==============

本章节详细介绍了 HRAG 系统中的深度搜索组件在 MultiHop-RAG 数据集上的使用。其能够处理需要多步推理的复杂问题。该组件通过结合语义检索、信息提取和批判性推理，逐步收集和整合信息，最终生成准确的答案。

深度搜索在 MultiHop-RAG 数据集上的使用流程包括四个主要步骤：

1. 下载并准备多跳数据
2. 为语料创建向量索引（Milvus）
3. 进行多跳问答生成
4. 对生成结果进行评估

.. contents:: 内容提要
   :local:
   :depth: 2

数据准备
--------------------

运行以下脚本以下载并保存所需的数据集：

.. code-block:: bash

    python src/deepsearch/data/get_data.py

该脚本将自动从 Hugging Face 下载 `yixuantt/MultiHopRAG` 数据集，并将以下两个文件保存到本地：

- `src/deepsearch/data/MultiHopRAG.json`：用于问答生成与评估的测试集
- `src/deepsearch/data/corpus.json`：用于构建 Milvus 检索库的语料内容

语料嵌入与索引
--------------------

语料库需先进行向量化嵌入，并建立 Milvus 索引。执行以下脚本：

.. code-block:: bash

    python src/deepsearch/data/retrieval_corpus.py

该脚本功能包括：

- 使用 EmbeddingProcessor 对 `corpus.json` 中的文本进行语义嵌入
- 调用 Milvus 接口创建向量集合 `Multi_hop`
- 分批写入嵌入后的文本块与元信息字段（如标题、作者、时间、类别等）
- 为向量字段建立默认的 COSINE 索引

问答生成
--------------------

问答系统基于 Agent 框架自动执行多轮检索与思考。使用以下命令运行生成模块：

.. code-block:: bash

    python src/deepsearch/multi_hop_qa.py

或运行完整流程测试脚本：

.. code-block:: bash

    python tests/test_multi_hop_qa.py

该模块执行以下功能：

- 读取多跳问题文件 `MultiHopRAG.json`
- 针对每个问题，通过 Milvus 检索相关文本块
- 自动完成多轮“思考-检索-推理”过程，直到获得最终答案
- 生成结构化输出并写入 `tests/deepsearch/multi_hop_data/answer.json`，包括模型思路、记忆片段和最终回答

结果评估
--------------------

完成问答生成后，可以执行以下命令对结果进行评估：

.. code-block:: bash

    python src/deepsearch/qa_evaluate.py

或继续运行测试脚本 `test_multi_hop_qa.py`（已包含评估步骤）：

.. code-block:: bash

    python tests/test_multi_hop_qa.py


评估结果会按照问题类型（如 multi-hop、comparison 等）分类展示，同时给出整体平均指标。

