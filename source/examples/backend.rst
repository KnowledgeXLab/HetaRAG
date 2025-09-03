.. _examples_backend:

后端示例
============

本章节提供了启动 HRAG 对应后端功能的示例。具体代码在 ``src/backend`` 中。


data_search_services.py
-----------------------

数据搜索服务，聚合多种数据库的搜索能力。

**功能描述**:

- 提供统一接口访问Elasticsearch、Milvus和Neo4j

- 支持关键词搜索、向量搜索和图谱搜索


**API端点**:

.. list-table:: 
   :widths: 20 30 50
   :header-rows: 1

   * - 端点
     - 方法
     - 描述
   * - ``/milvus/services``
     - POST
     - Milvus向量数据库搜索
   * - ``/neo4j/services``
     - POST
     - Neo4j图谱数据库搜索
   * - ``/elasticsearch/services``
     - POST
     - Elastic关键词搜索

.. raw:: html

    <div class="api-endpoint">
        <h5>API 端点示例</h5>
        <p><span class="method">POST</span> <span class="url">http://0.0.0.0:1242/milvus/services</span></p>
        <p>用于Milvus向量检索的 API 端点</p>
    </div>


**请求示例**:

.. code-block:: json

    {
        "query": "搜索查询",
        "retrieval_setting": {
            "milvus_collection": "collection_name",
            "top_k": 5,
            "score_threshold": 0.6
        }
    }

deepwriter_services.py
----------------------

深度写作服务，根据查询生成结构化报告。

**功能描述**:

- 基于检索到的文档生成详细报告


**API端点**:

.. list-table:: 
   :widths: 20 30 50
   :header-rows: 1

   * - 端点
     - 方法
     - 描述
   * - ``/deepwriter/retrieval``
     - POST
     - 根据查询生成详细报告

.. raw:: html

    <div class="api-endpoint">
        <h5>API 端点示例</h5>
        <p><span class="method">POST</span> <span class="url">http://0.0.0.0:1244/deepwriter/retrieval</span></p>
        <p>用于根据查询生成详细报告的 API 端点</p>
    </div>

**请求示例**:

.. code-block:: json

    {
        "knowledge_id": "知识库ID",
        "query": "报告主题"
    }


deepsearch_services.py
---------------------

深度搜索服务，支持通过多轮检索和推理回答复杂问题。

**功能描述**:

- 提供基于RAG的文档检索功能

- 支持深度搜索流程，可组合多个工具进行信息检索

- 包含网页搜索功能，可获取网页内容并解析

**API端点**:

.. list-table:: 
   :widths: 20 30 50
   :header-rows: 1

   * - 端点
     - 方法
     - 描述
   * - ``/web_search``
     - POST
     - 使用Serper API进行网页搜索并返回内容
   * - ``/milvus_retrieval``
     - POST
     - 从Milvus向量数据库检索相关文档
   * - ``/multi_hop_qa``
     - POST
     - 执行深度搜索流程，组合多个工具获取最终答案

.. raw:: html

    <div class="api-endpoint">
        <h5>API 端点示例</h5>
        <p><span class="method">POST</span> <span class="url">http://0.0.0.0:1246/multi_hop_qa</span></p>
        <p>用于执行深度搜索流程的 API 端点</p>
    </div>


**请求示例**:

.. code-block:: json

    {
        "query": "question for deepsearch",
        "max_rounds": 3,
        "selected_tools": ["rag_retrieve"],
        "retrieval_setting": {
            "milvus_collection": "challenge_data",
            "top_k": 3,
            "score_threshold": 0.5
        }
    }






**启动方式**:

所有服务均可使用以下命令启动:

.. code-block:: bash

    python src/backend/<服务文件名>.py

服务默认监听所有网络接口(0.0.0.0)，端口号通过各自配置获取，详见 :ref:`backend_configuration` 。