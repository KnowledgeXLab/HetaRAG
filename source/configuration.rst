.. _configuration:

系统参数配置指南
=================

本章节介绍了 HRAG 系统的各种系统参数配置选项。


参数配置内容
^^^^^^^^^^^^^^^

H-RAG的相关配置均在 `src/config` 中设置，包含了：

* 数据库参数配置
* 后端端口参数配置
* Embedding 模型参数配置
* LLM 模型参数配置
* knowledge_graph 相关参数配置

数据库配置、Embedding 模型配置、LLM 模型配置可直接在 `src/config/config.ini` 中更改
knowledge_graph 相关参数配置可直接在 `src/config/knowledge_graph/create_kg_conf.yaml` 中更改


数据库配置指南
^^^^^^^^^^^^^^^^^

本指南说明如何配置各数据库的连接参数，请先完成 :ref:`database_installation` 中的数据库安装。

**配置文件位置**： ``src/config/config.ini``

配置结构说明
----------------
每个数据库配置包含以下参数：

.. list-table:: 数据库配置参数说明
   :header-rows: 1
   :widths: 20 25 55
   
   * - 参数
     - 示例值
     - 说明
   * - host
     - 127.0.0.1
     - 数据库服务器IP（本地开发可保留127.0.0.1）
   * - front_end_port
     - 8080
     - 前端可视化工具连接端口
   * - read_write_port
     - 3306
     - 程序读写操作端口
   * - username
     - root
     - 数据库登录账号
   * - password
     - (对应密码)
     - 数据库登录密码


配置示例
--------------

.. code-block:: ini
   :linenos:

    [Elasticsearch]
    host = 127.0.0.1
    front_end_port = 5601
    read_write_port = 9200
    username = elastic
    password = elastic

    [Milvus]
    host = 127.0.0.1
    front_end_port = 8000
    read_write_port = 19530
    username = 
    password = 
    min_content_len = 200

    [Neo4j]
    host = 127.0.0.1
    front_end_port = 7474   
    read_write_port = 7687
    username = neo4j
    password = neo4j2025


    [MySQL]
    host = 127.0.0.1
    front_end_port = 8080
    read_write_port = 3306
    username = root
    password = 123456


各数据库特殊说明
----------------------

Elasticsearch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 默认用户：``elastic``

* 默认密码：``elastic``

* 端口用途：

    * 9200： 读写端口

    * 5601： Kibana可视化Web前端端口


Milvus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* 特殊参数：

  * ``min_content_len = 200``：设置插入文本块最小长度（字符数）

* 端口用途：

    * 19530： 读写端口

    * 8000： attu可视化Web前端端口


Neo4j
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* 默认用户：``neo4j``

* 默认密码：``neo4j2025``

* 端口用途：

  * 7687： 基于Bolt协议的读写端口
  
  * 7474： Neo4j可视化Web前端端口


MySQL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* 默认用户：``root``

* 默认密码：``123456``

* 端口用途：

    * 3306： 读写端口

    * 8080： adminer可视化Web前端端口



.. _backend_configuration:

后端端口配置指南
^^^^^^^^^^^^^^^^^

本指南说明后端端口配置的格式与使用方法。

**配置文件位置**： ``src/config/config.ini``

配置结构说明
----------------
后端端口参数配置均在 ``[backend_api]`` 中，设置每个服务的名称与其对应的端口号。

配置示例
------------

.. code-block:: ini

    [backend_api]
    data_search_port = 1242
    deepwriter_port = 1244
    deepsearch_port = 1246

即运行 ``src/backend/data_search_services.py`` 时，启动的端口为 data_search_port 确定的1242端口。

.. note::
    
    具体的后端运行实例见 :ref:`examples_backend` 。


Embedding 与 LLM 模型配置指南
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


本指南说明如何配置 Embedding 和 LLM 模型参数。

**配置文件位置**： ``src/config/config.ini``

配置结构说明
-----------------

1. **框架选择**：

   * 需指定使用的框架类型

   * 支持 ``ollama`` 、 ``vllm`` 两种框架

2. **参数设置**：

.. list-table:: 模型配置参数说明
   :header-rows: 1
   :widths: 20 25 55
   
   * - 参数
     - 示例值
     - 说明
   * - framework
     - vllm
     - 框架类型
   * - host
     - 127.0.0.1
     - 模型服务IP地址（本地部署填127.0.0.1）
   * - port
     - 8004
     - 模型服务端口号
   * - model_name
     - qwen2.5:72b
     - 需与部署的模型名称完全一致

配置示例
------------


1. **Ollama 框架示例**：

.. code-block:: ini

   [ollama_embedding]
   framework = ollama
   host = 127.0.0.1  # 修改为实际IP
   port = 11434
   model_name = bge-m3

   [ollama_llm]
   framework = ollama
   host = 127.0.0.1
   port = 11434 
   model_name = qwen2.5:72b

2. **vLLM 框架示例**：

.. code-block:: ini

   [vllm_embedding]
   framework = vllm
   host = 127.0.0.1  # 模型服务IP地址
   port = 8001
   model_name = bge-m3

   [vllm_llm] 
   framework = vllm
   host = 127.0.0.1
   port = 8002
   model_name = Qwen2.5-72B-Instruct


.. _configuration_knowledge_graph:

knowledge_graph 相关参数配置指南
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


本指南说明如何配置 knowledge_graph 相关参数。

**配置文件位置**： ``src/config/knowledge_graph/create_kg_conf.yaml``

.. note::
    
    具体的知识图谱构建参数使用见 :ref:`components_knowledge_graph` 。

配置结构说明
-----------------

1. **LLM 模型配置**：

    * 使用场景：三元组生成、对应描述生成、图知识库构建

    * 配置 LLM 框架、模型服务IP地址、模型服务端口（支持多个端口并行推理）、部署模型名称等。

2. **任务参数配置**：

    * 使用场景：三元组生成、对应描述生成、图知识库构建

    * 配置任务多进程数量、头实体匹配路径、参考的开源三元组文件路径、生成图谱的层数等。


配置示例
------------

1. **LLM 模型配置示例**：

.. code-block:: yaml
   :linenos:

   ## LLM参数
   llm_conf:

     llm_framework: "vllm"

     ## LLM url
     llm_host: "127.0.0.1"
     llm_ports: [8001, 8002, 8003, 8004] # 部署模型的端口


     ## LLM key
     llm_api_key: ""

     ## LLM模型
     llm_model: "qwen3_32b" 

     ## 在线调用LLM最大尝试次数
     max_error: 3


1. **任务参数配置示例**：

.. code-block:: yaml
   :linenos:
   
   ## 任务参数
   task_conf:
     ## 生成图谱的层数
     level_num: 2

     ## 头实体匹配多进程数量（-1表示使用所有CPU核心）
     num_processes_match: -1

     ## 推理多进程数量（-1表示使用所有CPU核心）
     num_processes_infer: 16

     ## 头实体路径
     pedia_entity_path:  src/resources/temp/knowledge_graph/dbpedia_entities_clean_valid.txt

     ## 参考的开源三元组文件路径
     ref_kg_path: src/resources/temp/knowledge_graph/triple_ref_test.txt


