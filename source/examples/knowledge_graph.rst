.. _examples_knowledge_graph:

知识图谱示例
============

本章节提供了 HiRAG 与 TRAG 两种方法的知识图谱构建和使用的示例。


两种方法均由：实体-关系三元组抽取、知识图谱构建、基于图的回答三个阶段构成。其中，实体-关系三元组抽取提供了CommonKG、GraphRAG两种三元组抽取方法。


构建实体关系三元组：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将 MinerU 处理后的结果提取实体关系三元组：


.. code-block:: python

    from src.data_processor.knowledge_graph.entity_relation_extractor import entity_relation_extractor
    
    # 根据MinerU生成的文件得到三元组
    mineru_path = "src/resources/pdf"
    
    # GraphRAG方法提取
    output_path = "src/resources/temp/knowledge_graph/graphrag"
    entity_relation_extractor(mineru_path, output_path, method="graphrag")
    

.. code-block:: python

    from src.data_processor.knowledge_graph.entity_relation_extractor import entity_relation_extractor
    
    # 根据MinerU生成的文件得到三元组
    mineru_path = "src/resources/pdf"

    # CommonKG方法提取
    output_path = "src/resources/temp/knowledge_graph/commonkg"
    entity_relation_extractor(mineru_path, output_path, method="CommonKG")


**参数说明：**

- ``mineru_path``: MinerU解析后的PDF文件路径
- ``output_path``: 三元组数据保存路径
- ``method``: 提取方法，可选 "graphrag" 或 "CommonKG"，任选一种即可


构建知识图谱
~~~~~~~~~~~~~~

根据提取的三元组数据构建知识图谱：

HiRAG 构建
^^^^^^^^^^

.. code-block:: python

    from src.data_processor.knowledge_graph.graph_builder import graph_builder
    
    # 实体关系三元组数据构建hirag
    data_path = "src/resources/temp/knowledge_graph/graphrag"
    working_dir = "src/resources/temp/knowledge_graph/hirag"  
    graph_builder(data_path, working_dir, method="hirag")

LearnRAG 构建
^^^^^^^^^^^^^

.. code-block:: python

    from src.data_processor.knowledge_graph.graph_builder import graph_builder
    
    # 实体关系三元组数据构建learnrag
    data_path = "src/resources/temp/knowledge_graph/graphrag"
    working_dir = "src/resources/temp/knowledge_graph/learnrag"  
    graph_builder(data_path, working_dir, method="learnrag")

**参数说明：**

- ``data_path``: 三元组数据路径
- ``working_dir``: 构建好的知识图谱保存路径
- ``method``: 构建方法，可选 "hirag" 或 "learnrag"


查询知识图谱
~~~~~~~~~~~~

对构建好的知识图谱进行查询：

HiRAG 查询
^^^^^^^^^^

.. code-block:: python

    from src.data_processor.knowledge_graph.query_graph import query_graph
    
    query = "Which leadership positions changed at Datalogic in the reporting period?"
    working_dir = "src/resources/temp/knowledge_graph/hirag"  
    result = query_graph(query, working_dir, method="hirag")
    print(result)

LearnRAG 查询
^^^^^^^^^^^^^

.. code-block:: python

    from src.data_processor.knowledge_graph.query_graph import query_graph
    
    query = "Which leadership positions changed at Datalogic in the reporting period?"
    working_dir = "src/resources/temp/knowledge_graph/learnrag"  
    result = query_graph(query, working_dir, method="learnrag")
    print(result)

**参数说明：**

- ``query``: 查询问题
- ``working_dir``: 知识图谱保存路径
- ``method``: 查询方法，与构建方法对应