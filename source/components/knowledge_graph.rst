.. _components_knowledge_graph:

知识图谱组件
============

本章节详细介绍了 HRAG 系统中的知识图谱组件。

知识图谱构建提供 HiRAG 与 TRAG 两种方法。两种方法均由以下三部分组成：

1. 构建实体关系三元组
2. 生成实体关系对应描述
3. 构建知识图谱

其中两种方法共用同一个构建实体关系三元组的方法。对应的参数设置见 :ref:`configuration_knowledge_graph`

从语料中抽取实体关系：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

提供两种抽取实体关系的方法

1. ConmmonKG方法，使用代码：

.. code-block:: python

    from src.data_processor.knowledge_graph.entity_relation_extractor import entity_relation_extractor


    # 根据MinerU生成的文件得到三元组
    mineru_path = "src/resources/pdf"
    output_path = "src/resources/temp/knowledge_graph/commonkg"
    entity_relation_extractor(mineru_path, output_path, corpus_dir = corpus_path, method="CommonKG")
   
   
2. GraphRAG方法，使用代码：

.. code-block:: python

    from src.data_processor.knowledge_graph.entity_relation_extractor import entity_relation_extractor


    # 根据MinerU生成的文件得到三元组
    mineru_path = "src/resources/pdf"
    output_path = "src/resources/temp/knowledge_graph/graphrag"
    entity_relation_extractor(mineru_path, output_path, method="graphrag")




构建知识图谱：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    
    # HiRAG
    from src.data_processor.knowledge_graph.graph_builder import graph_builder

    # 实体关系三元组等数据构建hirag，并存入working_dir
   
    # 选择合适的实体关系提取方法
    data_path = "src/resources/temp/knowledge_graph/graphrag" # or "src/resources/temp/knowledge_graph/commonkg"
    working_dir = "src/resources/temp/knowledge_graph/hirag"  
    graph_builder(data_path, working_dir,method="hirag")

    # LearnRAG
    from src.data_processor.knowledge_graph.graph_builder import graph_builder

    # 实体关系三元组等数据构建learnrag，并存入working_dir
    data_path = "src/resources/temp/knowledge_graph/trag_data"
    working_dir = "src/resources/temp/knowledge_graph/trag"  
    graph_builder(data_path, working_dir,method="trag")


查询知识图谱：
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # HiRAG
    from src.data_processor.knowledge_graph.query_graph import query_graph
    
    query = "Which leadership positions changed at Datalogic in the reporting period?"
    working_dir = "src/resources/temp/knowledge_graph/hirag"  
    result = query_graph(query, working_dir, method="hirag")
    print(result)

    # LearnRAG
    from src.data_processor.knowledge_graph.query_graph import query_graph
    
    query = "Which leadership positions changed at Datalogic in the reporting period?"
    working_dir = "src/resources/temp/knowledge_graph/learnrag"  
    result = query_graph(query, working_dir, method="learnrag")
    print(result)


