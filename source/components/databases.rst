.. _components_databases:

数据库组件
==========

本章节详细介绍了 HRAG 系统中的数据库组件。


数据库介绍
^^^^^^^^^^^^

本项目支持 Elasticsearch、Milvus、MySQL、Neo4j 四种数据库操作。具体安装流程见 :ref:`database_installation`。


数据库操作
^^^^^^^^^^^^

下文详细介绍四种数据库的连接、数据插入删除、查找等操作。

Elastic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.database.operations.elastic_operations import upload_es, search_top, delete
    from src.database.db_connection import es_connection

    # 创建连接
    client = es_connection()

    # 指定 index 名称（要与 elastic_operations.py 中的一致）
    index_name = "knowledge_test"

    # 删除旧的 index（如果存在）
    delete(index_name)

    # 上传新的 pkl 文件（请提前确认路径下的 .pkl 文件存在）
    pkl_path = "src/pkl_files/es_test"
    upload_es(pkl_path, client)

    # 执行搜索测试
    result = search_top("东数西算", 3, index_name, client)

    print("\n=== 检索结果 ===")
    for item in result:
        print(item)
下载 Elastic 所需要的测试数据，:download:`es_test_data.pkl </_static/es_test_data.pkl>`。数据格式为 List[dict]，每个数据条目为包含以下字段的字典：

.. list-table:: 数据字段说明
   :header-rows: 1
   :widths: 20 80

   * - 字段
     - 说明
   * - filename
     - 原始PDF文件名
   * - chunk_id
     - 文本块唯一标识
   * - text
     - 解析的文本内容（包含原文格式和换行）



Milvus 
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

    question = "2020年世界贸易报告的主要内容是什么？"
    # 测试检索
    search(collection_name, question, image_embedding=image_embedding, result_type="text")

    # （可选）删除集合
    # delete_collection(collection_name)

milvus 插入的数据格式见 :ref:`data_frame`

.. tip::
    embedding_dim 具体由对应的 Embedding 模型确定。本项目使用的纯文本 Embedding 模型 bge-reranker-large ，向量化后的向量为1024维；使用图片文本多模态 QwenVL 模型编码器，向量化后的向量为1536维。



MySQL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from tabulate import tabulate
    from src.database.db_connection import mysql_connnection
    from src.database.operations.mysql_operations import (
        create_database,
        create_table,
        import_pkl_to_mysql,
        query_all_records,
        query_with_conditions,
        query_aggregate,
        delete_table_alldata,
        drop_table
    )

    # 数据库和表配置
    database_name = "mysqldb_test"
    table_name = "pkltomysql"
    pkl_path = "src/pkl_files/vector_db.pkl"

    # 建立连接
    connection = mysql_connnection(database_name=database_name)

    # 可选：创建数据库（如未手动创建）
    # create_database(connection, database_name)

    # 创建表（如不存在）
    create_table(connection, table_name)

    # 导入数据
    import_pkl_to_mysql(connection, table_name, pkl_path)

    # 查询前5条记录
    print("\n=== 查询前5条记录 ===")
    records = query_all_records(connection, table_name, limit=5)
    print(tabulate(records, headers="keys", tablefmt="grid"))

    # 条件查询
    print("\n=== 条件查询 page_number = 5 且 block_type = 'text' ===")
    conditions = {'page_number': 5, 'block_type': 'text'}
    result = query_with_conditions(connection, table_name, conditions)
    print(tabulate(result, headers="keys", tablefmt="grid"))

    # 聚合查询
    print("\n=== 按 block_type 分组统计 ===")
    stats = query_aggregate(connection, table_name, 'block_type')
    print(tabulate(stats, headers="keys", tablefmt="grid"))

    # ✅ 关闭连接
    if connection:
        connection.close()
        print("\n✅ MySQL连接已关闭")





Neo4j
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from src.database.db_connection import neo4j_connection_driver, neo4j_connection
    from src.database.operations.neo4j_operation import (
        import_csv_to_neo4j,
        delete,
        Key_search_bytoken
    )

    # CSV 数据路径（可选导入构图）
    csv_file_path = "src/resources/temp/database/all_data.csv"

    # 用户词典（与 CSV 一致）
    user_dict = "src/resources/temp/database/all_data.csv"

    # 初始化 Neo4j 连接（graph 和 driver）
    graph = neo4j_connection()
    driver = neo4j_connection_driver()

    # （可选）清空现有图数据
    # delete(graph)

    # （可选）导入 CSV 数据建图
    # import_csv_to_neo4j(csv_file_path, graph)

    # 设定用户问题和 top-k 返回数量
    question = "在银屑病治疗过程中，糖皮质激素的作用是什么？"
    top_k = 5

    # 初始化搜索器并运行 pipeline
    key_searcher = Key_search_bytoken(driver, question, top_k, user_dict)
    print("Neo4j 图谱节点列表：", key_searcher.neo4j_nodes)

    result = key_searcher.pipeline()

    # 打印输出结果
    print("\n=== Neo4j Top-k 结果关系 ===")
    for item, score in result:
        print(f"{item} | Score: {score}")

    # 关闭连接
    driver.close()

下载 Neo4j 所需要的测试数据，:download:`neo4j_data.csv </_static/neo4j_data.csv>`