import sys, os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(project_root)

from src.database.db_connection import neo4j_connection_driver, neo4j_connection
from src.database.operations.neo4j_operation import (
    import_csv_to_neo4j,
    delete,
    Key_search_bytoken,
)

if __name__ == "__main__":
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
