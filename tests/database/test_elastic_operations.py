from src.database.operations.elastic_operations import upload_es, search_top, delete
from src.database.db_connection import es_connection

if __name__ == "__main__":
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
