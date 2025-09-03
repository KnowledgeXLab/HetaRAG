from pymilvus import utility, Collection
from src.database.db_connection import milvus_connection
from src.database.operations.milvus_operations import (
    create_collection,
    pkl_insert,
    search,
    delete_collection,
)

if __name__ == "__main__":
    # 初始化连接
    milvus_connection()

    # 配置参数
    collection_name = "world_trade_report"
    pkl_path = "src/pkl_files/wtr_vector_db.pkl"  # 确保此路径存在一个有效的.pkl
    embedding_dim = 1024
    image_embedding = False  # 是否包含图片向量
    question = "2020年世界贸易报告的主要内容是什么？"

    # 删除旧集合（如果存在）
    if utility.has_collection(collection_name):
        delete_collection(collection_name)

    # 创建新集合
    collection = create_collection(collection_name, embedding_dim)

    # 插入数据
    pkl_insert(collection, pkl_path, image_embedding=image_embedding)

    # 测试检索
    search(
        collection_name, question, image_embedding=image_embedding, result_type="text"
    )

    # （可选）删除集合
    # delete_collection(collection_name)
