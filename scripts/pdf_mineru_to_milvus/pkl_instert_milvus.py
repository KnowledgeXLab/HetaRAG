import argparse

from pymilvus import utility, Collection
from src.database.db_connection import milvus_connection
from src.database.operations.milvus_operations import (
    create_collection,
    pkl_insert,
    search,
    delete_collection,
)


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="统计指定文件夹中所有 PDF 文件的总页数、总大小、有效 PDF 文件数量和无法处理的文件数量"
    )
    parser.add_argument(
        "--collection_name", type=str, default="challenge_data", help="要创建的集合名称"
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="src/pkl_files/vector_db.pkl",
        help="插入的pkl路径",
    )
    parser.add_argument(
        "--image_embedding", action="store_true", help="是否进行图片embedding"
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default="",
        help="需要检索的问题，如果不提供则不进行检索",
    )

    # 解析命令行参数
    args = parser.parse_args()
    # 初始化连接
    milvus_connection()

    # 配置参数
    collection_name = args.collection_name
    pkl_path = args.pkl_path
    image_embedding = args.image_embedding  # 是否包含图片向量
    if image_embedding:
        embedding_dim = 1536
    else:
        embedding_dim = 1024
    question = args.question

    # 删除旧集合（如果存在）
    if utility.has_collection(collection_name):
        delete_collection(collection_name)

    # 创建新集合
    collection = create_collection(collection_name, embedding_dim)

    # 插入数据
    pkl_insert(collection, pkl_path, image_embedding=image_embedding)

    # 测试检索
    if question:
        search(
            collection_name,
            question,
            image_embedding=image_embedding,
            result_type="text",
        )


if __name__ == "__main__":
    main()
