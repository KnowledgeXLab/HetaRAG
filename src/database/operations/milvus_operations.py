import os
from pathlib import Path
import numpy as np

from src.config.db_config import get_min_content_len
from src.database.db_connection import milvus_connection
from src.utils.api_llm_requests import EmbeddingProcessor

# 读取pkl，将数据写入milvus中
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from pymilvus import connections

import pickle


def create_collection(collection_name: str, dim: int) -> Collection:
    # milvus_connection()

    if utility.has_collection(collection_name):
        print(f"集合 {collection_name} 已存在")
        collection = Collection(collection_name)
        if not collection.is_empty and not collection.has_index():
            print("集合存在但没有索引，正在创建默认索引...")
            create_default_index(collection, "block_embedding")
        return collection
    else:
        """创建新的集合"""
        fields = [
            FieldSchema(name="block_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(
                name="block_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim
            ),  # 假设embedding维度是1024
            FieldSchema(name="pdf_path", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="num_pages", dtype=DataType.INT64),
            FieldSchema(name="page_number", dtype=DataType.INT64),
            FieldSchema(name="page_height", dtype=DataType.FLOAT),
            FieldSchema(name="page_width", dtype=DataType.FLOAT),
            FieldSchema(name="num_blocks", dtype=DataType.INT64),
            FieldSchema(name="block_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="block_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="block_summary", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="image_caption", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="image_footer", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="block_bbox", dtype=DataType.JSON),  # 使用JSON存储复杂结构
            FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="section_title", dtype=DataType.VARCHAR, max_length=2048),
        ]

        schema = CollectionSchema(fields, description="pkl2milvus")
        collection = Collection(collection_name, schema)

    # 创建默认索引
    create_default_index(collection, "block_embedding")

    collection.load()  # 确保集合已加载

    return collection


def create_default_index(collection: Collection, field_name: str):
    """创建默认索引"""
    index_params = {
        "index_type": "FLAT",  # IVF_FLAT
        "metric_type": "COSINE",
        "params": {},
    }

    collection.create_index(
        field_name=field_name, index_params=index_params, index_name=field_name + "_idx"
    )

    # 创建标量索引（可选）
    # collection.create_index(
    #     field_name="block_type",
    #     index_name="scalar_index_block_type",
    #     index_type="STL_SORT"
    # )

    print(f"已为字段 {field_name} 创建 {index_params['index_type']} 索引")


from tqdm import tqdm  # 可选，用于显示进度条


def batch_insert(collection, data, batch_size=100):
    """分批插入数据，避免 gRPC 消息过大"""
    total = len(data[0])
    for i in tqdm(range(0, total, batch_size)):
        batch_data = [row[i : i + batch_size] for row in data]
        collection.insert(batch_data)

    print("插入完成！")


def pkl_insert(collection, pkl_path, image_embedding=False):

    path = Path(pkl_path)
    pkl_data = []
    if path.is_dir():
        for filename in os.listdir(pkl_path):  # 查看pkl文件夹中的所有文件
            if filename.endswith(".pkl"):
                now_pkl_path = os.path.join(pkl_path, filename)
                # 读取now_pkl_path指定的pkl文件，对其操作
                # 读取 pkl 文件
                with open(now_pkl_path, "rb") as f:
                    pkl_data.append(pickle.load(f))
    elif path.is_file():
        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)

    # 数据筛选与处理,如果没有image_embedding，则只插入文本数据

    if not image_embedding:
        data = []
        min_content_len = get_min_content_len("Milvus")[
            "min_content_len"
        ]  # 限定插入文本的最小长度
        for i in pkl_data:
            if (
                i["block_type"] == "text"
                and len(i["block_content"]) >= int(min_content_len)
                and len(i["block_content"]) < 65535
                and i["block_embedding"] is not None
                and len(i["block_embedding"]) > 0
            ):
                data.append(i)
            else:
                continue
    else:
        data = []
        min_content_len = get_min_content_len("Milvus")["min_content_len"]
        for i in pkl_data:
            if (
                i["block_type"] == "text"
                and len(i["block_content"]) >= int(min_content_len)
                and len(i["block_content"]) < 65535
            ):
                if i["block_embedding"] is not None and len(i["block_embedding"]) > 0:
                    data.append(i)
            elif i["block_type"] != "text":
                if i["block_embedding"] is not None and len(i["block_embedding"]) > 0:
                    data.append(i)
            else:
                # print(len(i["block_content"]),i["block_type"])
                continue

    insert_data = {
        "block_id": [int(item["block_id"]) for item in data],
        "block_embedding": [
            (
                item["block_embedding"].tolist()
                if item["block_embedding"] is not None
                and len(item["block_embedding"]) > 0
                and type(item["block_embedding"]) is not list
                else item["block_embedding"]
            )
            for item in data
        ],  # numpy数组转list
        "pdf_path": [item["pdf_path"] for item in data],
        "num_pages": [item["num_pages"] for item in data],
        "page_number": [item["page_number"] for item in data],
        "page_height": [
            item["page_height"] if item["page_height"] else 0 for item in data
        ],
        "page_width": [
            item["page_width"] if item["page_width"] else 0 for item in data
        ],
        "num_blocks": [
            item["num_blocks"] if item["num_blocks"] else 0 for item in data
        ],
        "block_type": [item["block_type"] for item in data],
        "block_content": [item["block_content"] for item in data],
        "block_summary": [item["block_summary"] for item in data],
        "image_path": [item["image_path"] for item in data],
        "image_caption": [item["image_caption"] for item in data],
        "image_footer": [item["image_footer"] for item in data],
        "block_bbox": [str(item["block_bbox"]) for item in data],  # 列表转字符串或JSON
        "document_title": [item["document_title"] for item in data],
        "section_title": [item["section_title"] for item in data],
    }
    insert_data_final = [
        insert_data["block_id"],
        insert_data["block_embedding"],
        insert_data["pdf_path"],
        insert_data["num_pages"],
        insert_data["page_number"],
        insert_data["page_height"],
        insert_data["page_width"],
        insert_data["num_blocks"],
        insert_data["block_type"],
        insert_data["block_content"],
        insert_data["block_summary"],
        insert_data["image_path"],
        insert_data["image_caption"],
        insert_data["image_footer"],
        insert_data["block_bbox"],
        insert_data["document_title"],
        insert_data["section_title"],
    ]
    print("插入数据属性个数：", len(insert_data_final))
    print("插入数据条数：", len(data))
    print("start insert data to milvus")

    try:
        # 执行插入
        batch_insert(collection, insert_data_final, batch_size=200)
        collection.flush()
    except Exception as e:
        print(f"插入失败: {str(e)}")
        raise


def delete_collection(collection_name):
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")

    else:
        collection = Collection(collection_name)
        collection.drop()
        print(f"集合 {collection_name} 已删除")


def delete_data(collection_name):
    if not utility.has_collection(collection_name):
        print(f"集合 {collection_name} 不存在")

    else:
        collection = Collection(collection_name)
        collection.delete(expr="block_id >= 0")
        print(f"已清空集合 {collection_name}中的数据")


def delete_index(collection, index_name):
    # 删除指定的索引
    # 如果不指定名字，则删除所有索引
    collection.drop_index(index_name)


import time
from src.config.db_config import get_ollama_embedding
from ollama import Client
from src.utils.query2vec import (
    GmeQwen2VL,
    get_embedding_Qwen2VL,
)


def search_output_fields(top_k, embedding, collection, OUTPUT_FIELDS):

    search_params = {
        "ef": 5 * top_k
    }  # 资料推荐，ef为top_k的2-5倍为优，值越大，精度越大，速度也越慢
    # 开始时间
    start_time = time.time()
    ###HNSW搜索

    result = collection.search(
        data=embedding,
        anns_field="block_embedding",
        param=search_params,
        limit=top_k,
        output_fields=OUTPUT_FIELDS,
    )

    # 记录时间
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算运行时间
    # print(f"搜索时间为: {execution_time} 秒")

    return result


def search_top(top_k, embedding, collection, result_type="text"):
    OUTPUT_FIELDS = [
        "block_id",
        "block_bbox",
        "pdf_path",
        "num_pages",
        "page_number",
        "page_height",
        "page_width",
        "num_blocks",
        "block_type",
        "block_content",
        # "block_summary",
        # "image_path",
        # "image_caption",
        # "image_footer",
        # "document_title",
        # "block_embedding",
    ]

    # embedding = [llm_client.embeddings(model=llm_model,prompt=question)['embedding']]#需要注意，milvus的查询必须是二维数组

    search_params = {
        "ef": 5 * top_k,  # 资料推荐，ef为top_k的2-5倍为优，值越大，精度越大，速度也越慢
        "metric_type": "COSINE",  # 明确指定使用余弦相似度
    }
    # 开始时间
    start_time = time.time()
    ###HNSW搜索

    if result_type == "text":
        result = collection.search(
            data=embedding,
            anns_field="block_embedding",
            param=search_params,
            limit=top_k,
            expr=f'block_type == "text"',
            output_fields=OUTPUT_FIELDS,
        )
    elif result_type == "image":
        result = collection.search(
            data=embedding,
            anns_field="block_embedding",
            param=search_params,
            limit=top_k,
            expr=f'block_type == "image"',
            output_fields=["pdf_path", "num_pages", "image_path", "image_caption"],
        )
    elif result_type == "table":
        result = collection.search(
            data=embedding,
            anns_field="block_embedding",
            param=search_params,
            limit=top_k,
            expr=f'block_type == "table"',
            output_fields=["pdf_path", "num_pages", "image_path", "image_caption"],
        )
    else:
        result = collection.search(
            data=embedding,
            anns_field="block_embedding",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["pdf_path", "num_pages", "block_content"],
        )

    # 记录时间
    end_time = time.time()  # 记录结束时间
    execution_time = end_time - start_time  # 计算运行时间
    # print(f"搜索时间为: {execution_time} 秒")

    return result


def search(collection_name, question, image_embedding=False, result_type="text"):
    """
    result_type: text, image, table
    其中image与table均为对图片的embedding
    """
    collection = Collection(collection_name)
    # 加载进来数据
    collection.load()
    # 释放内存里面的内容
    # connections.release_collection(collection_name)
    top_k = 10

    if image_embedding:
        gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
        vector = get_embedding_Qwen2VL(gme, "text", question)
        embedding = [vector.tolist()]
        result = search_top(top_k, embedding, collection, result_type)
    else:
        # embeddingmodel_config = get_ollama_embedding()
        # #指定ollama
        # llm_client = Client(host=f"http://{embeddingmodel_config['host']}:{embeddingmodel_config['port']}")
        # llm_model = embeddingmodel_config['model_name']
        # embedding = [llm_client.embeddings(model=llm_model,prompt=question)['embedding']]
        embeddingprocessor = EmbeddingProcessor()
        embedding = [embeddingprocessor.get_embedding(prompt=question)]
        result = search_top(top_k, embedding, collection)

    if result_type == "image" or result_type == "table":
        for i, hits in enumerate(result):
            print(f"Query {i + 1} results:")
            for hit in hits:
                print(f"  ID: {hit.id}, Distance: {hit.distance}")
                print(
                    f"  pdf_path: {hit.entity.get('pdf_path')}, num_pages: {hit.entity.get('num_pages')}, image_path: {hit.entity.get('image_path')}, image_caption: {hit.entity.get('image_caption')}"
                )
    else:
        for i, hits in enumerate(result):
            print(f"Query {i + 1} results:")
            for hit in hits:
                print(f"  ID: {hit.id}, Distance: {hit.distance}")
                print(
                    f"  pdf_path: {hit.entity.get('pdf_path')}, num_pages: {hit.entity.get('num_pages')}, block_content: {hit.entity.get('block_content')}"
                )


def milvus_search(
    collection_name,
    query,
    top_k,
    score_threshold,
    image_embedding=False,
    result_type="text",
):
    from src.database.db_connection import get_milvus_collection

    collection = get_milvus_collection(collection_name)
    # 释放内存里面的内容
    # connections.release_collection(collection_name)

    if image_embedding:
        gme = GmeQwen2VL("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct")
        vector = get_embedding_Qwen2VL(gme, "text", query)
        embedding = [vector.tolist()]
        result = search_top(top_k, embedding, collection, result_type)
    else:
        embeddingprocessor = EmbeddingProcessor()
        embedding = [embeddingprocessor.get_embedding(prompt=query)]
        result = search_top(top_k, embedding, collection)

    record = []
    for i, hits in enumerate(result):
        for hit in hits:
            page_number = hit.entity.get("page_number")
            block_bbox = hit.entity.get("block_bbox")
            block_content = hit.entity.get("block_content")
            if hit.distance > score_threshold:
                # 构建每一条记录
                row = {
                    "score": hit.distance,  # 评分使用距离
                    "pdf_path": hit.entity.get("pdf_path"),
                    "num_pages": hit.entity.get("num_pages"),
                    "page_number": hit.entity.get("page_number"),
                    "page_height": hit.entity.get("page_height"),
                    "page_width": hit.entity.get("page_width"),
                    "num_blocks": hit.entity.get("num_blocks"),
                    "block_type": hit.entity.get("block_type"),
                    "block_bbox": hit.entity.get("block_bbox"),
                    "block_id": hit.entity.get("block_id"),
                    "block_content": hit.entity.get("block_content"),
                }
                record.append(row)
    return record


def search_challenge_data(collection, pdf_file, query_embedding, top_n, output_fields):
    # 设置搜索参数
    search_params = {"search_mode": "FLAT", "params": {"nprobe": 256}}

    # 执行搜索
    results = collection.search(
        data=query_embedding,
        anns_field="block_embedding",
        param=search_params,
        limit=top_n,
        expr=f'pdf_path == "{pdf_file}"',
        output_fields=output_fields,
    )

    return results


if __name__ == "__main__":
    # pkl存放位置
    pkl_path = "src/pkl_files/image_embedding/wtr_vector_db.pkl"
    milvus_connection()
    collection_name = "deepwriter"  # 确定collection名字

    # 连接创建collection
    collection = create_collection(collection_name, dim=1536)
    # collection = Collection("kg_test0427")

    # 插入数据。传入参数image_embedding，确定是否对图片进行了向量化
    pkl_insert(collection, pkl_path, image_embedding=True)

    # question = "2020年世界贸易报告的主要内容是什么？"

    # 搜索数据。传入参数image_embedding，确定是否对图片进行了向量化；参数result_type，确定搜索结果返回的类型（可为text, image, table）
    # search(collection_name,question,image_embedding=False)

    # 删除数据
    # delete_collection(collection_name)
