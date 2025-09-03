import json
from src.utils.api_llm_requests import EmbeddingProcessor
from src.utils.file_utils import write, read
from src.database.db_connection import milvus_connection

from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility


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
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="block_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="block_content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="author", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="published_at", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=2048),
        ]

        schema = CollectionSchema(fields, description="Multi-hop corpus for RAG")

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

    print(f"已为字段 {field_name} 创建 {index_params['index_type']} 索引")


from tqdm import tqdm


def batch_insert(collection, data, batch_size=100):
    """分批插入数据，避免 gRPC 消息过大"""
    total = len(data[0])
    for i in tqdm(range(0, total, batch_size)):
        batch_data = [row[i : i + batch_size] for row in data]
        collection.insert(batch_data)

    print("插入完成！")


def data_insert(collection, data):

    insert_data = []
    for i in data:
        if (
            len(i["block_content"]) < 65535
            and i["block_embedding"] is not None
            and len(i["block_embedding"]) > 0
        ):
            insert_data.append(i)
        else:
            continue

    fields = [
        "block_embedding",
        "block_content",
        "author",
        "title",
        "published_at",
        "category",
        "url",
    ]
    data_columns = [[] for _ in fields]

    for item in insert_data:
        for i, field in enumerate(fields):
            data_columns[i].append(item[field])

    print("插入数据属性个数：", len(data_columns))

    try:
        # 执行插入
        batch_insert(collection, data_columns, batch_size=200)
        collection.flush()
    except Exception as e:
        print(f"插入失败: {str(e)}")
        raise


def data_embedding(corpus_data):  # 按照换行切割文本块
    # 初始化 embedding 处理器
    embeddingprocessor = EmbeddingProcessor()

    # 存储最终结果
    indexed_corpus = []

    # 处理每条文档
    for doc in corpus_data:
        body_chunks = doc["body"].split("\n\n")
        for chunk in body_chunks:
            chunk = chunk.strip()
            if chunk:  # 跳过空段
                embedding = embeddingprocessor.get_embedding(prompt=chunk)
                indexed_corpus.append(
                    {
                        "category": doc["category"],
                        "author": str(doc["author"]),
                        "published_at": doc["published_at"],
                        "title": doc["title"],
                        "url": doc["url"],
                        "source": doc["source"],
                        "block_content": chunk,
                        "block_embedding": embedding,
                    }
                )

    return indexed_corpus


from transformers import AutoTokenizer
from src.utils.api_llm_requests import EmbeddingProcessor


def data_embedding(corpus_data, model_name="bert-base-uncased", chunk_token_size=256):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embeddingprocessor = EmbeddingProcessor()

    indexed_corpus = []

    for doc in corpus_data:
        body_text = doc.get("body", "")
        if not body_text.strip():
            continue

        # 1. 将整段文本编码成 token ids
        encoded = tokenizer(
            body_text, return_offsets_mapping=True, add_special_tokens=False
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        # 2. 逐 chunk 切分 token，并恢复为原始文本
        for i in range(0, len(input_ids), chunk_token_size):
            chunk_ids = input_ids[i : i + chunk_token_size]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if not chunk_text:
                continue

            # 3. 获取 embedding
            embedding = embeddingprocessor.get_embedding(prompt=chunk_text)

            # 4. 添加到最终结果
            indexed_corpus.append(
                {
                    "category": str(doc.get("category", "")),
                    "author": str(doc.get("author", "")),
                    "published_at": str(doc.get("published_at", "")),
                    "title": str(doc.get("title", "")),
                    "url": str(doc.get("url", "")),
                    "source": str(doc.get("source", "")),
                    "block_content": chunk_text,
                    "block_embedding": embedding,
                }
            )

    return indexed_corpus


if __name__ == "__main__":

    # corpus 数据
    corpus = read("src/multi_hop_agent/data/corpus.json")
    indexed_corpus = data_embedding(corpus)

    # import pickle
    # with open("src/multi_hop_agent/data/corpus.pkl", 'wb') as f:
    #     pickle.dump(indexed_corpus, f)

    # with open("src/multi_hop_agent/data/corpus.pkl", 'rb') as f:
    #     indexed_corpus = pickle.load(f)

    # 链接milvus数据库
    milvus_connection()
    collection_name = "Multi_hop"  # 确定collection名字

    # 连接创建collection
    collection = create_collection(collection_name, dim=1024)

    # 插入数据。传入参数image_embedding，确定是否对图片进行了向量化
    data_insert(collection, indexed_corpus)
