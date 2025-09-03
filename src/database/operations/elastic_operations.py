import os
from elasticsearch_dsl import (
    Search,
    Document,
    Text,
    Index,
    Integer,
    connections,
    DenseVector,
)
from elasticsearch_dsl.query import ScriptScore, Q, MatchAll, MatchPhrase
import time
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ✅ 引入你自己的连接逻辑
from src.database.db_connection import es_connection

# ✅ 第一步：先创建连接
client = es_connection()

# ✅ 第二步：手动注册为 default（这一步是关键！！）
connections.add_connection("default", client)

# ✅ 第三步：再继续创建 index，使用 DSL API
index_name = "knowledge_test"
index = Index(index_name)


# 定义文档结构
class TextDocument(Document):
    filename = Text()
    chunk_id = Integer()
    text = Text()
    page = Integer()  # 添加页面字段

    class Index:
        name = index_name
        settings = {
            "analysis": {
                "analyzer": {"ik_smart": {"tokenizer": "ik_max_word"}},
                "search_analyzer": {"ik_smart": {"tokenizer": "ik_max_word"}},
            }
        }


# 只在索引不存在时初始化
if not index.exists():
    TextDocument.init()


def upload_es(pkl_path, client):
    """上传数据到ES索引"""
    print(f"[INFO] 开始上传数据到ES索引，路径: {pkl_path}")

    # 检查路径是否存在
    if not os.path.exists(pkl_path):
        print(f"[ERROR] 路径不存在: {pkl_path}")
        return

    # 如果是目录，查找pkl文件
    if os.path.isdir(pkl_path):
        pkl_files = [f for f in os.listdir(pkl_path) if f.endswith(".pkl")]
        if not pkl_files:
            print(f"[ERROR] 目录中没有找到pkl文件: {pkl_path}")
            return
        pkl_file = os.path.join(pkl_path, pkl_files[0])
    else:
        pkl_file = pkl_path

    print(f"[INFO] 使用pkl文件: {pkl_file}")

    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        print(f"[INFO] 加载了 {len(data)} 条数据")

        # 批量上传数据（使用rebuild_es_index.py的方法）
        batch_size = 2000  # 每批2000条
        total_batches = (len(data) + batch_size - 1) // batch_size
        print(f"开始批量上传，共 {total_batches} 批次...")

        success_count = 0
        for i in tqdm(
            range(0, len(data), batch_size), desc="批量上传到ES", unit="batch"
        ):
            batch_data = data[i : i + batch_size]
            bulk_actions = []

            for row in batch_data:
                try:
                    action = {
                        "_index": index_name,
                        "_source": {
                            "filename": row["filename"],
                            "chunk_id": row["chunk_id"],
                            "text": row["text"],
                            "page": row.get("page", 0),
                        },
                    }
                    bulk_actions.append(action)
                except Exception as e:
                    print(f"[WARN] 处理数据失败: {e}")
                    continue

            if bulk_actions:
                try:
                    from elasticsearch import helpers

                    # helpers.bulk返回(success_count, errors)元组
                    success_num, errors = helpers.bulk(
                        client, bulk_actions, chunk_size=1000, request_timeout=60
                    )
                    success_count += success_num
                    print(f"[INFO] 批次 {i//batch_size + 1} 成功上传 {success_num} 条")
                    if errors:
                        print(
                            f"[WARN] 批次 {i//batch_size + 1} 有 {len(errors)} 个错误"
                        )

                except Exception as e:
                    print(f"[ERROR] 批量上传失败: {e}")
                    # 回退到逐条上传
                    for action in bulk_actions:
                        try:
                            # 修正ES API调用方式
                            client.index(index=action["_index"], body=action["_source"])
                            success_count += 1
                        except Exception as e:
                            print(f"[WARN] 单条上传失败: {e}")
                            continue

        print(f"[INFO] ES数据上传完成，成功上传 {success_count} 条数据")

        print(f"[INFO] ES数据上传完成")

    except Exception as e:
        print(f"[ERROR] 上传ES数据失败: {e}")
        raise


def delete(index_name):
    index = Index(index_name)
    if index.exists():
        index.delete()


def search_top(question, top_k, index_name, client):
    try:
        # 检查索引是否存在
        index = Index(index_name)
        if not index.exists():
            print(f"[ERROR] ES索引 '{index_name}' 不存在")
            return []

        # 检查索引中的文档数量
        count_result = client.count(index=index_name)
        doc_count = count_result["count"]

        if doc_count == 0:
            print(f"[ERROR] ES索引 '{index_name}' 中没有数据")
            return []

    except Exception as e:
        print(f"[ERROR] 检查ES索引状态失败: {e}")
        return []

    search = Search(index=index_name, using=client)

    # 尝试不同的查询方式
    try:
        # 方式1: multi_match查询
        s = search.query("multi_match", query=question, fields=["filename", "text"])[
            :top_k
        ]
        result = s.execute()

        if len(result.hits) == 0:
            # 方式2: match查询
            s = search.query("match", text=question)[:top_k]
            result = s.execute()

            if len(result.hits) == 0:
                # 方式3: 模糊查询
                s = search.query("match_phrase", text=question)[:top_k]
                result = s.execute()

                if len(result.hits) == 0:
                    # 方式4: 获取所有文档
                    s = search.query("match_all")[:top_k]
                    result = s.execute()

    except Exception as e:
        print(f"[ERROR] ES查询失败: {e}")
        return []

    out = []
    for hit in result.hits:
        hit_dict = hit.to_dict()
        # 添加ES的分数信息
        hit_dict["meta"] = {"score": hit.meta.score, "index": hit.meta.index}
        # 添加页面信息
        hit_dict["page"] = hit_dict.get("page", 0)
        out.append(hit_dict)

    return out


if __name__ == "__main__":
    pkl_path = "src/pkl_files/es_test"
    upload_es(pkl_path, client)
    out = search_top("东数西算", 1, index_name, client)
