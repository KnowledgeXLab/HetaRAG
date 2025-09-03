# 第二版api，聚合elasticsearch的关键词搜索，milvus的向量搜索，以及neo4j的图谱搜索
# api的定义需要根据官方文档所说的请求示例来链接如下
# https://docs.dify.ai/zh-hans/guides/knowledge-base/external-knowledge-api-documentation
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from pymilvus import connections as milvus_connections
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel
from typing import Optional


from src.database.operations.milvus_operations import search_top as milvus_search_top
from src.database.operations.milvus_operations import milvus_search
import numpy as np
import json
import logging
from ollama import Client
from src.database.operations.neo4j_operation import (
    neo4j_key_search_bytoken_with_more_relation,
)
from neo4j import GraphDatabase
import re

from src.database.operations.elastic_operations import search_top as es_search_top
from elasticsearch_dsl import connections as es_connections
from elasticsearch_dsl.query import ScriptScore, Q, MatchAll, MatchPhrase
import warnings


# 这是我们存储的有效 API 密钥
API_KEY = "123456"


# 创建api的app
app = FastAPI()


# 定义请求体的 Pydantic 模型
class RetrievalSetting(BaseModel):
    milvus_collection: str
    top_k: int
    score_threshold: float


class RetrievalRequest(BaseModel):
    query: str
    retrieval_setting: RetrievalSetting


# 获取配置信息,因为这三个库中的get_ollama_embedding返回信息完全相同，因此只引用了es里的

from src.config.db_config import get_ollama_embedding
from src.config.db_config import get_config, get_data_search_port


es_config = get_config("Elasticsearch")
milvus_config = get_config("Milvus")
neo4j_config = get_config("Neo4j")
embeddingmodel_config = get_ollama_embedding()


# 定义一个函数来获取请求头中的 Authorization 字段
def get_api_key(authorization: str = Header(...)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="API Key 不正确")
    return authorization


# 调用neo4j_search来进行搜索,不接受评分限制
def get_neo4jknow(question, top_k):
    # 调用neo4j_searh，而后将数据进行包装
    user_dict = (
        "/data/hrag-backend/hrag-backend-master/src/data/words_dict/user_dict demo1.txt"
    )
    uri = f"bolt://{neo4j_config['host']}:{neo4j_config['read_write_port']}"
    username = neo4j_config["username"]  # 你的数据库用户名
    password = neo4j_config["password"]  # 你的数据库密码
    driver = GraphDatabase.driver(uri, auth=(username, password))

    key_searcher = neo4j_key_search_bytoken_with_more_relation(
        driver, question, top_k, user_dict
    )
    # key_searcher = neo4j_key_search_bytoken_with_more_relation(driver,'WTO and agriculture',10,user_dict)
    result = key_searcher.pipeline()
    record = []

    # #构建结构
    # if len(result) == 3:
    #     result_dict = result[0]
    #     doc_dict = result[1]
    #     page_dict = result[2]
    #     print(result_dict)

    for item in result:
        text, score = item

        # 初始化空列表
        node1_list = []
        node2_list = []
        relation_list = []
        relations = text.split(".--")
        # 遍历每个关系
        for relation in relations:
            try:
                # 使用正则表达式提取 node1, node2, relation
                pattern = r"node1:([^,]+),node2:([^,]+),relation:([^,]+),text:([^\.]+)"
                match = re.search(pattern, relation)

                if match:
                    node1 = match.group(1).strip()
                    node2 = match.group(2).strip()
                    relation = match.group(3).strip()
                    relation_text = match.group(4).strip()

                    relation_list.append(
                        {
                            "node1": node1,
                            "node2": node2,
                            "relation": relation,
                            "text": relation_text,
                        }
                    )
                else:
                    continue
            except Exception as e:
                continue

        record.append({"score": score, "relations": relation_list})

    return record


# 调用es_search来进行搜索，不接受评分限制
def get_esknow(question, top_k):
    # 与es建立连接
    from src.database.db_connection import es_connection

    client = es_connection()

    # 指定索引
    index_name = "knowledge_test"  # 索引名字

    result = es_search_top(question, top_k, index_name, client)
    record = []

    for item in result:
        # 构建row：
        row = {
            "metadata": {
                "path": item["filename"],  # 获取 filename 作为 path
                "description": item["chunk_id"],  # 获取 num 作为 description
            },
            "score": 0,  # 评分使用距离
            "title": item["filename"],  # title 使用 filename
            "content": item["text"],  # content 使用 text
        }
        record.append(row)

    # print(record)
    return record


#################################################################################
# 下面的都是定义请求的函数


@app.get("/")
async def hello():
    print("hello api!")
    return "hello api"


# 定义 POST 请求的处理函数,milvus向量数据库搜索的api
@app.post("/milvus/services")
async def milvus_retrieval(
    request: RetrievalRequest, authorization: str = Depends(get_api_key)
):

    # # 获取请求数据
    # query = request.query
    # top_k = request.retrieval_setting.top_k
    # score_threshold = request.retrieval_setting.score_threshold

    # # 调用结果函数
    # records = get_milvusknow_test(query,top_k,score_threshold)
    # # 返回符合要求的响应格式
    # return {"records": records}

    # 获取请求数据
    query = request.query
    milvus_collection = request.retrieval_setting.milvus_collection
    top_k = request.retrieval_setting.top_k
    score_threshold = request.retrieval_setting.score_threshold
    # 调用结果函数
    records = milvus_search(milvus_collection, query, top_k, score_threshold)
    # 返回符合要求的响应格式
    return {"records": records}


# 定义 POST 请求的处理函数,neo4j知识图谱搜索的api
@app.post("/neo4j/services")
async def neo4j_retrieval(
    request: RetrievalRequest, authorization: str = Depends(get_api_key)
):

    # 获取请求数据
    query = request.query
    top_k = request.retrieval_setting.top_k
    score_threshold = request.retrieval_setting.score_threshold

    # 调用结果函数
    records = get_neo4jknow(query, top_k)

    # 返回符合要求的响应格式
    return {"records": records}


# 定义 POST 请求的处理函数,Elasticsearch的api
@app.post("/elasticsearch/services")
async def es_retrieval(
    request: RetrievalRequest, authorization: str = Depends(get_api_key)
):

    query = request.query
    top_k = request.retrieval_setting.top_k

    # 执行搜索
    records = get_esknow(query, top_k)

    # 返回符合要求的响应格式
    return {"records": records}


# 启动服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_data_search_port())
