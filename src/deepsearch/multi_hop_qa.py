from tqdm import tqdm
from src.utils.file_utils import write, read


import os
import json5
from qwen_agent.tools.base import BaseTool, register_tool
import torch

from src.deepsearch.agent import HAgent
from src.database.operations.milvus_operations import search_output_fields
from src.database.db_connection import milvus_connection
from pymilvus import Collection
from src.config.api_config import get_vllm_llm, get_openai_key
from src.utils.api_llm_requests import EmbeddingProcessor

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


# openai
# OPENAI_API_KEY = get_openai_key()
# model = "gpt-4o"
# llm_cfg = {
#     'model': model,
#     'api_key': OPENAI_API_KEY,
#     'generate_cfg': {
#         'top_p': 0.8,
#         'max_input_tokens': 120000,
#         'max_retries': 20
#     },
# }

# vllm
llm_config = get_vllm_llm()
model = llm_config["model_name"]
llm_cfg = {
    "model": model,
    "api_key": "EMPTY",
    "model_server": f"http://{llm_config['host']}:{llm_config['port']}/v1",
    "generate_cfg": {"top_p": 0.8, "max_input_tokens": 120000, "max_retries": 20},
}


OUTPUT_FIELDS = [
    "id",
    "block_content",
    "author",
    "title",
    "published_at",
    "category",
    "url",
]

from pymilvus import Collection
from collections import defaultdict


def aggregate_blocks_by_title(
    records, collection, title_field="title", block_field="block_content"
):
    """
    基于初始 records 的 title 聚合其所有 block_content 并拼接为 text
    """
    # 1. 获取所有的 title
    titles = [r[title_field] for r in records if r.get(title_field)]

    # 2. 去重
    unique_titles = list(set(titles))

    # 3. 构造聚合结果容器
    aggregated_results = []

    for idx, title in enumerate(unique_titles):
        # 4. 构造查询表达式（注意转义引号）
        expr = f'{title_field} == "{title}"'

        # 5. 在 Milvus 中使用 filter 表达式查询所有 block_content
        search_result = collection.query(
            expr=expr, output_fields=[block_field], consistency_level="Strong"
        )

        # 6. 拼接所有 block_content
        all_blocks = [res[block_field] for res in search_result if res.get(block_field)]
        full_text = "\n\n".join(all_blocks)

        # 7. 构建返回结构
        aggregated_results.append({"id": idx, "title": title, "text": full_text})

    return aggregated_results


# RAG检索工具
@register_tool("rag_retrieve", allow_overwrite=True)
class RAGRetrieve(BaseTool):
    """
    A tool that performs a single step of RAG retrieval.
    It takes a query and retrieves relevant documents based on semantic similarity.
    """

    description = "A tool that performs a single step of RAG retrieval. It takes a query and retrieves relevant documents based on semantic similarity."
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "The query to retrieve documents for.",
            "required": True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        params = json5.loads(params)
        query = params["query"]

        milvus_collection = kwargs.get("milvus_collection")
        top_k = kwargs.get("top_k")
        score_threshold = kwargs.get("score_threshold")

        embeddingprocessor = EmbeddingProcessor()
        # print("embedding start")
        embedding = [embeddingprocessor.get_embedding(prompt=query)]
        # print("embedding ok")
        milvus_connection()
        collection = Collection(milvus_collection)
        result = search_output_fields(top_k, embedding, collection, OUTPUT_FIELDS)
        records = []
        for i, hits in enumerate(result):
            for hit in hits:
                if hit.distance > score_threshold:
                    # 构建每一条记录
                    row = {
                        "score": hit.distance,  # 评分使用距离
                        "id": i,
                        "text": hit.entity.get("block_content"),
                        "author": hit.entity.get("author"),
                        "title": hit.entity.get("title"),
                        "published_at": hit.entity.get("published_at"),
                        "category": hit.entity.get("category"),
                        "url": hit.entity.get("url"),
                    }
                    records.append(row)

        # aggregated = aggregate_blocks_by_title(records, collection)

        retrieved_docs = [
            {"id": record["id"], "text": record["text"]} for record in records
        ]
        # print(f"Retrieved documents: {retrieved_docs}")
        # 格式化返回结果

        response = "Retrieved documents:\n"
        for doc in retrieved_docs:
            response += f"- Document {doc['id']}: {doc['text']}\n\n\n"

        return response


class MultiHopAgent:
    def __init__(self) -> None:
        pass

    def answer(
        self,
        query: str,
        top_n: int = 10,
        score_threshold: float = 0.0,
        max_rounds: int = 3,
        collection_name: str = "Multi_hop",
    ):
        llm_cfg["query"] = query
        llm_cfg["action_count"] = max_rounds  # 设置最大检索次数
        bot = HAgent(llm=llm_cfg, function_list=["rag_retrieve"])
        messages = []  # This stores the chat history.
        start_prompt = "query:\n{query}".format(query=query)

        retrieval_params = {
            "top_k": top_n,
            "milvus_collection": collection_name,
            "score_threshold": score_threshold,
        }

        messages.append({"role": "user", "content": start_prompt})
        # print("bot.run start")
        response = bot.run(messages=messages, lang="zh", **retrieval_params)
        response_jsons = []
        r = 0
        for i in response:
            response_json = {}
            if '"}' in i[0]["content"] and "Memory" not in i[0]["content"]:
                thoughts_str = i[0]["content"].split("Action")[0]
                if r == 0:
                    response_json["thoughts"] = thoughts_str
                elif (
                    "thoughts" in response_jsons[r - 1]
                    and response_jsons[r - 1]["thoughts"] != thoughts_str
                ):
                    print(r - 1, "thoughts:\t", response_jsons[r - 1]["thoughts"])
                    response_json["thoughts"] = thoughts_str
                    print("thoughts_str", i)
            elif '"}' in i[0]["content"] and "Memory" in i[0]["content"]:
                memory_str = i[0]["content"][:-2]
                response_json["memory"] = memory_str
            # print("memory_str",i)
            if response_json is not None and response_json:
                response_jsons.append(response_json)
                r += 1
            if "Final Answer" in i[0]["content"]:
                answer_str = i[0]["content"]
                response_json["answer"] = answer_str
                # print("answer_str",i)
                response_jsons.append(response_json)

        # For now, we'll return a mock response
        # You'll need to implement the actual HAgent integration
        return response_jsons


def query_bot(messages, **kwargs):

    agent = MultiHopAgent()

    # 执行多跳推理
    answer = agent.answer(messages)
    # print(f"Answer: {answer}")
    # print(f"Answer type: {type(answer[-1])}")

    if answer and "answer" in answer[-1] and answer[-1]["answer"]:
        response = answer[-1]["answer"]
    else:
        response = "Insufficient information."

    return response


def multi_hop_qa(query_file, save_file):

    query_data = read(query_file)

    for d in tqdm(query_data):

        prompt = f"Question:{d['query']}\n"
        response = query_bot(prompt)
        # print(response)
        save = {}
        save["query"] = d["query"]
        save["prompt"] = prompt
        save["model_answer"] = response
        save["gold_answer"] = d["answer"]
        save["question_type"] = d["question_type"]

        write(save_file, [save])
