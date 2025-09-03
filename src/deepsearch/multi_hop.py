import os
import json5
from qwen_agent.tools.base import BaseTool, register_tool
import torch


from src.deepsearch.agent import HAgent
from src.config.api_config import get_vllm_llm, get_openai_key
from src.database.operations.milvus_operations import milvus_search

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
        # 调用结果函数
        records = milvus_search(milvus_collection, query, top_k, score_threshold)

        retrieved_docs = [
            {"id": record["block_id"], "text": record["block_content"]}
            for record in records
        ]
        # print(f"Retrieved documents: {retrieved_docs}")
        # 格式化返回结果

        response = "Retrieved documents:\n"
        for doc in retrieved_docs:
            response += f"- Document {doc['id']}: {doc['text']}\n"

        return response


class MultiHopAgent:
    def __init__(self) -> None:
        pass

    def answer(
        self,
        query: str,
        top_n: int = 14,
        score_threshold: float = 0.0,
        max_rounds: int = 3,
        collection_name: str = "challenge_data_vllm",
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
                    print(
                        r - 1,
                        "thoughts:\t",
                        response_jsons[r - 1]["thoughts"],
                        type(response_jsons[r - 1]["thoughts"]),
                    )
                    print("now thoughts:\t", thoughts_str, type(thoughts_str))
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


if __name__ == "__main__":
    # 初始化多跳代理
    agent = MultiHopAgent()

    # 执行多跳推理
    answer = agent.answer("需要多步推理的复杂问题")
