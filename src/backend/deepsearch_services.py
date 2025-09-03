from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from qwen_agent.tools.base import BaseTool, register_tool
from openai import OpenAI


from src.deepsearch.agent import HAgent
import json5
import torch
import torch.nn.functional as F
import os
import requests
import asyncio
import re
from typing import List, Optional, Dict, Any
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from loguru import logger
from src.config.db_config import get_multi_hop_port
from src.config.api_config import get_openai_key, get_vllm_llm

app = FastAPI(title="Multi-Hop RAG QA API")

# Define your document and query data structures
documents = []  # This should be populated with your actual documents
document_embeddings = []  # This should be populated with your document embeddings


# 定义请求体的 Pydantic 模型
class RetrievalSetting(BaseModel):  # milvus 检索参数
    milvus_collection: str = "challenge_data"
    top_k: int = 3
    score_threshold: float = 0.5


class RAGRetrieveRequest(BaseModel):
    query: str
    retrieval_setting: RetrievalSetting


class WebSearchRequest(BaseModel):
    query: str


class MultiHopQARequest(BaseModel):
    query: str
    max_rounds: int
    selected_tools: List[str] = ["rag_retrieve"]
    retrieval_setting: RetrievalSetting = RetrievalSetting()


OPENAI_API_KEY = get_openai_key()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# openai
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


def clean_markdown(res: str) -> str:
    """
    Clean markdown content by removing links and extra newlines.
    """
    pattern = r"\[.*?\]\(.*?\)"
    try:
        result = re.sub(pattern, "", res)
        url_pattern = pattern = (
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )
        result = re.sub(url_pattern, "", result)
        result = result.replace("* \n", "")
        result = re.sub(r"\n\n+", "\n", result)
        return result
    except Exception:
        return res


async def get_info(url: str, screenshot: bool = True) -> tuple:
    """
    Fetch URL content with optional screenshot.
    """
    run_config = CrawlerRunConfig(
        screenshot=True,
        screenshot_wait_for=1.0,
    )
    async with AsyncWebCrawler() as crawler:
        if screenshot:
            result = await crawler.arun(url, config=run_config)
            return result.html, clean_markdown(result.markdown), result.screenshot
        else:
            result = await crawler.arun(url, screenshot=screenshot)
            return result.html, clean_markdown(result.markdown)


async def fetch_all_urls(urls: List[str], screenshot: bool = True) -> List[tuple]:
    """
    Concurrently fetch multiple URLs.
    """
    tasks = [get_info(url, screenshot) for url in urls]
    return await asyncio.gather(*tasks)


@app.post("/web_search")
async def web_search(request: WebSearchRequest):
    """
    Perform web search using Serper API.
    """
    try:
        query = request.query
        # api_key = os.getenv('SERPER_API_KEY')
        api_key = SERPER_API_KEY
        if not api_key:
            raise HTTPException(
                status_code=500, detail="Serper API key not configured."
            )

        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query}
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            search_results = response.json()
            urls = [
                result.get("link", "")
                for result in search_results.get("organic", [])[:2]
            ]
            results = await fetch_all_urls(urls)

            formatted_results = []
            for url, (html, markdown, screenshot) in zip(urls, results):
                formatted_results.append(
                    {"url": url, "content": markdown, "screenshot": screenshot}
                )

            return JSONResponse(
                content={
                    "status": "success",
                    "query": query,
                    "results": formatted_results,
                }
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to retrieve search results.",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


from src.database.operations.milvus_operations import milvus_search


# API endpoints
@app.post("/milvus_retrieval")
async def milvus_retrieval(request: RAGRetrieveRequest):

    # 获取请求数据
    query = request.query
    milvus_collection = request.retrieval_setting.milvus_collection
    top_k = request.retrieval_setting.top_k
    score_threshold = request.retrieval_setting.score_threshold
    # 调用结果函数
    records = milvus_search(milvus_collection, query, top_k, score_threshold)
    # 返回符合要求的响应格式
    return {"records": records}


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


@register_tool("web_search", allow_overwrite=True)
class WebSearch(BaseTool):
    """
    A tool that performs a web search using Serper API.
    """

    description = "A tool that performs a web search using Serper API."
    parameters = [
        {
            "name": "query",
            "type": "string",
            "description": "The query to search for.",
            "required": True,
        }
    ]

    def call(self, params: str, **kwargs) -> str:
        params = json5.loads(params)
        query = params["query"]
        # 替换为你的Serper API Key
        api_key = os.getenv("SERPER_API_KEY")
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
        payload = {"q": query}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            search_results = response.json()
            # 格式化返回结果
            formatted_results = "Web search results:\n"
            formatted_results_show = "Web search results:\n"
            top_k = 2  # 获取前2个结果

            urls = [
                result.get("link", "")
                for result in search_results.get("organic", [])[:top_k]
            ]  # 多次并行爬取内容
            results = asyncio.run(fetch_all_urls(urls))
            for url, (html, markdown, screenshot) in zip(urls, results):
                formatted_results += f"\n Content: {markdown}\n"
                formatted_results_show += f"\n link: {url}\n"

            return formatted_results
        else:
            return f"Failed to retrieve search results. Status code: {response.status_code}"


@app.post("/multi_hop_qa")
async def multi_hop_qa(request: MultiHopQARequest):
    """
    Perform multi-hop question answering using selected tools.
    """
    try:
        query = request.query
        max_rounds = request.max_rounds
        selected_tools = request.selected_tools
        retrieval_params = {
            "milvus_collection": request.retrieval_setting.milvus_collection,
            "top_k": request.retrieval_setting.top_k,
            "score_threshold": request.retrieval_setting.score_threshold,
        }

        # Initialize your HAgent here
        llm_cfg["query"] = query
        llm_cfg["action_count"] = max_rounds  # 设置最大检索次数
        bot = HAgent(llm=llm_cfg, function_list=selected_tools)
        messages = []  # This stores the chat history.
        start_prompt = "query:\n{query}".format(query=query)

        messages.append({"role": "user", "content": start_prompt})
        response = bot.run(messages=messages, lang="zh", **retrieval_params)
        response_jsons = []
        r = 0
        for i in response:
            response_json = {}
            if '"}' in i[0]["content"] and "Memory" not in i[0]["content"]:
                thoughts_str = i[0]["content"].split("Action")[0]
                if r == 0:
                    response_json["thoughts"] = f"**💭Thoughts**\n {thoughts_str} "
                elif (
                    "thoughts" in response_jsons[r - 1]
                    and response_jsons[r - 1]["thoughts"]
                    != f"**💭Thoughts**\n {thoughts_str} "
                ):
                    # print(r-1,"thoughts:\t",response_jsons[r-1]["thoughts"],type(response_jsons[r-1]["thoughts"]))
                    # print("now thoughts:\t",thoughts_str,type(thoughts_str))
                    response_json["thoughts"] = f"**💭Thoughts**\n {thoughts_str} "
                    # print("thoughts_str",i)
            elif '"}' in i[0]["content"] and "Memory" in i[0]["content"]:
                memory_str = i[0]["content"][:-2]
                response_json["memory"] = f"**🤯Memory Update**\n {memory_str} "
                print("memory_str", i)
            if response_json is not None and response_json:
                response_jsons.append(response_json)
                r += 1
            if "Final Answer" in i[0]["content"]:
                answer_str = i[0]["content"]
                response_json["answer"] = f"***🙋Anwser**\n {answer_str} "
                print("answer_str", i)
                response_jsons.append(response_json)

        # For now, we'll return a mock response
        # You'll need to implement the actual HAgent integration
        return response_jsons
    except Exception as e:
        logger.error(f"Error in multi_hop_qa: {str(e)}", exc_info=True)  # 记录完整堆栈
        raise HTTPException(status_code=500, detail=str(e))  # 返回具体错误给客户端


# 启动服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_multi_hop_port())
