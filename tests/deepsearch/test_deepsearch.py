import streamlit as st
import os
import requests
import json5
from qwen_agent.tools.base import BaseTool, register_tool
from src.utils.api_llm_requests import EmbeddingProcessor
from src.deepsearch.agent import HAgent
import torch
import torch.nn.functional as F
import numpy as np

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from bs4 import BeautifulSoup
import re
from openai import OpenAI
from src.config.api_config import get_ollama_llm, get_vllm_llm, get_openai_key

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


# # 示例中文知识库（文档集合）
# documents = [
#     {"id": 1, "text": "苹果公司是一家总部位于美国加利福尼亚州库比蒂诺的跨国科技公司。"},
#     {"id": 2, "text": "苹果公司设计、制造和销售消费电子产品、软件和在线服务。"},
#     {"id": 3, "text": "苹果公司由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年创立。"},
#     {"id": 4, "text": "苹果公司的产品包括iPhone、iPad、Mac电脑和Apple Watch。"},
#     {"id": 5, "text": "苹果公司的软件包括macOS、iOS、iPadOS、watchOS和tvOS。"},
#     {"id": 6, "text": "苹果公司是全球知名的大型科技公司之一，与亚马逊、谷歌、微软和Facebook并列。"},
#     {"id": 7, "text": "史蒂夫·乔布斯从1997年到2011年去世期间担任苹果公司的首席执行官。"},
#     {"id": 8, "text": "苹果公司的总部位于美国加利福尼亚州库比蒂诺。"},
#     {"id": 9, "text": "苹果公司现任CEO是蒂姆·库克（Tim Cook），他自2011年起担任苹果公司的首席执行官"},
#     {"id": 10, "text": "苹果公司的股票在纳斯达克交易所上市，股票代码为AAPL。"}
# ]

# # 示例中文多跳查询
# queries = [
#     {"query": "苹果公司的股票代码是什么？它在哪里上市？它的总部在哪里？", "expected": [1, 8, 10]},
#     {"query": "谁创立了苹果公司？它有哪些产品？", "expected": [3, 4]},
#     {"query": "苹果公司有哪些软件产品？它2008年的CEO是谁？", "expected": [5, 7]},
#     {"query": "苹果公司2050年的CEO是谁？", "expected": []}
# ]


# 示例英文知识库（文档集合）
documents = [
    {
        "id": 1,
        "text": "Tesla, Inc. was founded in 2003 by engineers Martin Eberhard and Marc Tarpenning.",
    },
    {
        "id": 2,
        "text": "Elon Musk joined Tesla in early 2004 as an investor and became CEO in 2008.",
    },
    {"id": 3, "text": "Tesla's headquarters is located in Austin, Texas, USA."},
    {
        "id": 4,
        "text": "Tesla manufactures electric vehicles, battery energy storage systems, and solar panels.",
    },
    {
        "id": 5,
        "text": "The Tesla Model S, Model 3, Model X, and Model Y are its main electric vehicles.",
    },
    {"id": 6, "text": "Tesla’s stock is listed on the NASDAQ under the ticker TSLA."},
    {
        "id": 7,
        "text": "SpaceX was founded by Elon Musk in 2002 to reduce the cost of space travel.",
    },
    {"id": 8, "text": "SpaceX successfully launched the Falcon Heavy rocket in 2018."},
    {
        "id": 9,
        "text": "The Falcon Heavy is capable of delivering large payloads to orbit and beyond.",
    },
    {
        "id": 10,
        "text": "Austin is also home to a growing tech sector including companies like Oracle and Dell.",
    },
    {
        "id": 11,
        "text": "In 2020, Tesla announced the construction of its Gigafactory in Austin, Texas.",
    },
    {
        "id": 12,
        "text": "The Tesla Gigafactory in Austin is focused on producing the Cybertruck and Model Y.",
    },
    {
        "id": 13,
        "text": "Elon Musk is also the founder of Neuralink and The Boring Company.",
    },
    {"id": 14, "text": "Neuralink develops brain–computer interface technology."},
    {
        "id": 15,
        "text": "The Boring Company builds underground transportation systems using tunnel boring machines.",
    },
]


queries = [
    {
        "query": "Who is the founder of the company that launched the Falcon Heavy rocket? Where is the headquarters of the company he later became CEO of?",
        "expected": [2, 3, 7, 8],
    },
    {
        "query": "Which city is home to Tesla’s Gigafactory? What vehicle models are produced there?",
        "expected": [11, 12],
    },
    {
        "query": "What electric vehicles does Tesla produce, and what is its stock ticker on NASDAQ?",
        "expected": [5, 6],
    },
    {
        "query": "List all companies founded by Elon Musk and one key product or focus area of each.",
        "expected": [2, 7, 8, 13, 14, 15],
    },
    {
        "query": "Which founders started Tesla? Who became its CEO later, and in what year?",
        "expected": [1, 2],
    },
    {
        "query": "What major technological industries are represented in Austin, and which Tesla facility is located there?",
        "expected": [3, 10, 11],
    },
    {
        "query": "What rocket can deliver large payloads into orbit, and which company created it?",
        "expected": [8, 9, 7],
    },
    {
        "query": "What kind of transportation systems does The Boring Company develop, and what technology does Neuralink focus on?",
        "expected": [14, 15],
    },
    {
        "query": "Which vehicle models are currently being produced in Tesla’s Austin Gigafactory, and who is the CEO of the company?",
        "expected": [2, 12],
    },
    {
        "query": "Who became Tesla CEO in 2008, and what other companies has he founded?",
        "expected": [2, 7, 13],
    },
]


# 计算文档的嵌入向量（批量处理）
document_texts = [doc["text"] for doc in documents]
embeddingprocessor = EmbeddingProcessor()
document_embeddings = embeddingprocessor.get_list_embedding(text_list=document_texts)


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
        st.markdown("**🌐Now retrieving**")
        st.write(query)
        print("RAG query:", query)
        embeddingprocessor = EmbeddingProcessor()
        query_embedding = embeddingprocessor.get_embedding(prompt=query)
        if query_embedding is None:
            raise ValueError("Failed to retrieve query embedding.")

        # 计算查询与文档的相似度
        # 将查询嵌入向量和文档嵌入向量转换为 PyTorch 张量
        query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        document_embeddings_tensor = torch.tensor(
            document_embeddings, dtype=torch.float32
        )
        # 计算查询与文档的相似度
        # 使用 PyTorch 的 cosine_similarity 函数
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0), document_embeddings_tensor, dim=1
        )

        # 获取最相似的文档
        top_k = torch.topk(similarities, k=2)  # 获取最相似的文档
        print(f"Top K indices: {top_k.indices}")
        retrieved_docs = [documents[idx.item()] for idx in top_k.indices]
        print(f"Retrieved documents: {retrieved_docs}")

        # 格式化返回结果

        response = "Retrieved documents:\n"
        with col2:
            for doc in retrieved_docs:
                response += f"- Document {doc['id']}: {doc['text']}\n"
            st.markdown(query)
            st.markdown(response)

        return response


def clean_markdown(res):
    """
    Args:
        res (str): markdown content

    Returns:
        str: cleaned markdown content
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


async def get_info(url, screenshot=True) -> str:
    """
    Args:
        url (str): url
        screentshot (bool): whether to take a screenshot

    Returns:
        str: html content and cleaned markdown content
    """
    run_config = CrawlerRunConfig(
        screenshot=True,  # Grab a screenshot as base64
        screenshot_wait_for=1.0,  # Wait 1s before capturing
    )
    async with AsyncWebCrawler() as crawler:
        if screenshot:
            result = await crawler.arun(url, config=run_config)
            return result.html, clean_markdown(result.markdown), result.screenshot
        else:
            result = await crawler.arun(url, screenshot=screenshot)
            return result.html, clean_markdown(result.markdown)


async def fetch_all_urls(urls, screenshot=True):
    """
    并发抓取多个 URL 的内容
    Args:
        urls (list[str]): URL 列表
        screenshot (bool): 是否截图
    Returns:
        list[tuple]: 每个 URL 的抓取结果
    """
    tasks = [get_info(url, screenshot) for url in urls]
    results = await asyncio.gather(*tasks)
    return results


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
        with col2:
            if response.status_code == 200:
                search_results = response.json()
                # 格式化返回结果
                formatted_results = "Web search results:\n"
                formatted_results_show = "Web search results:\n"
                top_k = 2  # 获取前2个结果
                # # 单次爬取，比较耗时
                # # for result in search_results.get("organic", [])[:top_k]:
                # #     title = result.get('title', '')
                # #     link = result.get('link', '')
                # #     html, markdown, screenshot = asyncio.run(get_info(link)) # 调用函数获取链接内容
                # #     formatted_results += f"- {title}\n  Content: {markdown}\n"
                # #     formatted_results_show += f"- {title}\n  {link}\n"

                urls = [
                    result.get("link", "")
                    for result in search_results.get("organic", [])[:top_k]
                ]  # 多次并行爬取内容
                results = asyncio.run(fetch_all_urls(urls))
                for url, (html, markdown, screenshot) in zip(urls, results):
                    # print(f"URL: {url}")
                    # print("Main Content:")
                    # print(markdown)
                    formatted_results += f"\n Content: {markdown}\n"
                    formatted_results_show += f"\n link: {url}\n"
                st.markdown(query)
                st.markdown(formatted_results_show)
                return formatted_results
            else:
                return f"Failed to retrieve search results. Status code: {response.status_code}"


if __name__ == "__main__":
    st.title("🔍 Multi-Hop RAG QA")
    st.markdown("### 📚 Introduction")
    st.markdown(
        "👋 Welcome to Multi-Hop RAG QA! This tool helps you perform multi-hop RAG retrieval to answer complex questions."
    )

    MAX_ROUNDS = st.number_input(
        "Max Count Count：", min_value=1, max_value=50, value=20, step=1
    )
    # 注入自定义 CSS 样式，调整字体大小
    st.write(
        """
        <style>
        textarea {
            font-size: 12px !important; /* 调整字体大小 */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.form(key="my_form"):
            # 将文档内容格式化为列的形式
            documents_text = "\n".join(
                [f"{doc['id']}. {doc['text']}" for doc in documents]
            )
            form_1_text = st.text_area(
                "**documents**", value=documents_text, height=200
            )
            queries_text = "\n".join([f"Query: {query['query']}" for query in queries])
            form_2_text = st.text_area("**queries**", value=queries_text, height=100)
            query = st.text_area(
                "🤔Query", placeholder="Input the query you want to ask."
            )
            submit_button = st.form_submit_button("Start!!!!")
            selected_tools = st.multiselect(
                "Select Tools", ["rag_retrieve", "web_search"], default=["rag_retrieve"]
            )
            if submit_button:
                if query:
                    # tools = ["rag_retrieve", "web_search"]
                    llm_cfg["query"] = query
                    llm_cfg["action_count"] = MAX_ROUNDS  # 设置最大检索次数
                    bot = HAgent(llm=llm_cfg, function_list=selected_tools)
                    messages = []  # This stores the chat history.
                    start_prompt = "query:\n{query}".format(query=query)
                    # st.markdown('**🌐Now retrieving**')
                    # st.write(query)
                    with col2:
                        st.markdown("**📝Retrieved Documents**")

                    start_prompt = "query:\n{query} ".format(query=query)

                    messages.append({"role": "user", "content": start_prompt})

                    response = bot.run(messages=messages, lang="zh")
                    r = 0
                    response_jsons = []
                    for i in response:
                        response_json = {}
                        if '"}' in i[0]["content"] and "Memory" not in i[0]["content"]:

                            thoughts_str = i[0]["content"].split("Action")[0]
                            if r == 0:
                                st.markdown("**💭Thoughts**")
                                st.markdown(thoughts_str)
                                response_json["thoughts"] = thoughts_str
                            elif (
                                "thoughts" in response_jsons[r - 1]
                                and response_jsons[r - 1]["thoughts"] != thoughts_str
                            ):
                                print(
                                    r - 1,
                                    "thoughts:\t",
                                    response_jsons[r - 1]["thoughts"],
                                )
                                st.markdown("**💭Thoughts**")
                                st.markdown(thoughts_str)
                                response_json["thoughts"] = thoughts_str
                                print("thoughts_str", i)

                        elif '"}' in i[0]["content"] and "Memory" in i[0]["content"]:
                            st.text_area("**🤯Memory Update**", i[0]["content"][:-2])

                        if response_json is not None and response_json:
                            response_jsons.append(response_json)
                            r += 1

                        if "Final Answer" in i[0]["content"]:
                            st.session_state.answer = i[0]["content"]
                            st.markdown("**🙋Anwser**")
                            st.write(st.session_state.answer)
