import os
import sys
import json
import time
import argparse


from src.utils.logging_utils import setup_logger

import os
import logging
import numpy as np
import tiktoken
import yaml
from src.data_processor.knowledge_graph.hirag import HiRAG, QueryParam
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from src.data_processor.knowledge_graph.hirag.base import BaseKVStorage
from src.data_processor.knowledge_graph.hirag._utils import compute_args_hash
from src.utils.api_llm_requests import EmbeddingProcessor
from tqdm import tqdm


logger = setup_logger("query_HiRAG_graph")
with open("src/config/knowledge_graph/create_kg_conf.yaml", "r") as file:
    config = yaml.safe_load(file)
MODEL = config["llm_conf"]["llm_model"]
LLM_HOST = config["llm_conf"]["llm_host"]
LLM_POSTS = config["llm_conf"]["llm_ports"]
LLM_API_KEY = config["llm_conf"]["llm_api_key"]

tokenizer = tiktoken.get_encoding("cl100k_base")


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def wrap_embedding_func_with_attrs(**kwargs):
    """Wrap a function with attributes"""

    def final_decro(func) -> EmbeddingFunc:
        new_func = EmbeddingFunc(**kwargs, func=func)
        return new_func

    return final_decro


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def GLM_embedding(texts: list[str]) -> np.ndarray:
    embeddingprocessor = EmbeddingProcessor()
    final_embedding = embeddingprocessor.get_list_embedding(text_list=texts)
    return np.array(final_embedding)


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    global TOTAL_TOKEN_COST
    global TOTAL_API_CALL_COST

    port = LLM_POSTS[0]
    base_url = f"http://{LLM_HOST}:{port}/v1"

    openai_async_client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=base_url)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------
    # logging token cost
    cur_token_cost = len(tokenizer.encode(messages[0]["content"]))
    TOTAL_TOKEN_COST += cur_token_cost
    response = await openai_async_client.chat.completions.create(
        model=MODEL, messages=messages, **kwargs
    )

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


def query_HiRAG(query, working_dir):
    WORKING_DIR = working_dir

    global TOTAL_TOKEN_COST
    global TOTAL_API_CALL_COST
    TOTAL_TOKEN_COST = 0
    TOTAL_API_CALL_COST = 0

    graph_func = HiRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=False,
        embedding_func=GLM_embedding,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        enable_hierachical_mode=True,
        embedding_func_max_async=8,
        enable_naive_rag=True,
    )
    print(graph_func.query(query, param=QueryParam(mode="hi")))


if __name__ == "__main__":
    WORKING_DIR = f"src/resources/temp/knowledge_graph/hirag"
    MAX_QUERIES = 100
    TOTAL_TOKEN_COST = 0
    TOTAL_API_CALL_COST = 0

    graph_func = HiRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=False,
        embedding_func=GLM_embedding,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        enable_hierachical_mode=True,
        embedding_func_max_async=8,
        enable_naive_rag=True,
    )
    print(
        graph_func.query(
            "What's the relationship between WORLD TRADE REPORT and WTO",
            param=QueryParam(mode="hi"),
        )
    )
