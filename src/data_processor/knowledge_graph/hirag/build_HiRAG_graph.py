import os
import sys
import json
import time

from src.utils.logging_utils import setup_logger
from src.data_processor.knowledge_graph.hirag import HiRAG, QueryParam
import os
import logging
import numpy as np
import tiktoken
import yaml
from openai import AsyncOpenAI, OpenAI
from dataclasses import dataclass
from src.data_processor.knowledge_graph.hirag.base import BaseKVStorage
from src.data_processor.knowledge_graph.hirag._utils import compute_args_hash
from src.utils.api_llm_requests import EmbeddingProcessor

logging.basicConfig(level=logging.WARNING)
logger = setup_logger("build_HiRAG_graph")


with open("src/config/knowledge_graph/create_kg_conf.yaml", "r") as file:
    config = yaml.safe_load(file)
MODEL = config["llm_conf"]["llm_model"]
LLM_HOST = config["llm_conf"]["llm_host"]
LLM_POSTS = config["llm_conf"]["llm_ports"]
LLM_API_KEY = config["llm_conf"]["llm_api_key"]


TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

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
    # model_name = "bge-m3:latest"
    # client = OpenAI(
    #     api_key='ollama',
    #     base_url="http://127.0.0.1:11434/v1"
    # )
    # embedding = client.embeddings.create(
    #     input=texts,
    #     model=model_name,
    # )
    # final_embedding = [d.embedding for d in embedding.data]
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
    retry_time = 3
    try:
        # logging token cost
        cur_token_cost = len(tokenizer.encode(messages[0]["content"]))
        TOTAL_TOKEN_COST += cur_token_cost
        # logging api call cost
        TOTAL_API_CALL_COST += 1
        # request
        response = await openai_async_client.chat.completions.create(
            model=MODEL, messages=messages, **kwargs
        )
    except Exception as e:
        print(f"Retry for Error: {e}")
        retry_time -= 1
        response = ""

    if response == "":
        return response

    # Cache the response if having-------------------
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
        )
    # -----------------------------------------------------
    return response.choices[0].message.content


def get_common_rag_res(data_path):
    entity_path = os.path.join(data_path, "entity.jsonl")
    relation_path = os.path.join(data_path, "relation.jsonl")
    entity_results = []
    relation_results = []
    # i=0
    with open(entity_path, "r") as f:
        for xline in f:
            if not xline.strip():
                continue  # 跳过空行
            e_dic = {}
            line = json.loads(xline)
            if line is None:
                continue
            entity_name = line["entity_name"]
            entity_type = line["entity_type"]
            description = line["description"]
            source_id = line["source_id"]
            e_dic[entity_name] = [
                dict(
                    entity_name=str(entity_name),
                    entity_type=entity_type,
                    description=description,
                    source_id=source_id,
                )
            ]
            entity_results.append((e_dic, {}))

    with open(relation_path, "r") as f:
        for xline in f:
            if not xline.strip():
                continue  # 跳过空行
            r_dic = {}
            line = json.loads(xline)
            if line is None:
                continue
            line = json.loads(xline)
            src_tgt = '"' + str(line["src_tgt"]) + '"'
            tgt_src = '"' + str(line["tgt_src"]) + '"'
            description = '"' + line["description"] + '"'
            weight = 1
            source_id = line["source_id"]
            r_dic[(src_tgt, tgt_src)] = [
                {
                    "src_tgt": str(src_tgt),
                    "tgt_src": str(tgt_src),
                    "description": description,
                    "weight": weight,
                    "source_id": source_id,
                }
            ]
            relation_results.append(({}, r_dic))
            # i+=1
            # if i==1000:
            #     break
    return entity_results, relation_results


def hirag_graph_builder(data_path, working_dir):
    graph_func = HiRAG(
        working_dir=working_dir,
        enable_llm_cache=True,
        embedding_func=GLM_embedding,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        enable_hierachical_mode=True,
        embedding_func_max_async=10,
        enable_naive_rag=True,
    )

    entity_results, relation_results = get_common_rag_res(data_path)
    alldata_path = os.path.join(data_path, "data_chunk.jsonl")
    with open(alldata_path, mode="r") as f:
        x = []
        for line in f:
            x.append(json.loads(line))
        graph_func.insert(x, entity_results, relation_results)
        logging.info(f"[Total token cost: {TOTAL_TOKEN_COST}]")
        logging.info(f"[Total api call cost: {TOTAL_API_CALL_COST}]")


if __name__ == "__main__":
    file_path = f"data/wto_test.jsonl"
    WORKING_DIR = f"wto"
    data_path = "data"
    graph_func = HiRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        embedding_func=GLM_embedding,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        enable_hierachical_mode=True,
        embedding_func_max_async=10,
        enable_naive_rag=True,
    )

    entity_results, relation_results = get_common_rag_res(data_path)
    with open(file_path, mode="r") as f:
        x = []
        for line in f:
            x.append(json.loads(line))
        graph_func.insert(x, entity_results, relation_results)
        logging.info(f"[Total token cost: {TOTAL_TOKEN_COST}]")
        logging.info(f"[Total api call cost: {TOTAL_API_CALL_COST}]")
