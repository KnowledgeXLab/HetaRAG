import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import field
import json
import os
import shutil
import logging
import numpy as np
from openai import OpenAI
import tiktoken
from tqdm import tqdm
import yaml
from openai import AsyncOpenAI, OpenAI
from src.data_processor.knowledge_graph.learnrag._cluster_utils import (
    Hierarchical_Clustering,
)
from src.utils.file_utils import write_jsonl_list
from src.data_processor.knowledge_graph.tools.llm_processor import InstanceManager
from src.data_processor.knowledge_graph.learnrag.database_utils import (
    build_vector_search,
    create_db_table_mysql,
    insert_data_to_mysql,
)
from src.utils.api_llm_requests import EmbeddingProcessor

import multiprocessing

logger = logging.getLogger(__name__)

with open("src/config/knowledge_graph/create_kg_conf.yaml", "r") as file:
    config = yaml.safe_load(file)
MODEL = config["llm_conf"]["llm_model"]
LLM_HOST = config["llm_conf"]["llm_host"]
LLM_POSTS = config["llm_conf"]["llm_ports"]
LLM_API_KEY = config["llm_conf"]["llm_api_key"]

TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def get_common_rag_res(data_path):

    entity_path = f"{data_path}/entity.jsonl"
    relation_path = f"{data_path}/relation.jsonl"

    e_dic = {}
    with open(entity_path, "r") as f:
        for xline in f:

            line = json.loads(xline)
            entity_name = str(line["entity_name"])
            description = line["description"]
            source_id = line["source_id"]
            if entity_name not in e_dic.keys():
                e_dic[entity_name] = dict(
                    entity_name=str(entity_name),
                    description=description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                e_dic[entity_name]["description"] += (
                    "|Here is another description : " + description
                )
                if e_dic[entity_name]["source_id"] != source_id:
                    e_dic[entity_name]["source_id"] += "|" + source_id

    r_dic = {}
    with open(relation_path, "r") as f:
        for xline in f:

            line = json.loads(xline)
            src_tgt = str(line["src_tgt"])
            tgt_src = str(line["tgt_src"])
            description = line["description"]
            weight = 1
            source_id = line["source_id"]
            r_dic[(src_tgt, tgt_src)] = {
                "src_tgt": str(src_tgt),
                "tgt_src": str(tgt_src),
                "description": description,
                "weight": weight,
                "source_id": source_id,
            }

    return e_dic, r_dic


def embedding(texts: list[str]) -> np.ndarray:  # vllm serve
    embeddingprocessor = EmbeddingProcessor()
    final_embedding = embeddingprocessor.get_list_embedding(text_list=texts)
    return np.array(final_embedding)


def embedding_init(entities: list[dict]) -> list[dict]:
    texts = [truncate_text(i["description"]) for i in entities]
    embeddingprocessor = EmbeddingProcessor()
    final_embedding = embeddingprocessor.get_list_embedding(text_list=texts)
    for i, entity in enumerate(entities):
        entity["vector"] = np.array(final_embedding[i])
    return entities


tokenizer = tiktoken.get_encoding("cl100k_base")


def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


def embedding_data(entity_results):
    entities = [v for k, v in entity_results.items()]
    entity_with_embeddings = []
    embeddings_batch_size = 64
    num_embeddings_batches = (
        len(entities) + embeddings_batch_size - 1
    ) // embeddings_batch_size

    batches = [
        entities[
            i
            * embeddings_batch_size : min(
                (i + 1) * embeddings_batch_size, len(entities)
            )
        ]
        for i in range(num_embeddings_batches)
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(embedding_init, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            entity_with_embeddings.extend(result)

    for i in entity_with_embeddings:
        entiy_name = i["entity_name"]
        vector = i["vector"]
        entity_results[entiy_name]["vector"] = vector
    return entity_results


def hierarchical_clustering(global_config):
    entity_results, relation_results = get_common_rag_res(global_config["data_path"])
    all_entities = embedding_data(entity_results)
    hierarchical_cluster = Hierarchical_Clustering()
    all_entities, generate_relations, community = (
        hierarchical_cluster.perform_clustering(
            global_config=global_config,
            entities=all_entities,
            relations=relation_results,
            WORKING_DIR=global_config["working_dir"],
            max_workers=global_config["max_workers"],
        )
    )
    try:
        all_entities[-1]["vector"] = embedding(all_entities[-1]["description"])
        build_vector_search(all_entities, f"{global_config['working_dir']}")
    except Exception as e:
        print(f"Error in build_vector_search: {e}")
    for layer in all_entities:
        if type(layer) != list:
            if "vector" in layer.keys():
                del layer["vector"]
            continue
        for item in layer:
            if "vector" in item.keys():
                del item["vector"]
            if len(layer) == 1:
                item["parent"] = "root"

    save_relation = [v for k, v in generate_relations.items()]
    save_community = [v for k, v in community.items()]
    write_jsonl_list(
        save_relation, f"{global_config['working_dir']}/generate_relations.json"
    )
    write_jsonl_list(save_community, f"{global_config['working_dir']}/community.json")
    create_db_table_mysql(global_config["working_dir"])
    insert_data_to_mysql(global_config["working_dir"])


def learnrag_graph_builder(data_path, working_dir):
    try:
        multiprocessing.set_start_method("spawn", force=True)  # 强制设置
    except RuntimeError:
        pass  # 已经设置过，忽略

    instanceManager = InstanceManager(
        ports=LLM_POSTS, base_url=LLM_HOST, model=MODEL, startup_delay=30
    )
    num = len(LLM_POSTS)
    global_config = {}
    global_config["max_workers"] = num * 4
    global_config["data_path"] = data_path
    global_config["working_dir"] = working_dir
    global_config["use_llm_func"] = instanceManager.generate_text
    global_config["embeddings_func"] = embedding
    global_config["special_community_report_llm_kwargs"] = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    hierarchical_clustering(global_config)

    source_file = os.path.join(data_path, "data_chunk.jsonl")
    destination_file = os.path.join(working_dir, "data_chunk.jsonl")

    try:
        shutil.copy2(source_file, destination_file)
    except Exception as e:
        print(f"copy data_chunk file {source_file} error: {e}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)  # 强制设置
    except RuntimeError:
        pass  # 已经设置过，忽略
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="ttt")
    parser.add_argument("-n", "--num", type=int, default=2)
    args = parser.parse_args()

    WORKING_DIR = args.path
    num = args.num
    instanceManager = InstanceManager(
        ports=[8001 + i for i in range(num)],
        gpus=[i for i in range(num)],
        generate_model=MODEL,
        startup_delay=30,
    )
    global_config = {}
    global_config["max_workers"] = num * 4
    global_config["working_dir"] = WORKING_DIR
    global_config["use_llm_func"] = instanceManager.generate_text
    global_config["embeddings_func"] = embedding
    global_config["special_community_report_llm_kwargs"] = field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    hierarchical_clustering(global_config)
