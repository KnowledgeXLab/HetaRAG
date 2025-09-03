import time
import numpy as np
import tiktoken
import yaml
import os

from src.data_processor.knowledge_graph.tools.llm_processor import InstanceManager
from src.utils.api_llm_requests import EmbeddingProcessor
from src.data_processor.knowledge_graph.learnrag.database_utils import (
    search_vector_search,
    find_tree_root,
    search_nodes_link,
    search_community,
    get_text_units,
)
from src.data_processor.knowledge_graph.learnrag.prompt import GRAPH_FIELD_SEP, PROMPTS
from itertools import combinations
from src.utils.logging_utils import setup_logger

logger = setup_logger("query_HiRAG_graph")
with open("src/config/knowledge_graph/create_kg_conf.yaml", "r") as file:
    config = yaml.safe_load(file)
MODEL = config["llm_conf"]["llm_model"]
LLM_HOST = config["llm_conf"]["llm_host"]
LLM_POSTS = config["llm_conf"]["llm_ports"]
LLM_API_KEY = config["llm_conf"]["llm_api_key"]

TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def embedding(texts: list[str]) -> np.ndarray:
    embeddingprocessor = EmbeddingProcessor()
    final_embedding = embeddingprocessor.get_list_embedding(text_list=texts)
    return np.array(final_embedding)


tokenizer = tiktoken.get_encoding("cl100k_base")


def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text


def get_reasoning_chain(global_config, entities_set):
    maybe_edges = list(combinations(entities_set, 2))
    reasoning_path = []
    reasoning_path_information = []
    db_name = global_config["working_dir"].split("/")[-1]
    information_record = []
    for edge in maybe_edges:
        a_path = []
        b_path = []
        node1 = edge[0]
        node2 = edge[1]
        node1_tree = find_tree_root(db_name, node1)
        node2_tree = find_tree_root(db_name, node2)

        # if node1_tree[1]!=node2_tree[1] :
        #     print("debug")
        for index, (i, j) in enumerate(zip(node1_tree, node2_tree)):
            if i == j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i != j:
                a_path.append(i)
                b_path.append(j)

        reasoning_path.append(
            a_path + [b_path[len(b_path) - 1 - i] for i in range(len(b_path))]
        )
        a_path = list(set(a_path))
        b_path = list(set(b_path))
        for maybe_edge in list(combinations(a_path + b_path, 2)):
            if maybe_edge[0] == maybe_edge[1]:
                continue
            information = search_nodes_link(
                maybe_edge[0], maybe_edge[1], global_config["working_dir"]
            )
            if information == None:
                continue
            information_record.append(information)
            # reasoning_path_information.append([maybe_edge[0],maybe_edge[1],information[2]])
            reasoning_path_information.append(
                [maybe_edge[0], maybe_edge[1], information["description"]]
            )
    # columns=['src_tgt','tgt_src','path_description']
    # reasoning_path_information_description="\t\t".join(columns)+"\n"
    temp_relations_information = list(
        set([information[2] for information in reasoning_path_information])
    )
    reasoning_path_information_description = "\n".join(temp_relations_information)
    return reasoning_path, reasoning_path_information_description


def get_entity_description(global_config, entities_set, mode=0):

    columns = ["entity_name", "parent", "description"]
    entity_descriptions = "\t\t".join(columns) + "\n"
    entity_descriptions += "\n".join(
        [
            information[0] + "\t\t" + information[1] + "\t\t" + information[2]
            for information in entities_set
        ]
    )

    return entity_descriptions


def get_aggregation_description(global_config, reasoning_path, if_findings=False):

    aggregation_results = []

    communities = set(
        [community for each_path in reasoning_path for community in each_path]
    )
    for community in communities:
        temp = search_community(community, global_config["working_dir"])
        if temp == "":
            continue
        aggregation_results.append(temp)
    if if_findings:
        columns = ["entity_name", "entity_description", "findings"]
        aggregation_descriptions = "\t\t".join(columns) + "\n"
        # aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1])+"\t\t"+information[2] for information in aggregation_results])
        aggregation_descriptions += "\n".join(
            [
                information["entity_name"]
                + "\t\t"
                + str(information["entity_description"])
                + "\t\t"
                + information["findings"]
                for information in aggregation_results
            ]
        )
    else:
        columns = ["entity_name", "entity_description"]
        aggregation_descriptions = "\t\t".join(columns) + "\n"
        # aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1]) for information in aggregation_results])
        aggregation_descriptions += "\n".join(
            [
                information["entity_name"]
                + "\t\t"
                + str(information["entity_description"])
                for information in aggregation_results
            ]
        )
    return aggregation_descriptions, communities


def query_graph(global_config, db, query):
    use_llm_func: callable = global_config["use_llm_func"]
    embedding: callable = global_config["embeddings_func"]
    b = time.time()
    level_mode = global_config["level_mode"]
    topk = global_config["topk"]
    chunks_file = global_config["chunks_file"]
    entity_results = search_vector_search(
        global_config["working_dir"], embedding(query), topk=topk, level_mode=level_mode
    )
    v = time.time()
    res_entity = [i[0] for i in entity_results]
    chunks = [i[-1] for i in entity_results]
    entity_descriptions = get_entity_description(global_config, entity_results)
    reasoning_path, reasoning_path_information_description = get_reasoning_chain(
        global_config, res_entity
    )
    # reasoning_path,reasoning_path_information_description=get_path_chain(global_config,res_entity)
    aggregation_descriptions, aggregation = get_aggregation_description(
        global_config, reasoning_path
    )
    # chunks=search_chunks(global_config['working_dir'],aggregation)
    text_units = get_text_units(global_config["working_dir"], chunks, chunks_file, k=5)
    describe = f"""
    entity_information:
    {entity_descriptions}
    aggregation_entity_information:
    {aggregation_descriptions}
    reasoning_path_information:
    {reasoning_path_information_description}
    text_units:
    {text_units}
    """
    e = time.time()

    # print(describe)
    sys_prompt = PROMPTS["rag_response"].format(context_data=describe)
    response = use_llm_func(query, system_prompt=sys_prompt)
    g = time.time()
    print(f"embedding time: {v-b:.2f}s")
    print(f"query time: {e-v:.2f}s")

    print(f"response time: {g-e:.2f}s")
    return describe, response


def query_LearnRAG(query, working_dir):
    from src.database.db_connection import mysql_connnection

    db = mysql_connnection()
    global_config = {}
    WORKING_DIR = working_dir
    instanceManager = InstanceManager(
        ports=LLM_POSTS, base_url=LLM_HOST, model=MODEL, startup_delay=30
    )

    topk = 10
    global_config["chunks_file"] = os.path.join(working_dir, "data_chunk.jsonl")
    global_config["use_llm_func"] = instanceManager.generate_text
    global_config["embeddings_func"] = embedding
    global_config["working_dir"] = WORKING_DIR
    global_config["topk"] = topk
    global_config["level_mode"] = 1

    ref, response = query_graph(global_config, db, query)
    print(ref)
    print("#" * 20)
    print(response)
    # beginning=time.time()
    # for i in range(10):
    #     print(query_graph(global_config,db,query))
    # end=time.time()
    # print(f"total time: {end-beginning:.2f}s")
    db.close()
