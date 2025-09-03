from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import field
import json
import logging
import math
import numbers
import random
import re
import numpy as np
import tiktoken
import umap
import copy
import asyncio
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from itertools import combinations
from src.data_processor.knowledge_graph.learnrag._utils import (
    split_string_by_multi_markers,
    clean_str,
    is_float_regex,
)
from src.data_processor.knowledge_graph.learnrag.prompt import GRAPH_FIELD_SEP, PROMPTS
from src.utils.file_utils import write_jsonl_list

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("cluster")
ENCODER = None


def check_test(entities):
    e_l = []
    max_len = len(entities)
    for layer in entities:
        temp_e = []
        if type(layer) != list:
            temp_e.append(layer["entity_name"])
            e_l.append(temp_e)
            continue
        for item in layer:
            temp_e.append(item["entity_name"])
        e_l.append(temp_e)

    for index, layer in enumerate(entities):
        if type(layer) != list or index == max_len - 1:
            break
        for item in layer:
            if item["parent"] not in e_l[index + 1]:
                print(item["entity_name"], item["parent"])


def extract_first_complete_json(s: str):
    """Extract the first complete JSON object from the string using a stack to track braces."""
    stack = []
    first_json_start = None

    for i, char in enumerate(s):
        if char == "{":
            stack.append(i)
            if first_json_start is None:
                first_json_start = i
        elif char == "}":
            if stack:
                start = stack.pop()
                if not stack:
                    first_json_str = s[first_json_start : i + 1]
                    try:
                        # Attempt to parse the JSON string
                        return json.loads(first_json_str.replace("\n", ""))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"JSON decoding failed: {e}. Attempted string: {first_json_str[:50]}..."
                        )
                        return None
                    finally:
                        first_json_start = None
    logger.warning("No complete JSON object found in the input string.")
    return None


def extract_json_from_cluster(s: str):
    import re

    s = s.replace("*", "")
    entity_name = re.search(r"Aggregate Entity Name:\s*(.+)", s).group(1).strip()
    entity_description = (
        re.search(r"Aggregate Entity Description:\s*(.+?)\n\nFindings:", s, re.DOTALL)
        .group(1)
        .strip()
    )

    # 提取 findings
    pattern = (
        r"<summary_(\d+)>:\s*(.*?)\s*<explanation_\1>:\s*(.*?)(?=\n<summary_\d+>:|\Z)"
    )
    matches = re.findall(pattern, s, re.DOTALL)

    findings = []
    for _, summary, explanation in matches:
        findings.append(
            {
                "summary": summary.strip().replace("\n", " "),
                "explanation": explanation.strip().replace("\n", " "),
            }
        )

    # 构造最终 JSON
    result = {
        "entity_name": entity_name,
        "entity_description": entity_description,
        "findings": findings,
    }

    # 输出 JSON 字符串（可选择写入文件）
    return result


def parse_value(value: str):
    """Convert a string value to its appropriate type (int, float, bool, None, or keep as string). Work as a more broad 'eval()'"""
    value = value.strip()

    if value == "null":
        return None
    elif value == "true":
        return True
    elif value == "false":
        return False
    else:
        # Try to convert to int or float
        try:
            if "." in value:  # If there's a dot, it might be a float
                return float(value)
            else:
                return int(value)
        except ValueError:
            # If conversion fails, return the value as-is (likely a string)
            return value.strip('"')  # Remove surrounding quotes if they exist


def extract_values_from_json(
    json_string, keys=["reasoning", "answer", "data"], allow_no_quotes=False
):
    """Extract key values from a non-standard or malformed JSON string, handling nested objects."""
    extracted_values = {}

    # Enhanced pattern to match both quoted and unquoted values, as well as nested objects
    regex_pattern = r'(?P<key>"?\w+"?)\s*:\s*(?P<value>{[^}]*}|".*?"|[^,}]+)'

    for match in re.finditer(regex_pattern, json_string, re.DOTALL):
        key = match.group("key").strip('"')  # Strip quotes from key
        value = match.group("value").strip()

        # If the value is another nested JSON (starts with '{' and ends with '}'), recursively parse it
        if value.startswith("{") and value.endswith("}"):
            extracted_values[key] = extract_values_from_json(value)
        else:
            # Parse the value into the appropriate type (int, float, bool, etc.)
            extracted_values[key] = parse_value(value)

    if not extracted_values:
        logger.warning("No values could be extracted from the string.")

    return extracted_values


def convert_response_to_json(response: str) -> dict:
    """Convert response string to JSON, with error handling and fallback to non-standard JSON extraction."""
    prediction_json = extract_first_complete_json(response)

    if prediction_json is None:
        logger.info("Attempting to extract values from a non-standard JSON string...")
        prediction_json = extract_values_from_json(response, allow_no_quotes=True)

    if not prediction_json:
        logger.error("Unable to extract meaningful data from the response.")
    else:
        logger.info("JSON data successfully extracted.")

    return prediction_json


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.encoding_for_model(model_name)
    tokens = ENCODER.encode(content)
    return tokens


# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(encode_string_by_tiktoken(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int = 15,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def fit_gaussian_mixture(n_components, embeddings, random_state):
    gm = GaussianMixture(
        n_components=n_components,
        random_state=random_state,
        n_init=5,
        init_params="k-means++",
    )
    gm.fit(embeddings)
    return gm.bic(embeddings)


def get_optimal_clusters(embeddings, max_clusters=50, random_state=0, rel_tol=1e-3):
    max_clusters = min(len(embeddings), max_clusters)
    n_clusters = np.arange(1, max_clusters)
    bics = []
    prev_bic = float("inf")
    for n in tqdm(n_clusters):
        bic = fit_gaussian_mixture(n, embeddings, random_state)
        # print(bic)
        bics.append(bic)
        # early stop
        if (abs(prev_bic - bic) / abs(prev_bic)) < rel_tol:
            break
        prev_bic = bic
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(
    embeddings: np.ndarray,
    threshold: float,
    random_state: int = 0,
    cluster_size: int = 20,
):
    n_clusters = max(len(embeddings) // cluster_size, get_optimal_clusters(embeddings))
    gm = GaussianMixture(
        n_components=n_clusters,
        random_state=random_state,
        n_init=5,
        init_params="k-means++",
    )
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)  # [num, cluster_num]
    # labels = [np.where(prob > threshold)[0] for prob in probs]
    labels = [[np.argmax(prob)] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
    verbose: bool = False,
    cluster_size: int = 20,
) -> List[np.ndarray]:
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )
    global_clusters, n_global_clusters = GMM_cluster(  # (num, 2)
        reduced_embeddings_global, threshold, cluster_size=cluster_size
    )
    if len(global_clusters) != len(embeddings):
        print("debug")
    # if verbose:
    #     logging.info(f"Global Clusters: {n_global_clusters}")

    # all_clusters = [[] for _ in range(len(embeddings))]
    # embedding_to_index = {tuple(embedding): idx for idx, embedding in enumerate(embeddings)}
    # for i in tqdm(range(n_global_clusters)):
    #     global_cluster_embeddings_ = embeddings[
    #         np.array([i in gc for gc in global_clusters])
    #     ]  #找到当前簇的embedding
    #     if verbose:
    #         logging.info(
    #             f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
    #         )
    #     if len(global_cluster_embeddings_) == 0:
    #         continue

    #     # embedding indices #反向取idx
    #     indices = [
    #         embedding_to_index[tuple(embedding)]
    #         for embedding in global_cluster_embeddings_
    #     ]

    #     # update
    #     for idx in indices:
    #         all_clusters[idx].append(i)

    # all_clusters = [np.array(cluster) for cluster in all_clusters]

    # if verbose:
    #     logging.info(f"Total Clusters: {len(n_global_clusters)}")
    return global_clusters


def enclose_string_with_quotes(content: Any) -> str:
    """Enclose a string with quotes"""
    if isinstance(content, numbers.Number):
        return str(content)
    content = str(content)
    content = content.strip().strip("'").strip('"')
    return f'"{content}"'


def list_of_list_to_csv(data: list[list]):
    return "\n".join(
        [
            ",\t".join([f"{enclose_string_with_quotes(data_dd)}" for data_dd in data_d])
            for data_d in data
        ]
    )


def get_direct_relations(set1, set2, relations):
    results = {
        k: v
        for k, v in relations.items()
        if (k[0] in set1 and k[1] in set2) or (k[0] in set2 and k[1] in set1)
    }
    return results


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


def _pack_single_community_describe(
    entitys,
    relations,
    max_token_size: int = 12000,
    global_config: dict = {},
) -> str:

    node_fields = ["id", "entity", "type", "description", "degree"]
    edge_fields = ["id", "source", "target", "description", "rank"]
    nodes_list_data = [
        [
            i,
            entity.get("entity_name"),
            entity.get("entity_type", "UNKNOWN"),
            entity.get("description", "UNKNOWN"),
            entity.get("degree", 1),
        ]
        for i, entity in enumerate(entitys)
    ]
    nodes_list_data = sorted(nodes_list_data, key=lambda x: x[-1], reverse=True)
    nodes_may_truncate_list_data = truncate_list_by_token_size(
        nodes_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )
    edges_list_data = [
        [
            i,
            relation.get("src_tgt"),
            relation.get("tgt_src"),
            relation.get("description", "UNKNOWN"),
        ]
        for i, relation in enumerate(relations.values())
    ]
    edges_list_data = sorted(edges_list_data, key=lambda x: x[-1], reverse=True)
    edges_may_truncate_list_data = truncate_list_by_token_size(
        edges_list_data, key=lambda x: x[3], max_token_size=max_token_size // 2
    )

    truncated = len(nodes_list_data) > len(nodes_may_truncate_list_data) or len(
        edges_list_data
    ) > len(edges_may_truncate_list_data)

    nodes_describe = list_of_list_to_csv([node_fields] + nodes_may_truncate_list_data)
    edges_describe = list_of_list_to_csv([edge_fields] + edges_may_truncate_list_data)
    return f"""
-----Entities-----
```csv
{nodes_describe}
```
-----Relationships-----
```csv
{edges_describe}
```"""


def process_cluster(
    use_llm_func,
    embeddings_func,
    clusters,
    label,
    nodes,
    community_report_prompt,
    relations,
    generate_relations,
    layer,
    temp_clusters_nodes,
):
    indices = [i for i, cluster in enumerate(clusters) if label in cluster]
    # Add the corresponding nodes to the node_clusters list
    cluster_nodes = [nodes[i] for i in indices]

    # Base case: if the cluster only has one node, do not attempt to recluster it
    logging.info(f"[Label{str(int(label))} Size: {len(cluster_nodes)}]")
    if len(cluster_nodes) == 1:
        cluster_nodes[0]["parent"] = cluster_nodes[0]["entity_name"]
        return {"community_data": None, "temp_node": cluster_nodes[0], "index": indices}
    name_set = [node["entity_name"] for node in cluster_nodes]
    cluster_intern_relation = {
        **get_direct_relations(name_set, name_set, relations),
        **get_direct_relations(name_set, name_set, generate_relations),
    }
    describe = _pack_single_community_describe(cluster_nodes, cluster_intern_relation)
    hint_prompt = community_report_prompt.format(input_text=describe)
    response = use_llm_func(hint_prompt)
    data = convert_response_to_json(response)
    data["level"] = layer
    data["children"] = [n["entity_name"] for n in cluster_nodes]
    data["source_id"] = "|".join(set([n["source_id"] for n in cluster_nodes]))

    temp_node = {
        "entity_name": data["entity_name"],
        "description": data["entity_description"],
        "source_id": data["source_id"],
        "entity_type": "aggregate entity",
        "degree": 1,
        "vector": embeddings_func(data["entity_description"]),
    }

    return {"community_data": data, "temp_node": temp_node, "index": indices}


def process_relation(
    use_llm_func,
    community_report,
    maybe_edge,
    relations,
    generate_relations,
    cluster_cluster_relation_prompt,
    layer,
    tokenizer,
    max_depth,
):
    cluster1_nodes = community_report[maybe_edge[0]]["children"]
    cluster2_nodes = community_report[maybe_edge[1]]["children"]

    threshold = min(len(cluster1_nodes) * 0.2, len(cluster2_nodes) * 0.2)
    # threshold=1
    exists_relation = {
        **get_direct_relations(cluster1_nodes, cluster2_nodes, relations),
        **get_direct_relations(cluster1_nodes, cluster2_nodes, generate_relations),
    }
    if exists_relation == {}:
        return None

    cluster1_description = community_report[maybe_edge[0]]["findings"]
    cluster2_description = community_report[maybe_edge[1]]["findings"]
    relation_infromation = [
        f"relationship<|>{v['src_tgt']}<|>{v['tgt_src']}<|>{v['description']} "
        for k, v in exists_relation.items()
    ]
    temp_relations = {}
    tokens = len(tokenizer.encode("\n".join(relation_infromation)))
    gene_tokens = (layer + 1) * 40
    allowed_tokens = (max_depth - layer) * 40 * 2
    # allowed_tokens=1
    if tokens > allowed_tokens:
        print(
            f"{tokens}大于{allowed_tokens}，进行llm生成\n{maybe_edge[0]}和{maybe_edge[1]} in processing"
        )
        exact_prompt = cluster_cluster_relation_prompt.format(
            entity_a=maybe_edge[0],
            entity_b=maybe_edge[1],
            entity_a_description=cluster1_description,
            entity_b_description=cluster2_description,
            relation_information="\n".join(relation_infromation),
            tokens=gene_tokens,
        )

        response = use_llm_func(exact_prompt)
        temp_relations[maybe_edge] = {
            "src_tgt": maybe_edge[0],
            "tgt_src": maybe_edge[1],
            "description": response,
            "weight": 1,
            "level": layer + 1,
        }
    else:
        print(f"{tokens}小于{allowed_tokens}，不进行llm生成")
        temp_relations[maybe_edge] = {
            "src_tgt": maybe_edge[0],
            "tgt_src": maybe_edge[1],
            "description": "\n".join(relation_infromation),
            "weight": 1,
            "level": layer + 1,
        }
    return temp_relations


class Hierarchical_Clustering(ClusteringAlgorithm):
    def perform_clustering(
        self,
        global_config: dict,
        entities: dict,
        relations: dict,
        max_length_in_cluster: int = 60000,
        tokenizer=tiktoken.get_encoding("cl100k_base"),
        reduction_dimension: int = 2,
        cluster_threshold: float = 0.1,
        verbose: bool = False,
        threshold: float = 0.98,  # 0.99
        thredshold_change_rate: float = 0.05,
        WORKING_DIR: str = None,
        max_workers: int = 8,
        cluster_size: int = 20,
    ) -> List[dict]:
        use_llm_func: callable = global_config["use_llm_func"]
        embeddings_func: callable = global_config["embeddings_func"]
        # Get the embeddings from the nodes
        nodes = list(entities.values())
        embeddings = np.array([x["vector"] for x in nodes])
        generate_relations = {}
        max_workers = global_config["max_workers"]
        community_report = {}
        all_nodes = []
        all_nodes.append(nodes)
        community_report_prompt = PROMPTS["aggregate_entities"]
        cluster_cluster_relation_prompt = PROMPTS["cluster_cluster_relation"]
        max_depth = round(math.log(len(nodes), cluster_size)) + 1
        for layer in range(max_depth):
            logging.info(f"############ Layer[{layer}] Clustering ############")
            # Perform the clustering
            if len(nodes) <= 2:
                print("当前簇数小于2，停止聚类")
                break
            clusters = perform_clustering(
                embeddings,
                dim=reduction_dimension,
                threshold=cluster_threshold,
                cluster_size=cluster_size,
            )
            temp_clusters_nodes = []
            # Initialize an empty list to store the clusters of nodes
            # Iterate over each unique label in the clusters
            unique_clusters = np.unique(np.concatenate(clusters))
            logging.info(
                f"[Clustered Label Num: {len(unique_clusters)} / Last Layer Total Entity Num: {len(nodes)}]"
            )
            # calculate the number of nodes belong to each cluster
            # cluster_sizes = Counter(np.concatenate(clusters))
            # # calculate cluster sparsity
            # cluster_sparsity = 1 - sum([x * (x - 1) for x in cluster_sizes.values()])/(len(nodes) * (len(nodes) - 1))
            # cluster_sparsity_change_rate = (abs(cluster_sparsity - pre_cluster_sparsity) / pre_cluster_sparsity)
            # pre_cluster_sparsity = cluster_sparsity
            # logging.info(f"[Cluster Sparsity: {round(cluster_sparsity, 4) * 100}%]")

            # if cluster_sparsity_change_rate <= thredshold_change_rate:
            #     logging.info(f"[Stop Clustering at Layer{layer} with Cluster Sparsity Change Rate {round(cluster_sparsity_change_rate, 4) * 100}%]")
            #     break
            # summarize
            if len(unique_clusters) <= 4:
                print(f"当前簇数小于5，停止聚类")
                break
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_cluster,
                        use_llm_func,
                        embeddings_func,
                        clusters,
                        label,
                        nodes,
                        community_report_prompt,
                        relations,
                        generate_relations,
                        layer,
                        temp_clusters_nodes,
                    )
                    for label in unique_clusters
                ]
                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    temp_clusters_nodes.append(result["temp_node"])
                    for index in result["index"]:
                        nodes[index]["parent"] = result["temp_node"]["entity_name"]
                    if result["community_data"] is not None:
                        title = result["community_data"]["entity_name"]
                        community_report[title] = result["community_data"]

            temp_cluster_relation = [
                i["entity_name"]
                for i in temp_clusters_nodes
                if i["entity_name"] in community_report.keys()
            ]
            temp_relations = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        process_relation,
                        use_llm_func,
                        community_report,
                        maybe_edge,
                        relations,
                        generate_relations,
                        cluster_cluster_relation_prompt,
                        layer,
                        tokenizer,
                        max_depth,
                    )
                    for maybe_edge in list(combinations(temp_cluster_relation, 2))
                ]

                for future in tqdm(as_completed(futures), total=len(futures)):
                    result = future.result()
                    if result != None:
                        for k, v in result.items():
                            temp_relations[k] = v
            for k, v in temp_relations.items():
                generate_relations[k] = v

            # update nodes to be clustered in the next layer
            nodes = copy.deepcopy(
                [x for x in temp_clusters_nodes if "entity_name" in x.keys()]
            )
            # filter the duplicate entities
            seen = set()
            unique_nodes = []
            for item in nodes:
                entity_name = item["entity_name"]
                if entity_name not in seen:
                    seen.add(entity_name)
                    unique_nodes.append(item)
            nodes = unique_nodes
            for index, i in enumerate(
                unique_nodes
            ):  # 再进行embedding时发现，有个元素的vector不是np而是list
                vec = i["vector"]
                if type(vec) == list or vec.shape != (1, 1024):
                    print(index)
                    unique_nodes[index]["vector"] = np.array(vec).reshape((1, 1024))

            embeddings = np.array(
                [x["vector"] for x in unique_nodes]
            ).squeeze()  # 为下一轮迭代做准备
            all_nodes.append(nodes)
            save_entities = copy.deepcopy(all_nodes)
            for layer in save_entities:
                if type(layer) != list:
                    if "vector" in layer.keys():
                        del layer["vector"]
                    continue
                for item in layer:
                    if "vector" in item.keys():
                        del item["vector"]
                    if len(layer) == 1:
                        item["parent"] = "root"
            # check_test(all_entities)
            write_jsonl_list(
                save_entities, f"{WORKING_DIR}/all_entities.json", mode="w+"
            )
            # check_test(all_nodes)
            # stop if the number of deduplicated cluster is too small
            # if len(embeddings) <= 2:
            #     logging.info(f"[Stop Clustering at Layer{layer} with entity num {len(embeddings)}]")
            #     break
        if len(all_nodes[-1]) != 1:
            temp_node = {}
            cluster_nodes = all_nodes[-1]
            cluster_intern_relation = get_direct_relations(
                cluster_nodes, cluster_nodes, generate_relations
            )  # 默认为顶层，从下层找关系就是在generate_relations中
            describe = _pack_single_community_describe(
                cluster_nodes, cluster_intern_relation
            )
            hint_prompt = community_report_prompt.format(input_text=describe)
            # response = use_llm_func(hint_prompt,**llm_extra_kwargs)
            response = use_llm_func(hint_prompt)
            data = convert_response_to_json(response)
            data["level"] = layer
            data["children"] = [i["entity_name"] for i in cluster_nodes]
            data["source_id"] = "|".join(set([i["source_id"] for i in cluster_nodes]))
            community_report[data["entity_name"]] = data

            temp_node["entity_name"] = data["entity_name"]
            temp_node["description"] = data["entity_description"]
            temp_node["source_id"] = "|".join(set(data["source_id"].split("|")))
            temp_node["entity_type"] = "community"
            temp_node["degree"] = 1
            temp_node["parent"] = "root"
            for i in cluster_nodes:
                i["parent"] = data["entity_name"]

            all_nodes.append(temp_node)
        save_entities = copy.deepcopy(all_nodes)
        for layer in save_entities:
            if type(layer) != list:
                if "vector" in layer.keys():
                    del layer["vector"]
                continue
            for item in layer:
                if "vector" in item.keys():
                    del item["vector"]
                if len(layer) == 1:
                    item["parent"] = "root"
        # check_test(all_entities)
        write_jsonl_list(save_entities, f"{WORKING_DIR}/all_entities.json", mode="w+")
        return all_nodes, generate_relations, community_report
