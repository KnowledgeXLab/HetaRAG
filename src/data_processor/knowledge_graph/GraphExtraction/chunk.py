import os
import json
from src.data_processor.knowledge_graph.GraphExtraction._utils import (
    split_string_by_multi_markers,
    _handle_single_entity_extraction,
    _handle_single_relationship_extraction,
    pack_user_ass_to_openai_messages,
)
from src.data_processor.knowledge_graph.tools.llm_processor import InstanceManager
from src.utils.file_utils import read_jsonl_list, write_jsonl_list
from collections import Counter, defaultdict
from src.data_processor.knowledge_graph.GraphExtraction.prompt import PROMPTS
import asyncio
import re
import copy
import yaml


def get_chunk(chunk_file):
    doc_name = os.path.basename(chunk_file).rsplit(".", 1)[0]
    # with open(chunk_file, "r") as f:
    #         corpus=json.load(f)
    corpus = read_jsonl_list(chunk_file)
    chunks = {item["hash_code"]: item["text"] for item in corpus}
    return chunks


async def triple_extraction(chunks, use_llm_func, output_dir):

    # extract entities
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings

    already_processed = 0
    already_entities = 0
    already_relations = 0
    ordered_chunks = list(chunks.items())

    async def _process_single_content_entity(
        chunk_key_dp, use_llm_func
    ):  # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        entity_extract_prompt = PROMPTS[
            "entity_extraction"
        ]  # give 3 examples in the prompt context
        relation_extract_prompt = PROMPTS["relation_extraction"]
        continue_prompt = PROMPTS[
            "entiti_continue_extraction"
        ]  # means low quality in the last extraction
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
        context_base_entity = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entity_types=",".join(PROMPTS["META_ENTITY_TYPES"]),
        )
        entity_extract_max_gleaning = 1
        hint_prompt = entity_extract_prompt.format(
            **context_base_entity, input_text=content
        )  # fill in the parameter
        final_result = await use_llm_func(hint_prompt)  # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(
            hint_prompt, final_result
        )  # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result
            )  # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = (
                await use_llm_func(  # judge if we still need the next iteration
                    if_loop_prompt, history_messages=history
                )
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(  # split entities from result --> list of entities
            final_result,
            [
                context_base_entity["record_delimiter"],
                context_base_entity["completion_delimiter"],
            ],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(  # split entity
                record, [context_base_entity["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(  # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1  # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][  # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    entity_results = await asyncio.gather(
        *[_process_single_content_entity(c, use_llm_func) for c in ordered_chunks]
    )
    print()  # clear the progress bar

    # fetch all entities from results
    all_entities = {}
    for item in entity_results:
        for k, v in item[0].items():
            value = v[0]
            all_entities[k] = v[0]
    context_entities = {
        key[0]: list(x[0].keys()) for key, x in zip(ordered_chunks, entity_results)
    }
    already_processed = 0

    async def _process_single_content_relation(
        chunk_key_dp, use_llm_func
    ):  # for each chunk, run the func
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        content = chunk_key_dp[1]
        entity_extract_prompt = PROMPTS[
            "entity_extraction"
        ]  # give 3 examples in the prompt context
        relation_extract_prompt = PROMPTS["relation_extraction"]
        continue_prompt = PROMPTS[
            "entiti_continue_extraction"
        ]  # means low quality in the last extraction
        if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]
        entities = context_entities[chunk_key]
        context_base_relation = dict(
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
            record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
            completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
            entities=",".join(entities),
        )
        entity_extract_max_gleaning = 1
        hint_prompt = relation_extract_prompt.format(
            **context_base_relation, input_text=content
        )  # fill in the parameter
        final_result = await use_llm_func(hint_prompt)  # feed into LLM with the prompt

        history = pack_user_ass_to_openai_messages(
            hint_prompt, final_result
        )  # set as history
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(
                continue_prompt, glean_result
            )  # add to history
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = (
                await use_llm_func(  # judge if we still need the next iteration
                    if_loop_prompt, history_messages=history
                )
            )
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(  # split entities from result --> list of entities
            final_result,
            [
                context_base_relation["record_delimiter"],
                context_base_relation["completion_delimiter"],
            ],
        )
        # resolve the entities
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(  # split entity
                record, [context_base_relation["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(  # get the name, type, desc, source_id of entity--> dict
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1  # already processed chunks
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][  # for visualization
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed}({already_processed*100//len(ordered_chunks)}%) chunks,  {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=True,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    relation_results = await asyncio.gather(
        *[_process_single_content_relation(c, use_llm_func) for c in ordered_chunks]
    )
    print()
    all_relations = {}
    for item in relation_results:
        for k, v in item[1].items():
            all_relations[k] = v
    save_entity = []
    save_relation = []
    for k, v in copy.deepcopy(all_entities).items():
        #     del v['embedding']
        save_entity.append(v)
    for k, v in copy.deepcopy(all_relations).items():
        save_relation.append(v)
    write_jsonl_list(save_entity, f"{output_dir}/entity.jsonl")
    write_jsonl_list(save_relation, f"{output_dir}/relation.jsonl")


def triple_extract(chunk_file, output_dir):
    with open("src/config/knowledge_graph/create_kg_conf.yaml", "r") as file:
        config = yaml.safe_load(file)
    MODEL = config["llm_conf"]["llm_model"]
    LLM_HOST = config["llm_conf"]["llm_host"]
    LLM_POSTS = config["llm_conf"]["llm_ports"]
    LLM_API_KEY = config["llm_conf"]["llm_api_key"]
    instanceManager = InstanceManager(
        ports=LLM_POSTS, base_url=LLM_HOST, model=MODEL, startup_delay=30
    )
    use_llm = instanceManager.generate_text_asy
    chunks = get_chunk(chunk_file)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(triple_extraction(chunks, use_llm, output_dir))


if __name__ == "__main__":
    output_dir = "src/resources/temp/knowledge_graph/graphrag"
    chunk_file = "src/resources/temp/knowledge_graph/data/data_chunk.jsonl"
    triple_extract(chunk_file, output_dir)
