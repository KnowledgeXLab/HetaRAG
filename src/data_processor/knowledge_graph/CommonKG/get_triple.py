from tqdm import tqdm
import os
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil


from src.utils.logging_utils import setup_logger
from src.utils.file_utils import (
    read_jsonl_list,
    write_jsonl_list,
    read_txt_line,
    write_txt_line,
)
from src.data_processor.knowledge_graph.tools.triple import Triple
from src.data_processor.knowledge_graph.CommonKG.corpus import Corpus

logger = setup_logger("triple_extractor")


def process_llm_batch(item_batch, llm_processer, ref_kg_path):
    """
    处理单个批次的LLM请求
    """
    doc_name, source_id, text, match_words = (
        item_batch["doc_name"],
        item_batch["source_id"],
        item_batch["text"],
        item_batch["match_words"],
    )

    # 生成大模型推理输入文件
    prompt = llm_processer.extract_triple_prompt(text, match_words, ref_kg_path)
    # 大模型推理
    response = llm_processer.infer(prompt)
    # 推理结果后处理（三元组过滤）
    infer_triples, head_entities, tail_entities = Triple.get_triple(
        match_words, response
    )
    # 再次调用大模型对实体进行验证
    verify_entities = llm_processer.entity_evaluate(tail_entities)

    return {
        "doc_name": doc_name,
        "source_id": source_id,
        "infer_triples": infer_triples,
        "head_entities": head_entities,
        "verify_entities": verify_entities,
    }


def process_single_file(corpus_path, task_conf, llm_processer, output_dir="output"):
    """处理单个文件的三元组抽取"""
    start_time = time.time()
    pedia_entity_path = task_conf["pedia_entity_path"]  # 头实体路径

    try:
        # 动态生成输出路径
        file_name = Path(corpus_path).stem
        output_subdir = Path(output_dir) / file_name
        if output_subdir.exists():
            logger.info(f"Target files: {file_name} already exists, overwrite\n")
        else:
            output_subdir.mkdir(parents=True, exist_ok=True)

        # 移动chunk文件
        shutil.copy(corpus_path, output_subdir)

        # 初始化输出文件路径
        result_triple_path = output_subdir / f"new_triples_{file_name}.jsonl"
        next_layer_entities_path = (
            output_subdir / f"next_layer_entities_{file_name}.txt"
        )
        all_entities_path = output_subdir / f"all_entities_{file_name}.txt"
        match_words_path = (
            output_subdir / f"match_words_{file_name}.jsonl"
        )  ## 匹配结果路径

        # 加载到头实体路径进行处理，假设有第0层，那么头实体直接next_layer_entities_path来匹配
        head_entities = read_txt_line(pedia_entity_path)
        next_layer_entities = set([item.strip() for item in head_entities])
        write_txt_line(next_layer_entities_path, next_layer_entities, mode="w")
        logger.info(f"Initialize next_layer_entities num: {len(next_layer_entities)}")

        # 初始化实体和三元组文件
        write_jsonl_list(data="", path=result_triple_path, mode="w")
        write_txt_line(data="", path=all_entities_path, mode="w")

        # 读取语料文件
        corpusfiles = read_jsonl_list(corpus_path)
        logger.info(f"corpus paragraph num: {len(corpusfiles)}")

        for iter in range(task_conf["level_num"]):
            logger.info(
                f"Processing {file_name} | Iteration {iter+1}/{task_conf['level_num']}"
            )
            layer_head_cnt, layer_tail_cnt, layer_triple_cnt = 0, 0, 0

            logger.info(f"[num_iteration]: {iter+1} ---------------------\n")
            logger.info("[corpus matching]-----------------------------------\n")

            # 检查文件是否存在，如果存在则删除
            if os.path.exists(match_words_path):
                os.remove(match_words_path)

            next_layer_entities = read_txt_line(next_layer_entities_path)
            next_layer_entities = [entity.strip("\n") for entity in next_layer_entities]

            source_id = "hash_code"
            text_key = "text"

            tasks_for_matching = [
                (item, file_name, next_layer_entities, source_id, text_key)
                for item in corpusfiles
            ]

            num_processes = (
                task_conf["num_processes_match"]
                if task_conf["num_processes_match"] != -1
                else multiprocessing.cpu_count()
            )
            logger.info(
                f"Starting AC matching for {len(tasks_for_matching)} paragraphs in {file_name} (Iter {iter+1}) using {num_processes} processes."
            )
            match_start_time = time.time()

            all_match_words = []
            if tasks_for_matching:  # Ensure there are tasks to process
                with multiprocessing.Pool(processes=num_processes) as pool:
                    results_iterator = pool.imap(
                        _process_paragraph_for_matching, tasks_for_matching
                    )
                    all_match_words = list(
                        tqdm(
                            results_iterator,
                            total=len(tasks_for_matching),
                            desc=f"AC Matching: {file_name} | Iter {iter+1}/{task_conf['level_num']}",
                        )
                    )

            logger.info(
                f"[corpus match finished for {file_name} | Iteration {iter+1}]-----------------------------------\n"
            )
            match_end_time = time.time()
            logger.info(
                f"Match time taken: {match_end_time - match_start_time} seconds"
            )
            # 每一层的头实体匹配结果写入文件（下一层覆盖上一层）
            write_jsonl_list(data=all_match_words, path=match_words_path, mode="w")
            logger.info(f"Save current match result to: {match_words_path}")

            logger.info("[LLM response]-----------------------------------\n")
            ref_kg_path = task_conf["ref_kg_path"]

            # 初始化下一层实体文件
            if os.path.exists(next_layer_entities_path):
                os.remove(next_layer_entities_path)

                # 初始化计数器和结果集合
            layer_head_cnt, layer_tail_cnt, layer_triple_cnt = 0, 0, 0
            current_all_triple = set()
            current_all_entity = set()

            # 读取现有的三元组和实体
            current_all_triple_item = read_jsonl_list(result_triple_path)
            current_all_triple = set(
                [item["triple"].lower() for item in current_all_triple_item]
            )
            current_all_entity = set(
                [item.strip().lower() for item in read_txt_line(all_entities_path)]
            )

            # 使用线程池并行处理LLM请求
            max_workers = (
                task_conf["num_processes_infer"]
                if task_conf["num_processes_infer"] != -1
                else multiprocessing.cpu_count()
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务到线程池
                future_to_item = {
                    executor.submit(
                        process_llm_batch, item, llm_processer, ref_kg_path
                    ): item
                    for item in all_match_words
                }

                # 使用tqdm显示处理进度
                for future in tqdm(
                    as_completed(future_to_item),
                    total=len(all_match_words),
                    desc=f"LLM Processing: {file_name} | Iter {iter+1}/{task_conf['level_num']}",
                ):
                    try:
                        result = future.result()

                        # 处理三元组
                        new_triples_item = []
                        if result["infer_triples"] is not None:
                            for triple in result["infer_triples"]:
                                if triple not in current_all_triple:
                                    layer_triple_cnt += 1
                                    current_all_triple.add(triple)
                                    triple_json = Triple.triple_json_format(
                                        triple, result["doc_name"], result["source_id"]
                                    )
                                    new_triples_item.append(triple_json)

                            if new_triples_item:
                                write_jsonl_list(
                                    data=new_triples_item,
                                    path=result_triple_path,
                                    mode="a",
                                )
                                logger.info(
                                    f"Add {len(new_triples_item)} triples to: {result_triple_path}"
                                )

                            # 处理头实体
                        if result["head_entities"] is not None:
                            head_entities_cnt = 0
                            for entity in result["head_entities"]:
                                if entity not in current_all_entity:
                                    current_all_entity.add(entity)
                                    head_entities_cnt += 1
                                    layer_head_cnt += 1

                                # 更新完整的实体清单, 覆写
                            if head_entities_cnt > 0:
                                write_txt_line(
                                    data=current_all_entity,
                                    path=all_entities_path,
                                    mode="w",
                                )
                                logger.info(
                                    f"Add {head_entities_cnt} entities to: {all_entities_path}"
                                )

                            # 处理验证实体
                        if result["verify_entities"] is not None:
                            tmp_next_layer_entities = set()
                            for entity in result["verify_entities"]:
                                entity_lower = entity.strip().lower()
                                if entity_lower not in current_all_entity:
                                    current_all_entity.add(entity_lower)
                                    tmp_next_layer_entities.add(entity_lower)
                                    layer_tail_cnt += 1

                            if tmp_next_layer_entities:
                                write_txt_line(
                                    data=tmp_next_layer_entities,
                                    path=next_layer_entities_path,
                                    mode="a",
                                )
                                logger.info(
                                    f"Save {len(tmp_next_layer_entities)} entities to: {next_layer_entities_path}"
                                )

                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        continue

            logger.info(
                f"layer: {iter+1}, add head: {layer_head_cnt}, tail: {layer_tail_cnt}, triple: {layer_triple_cnt}"
            )

        end_time = time.time()
        logger.info(f"extract_triple total time taken: {end_time - start_time} seconds")
        return True

    except Exception as e:
        logger.error(f"Error processing {corpus_path}: {str(e)}")
        return False


# Helper function for multiprocessing Aho-Corasick matching
def _process_paragraph_for_matching(args_tuple):
    """
    Worker function to process a single paragraph for entity matching.
    Unpacks arguments, creates a Corpus object, and performs matching.
    """
    item, file_name, local_next_layer_entities, source_id, text_key = args_tuple
    corpus = Corpus(
        doc_name=file_name, source_id=item[source_id], corpus=item[text_key]
    )
    match_words = corpus.get_match_words(local_next_layer_entities)
    return match_words


# def triple_extract(files_to_process, output_dir):

#     ## 读取LLM配置
#     conf_path = "src/config/knowledge_graph/create_kg_conf.yaml"
#     with open(conf_path, "r", encoding="utf-8") as file:
#         args = yaml.safe_load(file)

#     # logger.info(f"args:\n{args}\n")

#     task_conf = args["task_conf"]  ## 任务参数
#     llm_conf = args["llm_conf"]  ## llm参数
#     llm_processer = LLM_Processor(llm_conf)

#     output_dir = output_dir

#     # 批量处理
#     success_count = 0
#     for corpus_path in tqdm(files_to_process, desc="Processing files"):
#         if process_single_file(corpus_path, task_conf, llm_processer, output_dir):
#             success_count += 1

#     logger.info(f"Processed {success_count}/{len(files_to_process)} files successfully")

#     # 数据去重
#     process_all_folders(output_dir)

#     clean_next_layer_files(output_dir)
