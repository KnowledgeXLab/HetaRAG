import yaml
import json
import os
import time
from pathlib import Path
from tqdm import tqdm
import multiprocessing

from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.logging_utils import setup_logger
from src.data_processor.knowledge_graph.tools.llm_processor import LLM_Processor
from src.data_processor.knowledge_graph.tools.triple import Triple
from src.utils.file_utils import read_jsonl_list, write_jsonl_list
from src.data_processor.knowledge_graph.CommonKG.prompt import PROMPTS
from src.data_processor.knowledge_graph.tools.reports2corpus import reports2jsonl
from src.data_processor.knowledge_graph.CommonKG.get_triple import process_single_file
from src.data_processor.knowledge_graph.tools.sort_clean_files import (
    process_all_folders,
    clean_next_layer_files,
)

import tiktoken


logger = setup_logger("CommonKG_entity_relation_extractor")


class CommonKGExtractor:
    def __init__(
        self,
        mineru_dir,
        output_dir,
        conf_path="src/config/knowledge_graph/create_kg_conf.yaml",
    ) -> None:

        self.corpus_path_list = self._get_corpus(mineru_dir)
        self.output_path = output_dir
        self.llm_processor = self._get_llm_processer(conf_path)
        self.triple_root = os.path.join(output_dir, "triple")
        self.task_conf = self._get_task_conf(conf_path)
        self.threshold = 50

    def _get_corpus(self, mineru_dir):

        # 数据预处理：根据 MinerU 处理结果，生成包含"hash_code"、"text"的 jsonl 文件
        corpus_dir = reports2jsonl(mineru_dir, self.output_path)

        return [corpus_dir]

    def _get_llm_processer(
        self, conf_path="src/config/knowledge_graph/create_kg_conf.yaml"
    ):
        ## 读取LLM配置
        with open(conf_path, "r", encoding="utf-8") as file:
            args = yaml.safe_load(file)

        # logger.info(f"args:\n{args}\n")

        llm_conf = args["llm_conf"]  ## llm参数
        llm_processer = LLM_Processor(llm_conf)
        return llm_processer

    def _get_task_conf(
        self, conf_path="src/config/knowledge_graph/create_kg_conf.yaml"
    ):
        ## 读取LLM配置
        with open(conf_path, "r", encoding="utf-8") as file:
            args = yaml.safe_load(file)

        # logger.info(f"args:\n{args}\n")

        task_conf = args["task_conf"]  ## 任务参数
        return task_conf

    def extract_triple(self):
        """
        提取三元组
        """
        # 批量处理
        success_count = 0
        for corpus_path in tqdm(self.corpus_path_list, desc="Processing files"):
            if process_single_file(
                corpus_path, self.task_conf, self.llm_processor, self.triple_root
            ):
                success_count += 1

        logger.info(
            f"Processed {success_count}/{len(self.corpus_path_list)} files successfully"
        )

        # 数据去重
        process_all_folders(self.triple_root)
        clean_next_layer_files(self.triple_root)

    def extract_desc(self, triple_path, corpus_path):
        """
        为三元组抽取描述（支持多线程加速）
        """
        start_time = time.time()
        desc_output_path = str(triple_path).replace(".jsonl", "_descriptions.jsonl")
        corpus = read_jsonl_list(corpus_path)
        corpus_dict = {item["hash_code"]: item["text"] for item in corpus}

        # 读取三元组数据
        triples = read_jsonl_list(triple_path)
        logger.info(f"Total triples to add description: {len(triples)}")

        for item in triples:
            # 为每个三元组添加上下文信息
            source_id = item["source_id"]
            item["text"] = corpus_dict.get(source_id, "")

        # 线程池配置
        max_workers = (
            self.task_conf["num_processes_infer"]
            if self.task_conf["num_processes_infer"] != -1
            else multiprocessing.cpu_count()
        )
        all_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务到线程池
            future_to_triple = {
                executor.submit(self.process_single_description, triple): triple
                for triple in triples
            }

            # 使用tqdm显示处理进度
            for future in tqdm(
                as_completed(future_to_triple),
                total=len(triples),
                desc="Extracting descriptions...",
            ):
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)

                except Exception as e:
                    logger.error(f"Error processing description: {str(e)}")
                    continue

        # 将所有结果写入输出文件
        write_jsonl_list(data=all_results, path=desc_output_path, mode="w")

        end_time = time.time()
        logger.info(
            f"Description extraction completed in {end_time - start_time} seconds"
        )

        # 统计成功抽取了描述的三元组数量
        description_count = {
            "total": len(all_results),
            "with_description": 0,
            "without_description": 0,
        }
        for r in all_results:
            if len(r["triple"].split("\t")) == 6:
                description_count["with_description"] += 1
            else:
                description_count["without_description"] += 1

        logger.info(f"Description extraction statistics: {description_count}")
        return desc_output_path

    def process_single_description(self, triple) -> str:
        """
        处理单个三元组的描述
        """
        try:
            # 构造prompt
            text = triple["text"]
            triple_str = triple["triple"]
            prompt = self.llm_processor.extract_description_prompt(text, triple_str)

            # 调用LLM
            response = self.llm_processor.infer(prompt, output_json=True)

            result = Triple.parse_description_response(triple_str, response)

            triple["triple"] = result

            return triple

        except Exception as e:
            logger.error(f"Description generation failed: {str(e)}")

            return triple

    def summarize_entity(self, entity_name, description, summary_prompt, tokenizer):
        tokens = len(tokenizer.encode(description))
        if tokens > self.threshold:
            exact_prompt = summary_prompt.format(
                entity_name=entity_name, description=description
            )
            response = self.llm_processor.generate_text(exact_prompt)
            # response = use_llm(exact_prompt)
            return entity_name, response
        return entity_name, description  # 不需要摘要则返回原始 description

    def process_triple(self, file_path):
        """
        处理三元组与描述
        """
        if not os.path.exists(self.output_path):  # 如果目录不存在，递归创建该目录
            os.makedirs(self.output_path, exist_ok=True)
        with open(file_path, "r") as f:
            entities = {}
            relations = []
            for uline in f:
                line = json.loads(uline)
                triple = line["triple"].split("\t")
                doc_name = line["doc_name"]
                source_id = line["source_id"]
                head_entity = triple[0][1:-1]
                head_description = triple[1][1:-1]
                relation = triple[2][1:-1]
                relation_description = triple[3][1:-1]
                tail_entity = triple[4][1:-1]
                tail_description = triple[5][1:-1]

                if head_entity not in entities.keys():
                    entities[head_entity] = dict(
                        entity_name=str(head_entity),
                        description=head_description,
                        source_id=source_id,
                        doc_name=doc_name,
                        degree=0,
                    )
                else:
                    entities[head_entity]["description"] += " | " + head_description
                    if entities[head_entity]["source_id"] != source_id:
                        entities[head_entity]["source_id"] += "|" + source_id
                if tail_entity not in entities.keys():
                    entities[tail_entity] = dict(
                        entity_name=str(tail_entity),
                        description=tail_description,
                        source_id=source_id,
                        degree=0,
                    )
                else:
                    entities[tail_entity]["description"] += " | " + tail_description
                    if entities[tail_entity]["source_id"] != source_id:
                        entities[tail_entity]["source_id"] += "|" + source_id
                relations.append(
                    dict(
                        src_tgt=head_entity,
                        tgt_src=tail_entity,
                        source=relation,
                        description=relation_description,
                        weight=1,
                        source_id=source_id,
                    )
                )
        write_jsonl_list(relations, f"{self.output_path}/relation.jsonl")
        res_entity = []
        tokenizer = tiktoken.get_encoding("cl100k_base")
        to_summarize = []
        summary_prompt = PROMPTS["summary_entities"]
        for k, v in entities.items():
            v["source_id"] = "|".join(set(v["source_id"].split("|")))
            description = v["description"]
            tokens = len(tokenizer.encode(description))
            if tokens > self.threshold:
                to_summarize.append((k, description))
            else:
                res_entity.append(v)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self.summarize_entity, k, desc, summary_prompt, tokenizer
                ): k
                for k, desc in to_summarize
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Summarizing descriptions",
            ):
                k, summarized_desc = future.result()
                entities[k]["description"] = summarized_desc
                res_entity.append(entities[k])

        write_jsonl_list(res_entity, f"{self.output_path}/entity.jsonl")

    def deal_duplicate_entity(self):
        working_dir = self.triple_root
        output_path = self.output_path
        relation_path = f"{working_dir}/relation.jsonl"
        relation_output_path = f"{output_path}/relation.jsonl"
        entity_path = f"{working_dir}/entity.jsonl"
        entity_output_path = f"{output_path}/entity.jsonl"

        all_entities = []
        all_relations = []
        e_dic = {}
        summary_prompt = PROMPTS["summary_entities"]
        with open(entity_path, "r") as f:
            for xline in f:
                line = json.loads(xline)
                entity_name = str(line["entity_name"]).replace('"', "")
                entity_type = line["entity_type"].replace('"', "")
                description = line["description"].replace('"', "")
                source_id = line["source_id"]
                if entity_name not in e_dic.keys():
                    e_dic[entity_name] = dict(
                        entity_name=str(entity_name),
                        entity_type=entity_type,
                        description=description,
                        source_id=source_id,
                        degree=0,
                    )
                else:
                    e_dic[entity_name]["description"] += " | " + description
                    if e_dic[entity_name]["source_id"] != source_id:
                        e_dic[entity_name]["source_id"] += "|" + source_id

        tokenizer = tiktoken.get_encoding("cl100k_base")
        to_summarize = []
        for k, v in e_dic.items():
            v["source_id"] = "|".join(set(v["source_id"].split("|")))
            description = v["description"]
            tokens = len(tokenizer.encode(description))
            if tokens > self.threshold:
                to_summarize.append((k, description))
            else:
                all_entities.append(v)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(
                    self.summarize_entity, k, desc, summary_prompt, tokenizer
                ): k
                for k, desc in to_summarize
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Summarizing descriptions",
            ):
                k, summarized_desc = future.result()
                e_dic[k]["description"] = summarized_desc
                all_entities.append(e_dic[k])

        write_jsonl_list(all_entities, entity_output_path)
        with open(relation_path, "r") as f:
            for xline in f:
                line = json.loads(xline)[0]
                src_tgt = str(line["src_id"]).replace('"', "")
                tgt_src = str(line["tgt_id"]).replace('"', "")
                description = line["description"].replace('"', "")
                weight = 1
                source_id = line["source_id"]
                all_relations.append(
                    dict(
                        src_tgt=src_tgt,
                        tgt_src=tgt_src,
                        description=description,
                        weight=weight,
                        source_id=source_id,
                    )
                )
        write_jsonl_list(all_relations, relation_output_path)

    def extract(self):

        self.extract_triple()  # 抽取三元组

        for corpus_path in tqdm(
            self.corpus_path_list, desc="Processing files for description"
        ):

            dir = Path(corpus_path).stem
            result_triple_path = f"{self.triple_root}/{dir}/new_triples_{dir}.jsonl"
            if not os.path.getsize(result_triple_path) > 0:
                logger.warning(
                    f"No triples found in {result_triple_path}, skip extracting descriptions"
                )
            else:
                # desc_output_path = self.extract_desc(result_triple_path, corpus_path)
                # self.process_triple(desc_output_path)
                self.deal_duplicate_entity()
