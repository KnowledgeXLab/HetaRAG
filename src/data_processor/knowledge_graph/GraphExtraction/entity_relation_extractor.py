import yaml
import json
import os
import asyncio
import tiktoken
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.logging_utils import setup_logger

logger = setup_logger("GraphExtractor_entity_relation_extractor")

from src.utils.file_utils import read_jsonl_list, write_jsonl_list
from src.data_processor.knowledge_graph.tools.reports2corpus import reports2jsonl
from src.data_processor.knowledge_graph.tools.llm_processor import InstanceManager
from src.data_processor.knowledge_graph.GraphExtraction.chunk import triple_extraction
from src.data_processor.knowledge_graph.GraphExtraction.prompt import PROMPTS


class GraphExtractor:
    def __init__(
        self,
        mineru_dir,
        output_dir,
        conf_path="src/config/knowledge_graph/create_kg_conf.yaml",
    ) -> None:

        self.chunks = self._get_chunks(mineru_dir)
        self.output_path = output_dir
        self.instanceManager = self._get_instanceManager(conf_path)
        self.triple_root = os.path.join(output_dir, "triple")
        self.task_conf = self._get_task_conf(conf_path)
        self.threshold = 50

    def _get_chunks(self, mineru_dir):

        # 数据预处理：根据 MinerU 处理结果，生成包含"hash_code"、"text"的 jsonl 文件
        corpus_dir = reports2jsonl(mineru_dir, self.output_path)

        if os.path.isfile(corpus_dir):
            corpus = read_jsonl_list(corpus_dir)
            chunks = {item["hash_code"]: item["text"] for item in corpus}
            return chunks
        else:
            raise ValueError(f"Invalid input path: {corpus_dir}")

    def _get_instanceManager(
        self, conf_path="src/config/knowledge_graph/create_kg_conf.yaml"
    ):
        ## 读取LLM配置
        with open(conf_path, "r") as file:
            config = yaml.safe_load(file)
        MODEL = config["llm_conf"]["llm_model"]
        LLM_HOST = config["llm_conf"]["llm_host"]
        LLM_POSTS = config["llm_conf"]["llm_ports"]
        instanceManager = InstanceManager(
            ports=LLM_POSTS, base_url=LLM_HOST, model=MODEL, startup_delay=30
        )
        return instanceManager

    def _get_task_conf(
        self, conf_path="src/config/knowledge_graph/create_kg_conf.yaml"
    ):
        ## 读取LLM配置
        with open(conf_path, "r", encoding="utf-8") as file:
            args = yaml.safe_load(file)
        task_conf = args["task_conf"]  ## 任务参数
        return task_conf

    def extract_triple(self):

        use_llm = self.instanceManager.generate_text_asy
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            triple_extraction(self.chunks, use_llm, self.triple_root)
        )

    def summarize_entity(self, entity_name, description, summary_prompt, tokenizer):
        tokens = len(tokenizer.encode(description))
        if tokens > self.threshold:
            exact_prompt = summary_prompt.format(
                entity_name=entity_name, description=description
            )
            use_llm = self.instanceManager.generate_text
            response = use_llm(exact_prompt)
            return entity_name, response
        return entity_name, description  # 不需要摘要则返回原始 description

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

        self.extract_triple()

        self.deal_duplicate_entity()
