import yaml
from src.utils.logging_utils import setup_logger
from src.data_processor.knowledge_graph.tools.llm_processor import LLM_Processor
from src.data_processor.knowledge_graph.CommonKG.entity_relation_extractor import (
    CommonKGExtractor,
)
from src.data_processor.knowledge_graph.GraphExtraction.entity_relation_extractor import (
    GraphExtractor,
)
from src.data_processor.knowledge_graph.tools.reports2corpus import reports2jsonl


logger = setup_logger("entity_relation_extractor")


def entity_relation_extractor(mineru_dir, output_dir, method="GraphRAG"):

    ## 确定配置文件路径
    conf_path = "src/config/knowledge_graph/create_kg_conf.yaml"

    if method.lower() == "commonkg":
        extractor = CommonKGExtractor(mineru_dir, output_dir, conf_path=conf_path)
    elif method.lower() == "graphrag":
        extractor = GraphExtractor(mineru_dir, output_dir, conf_path=conf_path)
    else:
        logger.error(f"Unsupported RAG method: {method}")
        raise ValueError(f"Unsupported RAG method: {method}")

    extractor.extract()
