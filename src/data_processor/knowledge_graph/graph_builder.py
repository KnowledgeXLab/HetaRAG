from src.utils.logging_utils import setup_logger
import logging

logging.basicConfig(level=logging.WARNING)
logger = setup_logger("graph_builder")


from src.data_processor.knowledge_graph.hirag.build_HiRAG_graph import (
    hirag_graph_builder,
)
from src.data_processor.knowledge_graph.learnrag.build_LearnRAG_graph import (
    learnrag_graph_builder,
)


def graph_builder(data_path, working_dir, method="hirag"):
    if method.lower() == "hirag":
        hirag_graph_builder(data_path, working_dir)
    elif method.lower() == "learnrag":
        learnrag_graph_builder(data_path, working_dir)
    else:
        logger.error(f"Unsupported RAG method: {method}")
        raise ValueError(f"Unsupported RAG method: {method}")
