from src.utils.logging_utils import setup_logger

import logging

logging.basicConfig(level=logging.WARNING)
logger = setup_logger("query_graph")


from src.data_processor.knowledge_graph.hirag.query_HiRAG_graph import query_HiRAG
from src.data_processor.knowledge_graph.learnrag.query_LearnRAG_graph import (
    query_LearnRAG,
)


def query_graph(query, working_dir, method="hirag"):
    if method.lower() == "hirag":
        query_HiRAG(query, working_dir)
    elif method.lower() == "learnrag":
        query_LearnRAG(query, working_dir)
    else:
        logger.error(f"Unsupported RAG method: {method}")
        raise ValueError(f"Unsupported RAG method: {method}")
