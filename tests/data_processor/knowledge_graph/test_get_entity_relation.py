from src.data_processor.knowledge_graph.entity_relation_extractor import (
    entity_relation_extractor,
)

if __name__ == "__main__":

    # 根据MinerU生成的文件得到三元组
    mineru_path = "src/resources/pdf"
    # output_path = "src/resources/temp/knowledge_graph/commonkg"
    # entity_relation_extractor(mineru_path, output_path, corpus_dir = corpus_path, method="CommonKG")

    output_path = "src/resources/temp/knowledge_graph/graphrag"
    entity_relation_extractor(mineru_path, output_path, method="graphrag")
