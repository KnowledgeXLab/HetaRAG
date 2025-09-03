from src.data_processor.knowledge_graph.graph_builder import graph_builder


if __name__ == "__main__":

    # 实体关系三元组等数据构建hirag，并存入working_dir

    # 选择合适的实体关系提取方法
    data_path = "src/resources/temp/knowledge_graph/graphrag"  # or "src/resources/temp/knowledge_graph/commonkg"
    working_dir = "src/resources/temp/knowledge_graph/hirag"
    graph_builder(data_path, working_dir, method="hirag")
