from src.data_processor.knowledge_graph.query_graph import query_graph


if __name__ == "__main__":

    query = "Which leadership positions changed at Datalogic in the reporting period?"
    working_dir = "src/resources/temp/knowledge_graph/hirag"
    query_graph(query, working_dir, method="hirag")
