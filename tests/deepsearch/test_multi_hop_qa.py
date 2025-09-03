from src.deepsearch.multi_hop_qa import multi_hop_qa
from src.deepsearch.qa_evaluate import qa_evaluate

# the save file for generated answer
save_file = "tests/multi_hop_agent/multi_hop_data/answer.json"

# the file for question and ground truth
doc_file = "src/multi_hop_agent/data/MultiHopRAG.json"

if __name__ == "__main__":
    multi_hop_qa(doc_file, save_file)
    qa_evaluate(doc_file, save_file)
