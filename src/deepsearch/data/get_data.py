from datasets import load_dataset

MultiHopRAG = load_dataset("yixuantt/MultiHopRAG", "MultiHopRAG")
corpus = load_dataset("yixuantt/MultiHopRAG", "corpus")

from src.utils.file_utils import write

write("src/multi_hop_agent/data/MultiHopRAG.json", MultiHopRAG["train"].to_list())
write("src/multi_hop_agent/data/corpus.json", corpus["train"].to_list())
