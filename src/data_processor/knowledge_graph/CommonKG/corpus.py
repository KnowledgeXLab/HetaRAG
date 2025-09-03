from ahocorasick import Automaton
from src.utils.logging_utils import setup_logger
from src.utils.common_utils import is_word_boundary, custom_lower_fast


logger = setup_logger("corpus_batch")


class Corpus(object):
    def __init__(self, doc_name, source_id, corpus):
        self.doc_name = doc_name
        self.source_id = source_id
        self.corpus = corpus

    def get_match_words(self, entities: list):
        match_words = {
            "doc_name": self.doc_name,
            "source_id": self.source_id,
            "text": self.corpus,
            "match_words": [],
        }
        blacklist = ["table. ", "tab. ", "fig. ", "figure. "]
        match_words["match_words"] = self.auto_match(entities)

        return match_words

    def auto_match(self, entities, lower_case=True):
        entities = list(set(entities))  ## 去重
        match_words = set()
        A = Automaton()
        for entity in entities:
            # 中英文兼容的小写转换, 替换keyword.lower()为自定义函数custom_lower_fast(keyword)
            entity_key = custom_lower_fast(entity) if lower_case else entity
            # 检索entity，输出对应的(subject, entity)
            A.add_word(entity_key, entity)
        A.make_automaton()  # 构造自动机
        # 初始化match_raw：idx记录文本id, text_len记录文本长度，unique_count记录匹配到的entity个数
        _text = custom_lower_fast(self.corpus) if lower_case else self.corpus
        try:
            for end_index, entity in A.iter(_text):
                end_index += 1
                start_index = end_index - len(entity)
                # 如果检测到的不是单词边界，则跳过
                if is_word_boundary(_text, start_index, end_index):
                    match_words.add(entity)
        except Exception as e:
            pass
        return list(match_words)
