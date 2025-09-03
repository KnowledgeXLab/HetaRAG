import os
from typing import List, Dict, Any
from flashrank import Ranker, RerankRequest
from FlagEmbedding import FlagAutoReranker, FlagLLMReranker


def setup_models(model_name, local_model_path):
    if model_name == "ms-marco-TinyBERT-L-2-v2":
        return Ranker(
            model_name="ms-marco-TinyBERT-L-2-v2",
            cache_dir=os.path.join("../", local_model_path),
        )
    elif model_name == "rank-T5":
        return Ranker(
            model_name="rank-T5-flan", cache_dir=os.path.join("../", local_model_path)
        )
    elif model_name == "ms-marco-MiniLM-L-12-v2":
        return Ranker(
            model_name="ms-marco-MiniLM-L-12-v2",
            cache_dir=os.path.join("../", local_model_path),
        )
    elif model_name == "bge-reranker-large":
        return FlagAutoReranker.from_finetuned(
            local_model_path,
            query_max_length=256,
            passage_max_length=512,
            use_fp16=True,
            devices=["cuda:0"],
        )
    elif model_name == "bge-reranker-v2-m3":
        return FlagAutoReranker.from_finetuned(
            local_model_path,
            query_max_length=256,
            passage_max_length=512,
            use_fp16=True,
            devices=["cuda:0"],
        )
    elif model_name == "bge-reranker-v2-gemma":
        return FlagLLMReranker(
            local_model_path,
            query_max_length=256,
            passage_max_length=512,
            use_fp16=True,
            devices=["cuda:1"],
        )
    elif model_name == "bge-reranker-v2-minicpm-layerwise":
        return FlagAutoReranker.from_finetuned(
            local_model_path,
            query_max_length=256,
            passage_max_length=512,
            use_fp16=True,
            devices=["cuda:0"],
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


class ModelReranker:
    def __init__(self, model, flash_rank=False, sample_size: int = 36):
        self.model = model
        self.flash_rank = flash_rank
        self.sample_size = sample_size

    def send_message_for_reranking(
        self, query: str, documents: List[Dict[str, Any]], top_n: int = 10
    ) -> List[Dict[str, Any]]:

        # Step 1: 构造 doc_id <-> passage 对
        docs = [(str(i), doc["text"]) for i, doc in enumerate(documents)]

        # Step 2: 调用本地 rerank 方法
        if self.flash_rank:
            scored_docs = self._rerank_flash(query, docs)
        else:
            scored_docs = self._rerank_with_model(query, docs)

        # Step 3: 映射回原始文档结构
        result_map = {doc_id: score for doc_id, score in scored_docs}
        reranked = []

        for idx, doc in enumerate(documents):
            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = result_map.get(str(idx), -1e9)
            reranked.append(doc_with_score)

        # Step 4: 按相关性得分排序
        reranked.sort(key=lambda x: x["relevance_score"], reverse=True)
        return reranked[:top_n]

    def _rerank_with_model(self, query, docs):

        query_doc_pairs = [[query, passage] for _, passage in docs]
        scores = []
        scores = self.model.compute_score(query_doc_pairs)
        doc_ids = [doc[0] for doc in docs]
        scored_docs = list(zip(doc_ids, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs

    def _rerank_flash(self, query, docs):
        passages = [{"id": doc[0], "text": doc[1].strip(), "meta": {}} for doc in docs]
        rerankrequest = RerankRequest(query=query, passages=passages)
        results = self.model.rerank(rerankrequest)
        scored_docs = [(res["id"], res["score"]) for res in results]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs


from openai import OpenAI


class VLLMReranker:
    def __init__(
        self,
        api_base: str = "http://127.0.0.1:8005/v1",
        api_key: str = "EMPTY",
        sample_size: int = 36,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.models = [model.id for model in self.client.models.list().data]

        self.sample_size = sample_size
        if not self.models:
            raise RuntimeError("No models available at the provided VLLM server.")

    def send_message_for_reranking(
        self, query: str, documents: List[Dict[str, Any]], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        """
        使用远程 VLLM 模型进行 reranking。

        Args:
            query (str): 用户查询语句
            documents (List[Dict]): 文档列表，每个文档必须包含 'title' 和 'context'
            top_n (int): 返回前 N 个最相关的文档

        Returns:
            List[Dict]: reranked 后的文档列表，包含 'text' 和 'relevance_score'
        """
        # Step 1: 构造 doc 格式
        docs = [f"Context:{doc['text']}" for doc in documents]
        retrieved_docs = [f"[{i + 1}]{doc}\n" for i, doc in enumerate(docs)]

        sys_prompt = "Please carefully read the following content and select the document number that can answer the question."

        messages = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": f"query: {query}\n\nContext:\n{''.join(retrieved_docs)}",
            },
        ]

        # Step 2: 调用远程模型 API 获取 rerank 结果
        resp = self.client.chat.completions.create(
            model=self.models[0], messages=messages, temperature=0
        )
        response = resp.choices[0].message.content

        # Step 3: 解析模型返回的索引
        try:
            indices = [int(index.strip()) for index in response.split(",")]
        except Exception as e:
            print(f"Error parsing response: {response}. Using fallback order.")
            indices = list(range(1, len(docs) + 1))  # Fallback to original order

        # Step 4: 过滤并构造最终结果（添加 relevance_score）
        selected_docs = []
        for idx in indices:
            if 0 < idx <= len(docs):
                selected_docs.append(documents[idx - 1])

        # 伪打分（按顺序赋值），也可以根据实际需求改进
        reranked = []
        top_n = min(top_n, len(selected_docs))
        for i, doc in enumerate(selected_docs[:top_n]):
            doc_with_score = doc.copy()
            doc_with_score["relevance_score"] = float(top_n - i)  # 伪相关性得分
            reranked.append(doc_with_score)

        return reranked
