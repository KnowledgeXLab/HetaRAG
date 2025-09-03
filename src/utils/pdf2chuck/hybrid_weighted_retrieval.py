import numpy as np
from pathlib import Path
from src.database.operations.milvus_operations import milvus_search
from src.database.operations.elastic_operations import search_top
from src.database.db_connection import es_connection, get_milvus_collection
from pymilvus import Collection
import json
import pandas as pd


class HybridWeightedRetriever:
    def __init__(
        self,
        collection_name: str,
        documents_dir: Path,
        es_index: str,
        alpha: float = 0.5,
        top_k: int = 5,
        return_parent_pages: bool = False,
    ):
        self.collection_name = collection_name
        self.documents_dir = documents_dir
        self.es_index = es_index
        self.alpha = alpha
        self.top_k = top_k
        self.return_parent_pages = return_parent_pages
        self.es_client = es_connection()
        self.search_top = search_top
        # 使用新的连接管理方式，避免重复连接
        self.collection = get_milvus_collection(collection_name)

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = None,
        return_parent_pages: bool = None,
    ):
        if top_n is None:
            top_n = self.top_k

        # 使用传入的参数，如果没有传入则使用实例变量
        if return_parent_pages is None:
            return_parent_pages = self.return_parent_pages

        # 1. Milvus向量检索
        milvus_results = milvus_search(self.collection_name, query, top_n, 0.0)

        # 2. ES检索
        es_results = self.search_top(query, top_n, self.es_index, self.es_client)

        # 3. 分别处理两个系统的结果
        milvus_items = []
        for r in milvus_results:
            # 根据return_parent_pages决定使用chunk还是page级别的文本
            if return_parent_pages:
                # 需要获取页面级别的文本
                page_text = self._get_page_text(r.get("pdf_path"), r.get("page_number"))
                text_content = page_text
                page_number = r.get("page_number")
            else:
                # 使用chunk级别的文本
                text_content = r["block_content"]
                page_number = r.get("page_number")

            milvus_items.append(
                {
                    "distance": r["score"],
                    "page": page_number,
                    "text": text_content,
                    "milvus_score": r["score"],
                    "es_score": 0.0,  # ES没有找到这个文档
                    "filename": r.get("pdf_path"),
                    "chunk_id": r.get("block_id"),
                }
            )

        es_items = []
        for r in es_results:
            es_score = r["meta"]["score"]
            # 根据return_parent_pages决定使用chunk还是page级别的文本
            if return_parent_pages:
                # 需要获取页面级别的文本
                page_text = self._get_page_text(r.get("filename"), r.get("page"))
                text_content = page_text
                page_number = r.get("page")
            else:
                # 使用chunk级别的文本
                text_content = r["text"]
                page_number = r.get("page")

            es_items.append(
                {
                    "distance": es_score,
                    "page": page_number,
                    "text": text_content,
                    "milvus_score": 0.0,  # Milvus没有找到这个文档
                    "es_score": es_score,
                    "filename": r.get("filename"),
                    "chunk_id": r.get("chunk_id"),
                }
            )

        # 4. 合并所有结果
        all_items = milvus_items + es_items

        # 5. 对Milvus和ES分数都进行归一化处理

        # 归一化Milvus分数
        milvus_scores = [
            item["milvus_score"] for item in all_items if item["milvus_score"] > 0
        ]
        if milvus_scores:
            max_milvus_score = max(milvus_scores)
            min_milvus_score = min(milvus_scores)
            milvus_score_range = max_milvus_score - min_milvus_score

            # 归一化Milvus分数到0-1范围
            for item in all_items:
                if item["milvus_score"] > 0:
                    if milvus_score_range > 0:
                        item["milvus_score_normalized"] = (
                            item["milvus_score"] - min_milvus_score
                        ) / milvus_score_range
                    else:
                        item["milvus_score_normalized"] = 1.0  # 如果所有分数相同，设为1
                else:
                    item["milvus_score_normalized"] = 0.0
        else:
            for item in all_items:
                item["milvus_score_normalized"] = 0.0

        # 归一化ES分数
        if es_items:
            es_scores = [item["es_score"] for item in all_items if item["es_score"] > 0]

            if es_scores:
                max_es_score = max(es_scores)
                min_es_score = min(es_scores)
                es_score_range = max_es_score - min_es_score

                # 归一化ES分数到0-1范围
                for item in all_items:
                    if item["es_score"] > 0:
                        if es_score_range > 0:
                            item["es_score_normalized"] = (
                                item["es_score"] - min_es_score
                            ) / es_score_range
                        else:
                            item["es_score_normalized"] = 1.0  # 如果所有分数相同，设为1
                    else:
                        item["es_score_normalized"] = 0.0
            else:
                for item in all_items:
                    item["es_score_normalized"] = 0.0
        else:
            for item in all_items:
                item["es_score_normalized"] = 0.0

        # 6. 计算融合分数（使用归一化后的两个分数）
        for i, item in enumerate(all_items):
            item["distance"] = float(
                self.alpha * item["milvus_score_normalized"]
                + (1 - self.alpha) * item["es_score_normalized"]
            )

        # 7. 按融合分数排序，取前top_n个
        results = sorted(all_items, key=lambda x: x["distance"], reverse=True)[:top_n]
        return results

    def _get_page_text(self, pdf_path: str, page_number: int) -> str:
        """
        根据PDF路径和页面号获取页面级别的文本

        Args:
            pdf_path: PDF文件路径
            page_number: 页面号

        Returns:
            页面级别的文本内容
        """
        try:
            # 从PDF路径提取SHA1
            if pdf_path:
                # 假设pdf_path格式为 /path/to/xxx.pdf，需要提取xxx部分
                import os

                pdf_name = os.path.basename(pdf_path)
                sha1_name = pdf_name.replace(".pdf", "")

                # 构建对应的JSON文件路径
                json_path = self.documents_dir / f"{sha1_name}.json"

                if json_path.exists():
                    with open(json_path, "r", encoding="utf-8") as f:
                        target_report = json.load(f)

                    # 查找对应页面的文本
                    pages = target_report["content"]["pages"]
                    for page in pages:
                        if page["page"] == page_number:
                            return page["text"]

                    # 如果没找到对应页面，返回空字符串
                    return ""
                else:
                    print(f"Warning: JSON file not found: {json_path}")
                    return ""
            else:
                return ""
        except Exception as e:
            print(f"Error getting page text: {e}")
            return ""
