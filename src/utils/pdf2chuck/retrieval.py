import json
import logging
from typing import List, Tuple, Dict, Union
from rank_bm25 import BM25Okapi
import pickle
from pathlib import Path
import faiss
import ollama
from dotenv import load_dotenv
import os
import numpy as np
from src.utils.pdf2chuck.reranking import LLMReranker
from src.utils.api_llm_requests import EmbeddingProcessor

from pymilvus import Collection
from src.database.operations.milvus_operations import (
    search_challenge_data,
    milvus_connection,
)
import pandas as pd

_log = logging.getLogger(__name__)


class BM25Retriever:
    def __init__(self, bm25_db_dir: Path, documents_dir: Path):
        self.bm25_db_dir = bm25_db_dir
        self.documents_dir = documents_dir

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 3,
        return_parent_pages: bool = False,
    ) -> List[Dict]:
        document_path = None
        for path in self.documents_dir.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
                if doc["metainfo"]["company_name"] == company_name:
                    document_path = path
                    document = doc
                    break

        if document_path is None:
            raise ValueError(f"No report found with '{company_name}' company name.")

        # Load corresponding BM25 index
        bm25_path = self.bm25_db_dir / f"{document['metainfo']['sha1_name']}.pkl"
        with open(bm25_path, "rb") as f:
            bm25_index = pickle.load(f)

        # Get the document content and BM25 index
        document = document
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]

        # Get BM25 scores for the query
        tokenized_query = query.split()
        scores = bm25_index.get_scores(tokenized_query)

        actual_top_n = min(top_n, len(scores))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            :actual_top_n
        ]

        retrieval_results = []
        seen_pages = set()

        for index in top_indices:
            score = round(float(scores[index]), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])

            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": score,
                        "page": parent_page["page"],
                        "text": parent_page["text"],
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": score,
                    "page": chunk["page"],
                    "text": chunk["text"],
                }
                retrieval_results.append(result)

        return retrieval_results


class VectorRetriever:
    def __init__(self, vector_db_dir, documents_dir):
        self.vector_db_dir = vector_db_dir
        self.documents_dir = documents_dir
        self.all_dbs = self._load_dbs()
        self.embeddingprocessor = EmbeddingProcessor()

    def _load_dbs(self):
        all_dbs = []
        # Get list of JSON document paths
        all_documents_paths = list(self.documents_dir.glob("*.json"))
        if self.vector_db_dir != "":
            vector_db_files = {
                db_path.stem: db_path for db_path in self.vector_db_dir.glob("*.faiss")
            }
        else:
            vector_db_files = {}

        for document_path in all_documents_paths:
            stem = document_path.stem
            if stem not in vector_db_files and self.vector_db_dir != "":
                _log.warning(
                    f"No matching vector DB found for document {document_path.name}"
                )
                continue
            try:
                with open(document_path, "r", encoding="utf-8") as f:
                    document = json.load(f)
            except Exception as e:
                _log.error(f"Error loading JSON from {document_path.name}: {e}")
                continue

            # Validate that the document meets the expected schema
            if not (
                isinstance(document, dict)
                and "metainfo" in document
                and "content" in document
            ):
                _log.warning(
                    f"Skipping {document_path.name}: does not match the expected schema."
                )
                continue
            if self.vector_db_dir != "":
                try:
                    vector_db = faiss.read_index(str(vector_db_files[stem]))
                except Exception as e:
                    _log.error(f"Error reading vector DB for {document_path.name}: {e}")
                    continue
            else:
                vector_db = None

            report = {"name": stem, "vector_db": vector_db, "document": document}
            all_dbs.append(report)
        return all_dbs

    @staticmethod
    def get_strings_cosine_similarity(str1, str2):
        embeddingprocessor = EmbeddingProcessor()
        embedding1 = embeddingprocessor.get_embedding(model="bge-m3", prompt=str1)
        embedding2 = embeddingprocessor.get_embedding(model="bge-m3", prompt=str2)
        similarity_score = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        similarity_score = round(similarity_score, 4)
        return similarity_score

    def retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 3,
        return_parent_pages: bool = False,
    ) -> List[Tuple[str, float]]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                _log.error(f"Report '{report.get('name')}' is missing 'metainfo'!")
                raise ValueError(
                    f"Report '{report.get('name')}' is missing 'metainfo'!"
                )
            if metainfo.get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        vector_db = target_report["vector_db"]
        chunks = document["content"]["chunks"]
        pages = document["content"]["pages"]

        actual_top_n = min(top_n, len(chunks))

        embedding = self.embeddingprocessor.get_embedding(prompt=query)
        embedding_array = np.array(embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = vector_db.search(x=embedding_array, k=actual_top_n)

        retrieval_results = []
        seen_pages = set()

        for distance, index in zip(distances[0], indices[0]):
            distance = round(float(distance), 4)
            chunk = chunks[index]
            parent_page = next(page for page in pages if page["page"] == chunk["page"])
            if return_parent_pages:
                if parent_page["page"] not in seen_pages:
                    seen_pages.add(parent_page["page"])
                    result = {
                        "distance": distance,
                        "page": parent_page["page"],
                        "text": parent_page["text"],
                    }
                    retrieval_results.append(result)
            else:
                result = {
                    "distance": distance,
                    "page": chunk["page"],
                    "text": chunk["text"],
                }
                retrieval_results.append(result)

        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        pages = document["content"]["pages"]

        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {"distance": 0.5, "page": page["page"], "text": page["text"]}
            all_pages.append(result)

        return all_pages


class MilvusVectorRetriever:
    def __init__(self, collection: Collection):
        self.collection = collection
        self.embeddingprocessor = EmbeddingProcessor()

    def _load_milvus_collection(self, collection_name: str):
        milvus_connection()
        collection = Collection(collection_name)
        collection.load()
        return collection

    def milvus_retrieve_by_company_name(
        self,
        company_name: str,
        query: str,
        top_n: int = 3,
        return_parent_pages: bool = False,
        documents_dir: Path = None,
        subset_path: Path = None,
        pdf_path: Path = None,
    ) -> List[Tuple[str, float]]:

        df = pd.read_csv(subset_path)

        try:
            sha_name = df[df["company_name"] == company_name]["sha1"].values

            if len(sha_name) == 0:
                print(f"Cannot found '{company_name}'")
                sha_name = None
            else:
                sha_name = sha_name[0]
        except Exception as e:
            sha_name = None
            _log.error(f"Error finding SHA1 for company '{company_name}': {str(e)}")
            raise ValueError(f"Error:{e}") from e

        query_embedding = [self.embeddingprocessor.get_embedding(prompt=query)]

        pdf_file = str(pdf_path / f"{sha_name}.pdf")

        output_fields = ["page_number", "pdf_path", "block_content"]

        results = search_challenge_data(
            self.collection, pdf_file, query_embedding, top_n, output_fields
        )
        target_report_path = Path(documents_dir / f"{sha_name}.json")

        with open(target_report_path, "r", encoding="utf-8") as f:
            target_report = json.load(f)

        retrieval_results = []
        seen_pages = set()
        pages = target_report["content"]["pages"]

        for hits in results:
            for rank, hit in enumerate(hits):
                distance = round(float(hit.distance), 4)
                parent_page = next(
                    page
                    for page in pages
                    if page["page"] == hit.entity.get("page_number")
                )
                if return_parent_pages:
                    if parent_page["page"] not in seen_pages:
                        seen_pages.add(parent_page["page"])
                        result = {
                            "distance": distance,
                            "page": parent_page["page"],
                            "text": parent_page["text"],
                        }
                        retrieval_results.append(result)
                else:
                    result = {
                        "distance": distance,
                        "page": hit.entity.get("page_number"),
                        "text": hit.entity.get("block_content"),
                    }
                    retrieval_results.append(result)

        _log.debug(f"Final retrieval results count: {len(retrieval_results)}")
        return retrieval_results

    def retrieve_all(self, company_name: str) -> List[Dict]:
        target_report = None
        for report in self.all_dbs:
            document = report.get("document", {})
            metainfo = document.get("metainfo")
            if not metainfo:
                continue
            if metainfo.get("company_name") == company_name:
                target_report = report
                break

        if target_report is None:
            _log.error(f"No report found with '{company_name}' company name.")
            raise ValueError(f"No report found with '{company_name}' company name.")

        document = target_report["document"]
        pages = document["content"]["pages"]

        all_pages = []
        for page in sorted(pages, key=lambda p: p["page"]):
            result = {"distance": 0.5, "page": page["page"], "text": page["text"]}
            all_pages.append(result)

        return all_pages
