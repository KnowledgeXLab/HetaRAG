import os
import json
import pickle
from typing import List, Union
from pathlib import Path
from tqdm import tqdm

from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt
import ollama
import concurrent.futures
from src.utils.api_llm_requests import EmbeddingProcessor


class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """Create a BM25 index from a list of text chunks."""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """Process all reports and save individual BM25 indices.

        Args:
            all_reports_dir (Path): Directory containing the JSON report files
            output_dir (Path): Directory where to save the BM25 indices
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # Load the report
            with open(report_path, "r", encoding="utf-8") as f:
                report_data = json.load(f)

            # Extract text chunks and create BM25 index
            text_chunks = [chunk["text"] for chunk in report_data["content"]["chunks"]]
            bm25_index = self.create_bm25_index(text_chunks)

            # Save BM25 index
            sha1_name = report_data["metainfo"]["sha1_name"]
            output_file = output_dir / f"{sha1_name}.pkl"
            with open(output_file, "wb") as f:
                pickle.dump(bm25_index, f)

        print(f"Processed {len(all_report_paths)} reports")


class VectorDBIngestor:
    def __init__(self):
        self.embeddingprocessor = EmbeddingProcessor()

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(
        self, text: Union[str, List[str]], model: str = "bge-m3"
    ) -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        embeddings = []
        for chunk in text_chunks:
            result = self.embeddingprocessor.get_embedding(prompt=chunk)
            embeddings.append(result)  # 直接使用返回的嵌入向量
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index

    def _process_report(self, report: dict):
        text_chunks = [chunk["text"] for chunk in report["content"]["chunks"]]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # for report_path in tqdm(all_report_paths, desc="Processing reports"):
        #     with open(report_path, 'r', encoding='utf-8') as file:
        #         report_data = json.load(file)
        #     index = self._process_report(report_data)
        #     sha1_name = report_data["metainfo"]["sha1_name"]
        #     faiss_file_path = output_dir / f"{sha1_name}.faiss"
        #     faiss.write_index(index, str(faiss_file_path))

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)
            sha1_name = report_data["metainfo"]["sha1_name"]
            faiss_file_path = output_dir / f"{sha1_name}.faiss"
            if faiss_file_path.exists():
                continue  # 已存在则跳过
            index = self._process_report(report_data)
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")


import re


class VectorIngestor:
    def __init__(self):
        self.embeddingprocessor = EmbeddingProcessor()

    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(
        self, text: Union[str, List[str]], model: str = "bge-m3"
    ) -> List[float]:
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")

        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        embeddings = []
        for chunk in text_chunks:
            result = self.embeddingprocessor.get_embedding(prompt=chunk)
            embeddings.append(result)  # 直接使用返回的嵌入向量
        return embeddings

    def _process_report(self, report: dict, pdf_path: Path):
        data = []
        # text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        # embeddings = self._get_embeddings(text_chunks)
        # i=0
        for chunk in report["content"]["chunks"]:
            chunk_dict = dict()
            chunk_dict["pdf_path"] = str(
                pdf_path / f'{report["metainfo"]["sha1_name"]}.pdf'
            )
            chunk_dict["num_pages"] = report["metainfo"]["pages_amount"]
            chunk_dict["page_number"] = chunk["page"]
            chunk_dict["page_height"] = None
            chunk_dict["page_width"] = None
            chunk_dict["num_blocks"] = None
            chunk_dict["block_type"] = "text"
            chunk_dict["block_content"] = chunk["text"]
            chunk_dict["block_summary"] = ""
            chunk_dict["block_embedding"] = self._get_embeddings(chunk["text"])[0]
            chunk_dict["image_path"] = ""
            chunk_dict["image_caption"] = ""
            chunk_dict["image_footer"] = ""
            chunk_dict["block_bbox"] = []
            chunk_dict["block_id"] = chunk["id"]
            chunk_dict["document_title"] = ""
            chunk_dict["section_title"] = ""
            # i=i+1
            data.append(chunk_dict)

        # print(data[0])

        return data

    def _process_single_report(self, report_path, pdf_path):
        with open(report_path, "r", encoding="utf-8") as file:
            report_data = json.load(file)
        return self._process_report(report_data, pdf_path)

    def process_reports_to_pkl(
        self,
        all_reports_dir: Path,
        output_dir: Path,
        pdf_path: Path,
        parallel_threads=10,
    ):
        all_report_paths = list(all_reports_dir.glob("*.json"))
        data_pkl = []
        parallel_threads = parallel_threads
        total_reports = len(all_report_paths)

        if parallel_threads <= 1:
            # 单线程处理
            for report_path in tqdm(
                all_report_paths, desc="Processing reports (single thread)"
            ):
                data = self._process_single_report(report_path, pdf_path)
                data_pkl.extend(data)
        else:
            # 多线程处理
            print(f"Parallel processing with {parallel_threads} threads")
            with tqdm(
                total=total_reports, desc="Processing reports (parallel)"
            ) as pbar:
                for i in range(0, total_reports, parallel_threads):
                    batch = all_report_paths[i : i + parallel_threads]

                    # 处理批处理
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=parallel_threads
                    ) as executor:
                        # 使用map保持顺序
                        batch_results = list(
                            executor.map(
                                self._process_single_report,
                                batch,
                                [pdf_path] * len(batch),
                            )
                        )

                    # 合并结果
                    for result in batch_results:
                        data_pkl.extend(result)

                    # 更新进度条
                    pbar.update(len(batch))

        # 最终保存结果
        if output_dir.suffix == ".pkl":
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(output_dir, "wb") as f:
                    pickle.dump(data_pkl, f)
                print(f"Processed {len(all_report_paths)} reports save to {output_dir}")
            except Exception as e:
                print(f"Save to PKL Error: {e}")
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            pkl_file_path = output_dir / "docling.pkl"

            with open(pkl_file_path, "wb") as f:
                pickle.dump(data_pkl, f)
            print(f"Processed {len(all_report_paths)} reports save to {pkl_file_path}")
