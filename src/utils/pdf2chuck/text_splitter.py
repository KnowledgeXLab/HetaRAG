import time
import json
from pathlib import Path
from typing import List, Dict, Optional
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from src.utils.query2vec import get_embedding_ollama
import numpy as np


class OllamaEmbeddings:
    """自定义的 Ollama 嵌入类，用于语义分块"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为嵌入向量列表"""
        embeddings = []
        for text in texts:
            embedding = get_embedding_ollama(text)
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """将单个文本转换为嵌入向量"""
        embedding = get_embedding_ollama(text)
        return embedding.tolist()


class TextSplitter:
    def __init__(
        self,
        chunk_mode: str = "base",  # 分块模式：base(基础分块), semantic(语义分块), fixed_page(页面固定长度), fixed_doc(文档固定长度)
        breakpoint_type: str = "percentile",
        breakpoint_amount: float = 90,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
    ):
        """
        初始化文本分块器

        Args:
            chunk_mode: 分块模式
                - base: 基础分块（按页面）
                - semantic: 语义分块（按页面）
                - fixed_page: 页面固定长度分块
                - fixed_doc: 文档固定长度分块
            breakpoint_type: 语义分块阈值类型 (percentile/standard_deviation/interquartile/gradient)
            breakpoint_amount: 语义分块阈值大小
            chunk_size: 分块大小
            chunk_overlap: 分块重叠度
        """
        self.chunk_mode = chunk_mode
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 基础分块器
        self.base_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # 语义分块器
        if chunk_mode == "semantic":
            self.semantic_chunker = SemanticChunker(
                embeddings=OllamaEmbeddings(),  # 使用自定义的嵌入类
                breakpoint_threshold_type=breakpoint_type,
                breakpoint_threshold_amount=breakpoint_amount,
            )

    def count_tokens(self, string: str, encoding_name="cl100k_base") -> int:
        """计算文本的token数量

        Args:
            string: 要计算token数量的文本
            encoding_name: 使用的编码器名称，默认为cl100k_base

        Returns:
            token数量
        """
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            tokens = encoding.encode(string)
            return len(tokens)
        except Exception as e:
            print(f"计算token数量时出错: {str(e)}")
            # 如果出错，返回一个估计值（每个字符算一个token）
            return len(string)

    def _split_page(self, page: Dict[str, any]) -> List[Dict[str, any]]:
        """分块入口函数

        Args:
            page: 包含页面信息的字典，必须包含 'text' 和 'page' 键

        Returns:
            分块后的文本列表，每个分块包含元数据
        """
        if not isinstance(page, dict) or "text" not in page or "page" not in page:
            raise ValueError("页面数据格式错误，必须包含 'text' 和 'page' 键")

        if self.chunk_mode == "semantic":
            return self._split_by_semantic(page["text"], page["page"])
        elif self.chunk_mode == "fixed_page":
            return self._split_by_fixed_size(page["text"], page["page"])
        else:  # base mode
            return self._split_by_base(page["text"], page["page"])

    def _split_by_semantic(self, text: str, page_num: int) -> List[Dict[str, any]]:
        """使用语义分块"""
        # 创建文档对象
        doc = Document(page_content=text, metadata={"page": page_num})

        # 使用语义分块器
        semantic_docs = self.semantic_chunker.create_documents([text])

        # 处理分块结果
        chunks_with_meta = []
        for doc in semantic_docs:
            # 如果分块太大，使用基础分块器继续分割
            if self.count_tokens(doc.page_content) > self.chunk_size * 2:
                sub_chunks = self.base_splitter.split_text(doc.page_content)
                for chunk in sub_chunks:
                    chunks_with_meta.append(
                        {
                            "page": page_num,
                            "length_tokens": self.count_tokens(chunk),
                            "text": chunk,
                            "type": "content",  # 保持与基础分块一致的类型
                            "metadata": doc.metadata,
                        }
                    )
            else:
                chunks_with_meta.append(
                    {
                        "page": page_num,
                        "length_tokens": self.count_tokens(doc.page_content),
                        "text": doc.page_content,
                        "type": "content",  # 保持与基础分块一致的类型
                        "metadata": doc.metadata,
                    }
                )

        return chunks_with_meta

    def _split_by_fixed_size(self, text: str, page_num: int) -> List[Dict[str, any]]:
        """使用严格的固定长度分块

        Args:
            text: 要分块的文本
            page_num: 页码

        Returns:
            分块后的文本列表，每个分块严格保持固定长度
        """
        # 计算文本的总token数
        total_tokens = self.count_tokens(text)

        # 如果文本长度小于chunk_size，直接返回
        if total_tokens <= self.chunk_size:
            return [
                {
                    "page": page_num,
                    "length_tokens": total_tokens,
                    "text": text,
                    "type": "content",
                }
            ]

        # 使用tiktoken进行严格的分块
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)

        # 计算需要多少个完整的分块
        num_chunks = (total_tokens - self.chunk_overlap) // (
            self.chunk_size - self.chunk_overlap
        )
        if (total_tokens - self.chunk_overlap) % (
            self.chunk_size - self.chunk_overlap
        ) > 0:
            num_chunks += 1

        chunks_with_meta = []
        for i in range(num_chunks):
            # 计算当前分块的起始和结束位置
            start = i * (self.chunk_size - self.chunk_overlap)
            end = min(start + self.chunk_size, total_tokens)

            # 获取当前分块的token
            chunk_tokens = tokens[start:end]

            # 将token转换回文本
            chunk_text = encoding.decode(chunk_tokens)

            chunks_with_meta.append(
                {
                    "page": page_num,
                    "length_tokens": len(chunk_tokens),
                    "text": chunk_text,
                    "type": "content",
                }
            )

            # 如果已经到达文本末尾，退出循环
            if end >= total_tokens:
                break

        return chunks_with_meta

    def _split_by_base(self, text: str, page_num: int) -> List[Dict[str, any]]:
        """使用基础分块"""
        chunks = self.base_splitter.split_text(text)
        return [
            {
                "page": page_num,
                "length_tokens": self.count_tokens(chunk),
                "text": chunk,
                "type": "content",
            }
            for chunk in chunks
        ]

    def _get_serialized_tables_by_page(
        self, tables: List[Dict]
    ) -> Dict[int, List[Dict]]:
        """处理序列化表格"""
        tables_by_page = {}
        for table in tables:
            if "serialized" not in table:
                continue

            page = table["page"]
            if page not in tables_by_page:
                tables_by_page[page] = []

            table_text = "\n".join(
                block["information_block"]
                for block in table["serialized"]["information_blocks"]
            )

            tables_by_page[page].append(
                {
                    "page": page,
                    "text": table_text,
                    "table_id": table["table_id"],
                    "length_tokens": self.count_tokens(table_text),
                    "type": "table",
                }
            )

        return tables_by_page

    def _split_report(
        self,
        file_content: Dict[str, any],
        serialized_tables_report_path: Optional[Path] = None,
    ) -> Dict[str, any]:
        """处理单个报告文件"""
        chunks = []
        chunk_id = 0

        # 处理序列化表格
        tables_by_page = {}
        if serialized_tables_report_path is not None:
            with open(serialized_tables_report_path, "r", encoding="utf-8") as f:
                parsed_report = json.load(f)
            tables_by_page = self._get_serialized_tables_by_page(
                parsed_report.get("tables", [])
            )

        # 如果是文档级别的固定长度分块，直接处理整个文档
        if self.chunk_mode == "fixed_doc":
            # 收集所有文本内容和页码信息
            all_text = ""
            page_positions = []  # 记录每个字符对应的页码
            current_pos = 0

            for page in file_content["content"]["pages"]:
                page_text = page["text"] + "\n"
                all_text += page_text
                # 记录这段文本对应的页码
                page_positions.extend([page["page"]] * len(page_text))

            # 对整体文本进行固定长度分块
            doc_chunks = self._split_by_fixed_size(all_text, -1)  # 使用-1表示文档级别

            for chunk in doc_chunks:
                chunk["id"] = chunk_id
                chunk_id += 1
                chunk["type"] = "doc_content"  # 标记为文档级别内容

                # 获取分块文本在原文中的位置
                chunk_start = all_text.find(chunk["text"])
                if chunk_start != -1:
                    # 获取分块开始位置的页码
                    chunk["page"] = page_positions[chunk_start]

                chunks.append(chunk)
        else:
            # 处理每一页
            for page in file_content["content"]["pages"]:
                # 分块文本
                page_chunks = self._split_page(page)
                for chunk in page_chunks:
                    chunk["id"] = chunk_id
                    chunk_id += 1
                    chunks.append(chunk)

                # 添加表格
                if tables_by_page and page["page"] in tables_by_page:
                    for table in tables_by_page[page["page"]]:
                        table["id"] = chunk_id
                        chunk_id += 1
                        chunks.append(table)

        file_content["content"]["chunks"] = chunks
        return file_content

    def split_all_reports(
        self,
        all_report_dir: Path,
        output_dir: Path,
        serialized_tables_dir: Optional[Path] = None,
    ):
        """处理所有报告文件"""
        all_report_paths = list(all_report_dir.glob("*.json"))

        for report_path in all_report_paths:
            # 处理序列化表格
            serialized_tables_path = None
            if serialized_tables_dir is not None:
                serialized_tables_path = serialized_tables_dir / report_path.name
                if not serialized_tables_path.exists():
                    print(
                        f"Warning: Could not find serialized tables report for {report_path.name}"
                    )

            # 读取和处理报告
            with open(report_path, "r", encoding="utf-8") as file:
                report_data = json.load(file)

            updated_report = self._split_report(report_data, serialized_tables_path)

            # 保存结果
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / report_path.name, "w", encoding="utf-8") as file:
                json.dump(updated_report, file, indent=2, ensure_ascii=False)

            print(f"Processed {report_path.name}")

        print(f"Split {len(all_report_paths)} files")
