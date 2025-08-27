# HetaRAG: Hybrid Deep Retrieval-Augmented Generation across Heterogeneous Data Stores

[![Custom badge](https://img.shields.io/badge/Paper-Arxiv-b31b1b?logo=arxiv&logoColor=white?style=flat-square)](https://arxiv.org/abs/2402.03830)
[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Read the Docs](https://img.shields.io/readthedocs/heta)](https://heta.readthedocs.io/en/latest/)
[![Stars](https://img.shields.io/github/stars/KnowledgeXLab/HetaRAG?style=social)](https://github.com/KnowledgeXLab/HetaRAG/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/KnowledgeXLab/HetaRAG?style=flat-square)](https://github.com/KnowledgeXLab/HetaRAG/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/KnowledgeXLab/HetaRAG/pulls)

HetaRAG is a hybrid, deep-retrieval RAG framework that unifies multiple heterogeneous data stores—vector indices, knowledge graphs, full-text search engines, and relational databases. 

## 🌟 Highlights
- **`2025-08-30`** Codes are now release!
- **`2025-08-30`** Project quick guide, now live [here](https://heta.readthedocs.io/en/latest/)🔗!
- **`2025-08-27`** Our paper is available on [Arxiv](https://)📄!

## ✨ Features

- **Advanced Document Parsing**: Supports multiple state-of-the-art document parsing backends, including MinerU and Docling, for handling complex layouts and multi-modal content.
- **Knowledge Graph Integration**: Automatically extracts entities and relations to build a knowledge graph (HiRAG or TRAG).

- **Flexible Database Support**: Integrates with various databases for different needs:
    - **Vector Stores**: Milvus
    - **Search Engines**: Elasticsearch
    - **Graph Databases**: Neo4j
    - **Relational Databases**: MySQL

- **DeepRetrieval**: Supports multiple advanced retrieval paradigms, including **Hybrid Retrieval** (combining vector search and keyword search), **Query Rewrite** ,**Rerank** , and **DeepSearch** modules.

- **DeepWriter**: A multimodal report generation module. Generate fact-grounded, query-driven reports, with fine-grained citations, from unstructured documents.

## 🚀 Getting Started

### Database Service Configuration
Configure the following four types of database services using Docker, including

- **Elasticsearch**: For full-text search and document indexing.
- **Milvus**: For vector similarity search.
- **Neo4j**: For graph database.
- **MySQL**: For relational database.

These databases can be installed with a single command via Docker. For detailed installation instructions, please refer to the [README](./docker/README.md). 

### Prerequisites

- Python 3.10+
- Conda for environment management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-github-username/hrag.git
    cd hrag
    ```

2.  **Create a virtual environment:**
    ```bash
    # Upgrade pip and install uv
    pip install --upgrade pip
    pip install uv

    # Create and activate a virtual environment using uv
    uv venv h-rag --python=3.10
    source h-rag/bin/activate      # For Unix/macOS
    h-rag\Scripts\activate         # For Windows

    # Alternatively, you can use conda to create and activate the environment
    conda create -n h-rag python=3.10
    conda activate h-rag
    ```

3.  **Install the required dependencies:**
    ```bash
    uv pip install -e .
    ```

## 💻 Usage

Please refer to the document [**Read the Docs**](https://heta.readthedocs.io/en/latest/).

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
We utilized the following repos during development:

- [MinerU](https://github.com/opendatalab/MinerU)
- [Qwen2.5](https://github.com/QwenLM/Qwen3/tree/v2.5)
- [WebAgent](https://github.com/Alibaba-NLP/WebAgent)
- [RAG-Challenge-2](https://github.com/IlyaRice/RAG-Challenge-2)
- [enterprise-rag-challenge](https://github.com/trustbit/enterprise-rag-challenge)


## Citation
If you find our paper and codes useful, please kindly cite us via:

```bibtex
@article{zhang2025rakg,
  title={RAKG: Document-level Retrieval Augmented Knowledge Graph Construction},
  author={Zhang, Hairong and Si, Jiaheng and Yan, Guohang and Qi, Boyuan and Cai, Pinlong and Mao, Song and Wang, Ding and Shi, Botian},
  journal={arXiv preprint arXiv:2504.09823},
  year={2025}
}

```