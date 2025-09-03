快速开始指南
============

本指南将帮助您快速上手 HRAG 系统，从安装到运行第一个示例。

.. raw:: html

    <div class="quick-start">
        <h3>⚡ 5分钟快速体验</h3>
        <p>如果您想立即体验 HRAG 的功能，可以按照以下步骤操作：</p>
    </div>

环境准备
--------

系统要求
^^^^^^^^^

* **Python**: 3.10 或更高版本
* **内存**: 至少 8GB RAM
* **存储**: 至少 10GB 可用空间
* **GPU**: 推荐 NVIDIA GPU (用于加速处理)

安装步骤
^^^^^^^^^

1. **创建虚拟环境**

   .. code-block:: bash

      # 安装 uv
      pip install --upgrade pip
      pip install uv

      # 使用 uv 创建 h-rag 环境并激活环境
      uv venv h-rag --python=3.10
      source h-rag/bin/activate  # On Unix/macOS
      h-rag\Scripts\activate     # On Windows

      # 或者使用 conda 创建 h-rag 环境并激活环境
      conda create -n h-rag python=3.10
      conda activate h-rag



2. **安装依赖**

   .. code-block:: bash

      uv pip install -e .

3. **验证安装**

   .. code-block:: bash

      python -c "import torch; print('PyTorch version:', torch.__version__)"
      python -c "import transformers; print('Transformers version:', transformers.__version__)"


demo示例
--------

1. **运行示例**

   .. code-block:: bash

      # mineru解析示例
      python tests/data_parser/test_mineru_pdf_parser.py
      
      # milvus数据库示例
      python tests/database/test_milvus_operations.py
      
      # 运行其他更多示例
      ...





