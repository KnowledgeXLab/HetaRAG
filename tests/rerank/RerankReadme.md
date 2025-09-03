# 使用指南

可以选择直接从 Huggingface 下载对应的 Rerank 模型，或者使用 VLLM 部署已有的 Rerank 模型

## 1.通过 Huggingface 下载对应的 Rerank 模型
### Rerank模型下载

#### 列出可用模型
```bash
python src/rerank/model_setup.py --list_models
```

#### 下载特定模型（如果不存在）
```bash
python src/model_setup.py --model_name bge-reranker-large
# 下载好的模型文件会默认保存在 src/rerank/model_path 
```

#### 强制重新下载模型
```bash
python src/rerank/model_setup.py --model_name bge-reranker-large --force_download
```

### 推荐模型
- bge-reranker-large
- bge-reranker-v2-m3
- bge-reranker-v2-gemma
- ms-marco-TinyBERT-L-2-v2
- ms-marco-MiniLM-L-12-v2


### 使用

#### 命令行执行
```bash
python tests/rerank/test_rerank_huggingface.py \
    --root_path src/resources/data \
    --parent_document_retrieval \
    --top_n_retrieval 14 \
    --rerank_model bge-reranker-large \
    --rerank_model_path src/rerank/model_path \
    --rerank_sample_size 36
```

#### 参数解释
- root_path 项目根目录
- parent_document_retrieval 是否使用父文档检索（chunk所在page均设为检索结果），默认使用
- top_n_retrieval 保留的top_n
- rerank_model 参考推荐模型
- rerank_model_path 模型路径，默认src/rerank/model_path
- rerank_sample_size 重排序前采样数，需大于top_n_retrieval



## 2.使用 VLLM 部署 Rerank 模型
### Rerank 模型部署


- 客户需要安装vllm
```bash
# 避免环境冲突，可新建环境使用vllm
conda create -n h-tag-vllm python=3.10
conda activate h-tag-vllm
pip install vllm -U
```

- 将对应的模型权重文件（如：`qwen2.5_hot_mus_sft\`）放在 `src/rerank` 路径下面

- 使用以下脚本启动部署服务端。具体的模型路径与创建的端口均可在 `src/rerank/deploy.sh` 中更改
```bash
bash src/rerank/deploy.sh
```


### 使用

#### 命令行执行
```bash
python tests/rerank/test_rerank_VLLM.py \
    --root_path src/resources/data \
    --parent_document_retrieval \
    --top_n_retrieval 14 \
    --rerank_api http://127.0.0.1:8005/v1 \
    --vector_db milvus
```

#### 参数解释
- root_path 项目根目录
- parent_document_retrieval 是否使用父文档检索（chunk所在page均设为检索结果），默认使用
- top_n_retrieval 保留的top_n
- rerank_api `src/rerank/deploy.sh` 中创建的端口，默认 http://127.0.0.1:8005/v1 
- rerank_sample_size 重排序前采样数，需大于top_n_retrieval
