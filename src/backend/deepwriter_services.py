from pymilvus import connections as milvus_connections
from fastapi import FastAPI, HTTPException, Request, Depends, Header
from pydantic import BaseModel


import argparse
from loguru import logger

from src.deepwriter.database.milvus_db import MilvusDB
from src.deepwriter.finders.doc_finder import DocFinder
from src.deepwriter.pipeline import DeepWriter
from src.config.db_config import get_deepwriter_port

logger.add("logs/deepwriter_report_generation_demo.log", mode="w")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for the report generation demo.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate reports from documents using DeepWriter"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output report"
    )
    parser.add_argument(
        "--query", type=str, required=True, help="Query to write the report for"
    )
    parser.add_argument(
        "--host",
        type=str,
        required=False,
        default="localhost",
        help="Host to connect to Milvus, default: localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        required=False,
        default=19530,
        help="Port to connect to Milvus, default: 19530",
    )
    parser.add_argument(
        "--user",
        type=str,
        required=False,
        default="",
        help="User to connect to Milvus, default: empty",
    )
    parser.add_argument(
        "--password",
        type=str,
        required=False,
        default="",
        help="Password to connect to Milvus, default: empty",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        required=False,
        default="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        help="Embedding model to use (currently only supports GME model), default: Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    )
    parser.add_argument(
        "--llm",
        type=str,
        required=False,
        default="ollama/qwen2:7b",
        help="LLM to use for report generation, default: ollama/qwen2:7b",
    )
    return parser.parse_args()


def deepwriter_report_generation(query) -> None:
    """Main function to run the report generation demo."""
    # args = parse_arguments()
    # Directly set the parameters in the code
    args = argparse.Namespace(
        output_path="output/",
        query=query,
        host="localhost",
        port=19530,
        user="",
        password="",
        embedding_model="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        base_url="http://10.6.8.115:11439",
        llm="ollama/qwen2:7b",
    )

    # 1. Initialize vector database
    milvus_db = MilvusDB(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        image_collection_name="deepwriter_image",
        text_collection_name="deepwriter_text",
        text_embedding_model=args.embedding_model,
        multimodal_embedding_model=args.embedding_model,
    )

    # 3. Initialize DocFinder
    logger.info("Setting up document finder")
    doc_finder = DocFinder(
        database=milvus_db,
        context_threshold=0.5,
        n_rerank=5,
    )

    # 4. Initialize DeepWriter
    logger.info("Initializing DeepWriter")
    deep_writer = DeepWriter(
        llm=args.llm,
        base_url=args.base_url,
        doc_finder=doc_finder,
    )

    # 5. Generate report
    logger.info(f"Generating report for query: '{args.query}'")
    report = deep_writer.generate_report(args.query, **{"param": {"ef": 10}})

    # 6. post-process report
    # report = citations.post_process_report(report, args.output_path)

    # 7. Save report
    # report_path = Path(args.output_path) / "report.md"
    # file_interface.save_markdown_report(report, report_path)

    row = {"query": query, "report": report}  # query  # report for query

    return row


# 这是我们存储的有效 API 密钥
API_KEY = "123456"

# 创建api的app
app = FastAPI()


# 配置 CORS 中间件
from fastapi.middleware.cors import CORSMiddleware

# 定义请求体的 Pydantic 模型


class RetrievalRequest(BaseModel):
    knowledge_id: str
    query: str


# 定义一个函数来获取请求头中的 Authorization 字段
def get_api_key(authorization: str = Header(...)):
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="API Key 不正确")
    return authorization


@app.get("/")
async def hello():
    print("hello api!")
    return "hello api"


# 定义 POST 请求的处理函数,milvus向量数据库搜索的api
@app.post("/deepwriter/retrieval")
async def milvus_retrieval(
    request: RetrievalRequest, authorization: str = Depends(get_api_key)
):

    # 获取请求数据
    knowledge_id = request.knowledge_id
    query = request.query

    # 调用结果函数
    records = deepwriter_report_generation(query)
    # records = get_milvusknow(query,top_k,score_threshold)

    # 返回符合要求的响应格式
    return {"records": records}


# 启动服务器
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=get_deepwriter_port())
