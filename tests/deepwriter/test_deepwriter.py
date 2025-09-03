import argparse
from pathlib import Path
from loguru import logger
from src.deepwriter.database.milvus_db import MilvusDB
from src.deepwriter.finders.doc_finder import DocFinder
from src.deepwriter.pipeline import DeepWriter
from src.deepwriter.processors.fitz_processor import FitzProcessor
from src.deepwriter.processors.MinerUProcessor import MinerUProcessor
from src.deepwriter.utils import file_interface, citations

logger.add("logs/report_generation_demo.log", mode="w")


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
    parser.add_argument(
        "--base_url",
        type=str,
        required=False,
        default=None,
        help="Base URL for the LLM, default: None",
    )
    return parser.parse_args()


def main(query) -> None:
    """Main function to run the report generation demo."""
    # args = parse_arguments()
    # Directly set the parameters in the code
    args = argparse.Namespace(
        output_path="output/",
        query=query,
        host="10.6.8.115",
        port=19530,
        user="",
        password="",
        embedding_model="Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
        llm="ollama_chat/qwen2.5:72b",
        base_url="http://10.6.8.115:11439",
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
    report = citations.post_process_report(report, args.output_path)

    # 7. Save report
    report_path = Path(args.output_path) / "report.md"
    # Check if report path exists and add a postfix if needed
    counter = 1
    original_path = report_path
    while report_path.exists():
        report_path = original_path.with_stem(f"{original_path.stem}_{counter}")
        counter += 1

    logger.info(f"Saving report to: {report_path}")
    file_interface.save_markdown_report(report, report_path)

    # save metadata
    metadata = {
        "query": args.query,
        "report_file": str(report_path),
        "model": args.llm,
        "embedding_model": args.embedding_model,
    }
    metadata_path = Path(args.output_path) / "metadata.json"
    if metadata_path.exists():
        existing_metadata = file_interface.load_json(metadata_path)
        existing_metadata.append(metadata)
    else:
        existing_metadata = [metadata]
    file_interface.export_json(existing_metadata, metadata_path)


if __name__ == "__main__":
    query = "What is the total trade volume of the world in 2024?"
    main(query)
