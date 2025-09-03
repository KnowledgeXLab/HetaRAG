# run_chunk_only.py
import argparse
from pathlib import Path


from src.data_processor.converters.challenge_pipeline import Pipeline, RunConfig


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the chunk reports.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Chunk reports pipeline")
    parser.add_argument(
        "--root_path",
        type=Path,
        required=False,
        default="src" / "resources" / "data",
        help="data root path for the pipeline",
    )
    parser.add_argument(
        "--chunk_mode",
        type=str,
        required=False,
        default="fixed_doc",
        choices=["base", "semantic", "fixed_page", "fixed_doc"],
        help="分块模式：base(基础分块), semantic(语义分块), fixed_page(页面固定长度), fixed_doc(文档固定长度)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        required=False,
        default=300,
        help="设置分块大小，默认300个token",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        required=False,
        default=50,
        help="设置分块重叠度，默认50个token",
    )
    parser.add_argument(
        "--breakpoint_type",
        type=str,
        required=False,
        default="percentile",
        help="语义分块阈值类型",
    )
    parser.add_argument(
        "--breakpoint_amount",
        type=float,
        required=False,
        default=90,
        help="语义分块阈值大小",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 初始化pipeline配置
    run_config = RunConfig(
        config_suffix="_debug_chunk_only",
        # TextSplitter 配置
        chunk_mode=args.chunk_mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        breakpoint_type=args.breakpoint_type,
        breakpoint_amount=args.breakpoint_amount,
    )

    # 初始化pipeline
    pipeline = Pipeline(root_path=args.root_path, run_config=run_config)

    # 打印pipeline的路径配置
    print("\n=== Pipeline路径配置 ===")
    print(f"root_path: {pipeline.paths.root_path}")
    print(f"debug_data_path: {pipeline.paths.debug_data_path}")
    print(f"merged_reports_path: {pipeline.paths.merged_reports_path}")
    print(f"documents_dir: {pipeline.paths.documents_dir}")
    print(f"databases_path: {pipeline.paths.databases_path}")

    # 打印TextSplitter配置
    print("\n=== TextSplitter配置 ===")
    print(f"chunk_mode: {run_config.chunk_mode}")
    print(f"chunk_size: {run_config.chunk_size}")
    print(f"chunk_overlap: {run_config.chunk_overlap}")

    # 执行分块
    print("\n=== 执行分块 ===")
    pipeline.chunk_reports()
    print("分块完成")


if __name__ == "__main__":
    main()


"""
使用示例：

1. 固定文档长度分块（fixed_doc）
python tests/chunk_test/test_chunk_reports.py \
    --chunk_mode fixed_doc \
    --chunk_size 300 \
    --chunk_overlap 50

2. 固定页面长度分块（fixed_page）
python tests/chunk_test/test_chunk_reports.py \
    --chunk_mode fixed_page \
    --chunk_size 400 \
    --chunk_overlap 40

3. 基础分块（base）
python tests/chunk_test/test_chunk_reports.py \
    --chunk_mode base \
    --chunk_size 256 \
    --chunk_overlap 32

4. 语义分块（semantic）
python tests/chunk_test/test_chunk_reports.py \
    --chunk_mode semantic \
    --chunk_size 300 \
    --chunk_overlap 50 \
    --breakpoint_type percentile \
    --breakpoint_amount 85

5. 查看所有参数说明
python tests/chunk_test/test_chunk_reports.py --help

"""
