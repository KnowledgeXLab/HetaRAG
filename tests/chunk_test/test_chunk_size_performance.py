import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from pyprojroot import here
from src.data_processor.converters.challenge_pipeline import Pipeline, RunConfig


class ChunkSizePerformanceTester:
    def __init__(self, root_path: Path, chunk_sizes: List[int]):
        self.root_path = root_path
        self.chunk_sizes = chunk_sizes

    def read_rank_results(self, chunk_size: int) -> Dict[str, Any]:
        """直接读取ranking.csv获取测评结果"""
        ranking_path = Path("src/resources/data/ranking.csv")
        if not ranking_path.exists():
            print("ranking.csv not found!")
            return {}
        df = pd.read_csv(ranking_path)
        submission_name = f"ChunkSize_{chunk_size}"
        result_row = df[df["team"] == submission_name]
        if result_row.empty:
            return {}
        return {
            "rank": int(result_row.iloc[0]["rank"]),
            "ref_score": float(result_row.iloc[0]["R"]),
            "val_score": float(result_row.iloc[0]["G"]),
            "total_score": float(result_row.iloc[0]["Score"]),
        }

    def _create_milvus_db_with_pkl(self, pipeline, pkl_dir: Path):
        """创建Milvus数据库并插入PKL数据"""
        from src.database.operations.milvus_operations import (
            create_collection,
            pkl_insert,
        )
        from src.database.db_connection import milvus_connection

        milvus_connection()
        collection_name = pipeline.run_config.collection_name
        dim = 1024
        print(f"[Milvus] 创建集合: {collection_name}")
        collection = create_collection(collection_name, dim)
        pkl_files = list(pkl_dir.glob("*.pkl"))
        if pkl_files:
            pkl_file = pkl_files[0]
            print(f"[Milvus] 使用PKL文件: {pkl_file}")
            pkl_insert(collection, str(pkl_file), image_embedding=False)
        else:
            print("[Milvus] 未找到PKL文件")
        print(f"[Milvus] 数据库创建完成: {collection_name}")

    def test_single_chunk_size(self, chunk_size: int) -> Dict[str, Any]:
        """测试单个chunk_size的性能"""
        print(f"\n=== 测试 Chunk Size: {chunk_size} ===")
        run_config = RunConfig(
            parent_document_retrieval=True,
            parallel_requests=1,
            top_n_retrieval=14,
            submission_name=f"ChunkSize_{chunk_size}",
            pipeline_details=f"ChunkSize_{chunk_size} + milvus + Parent Document Retrieval + SO CoT; llm = qwen2.5:72b; embedding = bge-m3:latest",
            api_provider="ollama",
            answering_model="qwen2.5:72b",
            config_suffix=f"_chunk_size_{chunk_size}",
            use_vector_dbs="milvus",
        )
        output_dir = here() / "src" / "pkl_files" / f"chunk_size_{chunk_size}"
        pkl_file = output_dir / "docling.pkl"
        milvus_collection_name = f"challenge_data_chunk_{chunk_size}"

        try:
            pipeline = Pipeline(self.root_path, run_config=run_config)
            start_time = time.time()

            if not pkl_file.exists():
                print(f"[Step1] 分块报告 (chunk_size={chunk_size})")
                pipeline.chunk_reports(chunk_size=chunk_size)
                output_dir.mkdir(parents=True, exist_ok=True)
                pipeline.chunk_to_pkl(output_dir)
                print(
                    f"[Step2] 创建Milvus向量数据库 (collection_name={milvus_collection_name})"
                )
                pipeline.run_config.collection_name = milvus_collection_name
                self._create_milvus_db_with_pkl(pipeline, output_dir)
            else:
                print(f"[Info] 已存在pkl和Milvus集合，跳过分块、向量化、建库步骤。")
                pipeline.run_config.collection_name = milvus_collection_name

            print(f"[Step3] 处理问题")
            question_start_time = time.time()
            # pipeline.process_questions()
            avg_response_time = time.time() - question_start_time

            print(f"[Step4] 结果测评")
            pipeline.get_rank()

            total_time = time.time() - start_time
            rank_results = self.read_rank_results(chunk_size)

            print(f"[Summary] Chunk Size {chunk_size} 测试完成:")
            print(f"  总耗时: {total_time:.2f}秒")
            print(f"  问题处理耗时: {avg_response_time:.2f}秒")
            if rank_results:
                print(f"  评分结果: {rank_results}")

            return {
                "chunk_size": chunk_size,
                "total_time": total_time,
                "avg_response_time": avg_response_time,
                "rank_results": rank_results,
                "config_suffix": run_config.config_suffix,
            }

        except Exception as e:
            print(f"[Error] Chunk Size {chunk_size} 测试失败: {str(e)}")
            return {
                "chunk_size": chunk_size,
                "error": str(e),
                "total_time": None,
                "avg_response_time": None,
                "rank_results": None,
            }

    def run_performance_test(self) -> List[Dict[str, Any]]:
        """运行所有chunk_size的性能测试"""
        print(f"开始测试 Chunk Sizes: {self.chunk_sizes}")
        results = []
        for chunk_size in self.chunk_sizes:
            results.append(self.test_single_chunk_size(chunk_size))
        self.generate_summary_report(results)
        return results

    def generate_summary_report(self, results: List[Dict[str, Any]]):
        """生成总结报告"""
        print("\n" + "=" * 60)
        print("性能测试总结报告")
        print("=" * 60)
        successful_results = [r for r in results if "error" not in r]
        if not successful_results:
            print("没有成功的测试结果")
            return
        summary_data = []
        for result in successful_results:
            chunk_size = result["chunk_size"]
            total_time = result["total_time"]
            avg_response_time = result["avg_response_time"]
            rank_results = result["rank_results"]
            summary_row = {
                "Chunk Size": chunk_size,
                "总耗时(秒)": f"{total_time:.2f}",
                "问题处理耗时(秒)": f"{avg_response_time:.2f}",
                "排名": rank_results.get("rank", "N/A"),
                "引用分数": f"{rank_results.get('ref_score', 0):.3f}",
                "答案分数": f"{rank_results.get('val_score', 0):.3f}",
                "总分": f"{rank_results.get('total_score', 0):.3f}",
            }
            summary_data.append(summary_row)
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        output_file = self.root_path / "chunk_size_performance_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存到: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="测试不同chunk_sizes对系统性能的影响")
    parser.add_argument("--root_path", type=Path, required=True, help="数据根路径")
    parser.add_argument(
        "--chunk_sizes",
        type=int,
        nargs="+",
        default=[128, 256],
        help="要测试的chunk_sizes列表，默认: 128 256",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from pyprojroot import here

    # 用here()拼接用户传入的相对路径，得到绝对路径
    args.root_path = (here() / args.root_path).resolve()
    tester = ChunkSizePerformanceTester(args.root_path, args.chunk_sizes)
    tester.run_performance_test()
    print(
        f"\n性能测试完成！测试了 {len(args.chunk_sizes)} 个chunk_sizes: {args.chunk_sizes}"
    )


if __name__ == "__main__":
    main()

"""
使用示例:

# 使用默认chunk_sizes (128, 256)
python tests/chunk_test/test_chunk_size_performance.py \
    --root_path src/resources/data

# 自定义chunk_sizes
python tests/chunk_test/test_chunk_size_performance.py \
    --root_path src/resources/data \
    --chunk_sizes 64 128 256 512

# 测试单个chunk_size
python tests/chunk_test/test_chunk_size_performance.py \
    --root_path src/resources/data \
    --chunk_sizes 256
"""
