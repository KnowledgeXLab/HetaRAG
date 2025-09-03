import os
from pathlib import Path
from tqdm import tqdm

from src.utils.logging_utils import setup_logger
from src.utils.common_utils import sort_large_file, remove_duplicates

logger = setup_logger("postprocess_data")


def process_all_folders(output_dir="output"):
    """批量处理所有子文件夹中的 all_entitys 文件 : 排序 & 去重"""
    output_dir = Path(output_dir)

    # 检查输出目录是否存在
    if not output_dir.exists():
        logger.info(f"Error: Output directory {output_dir} does not exist")
        return

    # 遍历输出目录下的所有子文件夹
    for folder in tqdm(output_dir.iterdir()):
        if not folder.is_dir():
            continue

        # 动态生成目标文件路径
        file_name = folder.name
        target_file = folder / f"all_entities_{file_name}.txt"

        # 跳过不存在的文件
        if not target_file.exists():
            logger.info(f"Warning: Target file {target_file} not found, skipping")
            continue

        logger.info(f"\nProcessing folder: {folder.name}")

        # 生成临时排序文件路径
        sorted_temp = folder / f"sorted_{target_file.name}"

        try:
            # Step 1: 排序大文件
            logger.info(f"Sorting {target_file.name}...")
            sort_large_file(str(target_file), str(sorted_temp))

            # Step 2: 去重排序后的文件（直接覆盖原文件）
            logger.info(f"Deduplicating {sorted_temp.name}...")

            remove_duplicates(str(sorted_temp), str(target_file))

            logger.info(f"Successfully processed {target_file.name}")
        except Exception as e:
            logger.info(f"Error processing {target_file}: {str(e)}")
        finally:
            # 清理临时排序文件
            if sorted_temp.exists():
                os.remove(sorted_temp)
                logger.info(f"Cleaned up temporary file: {sorted_temp.name}")


def clean_next_layer_files(root_dir="output"):
    """清理指定目录及其子目录中所有 next_layer_entities_*.txt 文件"""
    root_path = Path(root_dir)

    # 递归查找所有匹配文件
    for file_path in tqdm(root_path.glob("**/next_layer_entities_*.txt")):
        try:
            os.remove(file_path)
            logger.info(f"✅ 已删除：{file_path}")
        except FileNotFoundError:
            logger.info(f"⚠️ 文件不存在：{file_path}")
        except Exception as e:
            logger.info(f"❌ 删除失败：{file_path} | 错误：{str(e)}")


if __name__ == "__main__":

    data_path = "src/resources/temp/knowledge_graph/triple"

    process_all_folders(data_path)

    clean_next_layer_files(data_path)
