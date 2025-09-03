import os
import fitz  # PyMuPDF
import argparse
import shutil


def get_pdf_info(folder_path):
    """
    统计指定文件夹中所有 PDF 文件的总页数、总大小（单位：GB）、有效 PDF 文件数量和无法处理的文件数量。

    :param folder_path: 源文件夹路径
    :return: 总页数、总大小（GB）、有效 PDF 文件数量和无法处理的文件数量
    """
    total_pages = 0
    total_size_gb = 0.0
    valid_pdf_count = 0
    invalid_pdf_count = 0
    # target_dir = "/data/H-RAG/gov_decision/en-database-dash-clean"
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    # 遍历文件夹及其所有子目录
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(root, file)
                try:
                    # 打开 PDF 文件
                    pdf_document = fitz.open(file_path)
                    page_count = len(pdf_document)
                    file_size_gb = os.path.getsize(file_path) / (1024**3)  # 转换为 GB

                    # 累加总页数和总大小
                    total_pages += page_count
                    total_size_gb += file_size_gb

                    # 增加有效 PDF 文件计数
                    valid_pdf_count += 1

                    # 遍历每一页，尝试提取图像信息
                    for page_num in range(page_count):
                        page = pdf_document.load_page(page_num)
                        image_list = page.get_images(full=True)
                        for image_index, img in enumerate(image_list):
                            xref = img[0]
                            try:
                                # 尝试创建 Pixmap
                                pix = fitz.Pixmap(pdf_document, xref)
                                # 如果需要，可以在这里对图像进行进一步处理
                            except Exception as e:
                                print(
                                    f"无法处理文件 {file_path} 中的图像 (页面 {page_num}, 图像 {image_index}): {e}"
                                )
                                # 删除无法处理的文件
                                try:
                                    os.remove(file_path)
                                    print(f"已删除文件 {file_path}")
                                except Exception as e:
                                    print(f"无法删除文件 {file_path}: {e}")
                                # 增加无法处理的文件计数
                                invalid_pdf_count += 1
                                # 从统计结果中减去当前文件的页数和大小
                                total_pages -= page_count
                                total_size_gb -= file_size_gb
                                # 跳过当前文件的后续处理
                                raise  # 重新抛出异常，终止当前文件的处理

                    # try:
                    #     shutil.copy(file_path, target_dir)
                    #     print(f"已复制: {file_path} -> {target_dir}")
                    # except Exception as e:
                    #     print(f"无法复制 {file_path}: {e}")
                except Exception as e:
                    print(f"无法处理文件 {file_path}: {e}")
                    # 删除无法处理的文件
                    try:
                        os.remove(file_path)
                        print(f"已删除文件 {file_path}")
                    except Exception as e:
                        print(f"无法删除文件 {file_path}: {e}")
                    # 增加无法处理的文件计数
                    invalid_pdf_count += 1
                    # 如果文件在打开时就失败，不需要从统计结果中减去页数和大小

    return total_pages, total_size_gb, valid_pdf_count, invalid_pdf_count


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="统计指定文件夹中所有 PDF 文件的总页数、总大小、有效 PDF 文件数量和无法处理的文件数量"
    )
    parser.add_argument("folder_path", type=str, help="目标文件夹路径")

    # 解析命令行参数
    args = parser.parse_args()

    # 检查路径是否存在
    if not os.path.exists(args.folder_path):
        print(f"路径 {args.folder_path} 不存在，请检查后重新输入。")
        return

    # 调用函数统计 PDF 信息
    total_pages, total_size_gb, valid_pdf_count, invalid_pdf_count = get_pdf_info(
        args.folder_path
    )

    # 打印总统计结果
    print(f"总页数: {total_pages}")
    print(f"总大小: {total_size_gb:.4f} GB")
    print(f"有效 PDF 文件数量: {valid_pdf_count}")
    print(f"无法处理的文件数量: {invalid_pdf_count}")


if __name__ == "__main__":
    main()


"""
python scripts/get_info_clean_invalid_files.py /data/H-RAG/gov_decision/en-database-dash-clean
"""
