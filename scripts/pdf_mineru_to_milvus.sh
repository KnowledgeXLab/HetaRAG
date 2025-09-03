#!/bin/bash

# 定义共享参数
input_path="/data/H-RAG/gov_decision/en-database-dash-small"
pkl_path="/data/H-RAG/gov_decision/en-database-dash-small/small.pkl"
collection_name="en_database_dash_test"
question="" # 如果需要测试问答功能，可以设置一个问题


echo "-----------------预处理 PDF ，并统计 PDF 信息：---------------------"
# 处理pdf文件，统计可用pdf信息
python scripts/pdf_mineru_to_milvus/get_info_clean_invalid_files.py "$input_path" 

if [ $? -ne 0 ]; then
    echo "预处理 PDF 失败！！！！！"
    exit 1
fi


echo "------------------使用 MinerU 解析 PDF ： --------------------------"
# 使用mineru处理清洗过的pdf文件
python scripts/pdf_mineru_to_milvus/get_minerU_clean_invalid_files.py "$input_path"

if [ $? -ne 0 ]; then
    echo "MinerU 解析失败！！！！！"
    exit 1
fi


echo "--------------------向量化解析后的数据：---------------------------- "
# 向量化后存入pkl
# 进行图片的embedding，使用QwenVL
# python scripts/get_vector_to_pkl.py --input_path "$input_path" --output_path "$pkl_path" --image_embedding
# 不进行图片的embedding，使用BGE
python scripts/pdf_mineru_to_milvus/get_vector_to_pkl.py --input_path "$input_path" --output_path "$pkl_path"

if [ $? -ne 0 ]; then
    echo "向量化失败！！！！！"
    exit 1
fi


echo "---------------------数据插入 Milvus ： ---------------------------"
# pkl导入milvus
# python scripts/pkl_instert_milvus.py --collection_name "$collection_name" --pkl_path "$pkl_path" --question "$question" --image_embedding
python scripts/pdf_mineru_to_milvus/pkl_instert_milvus.py --collection_name "$collection_name" --pkl_path "$pkl_path" --question "$question"

if [ $? -ne 0 ]; then
    echo "Milvus 插入失败！！！！！"
    exit 1
fi
