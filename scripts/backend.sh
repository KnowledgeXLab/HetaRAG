#!/bin/bash

# 数据搜索
nohup python src/backend/data_search_services.py > logs/data_search.log 2>&1 &

# deepwriter 论文撰写
nohup python src/backend/deepwriter_services.py > logs/deepwriter.log 2>&1 &

# 多跳思考
nohup python src/backend/multi_hop_services.py > logs/multi_hop.log 2>&1 &
