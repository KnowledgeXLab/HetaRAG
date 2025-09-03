CUDA_VISIBLE_DEVICES=1 \
swift deploy \
    --adapters src/rerank/qwen_cl_v8 \
    --infer_backend vllm \
    --temperature 0 \
    --max_new_tokens 16384 \
    --port 8005 \
