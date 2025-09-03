# Embedding and LLM Model Configuration Guide

This guide explains how to configure parameters for Embedding and LLM models.

**Configuration File Location**: `src/config/config.ini`

## Configuration Structure

1. **Framework Selection**:

   * Specify the type of framework used
   * Supported frameworks: `ollama`, `vllm`

2. **Parameter Settings**:

| Parameter | Example Value | Description |
| :--- | :--- | :--- |
| framework | vllm | Framework type |
| host | 127.0.0.1 | Model service IP address (use 127.0.0.1 for local deployment) |
| port | 8004 | Model service port number |
| model_name | qwen2.5:72b | Must exactly match the deployed model name |

## Configuration Examples

**1. Ollama Framework Example**:

```ini
[ollama_embedding]
framework = ollama
host = 127.0.0.1  # Change to actual IP
port = 11434
model_name = bge-m3

[ollama_llm]
framework = ollama
host = 127.0.0.1
port = 11434 
model_name = qwen2.5:72b
```

**2. vLLM Framework Example**:

```ini
[vllm_embedding]
framework = vllm
host = 127.0.0.1  # Model service IP address
port = 8001
model_name = bge-m3

[vllm_llm] 
framework = vllm
host = 127.0.0.1
port = 8002
model_name = Qwen2.5-72B-Instruct
```
