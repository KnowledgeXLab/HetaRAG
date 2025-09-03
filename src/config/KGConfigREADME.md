# Knowledge Graph Related Parameters Configuration Guide

This guide explains how to configure parameters related to the knowledge graph.

**Configuration File Location**: `src/config/knowledge_graph/create_kg_conf.yaml`

> **Note**: For specific examples on using knowledge graph construction parameters, see :ref:`components_knowledge_graph`.

## Configuration Structure

1. **LLM Model Configuration**:

    * Use cases: Triplet generation, corresponding description generation, graph database construction
    * Configure LLM framework, model service IP address, model service ports (supports multiple ports for parallel inference), deployed model name, etc.

2. **Task Parameter Configuration**:

    * Use cases: Triplet generation, corresponding description generation, graph database construction
    * Configure task multiprocessing count, head entity matching path, reference open-source triplet file path, number of layers in the generated graph, etc.

## Configuration Examples

**1. LLM Model Configuration Example**:

```yaml
## LLM Parameters
llm_conf:
  llm_framework: "vllm"
  
  ## LLM URL
  llm_host: "127.0.0.1"
  llm_ports: [8001, 8002, 8003, 8004] # Ports where models are deployed
  
  ## LLM API Key
  llm_api_key: "EMPTY"
  
  ## LLM Model
  llm_model: "qwen3_32b" 

  ## Maximum retry attempts when calling LLM online
  max_error: 3
```

**2. Task Parameter Configuration Example**:

```yaml
## Task Parameters
task_conf:
  ## Number of levels in the generated graph
  level_num: 2

  ## Multiprocessing count for head entity matching (-1 means use all CPU cores)
  num_processes_match: -1

  ## Multiprocessing count for inference (-1 means use all CPU cores)
  num_processes_infer: 16

  ## Path to head entities
  pedia_entity_path: src/resources/temp/knowledge_graph/dbpedia_entities_clean_valid.txt

  ## Reference open-source triplet file path
  ref_kg_path: src/resources/temp/knowledge_graph/triple_ref_test.txt
```
