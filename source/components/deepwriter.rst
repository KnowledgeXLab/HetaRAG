.. _components_deepwriter:

Deepwriter
=============

本章节详细介绍了 HRAG 系统中的 Deepwriter 组件。

Deepwriter 是一个基于大语言模型（LLM）和向量数据库的自动报告生成工具。它从非结构化文档中提取语义信息，通过用户提供的查询，自动撰写结构化文本报告。

使用说明
----------------

输入参数
~~~~~~~~~~~~~~~~

你可以通过命令行参数或直接在代码中设置以下参数：

- `--output_path`：报告输出路径（必填）
- `--query`：用户查询文本，用于驱动报告生成（必填）
- `--host`：Milvus 服务器地址（默认：localhost）
- `--port`：Milvus 端口（默认：19530）
- `--user`：Milvus 用户名（默认：空）
- `--password`：Milvus 密码（默认：空）
- `--embedding_model`：嵌入模型名称（默认：Alibaba GME）
- `--llm`：用于生成报告的语言模型（默认：ollama/qwen2:7b）
- `--base_url`：LLM 服务的基础 URL（默认：None）

组件结构
~~~~~~~~~~~~~~~~

1. **MilvusDB**：用于管理图文向量数据的检索和存储。
2. **DocFinder**：通过向量检索获取上下文信息，支持语义相似度排序。
3. **DeepWriter**：结合 LLM，根据用户查询和检索到的上下文信息生成结构化报告。
4. **ReportProcessor**：对生成报告进行后处理，并保存为 Markdown 格式。
5. **日志系统**：使用 `loguru` 记录流程信息，默认日志输出至 `logs/report_generation_demo.log`。

使用示例
----------------

你可以通过调用 ``tests/deepwriter/test_deepwriter.py`` 中的主函数 `main()` 来生成报告，示例代码如下：

.. code-block:: python

    if __name__ == "__main__":
        query = "What is the total trade volume of the world in 2024?"
        main(query)

报告将保存至 `output/` 目录下，文件名为 `report.md`。如已有同名文件，将自动添加编号后缀以避免覆盖。

输出结果
~~~~~~~~~~~~~~~~

生成内容包括：

- 报告正文：Markdown 格式文件 `report.md`

- 报告元信息：`metadata.json`，包含查询语句、使用模型、报告路径等信息


日志文件
~~~~~~~~~~~~~~~~

日志记录默认输出到：

::

    logs/report_generation_demo.log

可用于问题排查和流程追踪。



注意事项
----------------

- 当前仅支持基于 GME 的嵌入模型。

- LLM 服务必须支持 RESTful API 接口。

- 多模态嵌入模型和文本嵌入模型应保持一致配置。

- 文档数据库需提前构建并包含有效数据。



