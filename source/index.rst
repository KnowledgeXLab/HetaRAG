.. HRAG documentation master file, created by
   sphinx-quickstart on Fri Jun 27 15:12:15 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

欢迎使用 HRAG 文档！
========================

**HRAG (Hybrid Retrieval-Augmented Generation)** 是一个强大的混合检索增强生成系统，专为处理复杂文档和生成高质量回答而设计。

.. raw:: html

    <div class="quick-start">
        <h3>🚀 快速开始</h3>
        <p>想要立即开始使用 HRAG？查看我们的 <a href="quickstart.html">快速开始指南</a> 或 <a href="installation.html">安装说明</a>。</p>
    </div>

系统特性
--------

* **多模态文档解析**: 支持 MinerU 和 Docling 两种文档解析方法
* **知识图谱构建**: 自动提取实体关系，构建 HiRAG 知识图谱
* **混合检索**: 结合向量检索和重排序技术，提供精准的文档检索
* **多数据库支持**: 支持 Milvus、Elasticsearch、Neo4j、MySQL 等多种数据库
* **高性能**: 针对大规模文档处理进行了优化
* **易于扩展**: 模块化设计，支持自定义组件和插件

.. raw:: html

    <div class="performance-metrics">
        <h4>📊 性能指标</h4>
        <ul>
            <li><strong>检索精度 (R)</strong>: 最高可达 79.7%</li>
            <li><strong>生成质量 (G)</strong>: 最高可达 77.2%</li>
            <li><strong>综合评分</strong>: 最高可达 117 分</li>
        </ul>
    </div>

.. toctree::
   :maxdepth: 2
   :caption: 用户指南
   :name: userguide

   quickstart
   database_Installation
   installation
   configuration
   basic_usage
   advanced_usage

.. toctree::
   :maxdepth: 2
   :caption: 功能组件
   :name: components

   components/data_parser
   components/data_processor
   components/databases
   components/knowledge_graph
   components/deepwriter
   components/deepsearch


.. toctree::
   :maxdepth: 2
   :caption: 检索方法
   :name: retrieve

   retrieve/data.rst
   retrieve/pipeline.rst
   retrieve/rank.rst

.. toctree::
   :maxdepth: 2
   :caption: 示例和教程
   :name: examples

   examples/basic_rag
   examples/knowledge_graph
   examples/rerank
   examples/deepsearch
   examples/backend

.. .. toctree::
..    :maxdepth: 2
..    :caption: 参考文档
..    :name: reference

..    reference/glossary
..    reference/faq
..    reference/changelog
..    reference/license

.. 索引和表格
.. ==========

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

.. .. raw:: html

..     <div class="hrag-component">
..         <h3>💡 需要帮助？</h3>
..         <p>如果您在使用过程中遇到问题，可以：</p>
..         <ul>
..             <li>查看 <a href="troubleshooting.html">故障排除指南</a></li>
..             <li>阅读 <a href="reference/faq.html">常见问题</a></li>
..             <li>在 <a href="https://github.com/your-github-username/hrag/issues">GitHub Issues</a> 中提交问题</li>
..         </ul>
..     </div>
