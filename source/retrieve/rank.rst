.. _retrieve_rank:

检索评估
===========

本章节详细介绍了 HRAG 系统的 RAG-Challenge 数据集回答结果的测评。

测评方法
-----------

调用 ``tests/data_processor/test_challenge_pipeline.py`` 中的

.. code-block:: python

    from src.data_processor.converters.challenge_pipeline import Pipeline,RunConfig

    def test_full_pipeline(root_path):
        # 设置日志级别
        logging.basicConfig(level=logging.INFO)
    
        # 初始化pipeline
        pipeline = Pipeline(root_path, run_config=RunConfig())
    
        print("\n=== 步骤7: 结果测评 ===")
        pipeline.get_rank()


.. note::
   仅运行检索结果评估时，需保证如下文件存在：
   
   1. 生成的答案文件在 root_path/answer/(``src/resources/data/answers``) 目录中
   2. 正确答案文件 root_path/answers.json(``/src/resources/data/answers.json``) 存在

   详细的文件组织见 :ref:`retrieve_data`

