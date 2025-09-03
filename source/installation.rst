.. _installation:

MinerU配置
=============


.. _MinerU_installation:
MinerU 模型下载
--------------------------------

模型文件可以从 Hugging Face 下载。

从 Hugging Face 下载模型
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

使用python脚本 从Hugging Face下载模型文件

    .. code-block:: bash
        
        pip install huggingface_hub
        python download_models.py

python脚本会自动下载模型文件并配置好配置文件中的模型目录

将配置文件 `magic-pdf.json` 放在用户目录中。

:download:`download_models.py </_static/download_models.py>`

:download:`magic-pdf.json </_static/magic-pdf.json>`

.. tip::
    
    windows的用户目录为 "C:\Users\用户名", linux用户目录为 "/home/用户名", macOS用户目录为 "/Users/用户名"

详情可查看 `MinerU 1.3.11 版本使用安装指南 <https://github.com/opendatalab/MinerU/tree/release-1.3.11#quick-start>`_ 