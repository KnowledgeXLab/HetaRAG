.. _database_installation:

数据库安装
============

本指南将帮助您快速安装4个数据库。

Docker 环境中使用的各种组件的默认版本，通过docker-compose.yml安装。

默认版本
^^^^^^^^^
* Docker: 4.35.0
* Elasticsearch: 7.17.1
* Kibana: 7.17.1
* Milvus: 2.4.0
* etcd: 3.5.5
* MinIO: RELEASE.2024-09-13T20-26-02Z
* Attu: 2.4.0
* Neo4j: 5.26.0
* MySQL: 5.7

安装步骤
^^^^^^^^^

1. **在docker目录下创建目录：**

   .. code-block:: bash

      cd docker
      mkdir -p elastic/data
      mkdir -p milvus
      mkdir -p neo4j
      mkdir -p mysql/data

* 修改 kibana 文件夹中的kibana.yml文件，并将elasticsearch.hosts更改为你自己的 IP 地址，并将下面的 elastic 密码更改为您稍后（在第二步中）想要设置的密码。（默认密码为 elastic）。启动服务


2. **一键安装数据库**

   .. code-block:: bash

      # 安装并启动数据库服务
      docker-compose up -d

此命令将下载并启动以下容器：

* elasticsearch：用于全文搜索和文档索引
* milvus：用于向量相似度搜索
* minio：用于对象存储
* etcd：用于分布式键值存储
* kibana：用于 elasticsearch 可视化

你需要进入 Elasticsearch 容器，在其容器的 exec 中输入命令：
   .. code-block:: bash

      # 设置密码
      bin/elasticsearch-setup-passwords interactive

有大约 5 到 6 个账户需要设置他们的密码。建议将所有密码修改为相同的密码。（为保持一致性，我将所有密码都修改为 “elastic”。）

3. **数据库端口**

.. list-table:: 服务端口映射
   :header-rows: 1
   :widths: 25 25 25

   * - Service
     - Front-end Port
     - Read/Write Port
   * - Elasticsearch
     - 5601
     - 9200
   * - Milvus
     - 8000
     - 19530
   * - Neo4j
     - 7474
     - 7687
   * - MySQL
     - 8080
     - 3306


.. note::
   Front-end Port用于浏览器查看数据服务的可视化界面，Read/Write Port用于数据库的读写操作。