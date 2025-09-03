from src.config.db_config import get_es_host, get_username_password, get_config

# === Elasticsearch Connection ===
from elasticsearch_dsl import Document, Text, Index, DenseVector, Integer
from elasticsearch_dsl.connections import connections as es_connections


def es_connection():
    host = get_es_host()
    username_password = get_username_password("Elasticsearch")
    client = es_connections.create_connection(
        hosts=[f"http://{host}"],
        timeout=60,
        http_auth=(username_password["username"], username_password["password"]),
        verify_certs=False,
    )
    return client


# === Milvus Connection ===
from pymilvus import connections as milvus_connections
from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility

# 全局连接状态
_milvus_connected = False


def milvus_connection():
    global _milvus_connected

    # 如果已经连接，直接返回
    if _milvus_connected:
        try:
            # 检查连接是否仍然有效
            utility.list_collections()
            return
        except Exception:
            # 连接已断开，重新连接
            _milvus_connected = False

    milvus_config = get_config("Milvus")
    host = milvus_config["host"]
    port = milvus_config["read_write_port"]
    username = milvus_config["username"]
    password = milvus_config["password"]

    print("开始连接Milvus...")
    milvus_connections.connect(
        alias="default", host=host, port=port, user=username, password=password
    )
    _milvus_connected = True
    print("Milvus连接成功")


def get_milvus_collection(collection_name: str) -> Collection:
    """获取Milvus集合，自动处理连接"""
    milvus_connection()
    collection = Collection(collection_name)
    collection.load()
    return collection


# === MySQL Connection ===
import pymysql


def mysql_connnection(database_name=None):
    mysql_config = get_config("MySQL")

    try:
        connection_params = {
            "host": mysql_config["host"],
            "user": mysql_config["username"],
            "password": mysql_config["password"],
            "charset": "utf8mb4",
            "cursorclass": pymysql.cursors.DictCursor,
        }

        # 只有当 database_name 不为空时才添加 database 参数
        if database_name:
            connection_params["database"] = database_name

        connection = pymysql.connect(**connection_params)
        return connection
    except pymysql.Error as e:
        print(f"连接数据库时出错: {e}")
        return False


# === Neo4j Connection using py2neo ===
from py2neo import Graph


def neo4j_connection():
    neo4j_config = get_config("Neo4j")
    host = neo4j_config["host"]
    read_write_port = neo4j_config["read_write_port"]
    username = neo4j_config["username"]
    password = neo4j_config["password"]

    neo4j_graph = Graph(
        f"bolt://{host}:{read_write_port}", user=username, password=password
    )
    return neo4j_graph


# === Neo4j Connection using official driver ===
from neo4j import GraphDatabase


def neo4j_connection_driver():
    neo4j_config = get_config("Neo4j")
    host = neo4j_config["host"]
    read_write_port = neo4j_config["read_write_port"]
    username = neo4j_config["username"]
    password = neo4j_config["password"]

    driver = GraphDatabase.driver(
        f"bolt://{host}:{read_write_port}", auth=(username, password)
    )
    return driver
