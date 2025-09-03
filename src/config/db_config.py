import configparser


class ServiceConfigLoader:
    """
    用于加载服务配置的类，配置文件路径默认为 ./../config.ini
    """

    def __init__(self):
        """
        初始化配置加载器，默认加载 ./config.ini 文件
        """
        self.config_file_path = "src/config/config.ini"  # 默认配置文件路径

        self.config = configparser.ConfigParser()
        self.config.read(self.config_file_path)

    def get_service_config(self, service_name):
        """
        获取指定服务的配置信息
        :param service_name: 服务名称（如Elasticsearch、Milvus、Neo4j）
        :return: 服务的host和端口配置
        """
        if service_name not in self.config.sections():
            raise ValueError(f"Service '{service_name}' not found in config file.")

        service_config = {
            "host": self.config.get(service_name, "host"),
            "front_end_port": self.config.getint(service_name, "front_end_port"),
            "read_write_port": self.config.getint(service_name, "read_write_port"),
            "username": self.config.get(service_name, "username"),
            "password": self.config.get(service_name, "password"),
        }
        return service_config

    def get_service_urls(self, service_name):
        """
        获取指定服务的host+port组合字符串
        :param service_name: 服务名称
        :return: 包含front_end_url和read_write_url的字典
        """
        config = self.get_service_config(service_name)
        return {
            "front_end_url": f"{config['host']}:{config['front_end_port']}",
            "read_write_url": f"{config['host']}:{config['read_write_port']}",
        }

    def get_service_username_password(self, service_name):
        """
        获取指定服务的username和passw
        :param service_name: 服务名称
        :return: 包含front_end_url和read_write_url的字典
        """
        if service_name not in self.config.sections():
            raise ValueError(f"Service '{service_name}' not found in config file.")

        username_password = {
            "username": self.config.get(service_name, "username"),
            "password": self.config.get(service_name, "password"),
        }

        return username_password

    def get_service_min_content_len(self, service_name):
        min_content_len = {
            "min_content_len": self.config.get(service_name, "min_content_len")
        }
        return min_content_len

    def get_embeddingmodel_config(self):
        """
        获取到embedding模型的各种信息
        """
        embeddingmodel_config = {
            "host": self.config.get("ollama_embedding", "host"),
            "port": self.config.get("ollama_embedding", "port"),
            "model_name": self.config.get("ollama_embedding", "model_name"),
            "username": self.config.get("ollama_embedding", "username"),
            "password": self.config.get("ollama_embedding", "password"),
        }

        return embeddingmodel_config

    def get_llmmodel_config(self):
        """
        获取到llm模型的各种信息
        """
        llmmodel_config = {
            "host": self.config.get("ollama_llm", "host"),
            "port": self.config.get("ollama_llm", "port"),
            "model_name": self.config.get("ollama_llm", "model_name"),
            "username": self.config.get("ollama_llm", "username"),
            "password": self.config.get("ollama_llm", "password"),
        }

        return llmmodel_config

    def get_knowledge_store(self, id):
        """
        获取到store的名字,这里要求各个数据库的store的名字相同
        """
        store_name = self.config.get("knowledge_store", str(id))

        return store_name

    def get_current_knowledge(self):
        """
        获取当前知识库的配置
        :return: 当前知识库的名称
        """
        # 从 current_knowledge 部分获取 select_knowledge_id 的值
        select_knowledge_id = self.config.get(
            "current_knowledge", "select_knowledge_id"
        )
        # 根据 select_knowledge_id 获取对应的知识库名称
        current_knowledge_name = self.get_knowledge_store(select_knowledge_id)
        return current_knowledge_name


def get_config(service_name):
    config_loader = ServiceConfigLoader()
    config = config_loader.get_service_config(service_name)
    return config


def get_username_password(service_name):
    config_loader = ServiceConfigLoader()
    username_password = config_loader.get_service_username_password(service_name)
    return username_password


def get_min_content_len(service_name):
    config_loader = ServiceConfigLoader()
    min_content_len = config_loader.get_service_min_content_len(service_name)
    return min_content_len


def get_ollama_embedding():
    config_loader = ServiceConfigLoader()
    embeddingmodel_config = config_loader.get_embeddingmodel_config()
    return embeddingmodel_config


def get_ollama_llm():
    config_loader = ServiceConfigLoader()
    llmmodel_config = config_loader.get_llmmodel_config()
    return llmmodel_config


def get_es_host():
    config_loader = ServiceConfigLoader()
    elasticsearch_urls = config_loader.get_service_urls("Elasticsearch")
    read_write_url = elasticsearch_urls["read_write_url"]
    return read_write_url


def get_store(id):
    config_loader = ServiceConfigLoader()
    store_name = config_loader.get_knowledge_store(id)
    return store_name


def get_data_search_port():
    config_loader = ServiceConfigLoader()
    # 从 backend_api 部分获取 data_search_port
    knowledge_search_port = config_loader.config.getint(
        "backend_api", "data_search_port"
    )
    return knowledge_search_port


def get_deepwriter_port():
    config_loader = ServiceConfigLoader()
    deepwriter_port = config_loader.config.getint("backend_api", "deepwriter_port")
    return deepwriter_port


def get_deepsearch_port():
    config_loader = ServiceConfigLoader()
    deepsearch_port = config_loader.config.getint("backend_api", "deepsearch")
    return deepsearch_port
