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

    def get_embedding_framework(self):
        """
        获取到 embedding 使用的框架
        """
        return self.config.get("embedding_service", "framework")

    def get_llm_framework(self):
        """
        获取到 llm 使用的框架
        """
        return self.config.get("llm_service", "framework")

    def get_embeddingmodel_config(self):
        """
        获取到embedding模型的各种信息
        """
        embeddingmodel_config = {
            "host": self.config.get("embedding_service", "host"),
            "port": self.config.get("embedding_service", "port"),
            "model_name": self.config.get("embedding_service", "model_name"),
        }

        return embeddingmodel_config

    def get_llmmodel_config(self):
        """
        获取到llm模型的各种信息
        """
        llmmodel_config = {
            "host": self.config.get("llm_service", "host"),
            "port": self.config.get("llm_service", "port"),
            "model_name": self.config.get("llm_service", "model_name"),
        }

        return llmmodel_config

    def get_llm_model(self):
        return self.config.get("llm_service", "model_name")

    def get_embedding_model(self):
        return self.config.get("embedding_service", "model_name")

    def get_openai_key(self):
        return self.config.get("openai_key", "key")


def get_embedding_framework():
    config_loader = ServiceConfigLoader()
    embedding_framework = config_loader.get_embedding_framework()
    return embedding_framework


def get_llm_framework():
    config_loader = ServiceConfigLoader()
    llm_framework = config_loader.get_llm_framework()
    return llm_framework


def get_ollama_embedding():
    config_loader = ServiceConfigLoader()
    embeddingmodel_config = config_loader.get_embeddingmodel_config()
    return embeddingmodel_config


def get_vllm_embedding():
    config_loader = ServiceConfigLoader()
    embeddingmodel_config = config_loader.get_embeddingmodel_config()
    return embeddingmodel_config


def get_ollama_llm():
    config_loader = ServiceConfigLoader()
    llmmodel_config = config_loader.get_llmmodel_config()
    return llmmodel_config


def get_vllm_llm():
    config_loader = ServiceConfigLoader()
    llmmodel_config = config_loader.get_llmmodel_config()
    return llmmodel_config


def get_llm_model():
    config_loader = ServiceConfigLoader()
    llm_model = config_loader.get_llm_model()
    return llm_model


def get_embedding_model():
    config_loader = ServiceConfigLoader()
    embedding_model = config_loader.get_embedding_model()
    return embedding_model


def get_openai_key():
    config_loader = ServiceConfigLoader()
    openai_key = config_loader.get_openai_key()
    return openai_key
