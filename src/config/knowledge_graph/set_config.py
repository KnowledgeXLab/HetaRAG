import yaml
import logging

logger = logging.getLogger("set_config")


def load_yaml_conf(config_path):

    with open(config_path, "r") as file:
        args = yaml.safe_load(file)
    return args


def write_yaml_conf(config_path, data):
    # 保存更新后的配置到新的YAML文件
    with open(config_path, "w") as file:
        yaml.safe_dump(data, file, default_flow_style=False)
