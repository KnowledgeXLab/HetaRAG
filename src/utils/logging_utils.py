import os
import time
import logging
from logging.handlers import RotatingFileHandler
import sys

# 如果需要颜色支持，建议使用 colorama 以确保在 Windows 上也能正确显示颜色
try:
    import colorama
    from colorama import Fore, Style

    colorama.init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """
    自定义格式化器，根据日志级别为日志消息添加颜色。
    INFO: 绿色
    WARNING: 黄色
    ERROR: 红色
    """

    # 定义颜色映射
    COLOR_MAP = {
        logging.DEBUG: Fore.CYAN if COLORAMA_AVAILABLE else "",
        logging.INFO: Fore.GREEN if COLORAMA_AVAILABLE else "",
        logging.WARNING: Fore.YELLOW if COLORAMA_AVAILABLE else "",
        logging.ERROR: Fore.RED if COLORAMA_AVAILABLE else "",
        logging.CRITICAL: Fore.MAGENTA if COLORAMA_AVAILABLE else "",
    }

    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ""

    def format(self, record):
        color = self.COLOR_MAP.get(record.levelno, "")
        message = super().format(record)
        if color:
            message = f"{color}{message}{self.RESET}"
        return message


def setup_logger(logger_name, level=logging.INFO):
    """
    初始化并返回一个日志记录器。

    该函数确保所有日志记录器共享相同的处理程序，包括：
    - 基于时间戳的日志文件
    - latest.log 文件
    - 控制台输出（带颜色）

    参数:
    - logger_name: 日志记录器的名称
    - level: 日志级别，默认为 logging.INFO

    返回:
    - logger: 配置好的日志记录器
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = True  # 确保日志传递到根记录器

    # 检查根记录器是否已配置处理程序
    if not logging.getLogger().hasHandlers():
        # 获取主脚本的路径
        main_script = sys.argv[0]
        if not main_script:
            # 在某些环境下（如交互式解释器），sys.argv[0] 可能为空
            main_script_dir = os.getcwd()
            main_script_name = "unknown"
        else:
            main_script_path = os.path.abspath(main_script)
            main_script_dir = os.path.dirname(main_script_path)
            main_script_name = os.path.splitext(os.path.basename(main_script_path))[0]

        # 创建日志目录：{主脚本目录}/logs/{主脚本名称}/
        log_dir = os.path.join("logs", main_script_name)
        os.makedirs(log_dir, exist_ok=True)

        # 生成基于时间戳的日志文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        timestamp_log_file = os.path.join(log_dir, f"{timestamp}.log")

        # 定义 latest.log 文件路径
        latest_log_file = os.path.join(log_dir, "latest.log")

        # 定义日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 定义带颜色的日志格式化器
        colored_formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 处理程序1：基于时间戳的日志文件（带轮转）
        timestamp_handler = RotatingFileHandler(
            timestamp_log_file,
            maxBytes=1000 * 1024 * 1024,  # 1000 MB
            backupCount=5,
            encoding="utf-8",
        )
        timestamp_handler.setFormatter(formatter)
        timestamp_handler.setLevel(level)
        logging.getLogger().addHandler(timestamp_handler)

        # 处理程序2：latest.log 文件（覆盖模式）
        latest_handler = logging.FileHandler(
            latest_log_file, mode="w", encoding="utf-8"  # 覆盖模式，每次运行覆盖文件
        )
        latest_handler.setFormatter(formatter)
        latest_handler.setLevel(level)
        logging.getLogger().addHandler(latest_handler)

        # 处理程序3：控制台输出（带颜色）
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(colored_formatter)
        console_handler.setLevel(level)
        logging.getLogger().addHandler(console_handler)

    return logger
