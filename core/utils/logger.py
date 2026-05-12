import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "pack_core",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    配置并返回一个日志记录器

    Args:
        name: 日志记录器名称
        log_level: 日志级别
        log_file: 可选的日志文件路径

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()
