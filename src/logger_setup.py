"""
Central logging configuration for the whole service.
Logs go to both the console and a log file, with a consistent format.
Call setup_logging() once, early, before anything else runs.
"""

import logging
import sys
from pathlib import Path

from src.config_loader import load_config, resolve_path


def setup_logging(config: dict = None) -> None:
    """Configure the root logger: level, format, console + file handlers."""
    config = config or load_config()
    log_config = config["logging"]

    level_name = log_config.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)

    log_file_path = resolve_path(log_config["log_file"])
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate handlers if setup_logging() is called more than once
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        f"Logging configured. level={level_name}, log_file={log_file_path}"
    )
