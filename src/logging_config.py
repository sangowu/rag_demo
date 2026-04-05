"""
logging_config.py
=================
结构化日志配置：输出 JSON 格式日志，包含 timestamp / level / module / message。

Usage:
    from src.logging_config import get_logger
    log = get_logger(__name__)
    log.info("chunks indexed", doc_id="ADI_2009", count=12)

生产环境中 JSON 日志可直接被 ELK / Datadog / CloudWatch 采集。
开发环境设置 LOG_FORMAT=text 可切换为可读格式。
"""

import logging
import os
import sys
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    """将 LogRecord 序列化为单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        import json

        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level":     record.levelname,
            "module":    record.name,
            "message":   record.getMessage(),
        }
        # 附加通过 extra= 传入的自定义字段
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            ):
                log_obj[key] = val
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """开发环境可读格式：2026-04-05 12:00:00 | INFO | src.api.main | message"""

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


def setup_logging(level: str = "INFO") -> None:
    """
    初始化全局日志配置。应在应用启动时调用一次。

    环境变量：
        LOG_LEVEL  : DEBUG / INFO / WARNING / ERROR（默认 INFO）
        LOG_FORMAT : json / text（默认 json）
    """
    log_level  = os.getenv("LOG_LEVEL", level).upper()
    log_format = os.getenv("LOG_FORMAT", "json").lower()

    formatter = _TextFormatter() if log_format == "text" else _JsonFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(log_level)
    # 避免重复添加 handler（uvicorn reload 时会多次调用）
    if not root.handlers:
        root.addHandler(handler)
    else:
        root.handlers = [handler]

    # 静默第三方噪声日志
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """获取具名 logger，供各模块使用。"""
    return logging.getLogger(name)
