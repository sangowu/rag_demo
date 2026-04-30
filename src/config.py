"""
config.py
=========
Load project settings from config/settings.yaml.

Usage:
    from src.config import config

    chunk_size = config["chunking"]["chunk_size"]
"""

from pathlib import Path
import yaml

# Project root = two levels up from this file (src/config.py → rag_demo/)
_ROOT = Path(__file__).parent.parent
_CONFIG_PATH = _ROOT / "config" / "settings.yaml"

# 加载 .env（存在时）— 不覆盖已有的 shell 环境变量
try:
    from dotenv import load_dotenv
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass


def _load() -> dict:
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# Single shared instance — loaded once at first import
config: dict = _load()
