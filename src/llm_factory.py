"""
llm_factory.py
==============
Centralized LLM client factory. Returns a ChatGoogleGenerativeAI instance
configured from settings.yaml. All modules should use get_llm() instead of
directly instantiating a provider-specific client.

Usage:
    from src.llm_factory import get_llm
    llm = get_llm()
    llm_strict = get_llm(temperature=0.0, max_output_tokens=64)
"""

import os

from src.config import config

_llm_cfg = config.get("llm", {})


def get_llm(**overrides):
    """
    Return a ChatGoogleGenerativeAI instance from config, with optional overrides.

    Args:
        **overrides: Any ChatGoogleGenerativeAI kwargs to override config defaults.
                     Common: temperature, max_output_tokens.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.environ.get(_llm_cfg.get("api_key_env", "GOOGLE_API_KEY"), "")
    kwargs = {
        "model": _llm_cfg.get("model", "gemini-2.5-flash-lite-preview"),
        "google_api_key": api_key,
        "temperature": _llm_cfg.get("temperature", 0.1),
        "max_output_tokens": _llm_cfg.get("max_tokens", 1024),
    }
    kwargs.update(overrides)
    return ChatGoogleGenerativeAI(**kwargs)
