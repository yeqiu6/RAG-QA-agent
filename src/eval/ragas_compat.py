"""Compatibility helpers for third-party RAGAS imports."""

from __future__ import annotations

import importlib.util
import sys
import types


def patch_missing_vertexai_chat_model() -> None:
    """Provide the old LangChain Vertex AI chat import path expected by RAGAS."""
    module_name = "langchain_community.chat_models.vertexai"
    if importlib.util.find_spec(module_name) is not None:
        return

    module = types.ModuleType(module_name)

    class ChatVertexAI:
        pass

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module
