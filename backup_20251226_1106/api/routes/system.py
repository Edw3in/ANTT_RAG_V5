from __future__ import annotations

import os
import platform
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter

from src.core.config import get_config

router = APIRouter(prefix="/system", tags=["System"])


def _safe_env(var: str) -> bool:
    """Retorna apenas se a variável existe (não expõe valor)."""
    return bool(os.getenv(var))


@router.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "ANTT RAG v4",
    }


@router.get("/config")
def config_view() -> Dict[str, Any]:
    c = get_config()
    return {
        "environment": c.environment,
        "debug": c.debug,
        "paths": {
            "base_dir": str(c.paths.base_dir),
            "data_dir": str(c.paths.data_dir) if c.paths.data_dir else None,
            "vectorstore_dir": str(c.paths.vectorstore_dir) if c.paths.vectorstore_dir else None,
        },
        "models": {
            "embedding": c.models.embedding,
            "embedding_device": c.models.embedding_device,
            "reranker_model": c.models.reranker_model,
            "reranker_device": c.models.reranker_device,
            "llm_provider": c.models.llm_provider,
            "llm_model": c.models.llm_model,
            "ollama_base_url": getattr(c.models, "ollama_base_url", None),
            "fallback_enabled": c.models.fallback_enabled,
            "fallback_provider": c.models.fallback_provider,
            "fallback_model": c.models.fallback_model,
        },
        "retrieval": {
            "use_hybrid": c.retrieval.use_hybrid,
            "use_reranker": c.retrieval.use_reranker,
            "default_k": c.retrieval.default_k,
            "max_k": c.retrieval.max_k,
        },
        "secrets_present": {
            "GOOGLE_API_KEY": _safe_env("GOOGLE_API_KEY"),
            "OPENAI_API_KEY": _safe_env("OPENAI_API_KEY"),
            "HUGGINGFACE_API_KEY": _safe_env("HUGGINGFACE_API_KEY"),
        },
    }


@router.get("/gpu")
def gpu_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch_available": False,
        "cuda_available": False,
        "device_name": None,
        "device_count": 0,
    }

    try:
        import torch  # noqa: F401

        info["torch_available"] = True
        import torch as _torch

        info["cuda_available"] = bool(_torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(_torch.cuda.device_count())
            info["device_name"] = _torch.cuda.get_device_name(0)
    except Exception as e:
        info["error"] = str(e)

    return info


@router.get("/runtime")
def runtime() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
