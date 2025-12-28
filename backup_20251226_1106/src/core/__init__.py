"""
Módulo Core - Componentes fundamentais do sistema ANTT RAG
"""

from src.core.config import (
    Config,
    get_config,
    reload_config
)

from src.core.embeddings import (
    EmbeddingManager,
    EmbeddingCache,
    EmbeddingResult,
    get_embedding_manager,
    get_embeddings_function,
    embed_texts,
    embed_query
)

from src.core.llm import (
    LLMManager,
    LLMProvider,
    LLMResponse,
    get_llm_manager,
    generate,
    agenerate
)

__all__ = [
    # Config
    "Config",
    "get_config",
    "reload_config",
    
    # Embeddings
    "EmbeddingManager",
    "EmbeddingCache",
    "EmbeddingResult",
    "get_embedding_manager",
    "get_embeddings_function",
    "embed_texts",
    "embed_query",
    
    # LLM
    "LLMManager",
    "LLMProvider",
    "LLMResponse",
    "get_llm_manager",
    "generate",
    "agenerate",
]
