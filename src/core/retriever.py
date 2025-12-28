"""
Módulo de compatibilidade para retriever
Re-exporta get_retriever do retrieval_service para manter compatibilidade
com imports antigos e facilitar uso em toda a aplicação
"""

from src.services.retrieval_service import (
    get_retriever,
    reset_retriever,
    HybridRetriever,
    RetrievalStrategy,
    RetrievalResult,
    DocumentFilter
)

__all__ = [
    "get_retriever",
    "reset_retriever", 
    "HybridRetriever",
    "RetrievalStrategy",
    "RetrievalResult",
    "DocumentFilter"
]