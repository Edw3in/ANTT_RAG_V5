"""
ANTT RAG System - Sistema de Recuperação e Geração Aumentada
Sistema completo de RAG para consulta de normativos da ANTT
"""

__version__ = "4.0.0"
__author__ = "ANTT RAG Team"

from src.core import (
    get_config,
    get_embedding_manager,
    get_llm_manager,
)

__all__ = [
    # Core
    "get_config",
    "get_embedding_manager",
    "get_llm_manager",
    
    # Services
    "HybridRetriever",
    "AnswerService",
    "IngestService",
    
    # Utils
    "ResponseValidator",
    "AuditLogger",
    "PromptManager",
    "TextProcessor",
    "MetadataManager",
]
