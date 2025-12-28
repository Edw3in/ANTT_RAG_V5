"""
Módulo Services - Serviços de alto nível do sistema ANTT RAG
"""

from src.services.retrieval_service import (
    HybridRetriever,
    RetrievalStrategy,
    RetrievalResult,
    DocumentFilter
)

from src.services.answer_service import (
    AnswerService,
    AnswerResult,
    Evidence,
    ConfidenceLevel
)

from src.services.ingest_service import (
    IngestService,
    IngestResult,
    DocumentProcessingResult,
    ProcessingStatus
)

__all__ = [
    # Retrieval
    "HybridRetriever",
    "RetrievalStrategy",
    "RetrievalResult",
    "DocumentFilter",
    
    # Answer
    "AnswerService",
    "AnswerResult",
    "Evidence",
    "ConfidenceLevel",
    
    # Ingest
    "IngestService",
    "IngestResult",
    "DocumentProcessingResult",
    "ProcessingStatus",
]
