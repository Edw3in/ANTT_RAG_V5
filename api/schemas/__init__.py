"""
Schemas da API - Modelos Pydantic para Request/Response
"""

from api.schemas.requests import AnswerRequest, QueryRequest, IngestRequest
from api.schemas.responses import (
    AnswerResponse,
    QueryResponse,
    IngestResultResponse,
    EvidenceResponse,
    DocumentResponse,
    HealthCheckResponse,
    StatsResponse,
    ErrorResponse,
)

__all__ = [
    # Requests
    "AnswerRequest",
    "QueryRequest",
    "IngestRequest",
    # Responses
    "AnswerResponse",
    "QueryResponse",
    "IngestResultResponse",
    "EvidenceResponse",
    "DocumentResponse",
    "HealthCheckResponse",
    "StatsResponse",
    "ErrorResponse",
]
