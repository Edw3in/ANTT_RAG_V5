"""
Schemas de Resposta da API - VERSÃO DEFINITIVA
Modelos Pydantic para formatação de saída com normalização de score e compatibilidade V2
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
import math


class EvidenceResponse(BaseModel):
    """Evidência que fundamenta a resposta"""
    fonte: str = Field(..., description="Nome do documento fonte")
    pagina: Optional[int] = Field(None, description="Número da página")
    tipo: str = Field(..., description="Tipo de documento")
    trecho: str = Field(..., description="Trecho relevante do documento")
    score: float = Field(..., description="Score de relevância")
    precedencia: Optional[int] = Field(None, description="Nível de precedência normativa")

    @field_validator("score", mode="before")
    @classmethod
    def normalize_score(cls, v: Any) -> float:
        """Normaliza score para intervalo [0, 1]"""
        try:
            val = float(v)
            if val < 0:
                return round(1.0 / (1.0 + math.exp(-val)), 4)
            if 0.0 <= val <= 1.0:
                return round(val, 4)
            return round(1.0 / (1.0 + math.log(val + 1.0)), 4)
        except (ValueError, TypeError):
            return 0.0

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fonte": "Resolução ANTT 5956/2024",
                "pagina": 3,
                "tipo": "Resolução",
                "trecho": "O prazo para renovação é de 30 dias...",
                "score": 0.92,
                "precedencia": 3,
            }
        }
    )


class DocumentResponse(BaseModel):
    """Documento recuperado na busca"""
    fonte: str
    pagina: Optional[int]
    tipo: str
    conteudo: str
    score: float
    metadata: Optional[Dict[str, Any]]
    
    @field_validator('score', mode='before')
    @classmethod
    def normalize_score(cls, v: Any) -> float:
        try:
            val = float(v)
            if val < 0: return round(1 / (1 + math.exp(-val)), 4)
            if val > 1: return 1.0
            return round(val, 4)
        except:
            return 0.0


class AnswerResponse(BaseModel):
    """Resposta completa para uma pergunta"""
    pergunta: str = Field(..., description="Pergunta original")
    resposta: str = Field(..., description="Resposta gerada")
    confiabilidade: str = Field(..., description="Nível de confiabilidade")
    evidencias: List[EvidenceResponse] = Field(default_factory=list)
    raciocinio: Optional[str] = Field(default=None)
    avisos: Optional[List[str]] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    tempo_processamento: Optional[float] = Field(default=None)


class QueryResponse(BaseModel):
    """Resposta de busca/retrieval"""
    query: str = Field(..., description="Query original")
    documentos: List[DocumentResponse] = Field(default_factory=list)
    total_encontrados: int = Field(0)
    estrategia: str = Field("unknown")
    tempo_processamento: float = Field(0.0)


class DocumentMetadataResponse(BaseModel):
    """Metadados de um documento"""
    doc_id: str
    title: str
    source_path: str
    sha256: str
    status: str
    precedencia: int
    tipo: str
    total_pages: Optional[int] = None
    vigencia_inicio: Optional[str] = None
    vigencia_fim: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    tags: Optional[List[str]] = None


class IngestResultResponse(BaseModel):
    """Resultado da ingestão de documentos"""
    total_arquivos: int
    sucesso: int
    ignorados: int
    erros: int
    total_chunks: int
    tempo_processamento: float
    detalhes: Optional[List[Dict[str, Any]]] = None


class HealthCheckResponse(BaseModel):
    """Resposta de health check"""
    status: str
    version: str
    timestamp: str
    components: Optional[Dict[str, str]] = None
    stats: Optional[Dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Resposta de erro padronizada"""
    error: str
    message: str
    detail: Optional[str] = None
    timestamp: str = ""
    path: Optional[str] = None


class StatsResponse(BaseModel):
    """Estatísticas do sistema"""
    total_documentos: int
    documentos_vigentes: int
    total_chunks: Optional[int] = None
    total_consultas: Optional[int] = None
    cache_hit_rate: Optional[str] = None
    uptime: Optional[str] = None