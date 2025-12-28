"""
Schemas de Request da API
Modelos Pydantic para validação de entrada
"""

from typing import Optional, List
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator
from enum import Enum


class RetrievalStrategyEnum(str, Enum):
    """Estratégias de retrieval disponíveis"""
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


class AnswerRequest(BaseModel):
    """Request para geração de resposta"""
    pergunta: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Pergunta a ser respondida",
        examples=["Qual o prazo para renovação de acreditação de verificador?"]
    )
    k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de documentos a recuperar"
    )
    
    # Campo oficial (novo)
    estrategia: Optional[RetrievalStrategyEnum] = Field(
        default=RetrievalStrategyEnum.HYBRID_RERANK,
        description="Estratégia de retrieval a utilizar"
    )
    
    # Campos legados (para compatibilidade com seus curls atuais)
    method: Optional[str] = Field(
        default=None, 
        description="LEGADO: vector|bm25|hybrid"
    )
    use_llm: Optional[bool] = Field(
        default=True, 
        description="LEGADO: usar LLM"
    )
    
    incluir_raciocinio: Optional[bool] = Field(
        default=False,
        description="Se deve incluir raciocínio na resposta"
    )
    filtros: Optional[dict] = Field(
        default=None,
        description="Filtros adicionais (status, tipo, precedência)"
    )
    
    @field_validator('pergunta')
    @classmethod
    def validate_pergunta(cls, v: str) -> str:
        """Valida e normaliza pergunta"""
        # Remove espaços excessivos
        v = " ".join(v.split()).strip()
        
        if not v:
            raise ValueError("Pergunta não pode ser vazia")
        
        return v
    
    @model_validator(mode='after')
    def normalize_strategy(self):
        """
        Normaliza estratégia:
        - Se estrategia não foi passada mas method foi, mapeia method -> estrategia
        - Garante compatibilidade com API legada
        """
        # Se estrategia já veio definida, mantém
        if self.estrategia and self.estrategia != RetrievalStrategyEnum.HYBRID_RERANK:
            return self
        
        # Se method legado foi passado, mapeia para estrategia
        if self.method:
            method_map = {
                "vector": RetrievalStrategyEnum.VECTOR_ONLY,
                "bm25": RetrievalStrategyEnum.BM25_ONLY,
                "hybrid": RetrievalStrategyEnum.HYBRID,
            }
            
            mapped_strategy = method_map.get(self.method.lower())
            if mapped_strategy:
                self.estrategia = mapped_strategy
        
        return self
    
    model_config = ConfigDict(
        populate_by_name=True,
        extra="ignore",
        json_schema_extra={
            "example": {
                "pergunta": "Qual o prazo para renovação de acreditação?",
                "k": 5,
                "estrategia": "hybrid_rerank",
                "incluir_raciocinio": False
            }
        }
    )


class QueryRequest(BaseModel):
    """Request para busca/retrieval"""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Texto de busca"
    )
    k: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Número de resultados"
    )
    estrategia: Optional[RetrievalStrategyEnum] = Field(
        default=RetrievalStrategyEnum.HYBRID,
        description="Estratégia de retrieval"
    )
    filtros: Optional[dict] = Field(
        default=None,
        description="Filtros de metadados"
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Valida e normaliza query"""
        v = " ".join(v.split()).strip()
        
        if not v:
            raise ValueError("Query não pode ser vazia")
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "acreditação verificador",
                "k": 10,
                "estrategia": "hybrid"
            }
        }
    )


class IngestRequest(BaseModel):
    """Request para ingestão de documentos"""
    force_reprocess: Optional[bool] = Field(
        default=False,
        description="Forçar reprocessamento de documentos já indexados"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "force_reprocess": False
            }
        }
    )


class DocumentMetadataRequest(BaseModel):
    """Request para atualização de metadados de documento"""
    doc_id: str = Field(..., description="ID do documento")
    status: Optional[str] = Field(None, description="Status do documento")
    precedencia: Optional[int] = Field(
        None, 
        ge=1, 
        le=100, 
        description="Nível de precedência"
    )
    tipo: Optional[str] = Field(None, description="Tipo de documento")
    vigencia_inicio: Optional[str] = Field(
        None, 
        description="Data de início de vigência (formato: YYYY-MM-DD)"
    )
    vigencia_fim: Optional[str] = Field(
        None, 
        description="Data de fim de vigência (formato: YYYY-MM-DD)"
    )
    tags: Optional[List[str]] = Field(None, description="Tags do documento")
    
    @field_validator('doc_id')
    @classmethod
    def validate_doc_id(cls, v: str) -> str:
        """Valida doc_id"""
        v = v.strip()
        if not v:
            raise ValueError("doc_id não pode ser vazio")
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "doc_id": "resolucao_123",
                "status": "Vigente",
                "precedencia": 3,
                "tipo": "Resolução"
            }
        }
    )


class HealthCheckRequest(BaseModel):
    """Request para health check detalhado"""
    include_stats: Optional[bool] = Field(
        default=False,
        description="Incluir estatísticas detalhadas"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "include_stats": True
            }
        }
    )