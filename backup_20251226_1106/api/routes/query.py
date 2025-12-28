"""
Rota de Query - Busca e Retrieval
Endpoint para busca de documentos sem geração de resposta
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional

from api.schemas import QueryRequest, QueryResponse, DocumentResponse
from src.services import HybridRetriever, RetrievalStrategy
from src.utils.audit_logger import get_audit_logger


router = APIRouter(prefix="/api/v1", tags=["Query"])

# Dependência: retriever (singleton)
_retriever: Optional[HybridRetriever] = None

def get_retriever() -> HybridRetriever:
    """Retorna instância do retriever"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        # Inicializa BM25 se necessário
        try:
            _retriever.initialize_bm25()
        except Exception as e:
            print(f"⚠️  Aviso: BM25 não inicializado: {e}")
    return _retriever


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Busca documentos relevantes",
    description="Recupera documentos relevantes para uma query sem gerar resposta",
    response_description="Lista de documentos recuperados com scores"
)
async def query_documents(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever)
):
    """
    Busca documentos relevantes para uma query.
    
    Útil para:
    - Exploração de documentos disponíveis
    - Validação de retrieval antes de gerar resposta
    - Integração com outros sistemas
    
    **Estratégias disponíveis:**
    - vector_only: Busca semântica pura (embeddings)
    - bm25_only: Busca lexical pura (BM25)
    - hybrid: Combinação de busca semântica e lexical
    - hybrid_rerank: Híbrido com reranking (recomendado)
    """
    try:
        # Mapeia estratégia
        strategy_map = {
            "vector_only": RetrievalStrategy.VECTOR_ONLY,
            "bm25_only": RetrievalStrategy.BM25_ONLY,
            "hybrid": RetrievalStrategy.HYBRID,
            "hybrid_rerank": RetrievalStrategy.HYBRID_RERANK,
        }
        
        retrieval_strategy = strategy_map.get(
            request.estrategia,
            RetrievalStrategy.HYBRID
        )
        
        # Executa retrieval
        result = retriever.retrieve(
            query=request.query,
            k=request.k,
            strategy=retrieval_strategy,
            filter_kwargs=request.filtros or {}
        )
        
        # Converte documentos para formato de resposta
        documents_response = [
            DocumentResponse(
                fonte=doc.metadata.get("source", "Desconhecido"),
                pagina=doc.metadata.get("page"),
                tipo=doc.metadata.get("tipo", "Normativo"),
                conteudo=doc.page_content[:1000],  # Limita tamanho
                score=score,
                metadata={
                    k: v for k, v in doc.metadata.items()
                    if k not in ["source", "page", "tipo"]
                }
            )
            for doc, score in zip(result.documents, result.scores)
        ]
        
        # Log da consulta
        auditor = get_audit_logger()
        auditor.log_query(
            query=request.query,
            results_count=len(documents_response),
            retrieval_strategy=request.estrategia,
            processing_time=result.processing_time
        )
        
        return QueryResponse(
            query=request.query,
            documentos=documents_response,
            total_encontrados=len(documents_response),
            estrategia=request.estrategia,
            tempo_processamento=result.processing_time
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Erro de validação: {str(e)}"
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Vectorstore não encontrado: {str(e)}"
        )
    
    except Exception as e:
        auditor = get_audit_logger()
        auditor.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={"endpoint": "/api/v1/query", "query": request.query}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao buscar documentos: {str(e)}"
        )


@router.get(
    "/query/stats",
    summary="Estatísticas do retriever",
    description="Retorna estatísticas e configuração do sistema de retrieval"
)
async def get_query_stats(
    retriever: HybridRetriever = Depends(get_retriever)
):
    """
    Retorna estatísticas do sistema de retrieval.
    """
    try:
        stats = retriever.get_stats()
        return {
            "status": "ok",
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )
