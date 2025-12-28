"""
Rota de Answer - Geração de Respostas
Endpoint principal para perguntas e respostas fundamentadas
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional

from api.schemas import AnswerRequest, AnswerResponse, EvidenceResponse
from src.services.answer_service import AnswerService, RetrievalStrategy
from src.utils.audit_logger import get_audit_logger


router = APIRouter(prefix="/api/v1", tags=["Answer"])

# Dependência: serviço de resposta (singleton)
_answer_service: Optional[AnswerService] = None

def get_answer_service() -> AnswerService:
    """Retorna instância do serviço de resposta"""
    global _answer_service
    if _answer_service is None:
        _answer_service = AnswerService(enable_audit=True)
    return _answer_service


@router.post(
    "/answer",
    response_model=AnswerResponse,
    summary="Gera resposta fundamentada",
    description="Recebe uma pergunta e retorna resposta baseada em documentos normativos",
    response_description="Resposta fundamentada com evidências e nível de confiabilidade"
)
async def generate_answer(
    request: AnswerRequest,
    service: AnswerService = Depends(get_answer_service)
):
    """
    Gera resposta fundamentada para uma pergunta usando RAG.
    
    O sistema:
    1. Recupera documentos relevantes usando estratégia configurada
    2. Gera resposta usando LLM baseado no contexto recuperado
    3. Valida a resposta e calcula confiabilidade
    4. Registra a interação para auditoria
    
    **Níveis de Confiabilidade:**
    - ALTA: Resposta bem fundamentada com múltiplas evidências de alta qualidade
    - MÉDIA: Resposta fundamentada mas com evidências limitadas
    - BAIXA: Resposta com poucas evidências ou baixa relevância
    - INSUFICIENTE: Informação não encontrada ou insuficiente
    """
    try:
        # Mapeia estratégia do enum para o tipo do serviço
        strategy_map = {
            "vector_only": RetrievalStrategy.VECTOR_ONLY,
            "bm25_only": RetrievalStrategy.BM25_ONLY,
            "hybrid": RetrievalStrategy.HYBRID,
            "hybrid_rerank": RetrievalStrategy.HYBRID_RERANK,
        }
        
        retrieval_strategy = strategy_map.get(
            request.estrategia,
            RetrievalStrategy.HYBRID_RERANK
        )
        
        # Gera resposta
        result = service.generate_answer(
            question=request.pergunta,
            k=request.k,
            retrieval_strategy=retrieval_strategy,
            include_reasoning=request.incluir_raciocinio,
            filter_kwargs=request.filtros
        )
        
        # Converte evidências para formato de resposta
        evidences_response = [
            EvidenceResponse(
                fonte=e.source,
                pagina=e.page,
                tipo=e.document_type,
                trecho=e.excerpt,
                score=e.score,
                precedencia=e.precedence
            )
            for e in result.evidences
        ]
        
        # Monta resposta
        return AnswerResponse(
            pergunta=result.question,
            resposta=result.answer,
            confiabilidade=result.confidence.value,
            evidencias=evidences_response,
            raciocinio=result.reasoning,
            avisos=result.warnings,
            metadata=result.metadata,
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
            detail=f"Serviço indisponível: {str(e)}"
        )
    
    except Exception as e:
        # Log do erro
        auditor = get_audit_logger()
        auditor.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={"endpoint": "/api/v1/answer", "question": request.pergunta}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno ao processar pergunta: {str(e)}"
        )


@router.post(
    "/answer/async",
    response_model=AnswerResponse,
    summary="Gera resposta fundamentada (assíncrono)",
    description="Versão assíncrona do endpoint de resposta para melhor performance"
)
async def generate_answer_async(
    request: AnswerRequest,
    service: AnswerService = Depends(get_answer_service)
):
    """
    Versão assíncrona da geração de resposta.
    Recomendado para alta concorrência.
    """
    try:
        strategy_map = {
            "vector_only": RetrievalStrategy.VECTOR_ONLY,
            "bm25_only": RetrievalStrategy.BM25_ONLY,
            "hybrid": RetrievalStrategy.HYBRID,
            "hybrid_rerank": RetrievalStrategy.HYBRID_RERANK,
        }
        
        retrieval_strategy = strategy_map.get(
            request.estrategia,
            RetrievalStrategy.HYBRID_RERANK
        )
        
        # Gera resposta assíncrona
        result = await service.agenerate_answer(
            question=request.pergunta,
            k=request.k,
            retrieval_strategy=retrieval_strategy
        )
        
        evidences_response = [
            EvidenceResponse(
                fonte=e.source,
                pagina=e.page,
                tipo=e.document_type,
                trecho=e.excerpt,
                score=e.score,
                precedencia=e.precedence
            )
            for e in result.evidences
        ]
        
        return AnswerResponse(
            pergunta=result.question,
            resposta=result.answer,
            confiabilidade=result.confidence.value,
            evidencias=evidences_response,
            raciocinio=result.reasoning,
            avisos=result.warnings,
            metadata=result.metadata,
            tempo_processamento=result.processing_time
        )
    
    except Exception as e:
        auditor = get_audit_logger()
        auditor.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={"endpoint": "/api/v1/answer/async", "question": request.pergunta}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno: {str(e)}"
        )


@router.get(
    "/answer/stats",
    summary="Estatísticas do serviço de resposta",
    description="Retorna estatísticas e métricas do serviço de geração de respostas"
)
async def get_answer_stats(
    service: AnswerService = Depends(get_answer_service)
):
    """
    Retorna estatísticas do serviço de resposta.
    """
    try:
        stats = service.get_stats()
        return {
            "status": "ok",
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )
