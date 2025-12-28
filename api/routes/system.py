"""
System Routes - Endpoints de Sistema e Monitoramento
"""

from __future__ import annotations

import os
import platform
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, status

from src.core.config import get_config

router = APIRouter(prefix="/system", tags=["System"])


def _safe_env(var: str) -> bool:
    """Retorna apenas se a variável existe (não expõe valor)."""
    return bool(os.getenv(var))


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
def health() -> Dict[str, Any]:
    """
    Health check básico
    
    Retorna status do serviço e timestamp atual
    """
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "service": "ANTT RAG v4",
    }


# ============================================================================
# STATUS COMPLETO DO SISTEMA (INCLUINDO BM25)
# ============================================================================

@router.get("/status")
def system_status() -> Dict[str, Any]:
    """
    Status completo do sistema incluindo retriever e BM25
    
    Retorna informações detalhadas sobre:
    - Status geral do sistema
    - Configuração do retriever
    - BM25 (inicializado, corpus size, etc.)
    - Estatísticas de uso
    """
    try:
        from src.core.retriever import get_retriever
        
        # Obtém retriever singleton
        retriever = get_retriever()
        
        # Obtém estatísticas do retriever
        retriever_stats = retriever.get_stats() if hasattr(retriever, 'get_stats') else {}
        
        # Informações adicionais sobre BM25
        bm25_info = {
            "initialized": hasattr(retriever, 'bm25') and retriever.bm25 is not None,
            "corpus_size": 0,
            "available": False
        }
        
        if bm25_info["initialized"]:
            bm25_info["available"] = True
            try:
                # Tenta obter tamanho do corpus
                if hasattr(retriever.bm25, 'corpus_size'):
                    bm25_info["corpus_size"] = retriever.bm25.corpus_size
                elif hasattr(retriever.bm25, 'doc_freqs'):
                    bm25_info["corpus_size"] = len(retriever.bm25.doc_freqs)
                else:
                    # Fallback: conta documentos do vectorstore
                    # tamanho do corpus BM25 (a fonte mais correta para BM25)
                    bm25_info["corpus_size"] = len(getattr(retriever, "bm25_corpus", []) or [])

            except Exception as e:
                bm25_info["corpus_size_error"] = str(e)
        
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "ANTT RAG v4",
            "components": {
                "retriever": {
                    "available": True,
                    "stats": retriever_stats,
                    "bm25": bm25_info
                }
            }
        }
    
    except ImportError as e:
        return {
            "status": "degraded",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "service": "ANTT RAG v4",
            "components": {
                "retriever": {
                    "available": False,
                    "error": f"Módulo retriever não encontrado: {str(e)}",
                    "bm25": {
                        "initialized": False,
                        "available": False
                    }
                }
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter status do sistema: {str(e)}"
        )


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

@router.get("/config")
def config_view() -> Dict[str, Any]:
    """
    Visualização da configuração do sistema
    
    Expõe configurações não sensíveis do sistema
    """
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


# ============================================================================
# INFORMAÇÕES DE GPU
# ============================================================================

@router.get("/gpu")
def gpu_info() -> Dict[str, Any]:
    """
    Informações sobre GPU disponível
    
    Retorna detalhes sobre PyTorch e CUDA se disponíveis
    """
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
            
            # Informações adicionais de GPU
            info["cuda_version"] = _torch.version.cuda
            info["devices"] = [
                {
                    "id": i,
                    "name": _torch.cuda.get_device_name(i),
                    "memory_total": f"{_torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB",
                }
                for i in range(info["device_count"])
            ]
    except Exception as e:
        info["error"] = str(e)

    return info


# ============================================================================
# INFORMAÇÕES DE RUNTIME
# ============================================================================

@router.get("/runtime")
def runtime() -> Dict[str, Any]:
    """
    Informações sobre o ambiente de execução
    
    Retorna versão do Python, plataforma e arquitetura
    """
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "N/A",
    }


# ============================================================================
# ESTATÍSTICAS DE RETRIEVAL
# ============================================================================

@router.get("/retrieval/stats")
def retrieval_stats() -> Dict[str, Any]:
    """
    Estatísticas detalhadas do sistema de retrieval
    
    Retorna métricas sobre:
    - Vectorstore (documentos, chunks, etc.)
    - BM25 (corpus, vocabulário, etc.)
    - Cache e performance
    """
    try:
        from src.core.retriever import get_retriever
        
        # Retriever
        retriever = get_retriever()
        
        # Estatísticas básicas do vectorstore
        stats = {
            "vectorstore": {
                "initialized": retriever.vectorstore is not None,
            },
            "bm25": {
                "initialized": hasattr(retriever, 'bm25') and retriever.bm25 is not None,
            }
        }
        
        # Conta documentos do vectorstore
        if retriever.vectorstore:
            try:
                all_data = retriever.vectorstore.get()
                doc_count = len(all_data.get("documents", []))
                stats["vectorstore"]["total_documents"] = doc_count
                stats["vectorstore"]["index_type"] = retriever.vectorstore.__class__.__name__
            except Exception as e:
                stats["vectorstore"]["error"] = str(e)
        
        # Estatísticas do BM25
        if stats["bm25"]["initialized"]:
            try:
                bm25 = retriever.bm25
                
                # Corpus size
                stats["bm25"]["corpus_size"] = len(retriever.bm25_corpus)
                
                # Vocabulário
                if hasattr(bm25, 'doc_freqs'):
                    stats["bm25"]["vocabulary_size"] = len(bm25.doc_freqs)
                
                # Parâmetros do BM25
                if hasattr(bm25, 'k1'):
                    stats["bm25"]["parameters"] = {
                        "k1": bm25.k1,
                        "b": bm25.b,
                        "epsilon": getattr(bm25, 'epsilon', 0.25)
                    }
                
                # Average document length
                if hasattr(bm25, 'avgdl'):
                    stats["bm25"]["avg_doc_length"] = bm25.avgdl
                
            except Exception as e:
                stats["bm25"]["error"] = str(e)
        
        # Estatísticas do retriever (se disponível)
        if hasattr(retriever, 'get_stats'):
            stats["retriever"] = retriever.get_stats()
        
        # Reranker info
        stats["reranker"] = {
            "enabled": retriever.enable_reranker,
            "available": retriever.reranker is not None,
            "model": retriever.config.models.reranker_model if retriever.enable_reranker else None
        }
        
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stats": stats
        }
    
    except FileNotFoundError as e:
        return {
            "status": "not_ready",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": "Vectorstore não encontrado",
            "detail": str(e),
            "hint": "Execute a ingestão de documentos primeiro (POST /api/v1/ingest)"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao obter estatísticas: {str(e)}"
        )


# ============================================================================
# VALIDAÇÃO DE BM25
# ============================================================================

@router.get("/retrieval/bm25/validate")
def validate_bm25() -> Dict[str, Any]:
    """
    Valida se BM25 está corretamente inicializado e funcionando
    
    Executa testes básicos para garantir que BM25 está operacional
    """
    try:
        from src.core.retriever import get_retriever
        
        retriever = get_retriever()
        
        # Verifica inicialização
        if not hasattr(retriever, 'bm25') or retriever.bm25 is None:
            return {
                "status": "not_initialized",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "bm25_initialized": False,
                "message": "BM25 não foi inicializado. Execute initialize_bm25() no startup.",
            }
        
        # Testa busca
        try:
            # Busca de teste
            test_query = "teste validação sistema"
            test_results = retriever.bm25.get_scores(test_query.split())
            
            return {
                "status": "ok",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "bm25_initialized": True,
                "bm25_operational": True,
                "test_query": test_query,
                "test_results_count": len(test_results),
                "message": "BM25 está funcionando corretamente"
            }
        
        except Exception as test_error:
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "bm25_initialized": True,
                "bm25_operational": False,
                "error": str(test_error),
                "message": "BM25 inicializado mas não está operacional"
            }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro na validação do BM25: {str(e)}"
        )