"""
ANTT RAG API - Aplicação FastAPI Principal
API completa para sistema de RAG da ANTT
"""

import time
from dotenv import load_dotenv
load_dotenv()
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime

from api.routes import answer_router, query_router, ingest_router, system_router
from api.schemas import ErrorResponse
from src.core import get_config
from src.utils.audit_logger import get_audit_logger
from src import __version__


# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia ciclo de vida da aplicação
    """
    # =========================================================================
    # STARTUP
    # =========================================================================
    print("🚀 Iniciando ANTT RAG API...")
    
    try:
        # Carrega configuração
        config = get_config()
        print(f"✅ Configuração carregada (ambiente: {config.environment})")
        
        # Valida ambiente
        config.validate_environment()
        print("✅ Ambiente validado")
        
        # =====================================================================
        # Inicializa componentes críticos
        # =====================================================================
        from src.core import get_embedding_manager, get_llm_manager
        
        # Embeddings
        _ = get_embedding_manager()
        print("✅ Sistema de embeddings inicializado")
        
        # LLM
        _ = get_llm_manager()
        print("✅ Sistema de LLM inicializado")
        
        # =====================================================================
        # Inicializa Retrieval (vectorstore + BM25)
        # CRÍTICO: Usa singleton get_retriever() para garantir instância única
        # =====================================================================
        try:
            from src.services.retrieval_service import get_retriever
            
            # Obtém singleton do retriever
            retriever = get_retriever()
            
            # Inicializa BM25 a partir do vectorstore
            retriever.initialize_bm25()
            print("✅ BM25 inicializado e pronto para estratégias HYBRID/HYBRID_RERANK")
            
            # Valida que BM25 tem documentos
            if hasattr(retriever, 'bm25') and retriever.bm25 is not None:
                try:
                    from src.core import get_vectorstore_manager
                    vs_manager = get_vectorstore_manager()
                    doc_count = len(vs_manager.vectorstore.docstore._dict)
                    print(f"   📚 {doc_count} documentos disponíveis para retrieval")
                except Exception:
                    print("   📚 BM25 inicializado (contagem de docs indisponível)")
            else:
                print("   ⚠️  BM25 não inicializado - verifique se há documentos indexados")
            
        except FileNotFoundError as e:
            print(f"⚠️  Aviso: Vectorstore não encontrado - {e}")
            print("   ℹ️  Execute a ingestão de documentos primeiro")
            print("   ℹ️  Endpoint: POST /api/v1/ingest")
        
        except ImportError as e:
            print(f"⚠️  Aviso: Módulo retriever não encontrado - {e}")
            print("   ℹ️  Verifique se src/core/retriever.py existe")
        
        except Exception as e:
            print(f"⚠️  Aviso ao inicializar retrieval: {e}")
            print("   ℹ️  Sistema continuará, mas pode haver problemas em buscas")
        
        # =====================================================================
        # Inicialização concluída
        # =====================================================================
        print(f"✅ API pronta na versão {__version__}")
        print(f"📡 Servidor rodando em http://{config.api.host}:{config.api.port}")
        print(f"📖 Documentação em http://{config.api.host}:{config.api.port}/docs")
    
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        raise
    
    yield
    
    # =========================================================================
    # SHUTDOWN
    # =========================================================================
    print("🛑 Encerrando ANTT RAG API...")
    print("   💾 Salvando estado...")
    print("   🔌 Fechando conexões...")
    print("✅ Shutdown concluído")


# ============================================================================
# CRIAÇÃO DA APLICAÇÃO FASTAPI
# ============================================================================
app = FastAPI(
    title="ANTT RAG API",
    description="""
    # API de Recuperação e Geração Aumentada para ANTT
    
    Sistema completo de RAG (Retrieval-Augmented Generation) para consulta de normativos 
    e documentos da Agência Nacional de Transportes Terrestres.
    
    ## Funcionalidades Principais
    
    * **Answer**: Gera respostas fundamentadas para perguntas
    * **Query**: Busca documentos relevantes sem gerar resposta
    * **Ingest**: Processa e indexa novos documentos
    * **System**: Monitoramento e informações do sistema
    
    ## Características
    
    * Busca híbrida (vetorial + BM25) com reranking
    * Validação de respostas e cálculo de confiabilidade
    * Auditoria completa de todas as interações
    * Suporte a múltiplos provedores de LLM
    * Cache inteligente de embeddings
    * Filtros de governança e precedência normativa
    
    ## Estratégias de Retrieval
    
    * **vector_only**: Busca puramente vetorial (semântica)
    * **bm25_only**: Busca léxica (keywords)
    * **hybrid**: Combinação vetorial + BM25
    * **hybrid_rerank**: Hybrid + reranking (recomendado)
    
    ## Autenticação
    
    Atualmente sem autenticação. Em produção, configure API keys via middleware.
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={
        "name": "Equipe ANTT",
        "email": "suporte@antt.gov.br"
    },
    license_info={
        "name": "Uso Interno ANTT"
    }
)

# ============================================================================
# CONFIGURAÇÃO DE MIDDLEWARES
# ============================================================================

# Carrega configuração
config = get_config()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)

# Compressão GZIP
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ============================================================================
# MIDDLEWARE DE LOGGING E MÉTRICAS
# ============================================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware para logging de requisições e cálculo de latência
    """
    start_time = time.time()
    
    # Processa requisição
    response = await call_next(request)
    
    # Calcula latência
    process_time = time.time() - start_time
    
    # Adiciona headers informativos
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    response.headers["X-API-Version"] = __version__
    
    # Log de acesso
    try:
        auditor = get_audit_logger()
        auditor.log_access(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            process_time=process_time
        )
    except Exception as e:
        # Não falha a requisição por erro no log
        print(f"⚠️  Erro ao registrar log: {e}")
    
    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handler para erros de validação do Pydantic
    """
    errors = []
    for error in exc.errors():
        errors.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="ValidationError",
            message="Erro de validação nos dados da requisição",
            detail=str(errors),
            timestamp=datetime.now().isoformat(),
            path=request.url.path
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handler geral para exceções não tratadas
    """
    # Log do erro
    try:
        auditor = get_audit_logger()
        auditor.log_error(
            error_type=type(exc).__name__,
            error_message=str(exc),
            context={
                "endpoint": request.url.path,
                "method": request.method,
                "client": request.client.host if request.client else None
            }
        )
    except Exception as log_error:
        print(f"⚠️  Erro ao registrar log de erro: {log_error}")
    
    # Resposta ao cliente
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=type(exc).__name__,
            message="Erro interno do servidor",
            detail=str(exc) if config.debug else "Entre em contato com o suporte",
            timestamp=datetime.now().isoformat(),
            path=request.url.path
        ).dict()
    )


# ============================================================================
# REGISTRO DE ROTAS
# ============================================================================

# Ordem de registro (mais específico primeiro)
app.include_router(system_router)    # /api/v1/health, /api/v1/info
app.include_router(answer_router)    # /api/v1/answer
app.include_router(query_router)     # /api/v1/query
app.include_router(ingest_router)    # /api/v1/ingest


# ============================================================================
# EVENTOS ADICIONAIS (LEGACY - considere usar apenas lifespan)
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """
    Evento executado no startup (além do lifespan)
    """
    print(f"📡 Servidor iniciado em {datetime.now().isoformat()}")
    print(f"🌍 Ambiente: {config.environment}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento executado no shutdown (além do lifespan)
    """
    print(f"📡 Servidor encerrado em {datetime.now().isoformat()}")


# ============================================================================
# ROTA RAIZ (HEALTHCHECK SIMPLES)
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """
    Rota raiz - healthcheck simples
    """
    return {
        "service": "ANTT RAG API",
        "version": __version__,
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "docs": "/docs",
        "health": "/api/v1/health"
    }


# ============================================================================
# PONTO DE ENTRADA
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    
    print("="*80)
    print("🚀 Iniciando ANTT RAG API via uvicorn")
    print("="*80)
    
    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers if not config.api.reload else 1,
        log_level=config.logging.level.lower(),
        access_log=True
    )