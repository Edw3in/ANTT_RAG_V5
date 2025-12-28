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
    # Startup
    print("🚀 Iniciando ANTT RAG API...")
    
    try:
        # Carrega configuração
        config = get_config()
        print(f"✅ Configuração carregada (ambiente: {config.environment})")
        
        # Valida ambiente
        config.validate_environment()
        print("✅ Ambiente validado")
        
        # Inicializa componentes críticos
        from src.core import get_embedding_manager, get_llm_manager
        
        _ = get_embedding_manager()
        print("✅ Sistema de embeddings inicializado")
        
        _ = get_llm_manager()
        print("✅ Sistema de LLM inicializado")
        
        print(f"✅ API pronta na versão {__version__}")
    
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        raise
    
    yield
    
    # Shutdown
    print("🛑 Encerrando ANTT RAG API...")


# Cria aplicação FastAPI
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
    
    ## Autenticação
    
    Atualmente sem autenticação. Em produção, configure API keys via middleware.
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Configuração de CORS
config = get_config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)

# Middleware de compressão
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Middleware de logging e métricas
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
    
    # Adiciona header de tempo de processamento
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log de acesso
    auditor = get_audit_logger()
    auditor.log_access(
        endpoint=request.url.path,
        method=request.method,
        status_code=response.status_code,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent")
    )
    
    return response


# Exception handlers
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
    auditor = get_audit_logger()
    auditor.log_error(
        error_type=type(exc).__name__,
        error_message=str(exc),
        context={
            "endpoint": request.url.path,
            "method": request.method
        }
    )
    
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


# Registra rotas
app.include_router(system_router)
app.include_router(answer_router)
app.include_router(query_router)
app.include_router(ingest_router)


# Eventos adicionais
@app.on_event("startup")
async def startup_event():
    """
    Evento executado no startup (além do lifespan)
    """
    print(f"📡 Servidor iniciado em {datetime.now().isoformat()}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento executado no shutdown (além do lifespan)
    """
    print(f"📡 Servidor encerrado em {datetime.now().isoformat()}")


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers if not config.api.reload else 1,
        log_level=config.logging.level.lower()
    )
