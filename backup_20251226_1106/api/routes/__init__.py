"""
API Routes - Rotas da API FastAPI
"""

from api.routes.answer import router as answer_router
from api.routes.query import router as query_router
from api.routes.ingest import router as ingest_router
from api.routes.system import router as system_router

__all__ = [
    "answer_router",
    "query_router",
    "ingest_router",
    "system_router",
]
