"""
Rota de Ingest - Ingestão de Documentos
Endpoint para processamento e indexação de documentos
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File
from typing import Optional, List
import shutil
from pathlib import Path

from api.schemas import IngestRequest, IngestResultResponse
from src.services import IngestService
from src.core import get_config
from src.utils.audit_logger import get_audit_logger


router = APIRouter(prefix="/api/v1", tags=["Ingest"])

# Dependência: serviço de ingestão (singleton)
_ingest_service: Optional[IngestService] = None

def get_ingest_service() -> IngestService:
    """Retorna instância do serviço de ingestão"""
    global _ingest_service
    if _ingest_service is None:
        _ingest_service = IngestService()
    return _ingest_service


@router.post(
    "/ingest",
    response_model=IngestResultResponse,
    summary="Processa documentos da inbox",
    description="Processa todos os PDFs da pasta inbox e indexa no vectorstore",
    response_description="Resultado do processamento com estatísticas"
)
async def ingest_documents(
    request: IngestRequest,
    service: IngestService = Depends(get_ingest_service)
):
    """
    Processa documentos PDF da pasta inbox.
    
    O sistema:
    1. Lê todos os PDFs da pasta inbox
    2. Extrai texto e metadados
    3. Cria chunks otimizados
    4. Gera embeddings
    5. Indexa no vectorstore
    6. Move arquivos para pasta de processados ou rejeitados
    
    **Comportamento:**
    - Documentos já processados são ignorados (baseado em hash)
    - Use force_reprocess=true para reprocessar todos
    - Arquivos com erro são movidos para pasta de rejeitados
    """
    try:
        # Executa ingestão
        result = service.ingest_all(force_reprocess=request.force_reprocess)
        
        # Log da ingestão
        auditor = get_audit_logger()
        for doc_result in result.results:
            auditor.log_ingest(
                filename=doc_result.filename,
                status=doc_result.status.value,
                chunks_created=doc_result.chunks_created,
                pages_processed=doc_result.pages_processed,
                file_hash=doc_result.file_hash,
                error_message=doc_result.error_message
            )
        
        # Prepara resposta
        return IngestResultResponse(
            total_arquivos=result.total_files,
            sucesso=result.successful,
            ignorados=result.skipped,
            erros=result.errors,
            total_chunks=result.total_chunks,
            tempo_processamento=result.processing_time,
            detalhes=[
                {
                    "arquivo": r.filename,
                    "status": r.status.value,
                    "chunks": r.chunks_created,
                    "paginas": r.pages_processed,
                    "erro": r.error_message
                }
                for r in result.results
            ]
        )
    
    except Exception as e:
        auditor = get_audit_logger()
        auditor.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={"endpoint": "/api/v1/ingest"}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar documentos: {str(e)}"
        )


@router.post(
    "/ingest/upload",
    summary="Upload e processa documento",
    description="Faz upload de um PDF e processa imediatamente"
)
async def upload_and_ingest(
    file: UploadFile = File(..., description="Arquivo PDF para upload"),
    service: IngestService = Depends(get_ingest_service)
):
    """
    Faz upload de um documento PDF e processa imediatamente.
    
    Útil para processar documentos individuais sem usar a pasta inbox.
    """
    # Valida tipo de arquivo
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Apenas arquivos PDF são aceitos"
        )
    
    try:
        config = get_config()
        inbox_path = config.paths.bcp_inbox
        
        # Salva arquivo na inbox
        file_path = inbox_path / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Processa documento
        result = service.process_document(file_path, force_reprocess=False)
        
        # Log
        auditor = get_audit_logger()
        auditor.log_ingest(
            filename=result.filename,
            status=result.status.value,
            chunks_created=result.chunks_created,
            pages_processed=result.pages_processed,
            file_hash=result.file_hash,
            error_message=result.error_message
        )
        
        return {
            "status": "ok",
            "arquivo": result.filename,
            "resultado": result.status.value,
            "chunks_criados": result.chunks_created,
            "paginas_processadas": result.pages_processed,
            "tempo_processamento": result.processing_time
        }
    
    except Exception as e:
        # Remove arquivo se houver erro
        if file_path.exists():
            file_path.unlink()
        
        auditor = get_audit_logger()
        auditor.log_error(
            error_type=type(e).__name__,
            error_message=str(e),
            context={"endpoint": "/api/v1/ingest/upload", "filename": file.filename}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar arquivo: {str(e)}"
        )


@router.get(
    "/ingest/stats",
    summary="Estatísticas de ingestão",
    description="Retorna estatísticas do sistema de ingestão"
)
async def get_ingest_stats(
    service: IngestService = Depends(get_ingest_service)
):
    """
    Retorna estatísticas do sistema de ingestão.
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


@router.delete(
    "/ingest/clear-inbox",
    summary="Limpa pasta inbox",
    description="Remove todos os arquivos da pasta inbox (use com cuidado!)"
)
async def clear_inbox(
    confirm: bool = False,
    service: IngestService = Depends(get_ingest_service)
):
    """
    Limpa todos os arquivos da pasta inbox.
    
    **ATENÇÃO:** Esta operação é irreversível!
    Use confirm=true para confirmar.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Confirmação necessária. Use confirm=true"
        )
    
    try:
        config = get_config()
        inbox_path = config.paths.bcp_inbox
        
        files = list(inbox_path.glob("*.pdf"))
        count = len(files)
        
        for file in files:
            file.unlink()
        
        auditor = get_audit_logger()
        auditor.log_error(
            error_type="InboxCleared",
            error_message=f"Inbox limpa: {count} arquivos removidos",
            context={"endpoint": "/api/v1/ingest/clear-inbox"}
        )
        
        return {
            "status": "ok",
            "message": f"{count} arquivos removidos da inbox"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao limpar inbox: {str(e)}"
        )
