"""
Sistema de Auditoria e Logging
Registra todas as interações do sistema para compliance e análise.
"""

import json
import gzip
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from src.core.config import get_config


class EventType(str, Enum):
    """Tipos de eventos auditáveis"""
    QUERY = "query"
    ANSWER = "answer"
    INGEST = "ingest"
    ERROR = "error"
    ACCESS = "access"
    SYSTEM = "system"


@dataclass
class AuditEvent:
    """Evento de auditoria estruturado"""
    timestamp: str
    event_type: EventType
    event_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class AuditLogger:
    """
    Logger de auditoria com suporte a JSONL e compressão
    """
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.config = get_config()
        self.log_dir = log_dir or self.config.paths.auditoria_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            "total_events": 0,
            "events_by_type": {},
            "last_event_time": None
        }
    
    def log_interaction(
        self,
        question: str,
        answer: str,
        confidence: str,
        evidences: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """
        Registra uma interação de pergunta-resposta
        """
        event = AuditEvent(
            timestamp=self._get_timestamp(),
            event_type=EventType.ANSWER,
            event_id=self._generate_event_id(),
            user_id=user_id,
            session_id=session_id,
            data={
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "evidences": evidences,
                "num_evidences": len(evidences)
            },
            metadata=metadata or {}
        )
        
        self._write_event(event)
    
    def log_query(
        self,
        query: str,
        results_count: int,
        retrieval_strategy: str,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """
        Registra uma consulta de retrieval
        """
        event = AuditEvent(
            timestamp=self._get_timestamp(),
            event_type=EventType.QUERY,
            event_id=self._generate_event_id(),
            user_id=user_id,
            data={
                "query": query,
                "results_count": results_count,
                "retrieval_strategy": retrieval_strategy,
                "processing_time": processing_time
            },
            metadata=metadata or {}
        )
        
        self._write_event(event)
    
    def log_ingest(
        self,
        filename: str,
        status: str,
        chunks_created: int,
        pages_processed: int,
        file_hash: str,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Registra ingestão de documento
        """
        event = AuditEvent(
            timestamp=self._get_timestamp(),
            event_type=EventType.INGEST,
            event_id=self._generate_event_id(),
            data={
                "filename": filename,
                "status": status,
                "chunks_created": chunks_created,
                "pages_processed": pages_processed,
                "file_hash": file_hash,
                "error_message": error_message
            },
            metadata=metadata or {}
        )
        
        self._write_event(event)
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ):
        """
        Registra erro do sistema
        """
        event = AuditEvent(
            timestamp=self._get_timestamp(),
            event_type=EventType.ERROR,
            event_id=self._generate_event_id(),
            user_id=user_id,
            data={
                "error_type": error_type,
                "error_message": error_message,
                "context": context or {}
            }
        )
        
        self._write_event(event)
    
    def log_access(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Registra acesso à API
        """
        event = AuditEvent(
            timestamp=self._get_timestamp(),
            event_type=EventType.ACCESS,
            event_id=self._generate_event_id(),
            user_id=user_id,
            data={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "ip_address": ip_address,
                "user_agent": user_agent
            }
        )
        
        self._write_event(event)
    
    def _write_event(self, event: AuditEvent):
        """
        Escreve evento no arquivo de log
        """
        # Determina arquivo de log (um por dia)
        log_file = self._get_log_file()
        
        # Converte evento para dict
        event_dict = asdict(event)
        
        # Escreve linha JSONL
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event_dict, ensure_ascii=False) + '\n')
        
        # Atualiza estatísticas
        self._update_stats(event)
        
        # Compacta logs antigos se necessário
        self._compress_old_logs()
    
    def _get_log_file(self) -> Path:
        """
        Retorna caminho do arquivo de log atual
        """
        today = datetime.now().strftime('%Y-%m-%d')
        return self.log_dir / f"auditoria_{today}.jsonl"
    
    def _get_timestamp(self) -> str:
        """
        Retorna timestamp ISO 8601 em UTC
        """
        return datetime.now(timezone.utc).isoformat()
    
    def _generate_event_id(self) -> str:
        """
        Gera ID único para o evento
        """
        import uuid
        return str(uuid.uuid4())
    
    def _update_stats(self, event: AuditEvent):
        """
        Atualiza estatísticas internas
        """
        self.stats["total_events"] += 1
        self.stats["last_event_time"] = event.timestamp
        
        event_type = event.event_type.value
        if event_type not in self.stats["events_by_type"]:
            self.stats["events_by_type"][event_type] = 0
        self.stats["events_by_type"][event_type] += 1
    
    def _compress_old_logs(self, days_threshold: int = 7):
        """
        Compacta logs com mais de X dias
        """
        try:
            from datetime import timedelta
            threshold_date = datetime.now() - timedelta(days=days_threshold)
            
            for log_file in self.log_dir.glob("auditoria_*.jsonl"):
                # Extrai data do nome do arquivo
                date_str = log_file.stem.replace("auditoria_", "")
                try:
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    
                    if file_date < threshold_date:
                        # Compacta arquivo
                        gz_file = log_file.with_suffix('.jsonl.gz')
                        if not gz_file.exists():
                            with open(log_file, 'rb') as f_in:
                                with gzip.open(gz_file, 'wb') as f_out:
                                    f_out.writelines(f_in)
                            
                            # Remove original
                            log_file.unlink()
                            print(f"📦 Log compactado: {log_file.name}")
                
                except ValueError:
                    # Nome de arquivo não segue padrão esperado
                    pass
        
        except Exception as e:
            print(f"⚠️  Erro ao compactar logs: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de auditoria
        """
        # Conta arquivos de log
        log_files = list(self.log_dir.glob("auditoria_*.jsonl"))
        compressed_files = list(self.log_dir.glob("auditoria_*.jsonl.gz"))
        
        return {
            **self.stats,
            "active_log_files": len(log_files),
            "compressed_log_files": len(compressed_files),
            "log_directory": str(self.log_dir)
        }
    
    def query_logs(
        self,
        event_type: Optional[EventType] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Consulta logs com filtros
        """
        results = []
        
        # Determina arquivos a serem lidos
        log_files = sorted(self.log_dir.glob("auditoria_*.jsonl"), reverse=True)
        
        for log_file in log_files:
            if len(results) >= limit:
                break
            
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                        
                        try:
                            event = json.loads(line)
                            
                            # Aplica filtros
                            if event_type and event.get("event_type") != event_type.value:
                                continue
                            
                            if user_id and event.get("user_id") != user_id:
                                continue
                            
                            if start_date and event.get("timestamp", "") < start_date:
                                continue
                            
                            if end_date and event.get("timestamp", "") > end_date:
                                continue
                            
                            results.append(event)
                        
                        except json.JSONDecodeError:
                            continue
            
            except Exception as e:
                print(f"⚠️  Erro ao ler log {log_file}: {e}")
        
        return results


# Função de conveniência para logging rápido
_logger_instance = None

def get_audit_logger() -> AuditLogger:
    """Retorna instância singleton do audit logger"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AuditLogger()
    return _logger_instance


def log_jsonl(
    log_dir: str,
    event_type: str,
    question: str,
    response: Any,
    sources: List[Dict[str, Any]],
    confidence: Optional[str] = None,
):
    """
    Função de compatibilidade com código legado
    """
    logger = AuditLogger(log_dir=Path(log_dir))
    
    logger.log_interaction(
        question=question,
        answer=str(response),
        confidence=confidence or "N/A",
        evidences=sources
    )


if __name__ == "__main__":
    # Teste do audit logger
    print("🧪 Testando audit logger...")
    
    logger = AuditLogger()
    
    # Teste 1: Log de interação
    logger.log_interaction(
        question="Qual o prazo?",
        answer="O prazo é de 30 dias",
        confidence="ALTA",
        evidences=[{"source": "Resolução 123", "page": 1}],
        user_id="test_user"
    )
    
    # Teste 2: Log de query
    logger.log_query(
        query="prazo renovação",
        results_count=5,
        retrieval_strategy="hybrid_rerank",
        processing_time=0.5
    )
    
    # Estatísticas
    stats = logger.get_stats()
    print(f"📊 Stats: {stats}")
    
    # Consulta logs
    recent_logs = logger.query_logs(limit=10)
    print(f"📋 Logs recentes: {len(recent_logs)}")
