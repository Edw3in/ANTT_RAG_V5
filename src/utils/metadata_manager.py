"""
Gerenciador de Metadados
Gerencia metadados de documentos usando SQLite.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.core.config import get_config


class MetadataManager:
    """
    Gerencia metadados de documentos em banco SQLite
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        self.config = get_config()
        self.db_path = db_path or self.config.paths.metadata_db
        
        if not self.db_path:
            self.db_path = self.config.paths.base_dir / "data" / "metadata" / "documents.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _connect(self) -> sqlite3.Connection:
        """Cria conexão com o banco"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Inicializa estrutura do banco de dados"""
        with self._connect() as conn:
            # Tabela principal de documentos
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL UNIQUE,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'Vigente',
                    precedencia INTEGER DEFAULT 99,
                    tipo TEXT DEFAULT 'Normativo',
                    total_pages INTEGER,
                    vigencia_inicio TEXT,
                    vigencia_fim TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Índices para performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_tipo ON documents(tipo)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_precedencia ON documents(precedencia)")
            
            # Tabela de tags/categorias
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
                    UNIQUE(doc_id, tag)
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_doc ON document_tags(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tags_tag ON document_tags(tag)")
            
            # Tabela de histórico de alterações
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            """)
            
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_doc ON document_history(doc_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_timestamp ON document_history(timestamp)")
    
    def upsert_document(self, metadata: Dict[str, Any]) -> bool:
        """
        Insere ou atualiza documento
        """
        now = datetime.now().isoformat()
        
        # Garante campos obrigatórios
        required_fields = ["doc_id", "title", "source_path", "sha256"]
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Campo obrigatório ausente: {field}")
        
        # Adiciona timestamps
        if "created_at" not in metadata:
            metadata["created_at"] = now
        metadata["updated_at"] = now
        
        try:
            with self._connect() as conn:
                # Verifica se documento já existe
                existing = self.get_document(metadata["doc_id"])
                
                if existing:
                    # Update
                    set_clause = ", ".join([f"{k} = ?" for k in metadata.keys() if k != "doc_id"])
                    values = [v for k, v in metadata.items() if k != "doc_id"]
                    values.append(metadata["doc_id"])
                    
                    conn.execute(
                        f"UPDATE documents SET {set_clause} WHERE doc_id = ?",
                        values
                    )
                    
                    action = "updated"
                else:
                    # Insert
                    placeholders = ", ".join(["?" for _ in metadata])
                    columns = ", ".join(metadata.keys())
                    
                    conn.execute(
                        f"INSERT INTO documents ({columns}) VALUES ({placeholders})",
                        list(metadata.values())
                    )
                    
                    action = "created"
                
                # Registra no histórico
                self._add_history(
                    conn,
                    metadata["doc_id"],
                    action,
                    f"Document {action}"
                )
                
                return True
        
        except Exception as e:
            print(f"❌ Erro ao salvar documento: {e}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Busca documento por ID
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_document_by_hash(self, sha256: str) -> Optional[Dict[str, Any]]:
        """
        Busca documento por hash
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE sha256 = ?",
                (sha256,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_document_by_path(self, source_path: str) -> Optional[Dict[str, Any]]:
        """
        Busca documento por caminho
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE source_path = ?",
                (source_path,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_documents(
        self,
        status: Optional[str] = None,
        tipo: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Lista documentos com filtros opcionais
        """
        query = "SELECT * FROM documents WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if tipo:
            query += " AND tipo = ?"
            params.append(tipo)
        
        query += " ORDER BY precedencia ASC, created_at DESC"
        
        if limit:
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        
        with self._connect() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_vigentes(self) -> List[Dict[str, Any]]:
        """
        Retorna apenas documentos vigentes
        """
        return self.list_documents(status="Vigente")
    
    def update_status(self, doc_id: str, new_status: str) -> bool:
        """
        Atualiza status de um documento
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    "UPDATE documents SET status = ?, updated_at = ? WHERE doc_id = ?",
                    (new_status, datetime.now().isoformat(), doc_id)
                )
                
                self._add_history(
                    conn,
                    doc_id,
                    "status_changed",
                    f"Status changed to {new_status}"
                )
                
                return True
        
        except Exception as e:
            print(f"❌ Erro ao atualizar status: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Remove documento do banco
        """
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                return True
        
        except Exception as e:
            print(f"❌ Erro ao deletar documento: {e}")
            return False
    
    def add_tag(self, doc_id: str, tag: str) -> bool:
        """
        Adiciona tag a um documento
        """
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO document_tags (doc_id, tag) VALUES (?, ?)",
                    (doc_id, tag)
                )
                return True
        
        except Exception as e:
            print(f"❌ Erro ao adicionar tag: {e}")
            return False
    
    def get_tags(self, doc_id: str) -> List[str]:
        """
        Retorna tags de um documento
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT tag FROM document_tags WHERE doc_id = ?",
                (doc_id,)
            )
            return [row["tag"] for row in cursor.fetchall()]
    
    def get_documents_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """
        Busca documentos por tag
        """
        with self._connect() as conn:
            cursor = conn.execute("""
                SELECT d.* FROM documents d
                JOIN document_tags t ON d.doc_id = t.doc_id
                WHERE t.tag = ?
            """, (tag,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_history(self, doc_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retorna histórico de alterações de um documento
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT * FROM document_history WHERE doc_id = ? ORDER BY timestamp DESC LIMIT ?",
                (doc_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def _add_history(
        self,
        conn: sqlite3.Connection,
        doc_id: str,
        action: str,
        details: Optional[str] = None
    ):
        """
        Adiciona entrada no histórico
        """
        conn.execute(
            "INSERT INTO document_history (doc_id, action, details, timestamp) VALUES (?, ?, ?, ?)",
            (doc_id, action, details, datetime.now().isoformat())
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do banco de metadados
        """
        with self._connect() as conn:
            # Total de documentos
            total = conn.execute("SELECT COUNT(*) as count FROM documents").fetchone()["count"]
            
            # Por status
            by_status = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM documents 
                GROUP BY status
            """).fetchall()
            
            # Por tipo
            by_tipo = conn.execute("""
                SELECT tipo, COUNT(*) as count 
                FROM documents 
                GROUP BY tipo
            """).fetchall()
            
            # Total de tags
            total_tags = conn.execute("SELECT COUNT(DISTINCT tag) as count FROM document_tags").fetchone()["count"]
            
            return {
                "total_documents": total,
                "by_status": {row["status"]: row["count"] for row in by_status},
                "by_tipo": {row["tipo"]: row["count"] for row in by_tipo},
                "total_tags": total_tags,
                "database_path": str(self.db_path)
            }
    
    def get_total_documents(self) -> int:
        """Retorna total de documentos"""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) as count FROM documents").fetchone()["count"]


if __name__ == "__main__":
    # Teste do gerenciador de metadados
    print("🧪 Testando gerenciador de metadados...")
    
    manager = MetadataManager()
    
    # Teste 1: Inserir documento
    doc_metadata = {
        "doc_id": "test_001",
        "title": "Resolução Teste",
        "source_path": "/path/to/test.pdf",
        "sha256": "abc123",
        "status": "Vigente",
        "tipo": "Resolução",
        "precedencia": 3
    }
    
    success = manager.upsert_document(doc_metadata)
    print(f"✅ Documento inserido: {success}")
    
    # Teste 2: Buscar documento
    doc = manager.get_document("test_001")
    print(f"✅ Documento recuperado: {doc['title'] if doc else 'Não encontrado'}")
    
    # Teste 3: Estatísticas
    stats = manager.get_stats()
    print(f"📊 Stats: {stats}")
    
    # Limpeza
    manager.delete_document("test_001")
    print("🧹 Documento de teste removido")
