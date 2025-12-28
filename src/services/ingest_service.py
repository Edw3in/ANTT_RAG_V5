"""
Serviço de Ingestão de Documentos
Processa PDFs, extrai texto, gera chunks, embeddings e armazena no vectorstore.
"""

import hashlib
import shutil
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pypdf
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from src.core.config import get_config
from src.core.embeddings import get_embeddings_function
from src.utils.metadata_manager import MetadataManager
from src.utils.text_processor import TextProcessor
from src.utils.text_cleaner import get_text_cleaner  # <-- NOVA IMPORTAÇÃO


class ProcessingStatus(str, Enum):
    """Status de processamento de documento"""
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"
    DUPLICATE = "duplicate"


@dataclass
class DocumentProcessingResult:
    """Resultado do processamento de um documento"""
    filename: str
    status: ProcessingStatus
    chunks_created: int
    pages_processed: int
    file_hash: str
    error_message: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class IngestResult:
    """Resultado completo da ingestão"""
    total_files: int
    successful: int
    skipped: int
    errors: int
    total_chunks: int
    processing_time: float
    results: List[DocumentProcessingResult]


class IngestService:
    """
    Serviço de ingestão de documentos com pipeline completo
    """
    
    def __init__(self):
        self.config = get_config()
        self.metadata_manager = MetadataManager()
        self.text_processor = TextProcessor()
        
        # Novo: cleaner centralizado para todo o projeto
        self.text_cleaner = get_text_cleaner()
        
        # Inicializa text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunking.chunk_size,
            chunk_overlap=self.config.chunking.chunk_overlap,
            separators=self.config.chunking.separators,
            length_function=len,
        )
        
        # Paths
        self.inbox_path = self.config.paths.bcp_inbox
        self.processed_path = self.config.paths.bcp_processado
        self.rejected_path = self.config.paths.bcp_rejeitado
        self.vectorstore_path = self.config.paths.vectorstore_dir
        
        # Garante que diretórios existem
        self.inbox_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.rejected_path.mkdir(parents=True, exist_ok=True)
    
    def ingest_all(self, force_reprocess: bool = False) -> IngestResult:
        """
        Ingere todos os PDFs da pasta inbox
        """
        start_time = time.time()
        
        pdf_files = list(self.inbox_path.glob("*.pdf"))
        
        if not pdf_files:
            print("📭 Nenhum arquivo para processar na inbox")
            return IngestResult(
                total_files=0,
                successful=0,
                skipped=0,
                errors=0,
                total_chunks=0,
                processing_time=0.0,
                results=[]
            )
        
        print(f"📥 Encontrados {len(pdf_files)} arquivos para processar")
        
        results = []
        all_chunks = []
        
        for pdf_path in pdf_files:
            # Calcular hash ANTES de processar
            file_hash = self._calculate_file_hash(pdf_path)
            
            # Pré-carregar chunks (para indexação posterior)
            chunks_for_this_doc = []
            if pdf_path.exists():
                try:
                    pages_text = self._extract_text_from_pdf(pdf_path)
                    if pages_text:
                        chunks_for_this_doc = self._create_chunks(pages_text, pdf_path, file_hash)
                except Exception as e:
                    print(f"   ⚠️  Erro ao pré-carregar chunks: {e}")
            
            # Processa o documento (move o arquivo, registra metadados, etc.)
            result = self.process_document(pdf_path, force_reprocess, file_hash)
            results.append(result)
            
            # Adiciona chunks só se sucesso
            if result.status == ProcessingStatus.SUCCESS and chunks_for_this_doc:
                all_chunks.extend(chunks_for_this_doc)
        
        # Indexa todos os chunks acumulados
        if all_chunks:
            print(f"\n📊 Indexando {len(all_chunks)} chunks no vectorstore...")
            self._index_chunks(all_chunks)
            print("✅ Indexação concluída")
        
        # Estatísticas finais
        successful = sum(1 for r in results if r.status == ProcessingStatus.SUCCESS)
        skipped = sum(1 for r in results if r.status in [ProcessingStatus.SKIPPED, ProcessingStatus.DUPLICATE])
        errors = sum(1 for r in results if r.status == ProcessingStatus.ERROR)
        
        processing_time = time.time() - start_time
        
        return IngestResult(
            total_files=len(pdf_files),
            successful=successful,
            skipped=skipped,
            errors=errors,
            total_chunks=len(all_chunks),
            processing_time=processing_time,
            results=results
        )
    
    def process_document(
        self,
        pdf_path: Path,
        force_reprocess: bool = False,
        file_hash: Optional[str] = None
    ) -> DocumentProcessingResult:
        """
        Processa um único documento PDF
        """
        start_time = time.time()
        filename = pdf_path.name
        
        print(f"\n📄 Processando: {filename}")
        
        try:
            # Calcula hash se não foi fornecido
            if file_hash is None:
                file_hash = self._calculate_file_hash(pdf_path)
            
            # Verifica duplicidade
            if not force_reprocess:
                existing_doc = self.metadata_manager.get_document_by_hash(file_hash)
                if existing_doc:
                    print(f"   ⏭️  Documento já processado (hash: {file_hash[:8]}...)")
                    self._move_to_processed(pdf_path)
                    return DocumentProcessingResult(
                        filename=filename,
                        status=ProcessingStatus.DUPLICATE,
                        chunks_created=0,
                        pages_processed=0,
                        file_hash=file_hash,
                        processing_time=time.time() - start_time
                    )
            
            # Extrai texto (com limpeza já aplicada)
            pages_text = self._extract_text_from_pdf(pdf_path)
            
            if not pages_text:
                raise ValueError("Nenhum texto extraído do PDF")
            
            print(f"   ✅ {len(pages_text)} páginas extraídas")
            
            # Cria chunks (apenas para contagem aqui)
            chunks = self._create_chunks(pages_text, pdf_path, file_hash)
            
            print(f"   ✅ {len(chunks)} chunks criados")
            
            # Registra metadados
            self._register_document_metadata(pdf_path, file_hash, len(pages_text))
            
            # Move para processados
            self._move_to_processed(pdf_path)
            
            processing_time = time.time() - start_time
            
            return DocumentProcessingResult(
                filename=filename,
                status=ProcessingStatus.SUCCESS,
                chunks_created=len(chunks),
                pages_processed=len(pages_text),
                file_hash=file_hash,
                processing_time=processing_time
            )
        
        except Exception as e:
            print(f"   ❌ ERRO: {e}")
            self._move_to_rejected(pdf_path)
            
            return DocumentProcessingResult(
                filename=filename,
                status=ProcessingStatus.ERROR,
                chunks_created=0,
                pages_processed=0,
                file_hash=file_hash or "",
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calcula SHA256 do arquivo"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> List[Tuple[int, str]]:
        """
        Extrai texto de todas as páginas do PDF.
        A limpeza é aplicada imediatamente após a extração bruta.
        Retorna lista de tuplas (page_number, texto_limpo)
        """
        pages_text = []
        
        reader = pypdf.PdfReader(str(pdf_path))
        
        for page_num, page in enumerate(reader.pages, 1):
            raw_text = page.extract_text() or ""
            
            # LIMPEZA CENTRALIZADA – aplicada logo após extração
            cleaned_text = self.text_cleaner.clean(raw_text)
            
            # Só inclui página se houver conteúdo após limpeza
            if cleaned_text.strip():
                pages_text.append((page_num, cleaned_text))
        
        return pages_text
    
    def _create_chunks(
        self,
        pages_text: List[Tuple[int, str]],
        pdf_path: Path,
        file_hash: str
    ) -> List[Document]:
        """
        Cria chunks a partir do texto já limpo
        """
        all_chunks = []
        
        for page_num, text in pages_text:
            base_metadata = {
                "source": pdf_path.name,
                "page": page_num,
                "hash": file_hash,
                "tipo": self._infer_document_type(pdf_path.name),
            }
            
            page_doc = Document(
                page_content=text,
                metadata=base_metadata
            )
            
            chunks = self.text_splitter.split_documents([page_doc])
            
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
            
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _index_chunks(self, chunks: List[Document]):
        """Indexa chunks no vectorstore"""
        embeddings = get_embeddings_function()
        
        if self.vectorstore_path.exists():
            vectorstore = Chroma(
                persist_directory=str(self.vectorstore_path),
                embedding_function=embeddings
            )
            vectorstore.add_documents(chunks)
        else:
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=str(self.vectorstore_path)
            )
    
    def _register_document_metadata(
        self,
        pdf_path: Path,
        file_hash: str,
        total_pages: int
    ):
        """Registra metadados do documento"""
        metadata = {
            "doc_id": pdf_path.stem,
            "title": pdf_path.name,
            "source_path": str(pdf_path),
            "sha256": file_hash,
            "status": "Vigente",
            "precedencia": self._infer_precedence(pdf_path.name),
            "tipo": self._infer_document_type(pdf_path.name),
            "total_pages": total_pages,
            "vigencia_inicio": None,
            "vigencia_fim": None,
        }
        
        self.metadata_manager.upsert_document(metadata)
    
    def _infer_document_type(self, filename: str) -> str:
        filename_lower = filename.lower()
        
        if "lei" in filename_lower:
            return "Lei"
        elif "decreto" in filename_lower:
            return "Decreto"
        elif "resolucao" in filename_lower or "resolução" in filename_lower:
            return "Resolução"
        elif "portaria" in filename_lower:
            return "Portaria"
        elif "instrucao" in filename_lower or "instrução" in filename_lower:
            return "Instrução Normativa"
        else:
            return "Normativo"
    
    def _infer_precedence(self, filename: str) -> int:
        doc_type = self._infer_document_type(filename)
        
        precedence_map = {
            "Lei": 1,
            "Decreto": 2,
            "Resolução": 3,
            "Portaria": 4,
            "Instrução Normativa": 5,
            "Normativo": 99
        }
        
        return precedence_map.get(doc_type, 99)
    
    def _move_to_processed(self, pdf_path: Path):
        """Move arquivo para pasta de processados"""
        destination = self.processed_path / pdf_path.name
        shutil.move(str(pdf_path), str(destination))
    
    def _move_to_rejected(self, pdf_path: Path):
        """Move arquivo para pasta de rejeitados"""
        destination = self.rejected_path / pdf_path.name
        shutil.move(str(pdf_path), str(destination))
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do serviço de ingestão"""
        return {
            "inbox_files": len(list(self.inbox_path.glob("*.pdf"))),
            "processed_files": len(list(self.processed_path.glob("*.pdf"))),
            "rejected_files": len(list(self.rejected_path.glob("*.pdf"))),
            "total_documents_indexed": self.metadata_manager.get_total_documents(),
            "chunk_size": self.config.chunking.chunk_size,
            "chunk_overlap": self.config.chunking.chunk_overlap,
        }


if __name__ == "__main__":
    print("🧪 Testando serviço de ingestão...")
    
    service = IngestService()
    stats = service.get_stats()
    
    print(f"📊 Estatísticas:")
    print(f"   Arquivos na inbox: {stats['inbox_files']}")
    print(f"   Arquivos processados: {stats['processed_files']}")
    print(f"   Arquivos rejeitados: {stats['rejected_files']}")
    print(f"   Documentos indexados: {stats['total_documents_indexed']}")