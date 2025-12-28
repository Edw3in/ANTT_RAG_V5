"""
Serviço de Retrieval Híbrido Avançado
Combina busca vetorial, BM25, reranking e filtros de governança para recuperação otimizada.
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.core.config import get_config
from src.core.embeddings import get_embeddings_function


class RetrievalStrategy(str, Enum):
    """Estratégias de retrieval disponíveis"""
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass
class RetrievalResult:
    """Resultado de uma operação de retrieval"""
    documents: List[Document]
    scores: List[float]
    strategy: str
    processing_time: float
    metadata: Dict[str, Any]


class DocumentFilter:
    """
    Filtro de documentos baseado em metadados e regras de governança
    """
    
    def __init__(self):
        self.config = get_config()
    
    def filter_by_status(self, documents: List[Document], status: str = "Vigente") -> List[Document]:
        """Filtra documentos por status (Vigente, Revogado, etc)"""
        return [
            doc for doc in documents 
            if doc.metadata.get("status", "Vigente") == status
        ]
    
    def filter_by_precedence(
        self, 
        documents: List[Document], 
        min_precedence: Optional[int] = None
    ) -> List[Document]:
        """Filtra documentos por nível de precedência normativa"""
        if min_precedence is None:
            return documents
        
        return [
            doc for doc in documents 
            if doc.metadata.get("precedencia", 99) <= min_precedence
        ]
    
    def filter_by_date_range(
        self, 
        documents: List[Document],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Document]:
        """Filtra documentos por período de vigência"""
        filtered = documents
        
        if start_date:
            filtered = [
                doc for doc in filtered
                if doc.metadata.get("vigencia_inicio", "") >= start_date
            ]
        
        if end_date:
            filtered = [
                doc for doc in filtered
                if doc.metadata.get("vigencia_fim", "9999-12-31") <= end_date
            ]
        
        return filtered
    
    def filter_by_source_type(
        self, 
        documents: List[Document],
        source_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Filtra documentos por tipo de fonte"""
        if not source_types:
            return documents
        
        return [
            doc for doc in documents
            if doc.metadata.get("tipo", "") in source_types
        ]
    
    def apply_all_filters(
        self,
        documents: List[Document],
        status: str = "Vigente",
        min_precedence: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Aplica todos os filtros em sequência"""
        filtered = self.filter_by_status(documents, status)
        filtered = self.filter_by_precedence(filtered, min_precedence)
        filtered = self.filter_by_date_range(filtered, start_date, end_date)
        filtered = self.filter_by_source_type(filtered, source_types)
        return filtered


class HybridRetriever:
    """
    Retriever híbrido que combina busca vetorial (semântica) e BM25 (lexical)
    com suporte a reranking e filtros avançados
    """
    
    def __init__(
        self,
        vectorstore: Optional[Chroma] = None,
        enable_reranker: Optional[bool] = None
    ):
        self.config = get_config()
        self.filter = DocumentFilter()
        
        # Inicializa vectorstore
        self.vectorstore = vectorstore or self._init_vectorstore()
        
        # Inicializa BM25
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_documents = []
        
        # Inicializa reranker
        self.enable_reranker = (
            enable_reranker 
            if enable_reranker is not None 
            else self.config.retrieval.use_reranker
        )
        self.reranker = None
        if self.enable_reranker:
            self.reranker = self._init_reranker()
    
    def _init_vectorstore(self) -> Chroma:
        """Inicializa o vectorstore ChromaDB"""
        print("🔄 Inicializando vectorstore...")
        
        vectorstore_path = self.config.paths.vectorstore_dir
        if not vectorstore_path or not vectorstore_path.exists():
            raise FileNotFoundError(f"Vectorstore não encontrado: {vectorstore_path}")
        
        embeddings = get_embeddings_function()
        
        return Chroma(
            persist_directory=str(vectorstore_path),
            embedding_function=embeddings
        )
    
    def _init_reranker(self) -> CrossEncoder:
        """Inicializa o modelo de reranking"""
        print(f"🔄 Inicializando reranker: {self.config.models.reranker_model}")
        
        return CrossEncoder(
            self.config.models.reranker_model,
            max_length=512,
            device=self.config.models.reranker_device
        )
    
    def initialize_bm25(self, documents: Optional[List[Document]] = None):
        """Inicializa o índice BM25 com todos os documentos Deve ser chamado antes de usar estratégias que incluem BM25"""
        print("🔄 Inicializando índice BM25...")
    
        if documents is None:
        # Carrega todos os documentos do vectorstore
            all_data = self.vectorstore.get()
            documents = [
                Document(
                    page_content=text if isinstance(text, str) else str(text),
                    metadata=meta or {}
                )
                for text, meta in zip(all_data["documents"], all_data["metadatas"])
            ]
    
        self.bm25_documents = documents
        # CORREÇÃO: Garante que usamos page_content (não content)
        self.bm25_corpus = [
            (doc.page_content if hasattr(doc, 'page_content') else str(doc)).lower().split() 
            for doc in documents
        ]
        self.bm25 = BM25Okapi(self.bm25_corpus)
    
        print(f"✅ BM25 inicializado com {len(documents)} documentos")
    
    def vector_search(
        self, 
        query: str, 
        k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], List[float]]:
        """Busca vetorial pura (semântica)"""
        results = self.vectorstore.similarity_search_with_score(
            query, 
            k=k,
            filter=filter_dict
        )
        
        documents = [doc for doc, _ in results]
        scores = [float(score) for _, score in results]
        
        return documents, scores
    
    def bm25_search(
        self, 
        query: str, 
        k: int = 10
    ) -> Tuple[List[Document], List[float]]:
        """Busca BM25 pura (lexical)"""
        if self.bm25 is None:
            raise RuntimeError("BM25 não inicializado. Chame initialize_bm25() primeiro.")
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pega top-k resultados
        top_indices = np.argsort(scores)[::-1][:k]
        
        documents = [self.bm25_documents[i] for i in top_indices]
        top_scores = [float(scores[i]) for i in top_indices]
        
        return documents, top_scores
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        vector_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], List[float]]:
        """
        Busca híbrida combinando vetorial e BM25 com fusão de scores
        """
        if self.bm25 is None:
            print("⚠️  BM25 não inicializado, usando apenas busca vetorial")
            return self.vector_search(query, k, filter_dict)
        
        # Usa pesos da configuração se não fornecidos
        vector_weight = vector_weight or self.config.retrieval.vector_weight
        bm25_weight = bm25_weight or self.config.retrieval.bm25_weight
        
        # Busca vetorial
        vector_docs, vector_scores = self.vector_search(query, k=k*2, filter_dict=filter_dict)
        
        # Busca BM25
        bm25_docs, bm25_scores = self.bm25_search(query, k=k*2)
        
        # Normaliza scores para [0, 1]
        vector_scores_norm = self._normalize_scores(vector_scores)
        bm25_scores_norm = self._normalize_scores(bm25_scores)
        
        # Combina resultados
        doc_scores = {}
        
        for doc, score in zip(vector_docs, vector_scores_norm):
            doc_id = self._get_doc_id(doc)
            doc_scores[doc_id] = {
                "doc": doc,
                "score": score * vector_weight
            }
        
        for doc, score in zip(bm25_docs, bm25_scores_norm):
            doc_id = self._get_doc_id(doc)
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += score * bm25_weight
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "score": score * bm25_weight
                }
        
        # Ordena por score combinado
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )[:k]
        
        documents = [r["doc"] for r in sorted_results]
        scores = [r["score"] for r in sorted_results]
        
        return documents, scores
    
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None
    ) -> Tuple[List[Document], List[float]]:
        """
        Aplica reranking nos documentos usando CrossEncoder
        """
        if not self.reranker:
            print("⚠️  Reranker não disponível, retornando documentos sem reranking")
            return documents, [1.0] * len(documents)
        
        if not documents:
            return [], []
        
        # Prepara pares query-documento
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Calcula scores de reranking
        scores = self.reranker.predict(pairs)
        
        # Ordena por score
        sorted_indices = np.argsort(scores)[::-1]
        
        if top_k:
            sorted_indices = sorted_indices[:top_k]
        
        reranked_docs = [documents[i] for i in sorted_indices]
        reranked_scores = [float(scores[i]) for i in sorted_indices]
        
        # Adiciona score de reranking aos metadados
        for doc, score in zip(reranked_docs, reranked_scores):
            doc.metadata["rerank_score"] = score
        
        return reranked_docs, reranked_scores
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        apply_filters: bool = True,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> RetrievalResult:
        """
        Método principal de retrieval com estratégia configurável
        """
        start_time = time.time()
        k = k or self.config.retrieval.default_k
        
        # Executa busca baseada na estratégia
        if strategy == RetrievalStrategy.VECTOR_ONLY:
            documents, scores = self.vector_search(query, k=k)
        
        elif strategy == RetrievalStrategy.BM25_ONLY:
            documents, scores = self.bm25_search(query, k=k)
        
        elif strategy == RetrievalStrategy.HYBRID:
            documents, scores = self.hybrid_search(query, k=k)
        
        elif strategy == RetrievalStrategy.HYBRID_RERANK:
            # Busca híbrida com mais documentos para reranking
            documents, scores = self.hybrid_search(query, k=k*2)
            # Aplica reranking
            documents, scores = self.rerank(query, documents, top_k=k)
        
        else:
            raise ValueError(f"Estratégia não suportada: {strategy}")
        
        # Aplica filtros de governança
        if apply_filters and filter_kwargs:
            documents = self.filter.apply_all_filters(documents, **filter_kwargs)
            # Ajusta scores se documentos foram removidos
            scores = scores[:len(documents)]
        
        processing_time = time.time() - start_time
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy=strategy,
            processing_time=processing_time,
            metadata={
                "query": query,
                "k": k,
                "total_results": len(documents),
                "filters_applied": apply_filters
            }
        )
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normaliza scores para o intervalo [0, 1]"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def _get_doc_id(self, doc: Document) -> str:
        """Gera ID único para um documento"""
        return f"{doc.metadata.get('source', '')}_{doc.metadata.get('page', 0)}"
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do retriever"""
        stats = {
            "vectorstore_initialized": self.vectorstore is not None,
            "bm25_initialized": self.bm25 is not None,
            "reranker_enabled": self.enable_reranker,
            "reranker_model": self.config.models.reranker_model if self.enable_reranker else None,
        }
        
        if self.bm25:
            stats["bm25_corpus_size"] = len(self.bm25_corpus)
        
        return stats


if __name__ == "__main__":
    # Teste do serviço de retrieval
    print("🧪 Testando serviço de retrieval...")
    
    retriever = HybridRetriever()
    print(f"📊 Stats: {retriever.get_stats()}")
    
    # Teste de busca vetorial
    query = "Qual o prazo para renovação de acreditação?"
    result = retriever.retrieve(query, k=5, strategy=RetrievalStrategy.VECTOR_ONLY)
    
    print(f"\n✅ Busca vetorial concluída:")
    print(f"   Documentos: {len(result.documents)}")
    print(f"   Tempo: {result.processing_time:.2f}s")
    print(f"   Estratégia: {result.strategy}")
