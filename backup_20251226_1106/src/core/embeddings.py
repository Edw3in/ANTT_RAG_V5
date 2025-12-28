"""
Sistema de Embeddings Otimizado com Cache e Pooling
Gerencia embeddings com cache em memória e disco, suporte a batch processing e múltiplos providers.
"""

import hashlib
import pickle
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from functools import lru_cache
from dataclasses import dataclass

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.core.config import get_config


@dataclass
class EmbeddingResult:
    """Resultado de embedding com metadados"""
    embeddings: List[List[float]]
    texts: List[str]
    model: str
    dimension: int
    cache_hit: bool
    processing_time: float


class EmbeddingCache:
    """
    Cache em disco para embeddings usando pickle
    Evita reprocessamento de textos já embedados
    """
    
    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return
            
        config = get_config()
        self.cache_dir = cache_dir or (config.paths.base_dir / ".cache" / "embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {"hits": 0, "misses": 0}
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Gera chave única para o cache"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Retorna caminho do arquivo de cache"""
        # Organiza em subdiretórios para evitar muitos arquivos em um diretório
        subdir = cache_key[:2]
        return self.cache_dir / subdir / f"{cache_key}.pkl"
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Recupera embedding do cache"""
        if not self.enabled:
            return None
            
        cache_key = self._get_cache_key(text, model)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.stats["hits"] += 1
                    return data["embedding"]
            except Exception as e:
                print(f"⚠️  Erro ao ler cache: {e}")
                cache_path.unlink(missing_ok=True)
        
        self.stats["misses"] += 1
        return None
    
    def set(self, text: str, model: str, embedding: List[float]):
        """Armazena embedding no cache"""
        if not self.enabled:
            return
            
        cache_key = self._get_cache_key(text, model)
        cache_path = self._get_cache_path(cache_key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    "embedding": embedding,
                    "text": text[:100],  # Armazena apenas início para debug
                    "model": model,
                    "timestamp": time.time()
                }, f)
        except Exception as e:
            print(f"⚠️  Erro ao salvar cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2%}",
            "cache_size_mb": self._get_cache_size_mb()
        }
    
    def _get_cache_size_mb(self) -> float:
        """Calcula tamanho do cache em MB"""
        if not self.enabled or not self.cache_dir.exists():
            return 0.0
        
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*.pkl"))
        return total_size / (1024 * 1024)
    
    def clear(self):
        """Limpa todo o cache"""
        if not self.enabled:
            return
            
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {"hits": 0, "misses": 0}


class EmbeddingManager:
    """
    Gerenciador de embeddings com suporte a múltiplos providers e cache
    """
    
    def __init__(self, cache_enabled: bool = True):
        self.config = get_config()
        self.cache = EmbeddingCache(enabled=cache_enabled and self.config.cache.enabled)
        self._embedding_model = None
        self._model_dimension = None
    
    @property
    def embedding_model(self):
        """Lazy loading do modelo de embedding"""
        if self._embedding_model is None:
            self._embedding_model = self._load_embedding_model()
        return self._embedding_model
    
    def _load_embedding_model(self):
        """Carrega o modelo de embedding baseado na configuração"""
        model_name = self.config.models.embedding
        device = self.config.models.embedding_device
        
        print(f"🔄 Carregando modelo de embedding: {model_name}")
        
        # HuggingFace embeddings (padrão)
        if "sentence-transformers" in model_name or "/" in model_name:
            model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={
                    "device": device,
                    "trust_remote_code": True
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": self.config.models.embedding_batch_size
                },
                show_progress=False
            )
            # Detecta dimensão
            test_embedding = model.embed_query("test")
            self._model_dimension = len(test_embedding)
            
        # Google embeddings
        elif model_name.startswith("models/embedding"):
            model = GoogleGenerativeAIEmbeddings(
                model=model_name,
                task_type="retrieval_document"
            )
            self._model_dimension = 768  # Dimensão padrão do Google
            
        # OpenAI embeddings
        elif "text-embedding" in model_name:
            model = OpenAIEmbeddings(
                model=model_name
            )
            self._model_dimension = 1536  # Dimensão padrão do OpenAI
            
        else:
            raise ValueError(f"Modelo de embedding não suportado: {model_name}")
        
        print(f"✅ Modelo carregado (dimensão: {self._model_dimension})")
        return model
    
    def embed_texts(
        self, 
        texts: List[str], 
        use_cache: bool = True,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Embeda múltiplos textos com cache e batch processing
        """
        start_time = time.time()
        embeddings = []
        cache_hits = 0
        texts_to_embed = []
        indices_to_embed = []
        
        # Verifica cache para cada texto
        if use_cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.config.models.embedding)
                if cached is not None:
                    embeddings.append(cached)
                    cache_hits += 1
                else:
                    embeddings.append(None)
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        else:
            texts_to_embed = texts
            indices_to_embed = list(range(len(texts)))
            embeddings = [None] * len(texts)
        
        # Embeda textos não cacheados
        if texts_to_embed:
            if show_progress:
                print(f"🔄 Embedando {len(texts_to_embed)} textos...")
            
            new_embeddings = self.embedding_model.embed_documents(texts_to_embed)
            
            # Armazena no cache e na lista de resultados
            for idx, text, embedding in zip(indices_to_embed, texts_to_embed, new_embeddings):
                embeddings[idx] = embedding
                if use_cache:
                    self.cache.set(text, self.config.models.embedding, embedding)
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=embeddings,
            texts=texts,
            model=self.config.models.embedding,
            dimension=self._model_dimension,
            cache_hit=cache_hits == len(texts),
            processing_time=processing_time
        )
    
    def embed_query(self, query: str, use_cache: bool = True) -> List[float]:
        """
        Embeda uma query única (otimizado para buscas)
        """
        if use_cache:
            cached = self.cache.get(query, self.config.models.embedding)
            if cached is not None:
                return cached
        
        embedding = self.embedding_model.embed_query(query)
        
        if use_cache:
            self.cache.set(query, self.config.models.embedding, embedding)
        
        return embedding
    
    def get_dimension(self) -> int:
        """Retorna a dimensão dos embeddings"""
        if self._model_dimension is None:
            _ = self.embedding_model  # Força carregamento
        return self._model_dimension
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache"""
        return self.cache.get_stats()
    
    def clear_cache(self):
        """Limpa o cache de embeddings"""
        self.cache.clear()


@lru_cache(maxsize=1)
def get_embedding_manager() -> EmbeddingManager:
    """
    Retorna instância singleton do gerenciador de embeddings
    """
    return EmbeddingManager()


def get_embeddings_function():
    """
    Retorna função de embeddings compatível com LangChain
    """
    manager = get_embedding_manager()
    return manager.embedding_model


# Funções de conveniência
def embed_texts(texts: List[str], use_cache: bool = True) -> List[List[float]]:
    """Função de conveniência para embedar textos"""
    manager = get_embedding_manager()
    result = manager.embed_texts(texts, use_cache=use_cache)
    return result.embeddings


def embed_query(query: str, use_cache: bool = True) -> List[float]:
    """Função de conveniência para embedar query"""
    manager = get_embedding_manager()
    return manager.embed_query(query, use_cache=use_cache)


if __name__ == "__main__":
    # Teste do sistema de embeddings
    print("🧪 Testando sistema de embeddings...")
    
    manager = get_embedding_manager()
    
    # Teste 1: Embedding único
    query = "Qual o prazo para renovação?"
    embedding = manager.embed_query(query)
    print(f"✅ Query embedada: dimensão {len(embedding)}")
    
    # Teste 2: Batch embedding
    texts = [
        "O prazo é de 30 dias",
        "A documentação deve ser completa",
        "O verificador deve ser acreditado"
    ]
    result = manager.embed_texts(texts)
    print(f"✅ Batch embedado: {len(result.embeddings)} textos em {result.processing_time:.2f}s")
    
    # Teste 3: Cache
    result2 = manager.embed_texts(texts)
    print(f"✅ Cache hit: {result2.cache_hit}")
    
    # Estatísticas
    stats = manager.get_cache_stats()
    print(f"📊 Cache stats: {stats}")
