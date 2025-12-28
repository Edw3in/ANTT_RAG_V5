"""
Sistema de Configuração Centralizado para ANTT RAG
Gerencia todas as configurações do sistema com validação e suporte a múltiplos ambientes.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from functools import lru_cache


class PathsConfig(BaseModel):
    """Configurações de caminhos do sistema"""
    base_dir: Path = Field(default_factory=lambda: Path.cwd())
    data_dir: Optional[Path] = None
    bcp_inbox: Optional[Path] = None
    bcp_processado: Optional[Path] = None
    bcp_rejeitado: Optional[Path] = None
    metadata_db: Optional[Path] = None
    vectorstore_dir: Optional[Path] = None
    auditoria_dir: Optional[Path] = None
    prompts_dir: Optional[Path] = None

    @validator("*", pre=True)
    def resolve_paths(cls, v, values):
        """Resolve caminhos relativos baseados no base_dir"""
        if v is None:
            return None
        if isinstance(v, str):
            path = Path(v)
            if not path.is_absolute():
                base = values.get("base_dir", Path.cwd())
                return base / path
            return path
        return v

    def ensure_directories(self):
        """Cria todos os diretórios necessários"""
        for field_name, field_value in self.__dict__.items():
            if field_value and isinstance(field_value, Path):
                if field_name.endswith("_db"):
                    field_value.parent.mkdir(parents=True, exist_ok=True)
                else:
                    field_value.mkdir(parents=True, exist_ok=True)


class ModelsConfig(BaseModel):
    """Configurações de modelos de ML/AI"""
    embedding: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu")
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    llm_provider: str = Field(default="google")  # google, openai, ollama, huggingface
    llm_model: str = Field(default="gemini-1.5-flash")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=256, le=8192)
    llm_timeout: int = Field(default=60, ge=10, le=300)

    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_device: str = Field(default="cpu")

    # Fallback (Opção 3)
    fallback_enabled: bool = Field(default=False)
    fallback_provider: str = Field(default="google")  # google, openai
    fallback_model: str = Field(default="gemini-1.5-flash")
    fallback_timeout: int = Field(default=60, ge=10, le=300)

    ollama_base_url: str = Field(default="http://127.0.0.1:11434")


class ChunkingConfig(BaseModel):
    """Configurações de chunking de documentos"""
    chunk_size: int = Field(default=1000, ge=100, le=4000)
    chunk_overlap: int = Field(default=200, ge=0, le=1000)
    separators: List[str] = Field(default=["\n\n", "\n", ". ", " ", ""])

    @validator("chunk_overlap")
    def overlap_must_be_less_than_size(cls, v, values):
        chunk_size = values.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) deve ser menor que chunk_size ({chunk_size})")
        return v


class RetrievalConfig(BaseModel):
    """Configurações de recuperação de documentos"""
    default_k: int = Field(default=5, ge=1, le=50)
    max_k: int = Field(default=20, ge=1, le=100)
    similarity_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    use_hybrid: bool = Field(default=True)
    use_reranker: bool = Field(default=True)
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    @validator("vector_weight")
    def weights_must_sum_to_one(cls, v, values):
        bm25_weight = values.get("bm25_weight", 0.5)
        total = bm25_weight + v
        if not (0.99 <= total <= 1.01):  # Tolerância para erros de float
            raise ValueError(f"bm25_weight + vector_weight deve somar 1.0 (atual: {total})")
        return v


class GovernancaConfig(BaseModel):
    """Configurações de governança e compliance"""
    matriz_precedencia: Optional[Path] = None
    taxonomia_temas: Optional[Path] = None
    politica_nao_resposta: Optional[Path] = None
    enable_audit: bool = Field(default=True)
    audit_level: str = Field(default="full")  # full, minimal, none


class APIConfig(BaseModel):
    """Configurações da API"""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    reload: bool = Field(default=False)

    cors_origins: List[str] = Field(default=["*"])
    cors_methods: List[str] = Field(default=["*"])
    cors_headers: List[str] = Field(default=["*"])

    rate_limit_enabled: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)  # segundos

    enable_metrics: bool = Field(default=True)
    enable_docs: bool = Field(default=True)


class CacheConfig(BaseModel):
    """Configurações de cache"""
    enabled: bool = Field(default=True)
    ttl: int = Field(default=3600, ge=60)  # segundos
    max_size: int = Field(default=1000, ge=10)
    backend: str = Field(default="memory")  # memory, redis


class LoggingConfig(BaseModel):
    """Configurações de logging"""
    level: str = Field(default="INFO")
    format: str = Field(default="json")  # json, text
    output: str = Field(default="stdout")  # stdout, file, both
    file_path: Optional[Path] = None
    rotation: str = Field(default="1 day")
    retention: str = Field(default="30 days")


class SecurityConfig(BaseModel):
    """Configurações de segurança"""
    enable_auth: bool = Field(default=False)
    api_key_header: str = Field(default="X-API-Key")
    allowed_api_keys: List[str] = Field(default=[])
    enable_https: bool = Field(default=False)
    ssl_cert_path: Optional[Path] = None
    ssl_key_path: Optional[Path] = None


class Config(BaseModel):
    """Configuração principal do sistema"""
    environment: str = Field(default="development")  # development, staging, production
    debug: bool = Field(default=False)

    paths: PathsConfig = Field(default_factory=PathsConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    governanca: GovernancaConfig = Field(default_factory=GovernancaConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Config":
        """Carrega configuração de arquivo YAML"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**data)

    @classmethod
    def from_env(cls) -> "Config":
        """Carrega configuração de variáveis de ambiente"""
        config_path = os.getenv("ANTT_RAG_CONFIG", "config/config.yaml")
        return cls.from_yaml(config_path)

    def to_yaml(self, yaml_path: str | Path):
        """Salva configuração em arquivo YAML"""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        # Converte Path objects para strings para serialização
        data = self.dict()
        self._convert_paths_to_str(data)

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _convert_paths_to_str(self, data: Dict[str, Any]):
        """Converte objetos Path para strings recursivamente"""
        for key, value in data.items():
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, dict):
                self._convert_paths_to_str(value)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Path):
                        value[i] = str(item)
                    elif isinstance(item, dict):
                        self._convert_paths_to_str(item)

    def validate_environment(self):
        """Valida se o ambiente está configurado corretamente (inclui fallback)"""
        errors: List[str] = []

        def require_key(provider: str):
            p = (provider or "").strip().lower()

            if p == "google":
                if not os.getenv("GOOGLE_API_KEY"):
                    errors.append("GOOGLE_API_KEY não definida")

            elif p == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    errors.append("OPENAI_API_KEY não definida")

            # ollama / huggingface: não exigem chave

        # Provider principal
        require_key(self.models.llm_provider)

        # Fallback (se habilitado)
        if getattr(self.models, "fallback_enabled", False):
            require_key(getattr(self.models, "fallback_provider", ""))

        # Verifica diretórios críticos
        if self.paths.vectorstore_dir and not self.paths.vectorstore_dir.exists():
            errors.append(f"Vectorstore não encontrado: {self.paths.vectorstore_dir}")

        if errors:
            raise ValueError(
                "Erros de validação de ambiente:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )

        return True


@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Retorna a configuração global do sistema (singleton com cache)
    """
    config_path = os.getenv("ANTT_RAG_CONFIG", "config/config.yaml")
    config = Config.from_yaml(config_path)
    config.paths.ensure_directories()
    return config


def reload_config():
    """
    Recarrega a configuração (limpa o cache)
    """
    get_config.cache_clear()
    return get_config()


if __name__ == "__main__":
    # Teste de configuração
    config = Config()
    print("✅ Configuração padrão criada com sucesso")
    print(f"Ambiente: {config.environment}")
    print(f"Modelo de embedding: {config.models.embedding}")
    print(f"LLM Provider: {config.models.llm_provider}")
    print(f"Chunk size: {config.chunking.chunk_size}")
    print(f"Default K: {config.retrieval.default_k}")
