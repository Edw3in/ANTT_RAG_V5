"""
Interface Unificada para Múltiplos Provedores de LLM
Suporta Google Gemini, OpenAI, Ollama e HuggingFace com fallback automático.
"""

import os
import time
import warnings
from typing import Optional, List, Dict, Any, Union
from functools import lru_cache
from dataclasses import dataclass, field
from enum import Enum

# Desabilitar telemetria do ChromaDB antes de qualquer import
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEndpoint

from src.core.config import get_config


class LLMProvider(str, Enum):
    """Provedores de LLM suportados"""
    GOOGLE = "google"
    OPENAI = "openai"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


@dataclass
class LLMResponse:
    """Resposta estruturada do LLM"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    processing_time: float = 0.0
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMManager:
    """
    Gerenciador unificado de LLMs com suporte a múltiplos providers
    """
    
    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.config = get_config()
        self.provider = provider or self.config.models.llm_provider
        self.model = model or self.config.models.llm_model
        self._llm = None
        self._fallback_providers = self._get_fallback_order()
    
    def _get_fallback_order(self) -> List[str]:
        """Define ordem de fallback para providers"""
        order = [self.provider]
        
        # Adiciona outros providers disponíveis
        all_providers = [LLMProvider.GOOGLE, LLMProvider.OPENAI, LLMProvider.OLLAMA]
        for p in all_providers:
            if p != self.provider and self._is_provider_available(p):
                order.append(p)
        
        return order
    
    def _is_provider_available(self, provider: str) -> bool:
        """Verifica se um provider está disponível"""
        if provider == LLMProvider.GOOGLE:
            return bool(os.getenv("GOOGLE_API_KEY"))
        elif provider == LLMProvider.OPENAI:
            return bool(os.getenv("OPENAI_API_KEY"))
        elif provider == LLMProvider.OLLAMA:
            # Assume que Ollama está disponível se configurado
            return True
        elif provider == LLMProvider.HUGGINGFACE:
            return bool(os.getenv("HUGGINGFACE_API_KEY"))
        return False
    
    @property
    def llm(self) -> BaseChatModel:
        """Lazy loading do modelo LLM"""
        if self._llm is None:
            self._llm = self._load_llm(self.provider, self.model)
        return self._llm
    
    def _load_llm(self, provider: str, model: str) -> BaseChatModel:
        """Carrega o modelo LLM baseado no provider"""
        print(f"🔄 Carregando LLM: {provider}/{model}")
        
        try:
            if provider == LLMProvider.GOOGLE:
                return self._load_google_llm(model)
            elif provider == LLMProvider.OPENAI:
                return self._load_openai_llm(model)
            elif provider == LLMProvider.OLLAMA:
                return self._load_ollama_llm(model)
            elif provider == LLMProvider.HUGGINGFACE:
                return self._load_huggingface_llm(model)
            else:
                raise ValueError(f"Provider não suportado: {provider}")
        
        except Exception as e:
            print(f"❌ Erro ao carregar {provider}: {e}")
            # Tenta fallback
            if len(self._fallback_providers) > 1:
                next_provider = self._fallback_providers[1]
                print(f"🔄 Tentando fallback para {next_provider}")
                return self._load_llm(next_provider, self._get_default_model(next_provider))
            raise
    
    def _load_google_llm(self, model: str) -> ChatGoogleGenerativeAI:
        """Carrega Google Gemini"""
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY não definida")
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=self.config.models.llm_temperature,
            max_output_tokens=self.config.models.llm_max_tokens,
            timeout=self.config.models.llm_timeout,
            convert_system_message_to_human=True
        )
    
    def _load_openai_llm(self, model: str) -> ChatOpenAI:
        """Carrega OpenAI"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY não definida")
        
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=self.config.models.llm_temperature,
            max_tokens=self.config.models.llm_max_tokens,
            timeout=self.config.models.llm_timeout
        )
    
    def _load_ollama_llm(self, model: str) -> Ollama:
        """Carrega Ollama (local)"""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=self.config.models.llm_temperature,
            num_predict=self.config.models.llm_max_tokens
        )
    
    def _load_huggingface_llm(self, model: str) -> HuggingFaceEndpoint:
        """Carrega HuggingFace Inference API"""
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY não definida")
        
        return HuggingFaceEndpoint(
            repo_id=model,
            huggingfacehub_api_token=api_key,
            temperature=self.config.models.llm_temperature,
            max_new_tokens=self.config.models.llm_max_tokens,
            timeout=self.config.models.llm_timeout
        )
    
    def _get_default_model(self, provider: str) -> str:
        """Retorna modelo padrão para um provider"""
        defaults = {
            LLMProvider.GOOGLE: "gemini-1.5-flash",
            LLMProvider.OPENAI: "gpt-4o-mini",
            LLMProvider.OLLAMA: "llama3.2",
            LLMProvider.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2"
        }
        return defaults.get(provider, "gemini-1.5-flash")
    
    def generate(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Gera resposta do LLM com tratamento de erros e métricas
        """
        start_time = time.time()
        
        # Prepara mensagens
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("human", prompt))
        
        # Override de parâmetros se fornecidos
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        
        try:
            # Invoca o LLM
            response = self.llm.invoke(messages, **kwargs)
            
            processing_time = time.time() - start_time
            
            # CORREÇÃO: Trata diferentes tipos de resposta
            # Caso 1: response é uma string direta (Ollama às vezes faz isso)
            if isinstance(response, str):
                return LLMResponse(
                    content=response,
                    model=self.model,
                    provider=self.provider,
                    tokens_used=None,
                    processing_time=processing_time,
                    finish_reason=None,
                    metadata={}
                )
            
            # Caso 2: response tem atributo 'content' (AIMessage)
            if hasattr(response, 'content'):
                content = response.content
                
                # Extrai metadados se disponíveis
                tokens_used = None
                finish_reason = None
                metadata = {}
                
                if hasattr(response, "response_metadata"):
                    metadata = response.response_metadata
                    tokens_used = metadata.get("token_usage", {}).get("total_tokens")
                    finish_reason = metadata.get("finish_reason")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider=self.provider,
                    tokens_used=tokens_used,
                    processing_time=processing_time,
                    finish_reason=finish_reason,
                    metadata=metadata
                )
            
            # Caso 3: Não é string nem tem content, converte para string
            return LLMResponse(
                content=str(response),
                model=self.model,
                provider=self.provider,
                tokens_used=None,
                processing_time=processing_time,
                finish_reason=None,
                metadata={}
            )
        
        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            raise
    
    async def agenerate(
        self, 
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Versão assíncrona da geração
        """
        start_time = time.time()
        
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("human", prompt))
        
        try:
            response = await self.llm.ainvoke(messages, **kwargs)
            processing_time = time.time() - start_time
            
            # CORREÇÃO: Trata diferentes tipos de resposta
            if isinstance(response, str):
                return LLMResponse(
                    content=response,
                    model=self.model,
                    provider=self.provider,
                    tokens_used=None,
                    processing_time=processing_time,
                    finish_reason=None,
                    metadata={}
                )
            
            if hasattr(response, 'content'):
                content = response.content
                tokens_used = None
                finish_reason = None
                metadata = {}
                
                if hasattr(response, "response_metadata"):
                    metadata = response.response_metadata
                    tokens_used = metadata.get("token_usage", {}).get("total_tokens")
                    finish_reason = metadata.get("finish_reason")
                
                return LLMResponse(
                    content=content,
                    model=self.model,
                    provider=self.provider,
                    tokens_used=tokens_used,
                    processing_time=processing_time,
                    finish_reason=finish_reason,
                    metadata=metadata
                )
            
            return LLMResponse(
                content=str(response),
                model=self.model,
                provider=self.provider,
                tokens_used=None,
                processing_time=processing_time,
                finish_reason=None,
                metadata={}
            )
        
        except Exception as e:
            print(f"❌ Erro ao gerar resposta assíncrona: {e}")
            raise
    
    def stream_generate(self, prompt: str, system_message: Optional[str] = None):
        """
        Gera resposta em streaming
        """
        messages = []
        if system_message:
            messages.append(("system", system_message))
        messages.append(("human", prompt))
        
        try:
            for chunk in self.llm.stream(messages):
                if isinstance(chunk, str):
                    yield chunk
                elif hasattr(chunk, 'content'):
                    yield chunk.content
                else:
                    yield str(chunk)
        except Exception as e:
            print(f"❌ Erro no streaming: {e}")
            raise
    
    def get_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o LLM atual"""
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.config.models.llm_temperature,
            "max_tokens": self.config.models.llm_max_tokens,
            "timeout": self.config.models.llm_timeout,
            "fallback_providers": self._fallback_providers
        }


@lru_cache(maxsize=1)
def get_llm_manager() -> LLMManager:
    """
    Retorna instância singleton do gerenciador de LLM
    """
    return LLMManager()


# Funções de conveniência
def generate(prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
    """Função de conveniência para geração síncrona"""
    manager = get_llm_manager()
    response = manager.generate(prompt, system_message, **kwargs)
    return response.content


async def agenerate(prompt: str, system_message: Optional[str] = None, **kwargs) -> str:
    """Função de conveniência para geração assíncrona"""
    manager = get_llm_manager()
    response = await manager.agenerate(prompt, system_message, **kwargs)
    return response.content


if __name__ == "__main__":
    # Teste do sistema de LLM
    print("🧪 Testando sistema de LLM...")
    
    manager = get_llm_manager()
    
    # Info
    info = manager.get_info()
    print(f"📊 LLM Info: {info}")
    
    # Teste de geração
    prompt = "Explique em uma frase o que é RAG."
    response = manager.generate(prompt)
    
    print(f"\n✅ Resposta gerada:")
    print(f"   Conteúdo: {response.content[:100]}...")
    print(f"   Modelo: {response.model}")
    print(f"   Provider: {response.provider}")
    print(f"   Tempo: {response.processing_time:.2f}s")
    if response.tokens_used:
        print(f"   Tokens: {response.tokens_used}")