#!/usr/bin/env python3
"""
Script de Inicialização do Sistema ANTT RAG
Prepara ambiente, valida configuração e inicializa componentes
"""

import sys
import os
from pathlib import Path

# Adiciona raiz do projeto ao PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import get_config, get_embedding_manager, get_llm_manager
from src.utils import MetadataManager
from src.services import HybridRetriever


def print_header(text: str):
    """Imprime cabeçalho formatado"""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_environment():
    """Verifica variáveis de ambiente necessárias"""
    print_header("Verificando Ambiente")
    
    required_vars = {
        "GOOGLE_API_KEY": "Chave da API do Google Gemini",
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"  ❌ {var}: {description}")
            print(f"❌ {var} não definida")
        else:
            print(f"✅ {var} definida")
    
    if missing:
        print("\n⚠️  Variáveis de ambiente faltando:")
        for m in missing:
            print(m)
        print("\nCopie .env.example para .env e preencha as variáveis necessárias.")
        return False
    
    return True


def validate_config():
    """Valida configuração do sistema"""
    print_header("Validando Configuração")
    
    try:
        config = get_config()
        print(f"✅ Configuração carregada")
        print(f"   Ambiente: {config.environment}")
        print(f"   Debug: {config.debug}")
        
        # Valida ambiente
        config.validate_environment()
        print(f"✅ Ambiente validado")
        
        # Cria diretórios necessários
        config.paths.ensure_directories()
        print(f"✅ Diretórios criados/verificados")
        
        return True
    
    except Exception as e:
        print(f"❌ Erro na configuração: {e}")
        return False


def initialize_embeddings():
    """Inicializa sistema de embeddings"""
    print_header("Inicializando Embeddings")
    
    try:
        manager = get_embedding_manager()
        dimension = manager.get_dimension()
        
        print(f"✅ Sistema de embeddings inicializado")
        print(f"   Modelo: {manager.config.models.embedding}")
        print(f"   Dimensão: {dimension}")
        print(f"   Device: {manager.config.models.embedding_device}")
        
        # Testa embedding
        test_text = "Teste de embedding"
        embedding = manager.embed_query(test_text)
        print(f"✅ Teste de embedding bem-sucedido ({len(embedding)} dims)")
        
        return True
    
    except Exception as e:
        print(f"❌ Erro ao inicializar embeddings: {e}")
        return False


def initialize_llm():
    """Inicializa sistema de LLM"""
    print_header("Inicializando LLM")
    
    try:
        manager = get_llm_manager()
        info = manager.get_info()
        
        print(f"✅ Sistema de LLM inicializado")
        print(f"   Provider: {info['provider']}")
        print(f"   Modelo: {info['model']}")
        print(f"   Temperatura: {info['temperature']}")
        
        # Testa geração
        print("🧪 Testando geração...")
        response = manager.generate("Responda apenas: OK")
        print(f"✅ Teste de geração bem-sucedido")
        print(f"   Resposta: {response.content[:50]}...")
        print(f"   Tempo: {response.processing_time:.2f}s")
        
        return True
    
    except Exception as e:
        print(f"❌ Erro ao inicializar LLM: {e}")
        return False


def initialize_database():
    """Inicializa banco de metadados"""
    print_header("Inicializando Banco de Dados")
    
    try:
        manager = MetadataManager()
        stats = manager.get_stats()
        
        print(f"✅ Banco de metadados inicializado")
        print(f"   Total de documentos: {stats['total_documents']}")
        print(f"   Por status: {stats['by_status']}")
        print(f"   Por tipo: {stats['by_tipo']}")
        
        return True
    
    except Exception as e:
        print(f"❌ Erro ao inicializar banco: {e}")
        return False


def check_vectorstore():
    """Verifica vectorstore"""
    print_header("Verificando Vectorstore")
    
    try:
        config = get_config()
        vectorstore_path = config.paths.vectorstore_dir
        
        if not vectorstore_path.exists():
            print(f"⚠️  Vectorstore não encontrado em {vectorstore_path}")
            print(f"   Execute o script de ingestão para criar o vectorstore")
            return True  # Não é erro crítico
        
        print(f"✅ Vectorstore encontrado")
        
        # Tenta inicializar retriever
        retriever = HybridRetriever()
        stats = retriever.get_stats()
        
        print(f"✅ Retriever inicializado")
        print(f"   Vectorstore: {'OK' if stats['vectorstore_initialized'] else 'Erro'}")
        print(f"   Reranker: {'Habilitado' if stats['reranker_enabled'] else 'Desabilitado'}")
        
        return True
    
    except Exception as e:
        print(f"⚠️  Aviso ao verificar vectorstore: {e}")
        return True  # Não bloqueia inicialização


def create_sample_prompts():
    """Cria prompts de exemplo se não existirem"""
    print_header("Verificando Prompts")
    
    try:
        config = get_config()
        prompts_dir = config.paths.prompts_dir
        
        base_prompt_path = prompts_dir / "base_system.txt"
        
        if not base_prompt_path.exists():
            print("📝 Criando prompt base de exemplo...")
            
            from src.utils import PromptManager
            manager = PromptManager()
            
            # Salva prompt padrão
            default_prompt = manager._get_default_system_prompt()
            manager.save_prompt("base_system.txt", default_prompt)
            
            print(f"✅ Prompt base criado")
        else:
            print(f"✅ Prompts existentes")
        
        return True
    
    except Exception as e:
        print(f"⚠️  Aviso ao verificar prompts: {e}")
        return True


def main():
    """Função principal"""
    print_header("ANTT RAG System - Inicialização")
    
    steps = [
        ("Ambiente", check_environment),
        ("Configuração", validate_config),
        ("Embeddings", initialize_embeddings),
        ("LLM", initialize_llm),
        ("Banco de Dados", initialize_database),
        ("Vectorstore", check_vectorstore),
        ("Prompts", create_sample_prompts),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        try:
            success = step_func()
            results.append((step_name, success))
            
            if not success:
                print(f"\n❌ Falha na etapa: {step_name}")
                break
        
        except Exception as e:
            print(f"\n❌ Erro inesperado em {step_name}: {e}")
            results.append((step_name, False))
            break
    
    # Resumo
    print_header("Resumo da Inicialização")
    
    for step_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n🎉 Sistema inicializado com sucesso!")
        print("\nPróximos passos:")
        print("  1. Execute 'python scripts/ingest_documents.py' para indexar documentos")
        print("  2. Execute 'python api/main.py' para iniciar a API")
        print("  3. Acesse http://localhost:8000/docs para documentação interativa")
        return 0
    else:
        print("\n❌ Inicialização falhou. Corrija os erros acima e tente novamente.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
