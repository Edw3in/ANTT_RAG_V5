"""
Testes Básicos do Sistema ANTT RAG
"""

import pytest
from pathlib import Path
import sys

# Adiciona raiz ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestConfig:
    """Testes de configuração"""
    
    def test_load_config(self):
        """Testa carregamento de configuração"""
        from src.core import get_config
        
        config = get_config()
        assert config is not None
        assert config.environment in ["development", "staging", "production"]
    
    def test_config_paths(self):
        """Testa caminhos da configuração"""
        from src.core import get_config
        
        config = get_config()
        assert config.paths.base_dir is not None
        assert config.paths.vectorstore_dir is not None


class TestEmbeddings:
    """Testes de embeddings"""
    
    def test_embedding_manager(self):
        """Testa gerenciador de embeddings"""
        from src.core import get_embedding_manager
        
        manager = get_embedding_manager()
        assert manager is not None
        assert manager.get_dimension() > 0
    
    def test_embed_query(self):
        """Testa embedding de query"""
        from src.core import embed_query
        
        embedding = embed_query("teste")
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)


class TestTextProcessor:
    """Testes de processamento de texto"""
    
    def test_clean_text(self):
        """Testa limpeza de texto"""
        from src.utils import TextProcessor
        
        processor = TextProcessor()
        
        dirty_text = "  Texto   com    espaços   "
        clean = processor.clean_text(dirty_text)
        
        assert clean == "Texto com espaços"
    
    def test_text_stats(self):
        """Testa estatísticas de texto"""
        from src.utils import TextProcessor
        
        processor = TextProcessor()
        
        text = "Este é um texto de teste. Tem duas sentenças."
        stats = processor.get_text_stats(text)
        
        assert stats["total_words"] > 0
        assert stats["total_sentences"] > 0


class TestValidator:
    """Testes de validação"""
    
    def test_validate_question(self):
        """Testa validação de pergunta"""
        from src.utils import ResponseValidator
        
        validator = ResponseValidator()
        
        # Pergunta válida
        result = validator.validate_question("Qual é o prazo?")
        assert result["is_valid"] is True
        
        # Pergunta muito curta
        result = validator.validate_question("Ok")
        assert result["is_valid"] is False


class TestMetadataManager:
    """Testes de gerenciador de metadados"""
    
    def test_create_manager(self):
        """Testa criação do gerenciador"""
        from src.utils import MetadataManager
        
        manager = MetadataManager()
        assert manager is not None
    
    def test_get_stats(self):
        """Testa obtenção de estatísticas"""
        from src.utils import MetadataManager
        
        manager = MetadataManager()
        stats = manager.get_stats()
        
        assert "total_documents" in stats
        assert isinstance(stats["total_documents"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
