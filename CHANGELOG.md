# Changelog - ANTT RAG System

Todas as mudanças notáveis neste projeto serão documentadas neste arquivo.

O formato é baseado em [Keep a Changelog](https://keepachangelog.com/pt-BR/1.0.0/),
e este projeto adere ao [Versionamento Semântico](https://semver.org/lang/pt-BR/).

---

## [4.0.0] - 2024-12-24

### 🎉 Lançamento Completo da Revisão

Esta é uma revisão completa e turbinada do sistema, representando uma reescrita significativa da arquitetura e dos componentes.

### ✨ Adicionado

#### Arquitetura
- Arquitetura modular de 3 camadas (Core, Services, API) com separação clara de responsabilidades
- Sistema de configuração robusto com Pydantic e suporte a variáveis de ambiente
- Gerenciadores abstratos para Embeddings e LLM, permitindo troca de provedores via configuração

#### Funcionalidades RAG
- Sistema de retrieval híbrido combinando busca vetorial (semântica) e BM25 (lexical)
- Reranking opcional usando modelos Cross-Encoder para refinar a relevância dos resultados
- Validação automática de respostas com cálculo de nível de confiabilidade (Alta, Média, Baixa, Insuficiente)
- Suporte a múltiplas estratégias de retrieval: `vector_only`, `bm25_only`, `hybrid`, `hybrid_rerank`

#### API
- API RESTful completa com FastAPI e documentação automática (Swagger/ReDoc)
- Schemas Pydantic para validação rigorosa de requests e responses
- Rotas modulares organizadas por funcionalidade (answer, query, ingest, system)
- Middleware para CORS, GZip, logging de requisições e tratamento de erros
- Endpoints de health check e estatísticas do sistema

#### Ingestão e Metadados
- Serviço de ingestão automatizado com detecção de duplicatas via hash SHA256
- Gerenciador de metadados com banco de dados SQLite para persistência
- Processador de texto avançado para limpeza, normalização e extração de metadados
- Movimentação automática de arquivos processados para pastas organizadas

#### Governança e Auditoria
- Sistema de auditoria completo com logging estruturado em formato JSONL
- Registro de todas as interações: consultas, ingestões, acessos e erros
- Rastreabilidade completa para fins de conformidade e análise

#### Infraestrutura
- Dockerfile multi-stage otimizado para produção
- Docker Compose para orquestração de serviços (API, Redis opcional)
- Scripts de inicialização, ingestão e teste de consultas
- Estrutura de testes com pytest

#### Documentação
- README.md completo com visão geral e instruções de início rápido
- INSTALL.md com guias detalhados para instalação (Docker e manual)
- PROJETO_REVISADO.md com detalhes técnicos da revisão
- ARCHITECTURE.md com arquitetura detalhada e diagramas
- Documentação inline em todos os módulos e funções

### 🔄 Modificado

- Refatoração completa da estrutura de diretórios para seguir padrões de projetos Python modernos
- Substituição de scripts monolíticos por serviços modulares e reutilizáveis
- Migração de configurações hardcoded para sistema de configuração centralizado
- Melhoria significativa na qualidade do código, com type hints e docstrings

### 🚀 Melhorias de Performance

- Cache de embeddings para evitar reprocessamento de textos idênticos
- Pooling de modelos de embedding para reduzir overhead de inicialização
- Busca híbrida com fusão otimizada (Reciprocal Rank Fusion)
- Compressão GZip automática para respostas grandes

### 🔒 Segurança

- Validação de entrada com Pydantic em todos os endpoints
- Suporte a variáveis de ambiente para credenciais sensíveis
- Preparação para autenticação via API keys (desabilitado por padrão)
- Execução de contêiner Docker como usuário não-root

### 📊 Observabilidade

- Logging estruturado com informações de contexto
- Métricas de tempo de processamento em todas as operações
- Endpoints de health check com status de componentes individuais
- Estatísticas detalhadas de uso do sistema

---

## [3.0.0] - 2024-XX-XX (Versão Original)

### Funcionalidades Iniciais

- Sistema básico de RAG com busca vetorial
- Scripts de ingestão e consulta
- API FastAPI simples
- Integração com ChromaDB e Google Gemini

---

## Roadmap Futuro

### [4.1.0] - Planejado

- [ ] Suporte a múltiplos idiomas
- [ ] Interface web para gerenciamento de documentos
- [ ] Integração com Redis para cache distribuído
- [ ] Métricas com Prometheus e dashboards Grafana
- [ ] Autenticação e autorização completa
- [ ] Suporte a documentos Word, Excel e outros formatos

### [5.0.0] - Planejado

- [ ] Sistema de feedback de usuários para melhoria contínua
- [ ] Fine-tuning de modelos de embedding específicos do domínio
- [ ] Integração com sistemas corporativos (Active Directory, etc.)
- [ ] Modo de geração de relatórios e análises
