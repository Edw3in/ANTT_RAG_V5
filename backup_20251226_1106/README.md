# ANTT RAG System v4.0

**Sistema de Recuperação e Geração Aumentada para consulta de normativos da ANTT**

---

## 📖 Visão Geral

O **ANTT RAG System** é uma plataforma de Inteligência Artificial de nível empresarial, projetada para transformar a maneira como a Agência Nacional de Transportes Terrestres (ANTT) interage com seu vasto corpo de documentos normativos. A solução utiliza uma arquitetura avançada de **Recuperação e Geração Aumentada (RAG)** para fornecer respostas precisas, rápidas e fundamentadas a partir de uma base de conhecimento documental.

Este projeto representa uma evolução significativa da versão anterior, incorporando as melhores práticas de engenharia de software, MLOps e arquitetura de sistemas de IA para entregar uma solução mais robusta, escalável, segura e eficiente.

### ✨ Principais Funcionalidades

- **Busca Híbrida Avançada**: Combina busca vetorial (semântica) com busca lexical (BM25) e reranking para máxima relevância.
- **Geração de Respostas Fundamentadas**: Gera respostas em linguagem natural, citando as fontes exatas e as páginas dos documentos consultados.
- **Análise de Confiabilidade**: Cada resposta é acompanhada de um nível de confiança (Alta, Média, Baixa, Insuficiente) baseado na qualidade das evidências.
- **API RESTful Completa**: Endpoints para geração de respostas, busca de documentos, ingestão de novos arquivos e monitoramento do sistema.
- **Ingestão Automatizada**: Processa e indexa automaticamente novos documentos PDF colocados em uma pasta de entrada (`inbox`).
- **Governança e Auditoria**: Registra todas as interações, desde consultas até ingestões, garantindo rastreabilidade e conformidade.
- **Arquitetura Modular**: Componentes desacoplados (core, serviços, API) que facilitam a manutenção, testes e futuras expansões.
- **Suporte Multi-Provider**: Interface agnóstica que permite a troca entre diferentes provedores de LLM (Google, OpenAI, Ollama) e modelos de embedding.

---

## 🏗️ Arquitetura do Sistema

O sistema é construído sobre uma arquitetura modular de 3 camadas, garantindo separação de responsabilidades e alta coesão.

| Camada | Componentes | Responsabilidades |
| :--- | :--- | :--- |
| 🤖 **API (Interface)** | `FastAPI`, `Pydantic`, `Uvicorn` | Expor endpoints, validar requisições e respostas, gerenciar ciclo de vida. |
| ⚙️ **Serviços (Lógica)** | `AnswerService`, `HybridRetriever`, `IngestService` | Orquestrar a lógica de negócio (RAG), processar documentos, interagir com o core. |
| 🧠 **Core (Fundação)** | `LLMManager`, `EmbeddingManager`, `Config` | Abstrair acesso a modelos de IA, gerenciar embeddings, centralizar configurações. |
| 🛠️ **Utilitários** | `MetadataManager`, `AuditLogger`, `TextProcessor` | Fornecer funcionalidades de suporte como logging, acesso a metadados e processamento de texto. |

![Arquitetura do Sistema](https://i.imgur.com/example.png)  <!-- Placeholder para diagrama de arquitetura -->

### Fluxo de Dados (RAG)

1.  **Pergunta do Usuário**: Uma requisição chega à API (`/api/v1/answer`).
2.  **Orquestração**: O `AnswerService` recebe a pergunta.
3.  **Recuperação (Retrieval)**: O `HybridRetriever` busca documentos relevantes no `Vectorstore` (ChromaDB) e no índice `BM25`.
4.  **Reranking**: Os resultados são reordenados por um modelo `Cross-Encoder` para refinar a relevância.
5.  **Construção do Contexto**: Os trechos mais relevantes são combinados com um template de prompt gerenciado pelo `PromptManager`.
6.  **Geração (Generation)**: O `LLMManager` envia o prompt formatado para o provedor de LLM configurado (ex: Google Gemini).
7.  **Validação e Auditoria**: A resposta do LLM é validada pelo `ResponseValidator` e a interação é registrada pelo `AuditLogger`.
8.  **Resposta Final**: A API retorna a resposta formatada, incluindo as evidências e o nível de confiança.

---

## 🚀 Como Começar

Siga os passos abaixo para ter o sistema rodando localmente.

### Pré-requisitos

- Python 3.11+
- Docker e Docker Compose (recomendado)
- Chave de API do Google Gemini (ou outro provedor de LLM)

### 1. Instalação

Para instruções detalhadas de instalação e configuração, consulte o arquivo **[INSTALL.md](INSTALL.md)**.

```bash
# 1. Clone o repositório
git clone <your-repo-url>
cd ANTT_RAG_REVISADO

# 2. Crie e configure o arquivo .env
cp .env.example .env
# Edite o .env e adicione sua GOOGLE_API_KEY

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Inicialize o sistema (valida tudo)
python scripts/init_system.py
```

### 2. Ingestão de Documentos

Antes de fazer consultas, você precisa indexar os documentos normativos.

1.  Copie seus arquivos PDF para a pasta `./data/bcp/inbox/`.
2.  Execute o script de ingestão:

```bash
python scripts/ingest_documents.py
```

### 3. Executando a API

Com os documentos indexados, inicie o servidor da API:

```bash
# Usando Uvicorn diretamente (desenvolvimento)
python api/main.py

# Ou com Docker Compose (produção)
docker-compose -f docker/docker-compose.yml up --build
```

### 4. Fazendo uma Consulta

A API estará disponível em `http://localhost:8000`.

- **Documentação Interativa**: Acesse [http://localhost:8000/docs](http://localhost:8000/docs) para testar os endpoints.
- **Via Linha de Comando**: Use o script `test_query.py`:

```bash
python scripts/test_query.py "Qual o prazo para renovação de acreditação de um OIA?"
```

---

## 📚 Documentação Adicional

- **[PROJETO_REVISADO.md](PROJETO_REVISADO.md)**: Detalhes técnicos sobre as melhorias e a nova arquitetura.
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Um mergulho profundo na arquitetura do sistema.
- **[docs/API_USAGE.md](docs/API_USAGE.md)**: Guia completo de uso da API.
- **[docs/GOVERNANCE.md](docs/GOVERNANCE.md)**: Detalhes sobre as funcionalidades de governança e auditoria.

---

## 🤝 Contribuição

Contribuições são bem-vindas. Por favor, siga as diretrizes de desenvolvimento e submeta um Pull Request.

## 📄 Licença

Este projeto é licenciado sob os termos da Licença MIT.
