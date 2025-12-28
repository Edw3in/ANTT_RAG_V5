# Arquitetura Detalhada - ANTT RAG System v4.0

Este documento fornece uma visão aprofundada da arquitetura do sistema, seus componentes e os fluxos de dados.

---

## 1. Princípios de Design

A arquitetura foi projetada com base nos seguintes princípios:

-   **Separação de Responsabilidades (SoC)**: Cada componente tem uma única responsabilidade bem definida.
-   **Inversão de Dependência (DI)**: Módulos de alto nível não dependem de módulos de baixo nível; ambos dependem de abstrações.
-   **Componentes Agnósticos**: Isolar o núcleo da lógica de negócio de implementações específicas de tecnologias (ex: modelos de LLM, bancos de dados vetoriais).
-   **Configuração sobre Código**: Permitir que o comportamento do sistema seja modificado através de arquivos de configuração em vez de alterações no código.
-   **Observabilidade**: Projetar o sistema para ser facilmente monitorado, com logging estruturado e endpoints de saúde.

## 2. Diagrama de Arquitetura

O diagrama abaixo ilustra a interação entre os principais componentes do sistema.

```mermaid
graph TD
    subgraph User Interface
        UI[Cliente HTTP / CLI]
    end

    subgraph API Layer (FastAPI)
        A[main.py]
        R1[routes/answer.py]
        R2[routes/query.py]
        R3[routes/ingest.py]
        R4[routes/system.py]
        S[schemas/]
    end

    subgraph Service Layer
        AS[services/AnswerService]
        HR[services/HybridRetriever]
        IS[services/IngestService]
    end

    subgraph Core Layer
        CM[core/Config]
        EM[core/EmbeddingManager]
        LM[core/LLMManager]
    end

    subgraph Utils
        MM[utils/MetadataManager]
        AL[utils/AuditLogger]
        TP[utils/TextProcessor]
        PM[utils/PromptManager]
    end

    subgraph Data & Models
        VS[Vector Store (ChromaDB)]
        DB[(Metadata DB (SQLite))]
        BM25[BM25 Index]
        EmbModel[Embedding Model]
        LLMModel[LLM Model]
        RerankModel[Reranker Model]
    end

    UI --> A
    A --> R1 & R2 & R3 & R4
    R1 & R2 & R3 & R4 -- Validação --> S

    R1 --> AS
    R2 --> HR
    R3 --> IS

    AS --> HR
    AS --> LM
    AS --> PM
    AS --> AL

    HR --> VS
    HR --> BM25
    HR --> EM
    HR --> RerankModel
    HR --> MM

    IS --> TP
    IS --> MM
    IS --> EM
    IS --> VS
    IS --> BM25

    EM --> EmbModel
    LM --> LLMModel

    subgraph Config & Logging
        CM
        AL
    end

    classDef api fill:#D6EAF8,stroke:#333,stroke-width:2px;
    classDef service fill:#D1F2EB,stroke:#333,stroke-width:2px;
    classDef core fill:#FDEDEC,stroke:#333,stroke-width:2px;
    classDef utils fill:#FEF9E7,stroke:#333,stroke-width:2px;
    classDef data fill:#E8DAEF,stroke:#333,stroke-width:2px;

    class A,R1,R2,R3,R4,S api;
    class AS,HR,IS service;
    class CM,EM,LM core;
    class MM,AL,TP,PM utils;
    class VS,DB,BM25,EmbModel,LLMModel,RerankModel data;
```

## 3. Detalhamento dos Componentes

### Camada Core (`src/core`)

É a fundação do sistema, responsável por abstrair as interações com os modelos de IA e gerenciar a configuração.

-   **`Config`**: Carrega e valida as configurações. É um singleton gerenciado por `functools.lru_cache` para garantir que a configuração seja lida apenas uma vez.
-   **`EmbeddingManager`**: Gerencia o ciclo de vida dos modelos de embedding. Implementa um cache para evitar o reprocessamento de textos idênticos. A interface é simples: `embed_documents()` e `embed_query()`.
-   **`LLMManager`**: Atua como uma fábrica para clientes de LLM. A configuração `llm_provider` determina qual cliente será instanciado (ex: `_get_google_client`). Todos os clientes implementam uma interface comum, retornando um objeto `LLMResponse` padronizado.

### Camada de Serviços (`src/services`)

Contém a lógica de negócio e orquestra os componentes do core e dos utilitários para realizar as tarefas.

-   **`IngestService`**: Orquestra o processo de ingestão de um documento. Ele utiliza o `TextProcessor` para extrair e limpar o texto, o `MetadataManager` para salvar os metadados, o `EmbeddingManager` para gerar os embeddings dos chunks e, finalmente, o `HybridRetriever` para adicionar os chunks ao `Vectorstore` e ao índice `BM25`.
-   **`HybridRetriever`**: Responsável por todas as operações de busca. Ele pode operar em diferentes estratégias (`VECTOR_ONLY`, `BM25_ONLY`, `HYBRID`, `HYBRID_RERANK`). A lógica de fusão dos resultados (Reciprocal Rank Fusion) está implementada aqui, assim como a chamada ao modelo de reranking.
-   **`AnswerService`**: É o coração do RAG. Ele utiliza o `HybridRetriever` para obter o contexto, o `PromptManager` para formatar o prompt, o `LLMManager` para gerar a resposta e o `ResponseValidator` para analisar a qualidade da resposta gerada.

### Camada de Utilitários (`src/utils`)

Fornece classes e funções de suporte usadas em todo o sistema.

-   **`MetadataManager`**: Abstrai todas as interações com o banco de dados SQLite de metadados. Usa `sqlite3` diretamente para simplicidade, sem a necessidade de um ORM completo.
-   **`AuditLogger`**: Fornece uma interface para registrar eventos de auditoria em formato JSONL. Isso é crucial para a observabilidade e conformidade.
-   **`TextProcessor`**: Contém toda a lógica para limpeza de texto extraído de PDFs, correção de OCR, divisão em seções, etc.
-   **`PromptManager`**: Gerencia a carga e formatação de templates de prompt, separando a lógica de formatação do prompt do serviço de geração de respostas.

## 4. Fluxos de Dados Principais

### Fluxo de Ingestão

1.  Um arquivo PDF é colocado na pasta `inbox`.
2.  O `IngestService` é acionado (via API ou script).
3.  Para cada arquivo:
    a. O hash SHA256 do arquivo é calculado.
    b. O `MetadataManager` verifica se o hash já existe. Se sim (e `force=False`), o arquivo é ignorado.
    c. O `TextProcessor` extrai o texto bruto do PDF.
    d. O `TextProcessor` limpa e normaliza o texto.
    e. O texto é dividido em chunks usando `RecursiveCharacterTextSplitter`.
    f. O `EmbeddingManager` gera os embeddings para todos os chunks.
    g. Os chunks (com seus embeddings e metadados) são adicionados ao `Vectorstore` (ChromaDB).
    h. Os textos dos chunks são adicionados ao índice `BM25`.
    i. Os metadados do documento são salvos no `MetadataManager`.
    j. O arquivo PDF é movido para a pasta `processado` ou `rejeitado`.

### Fluxo de Geração de Resposta (RAG)

1.  A `AnswerRequest` chega à rota `/answer`.
2.  O `AnswerService.generate_answer` é chamado.
3.  O `HybridRetriever.retrieve` é invocado com a pergunta e a estratégia definida.
    a. Busca vetorial e busca BM25 são executadas em paralelo.
    b. Os resultados são fundidos.
    c. Se a estratégia for `hybrid_rerank`, os resultados fundidos são reordenados pelo `Reranker`.
4.  Os documentos mais relevantes são formatados em um contexto.
5.  O `PromptManager` cria o prompt final, combinando o prompt do sistema, o contexto e a pergunta.
6.  O `LLMManager.generate` envia o prompt para o LLM.
7.  A resposta do LLM é recebida.
8.  O `ResponseValidator` analisa a resposta, verifica se ela se baseia no contexto (groundedness) e calcula um score de confiança.
9.  O `AuditLogger` registra o evento de resposta.
10. O `AnswerService` formata o resultado final no objeto `AnswerResult`.
11. A rota da API serializa o resultado em uma `AnswerResponse` JSON.

---

Este design modular e baseado em abstrações torna o sistema flexível para futuras evoluções, como a adição de novos tipos de fontes de dados, novos modelos de IA ou a integração com outros sistemas corporativos.
