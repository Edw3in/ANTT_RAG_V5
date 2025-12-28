# ANTT RAG System - Dockerfile
# Multi-stage build para otimização de tamanho

# ============================================
# Stage 1: Builder
# ============================================
FROM python:3.11-slim as builder

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /build

# Copia requirements
COPY requirements.txt .

# Instala dependências Python em um diretório isolado
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================
# Stage 2: Runtime
# ============================================
FROM python:3.11-slim

# Metadados
LABEL maintainer="ANTT RAG Team"
LABEL version="4.0.0"
LABEL description="Sistema de RAG para consulta de normativos da ANTT"

# Instala apenas dependências runtime necessárias
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cria usuário não-root
RUN useradd -m -u 1000 antt && \
    mkdir -p /app && \
    chown -R antt:antt /app

# Define diretório de trabalho
WORKDIR /app

# Copia dependências Python do builder
COPY --from=builder /root/.local /home/antt/.local

# Copia código da aplicação
COPY --chown=antt:antt . .

# Cria diretórios necessários
RUN mkdir -p \
    data/bcp/inbox \
    data/bcp/processado \
    data/bcp/rejeitado \
    data/metadata \
    vectorstore \
    00_GOVERNANCA/logs_auditoria \
    prompts \
    logs \
    && chown -R antt:antt data vectorstore 00_GOVERNANCA prompts logs

# Muda para usuário não-root
USER antt

# Adiciona .local/bin ao PATH
ENV PATH=/home/antt/.local/bin:$PATH
ENV PYTHONPATH=/app

# Expõe porta da API
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Comando padrão
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
