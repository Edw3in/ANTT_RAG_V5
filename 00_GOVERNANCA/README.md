# Diretório de Governança

Este diretório contém arquivos e logs relacionados à governança do sistema ANTT RAG.

## Conteúdo

- **logs_auditoria/**: Logs estruturados em formato JSONL de todas as interações com o sistema (consultas, ingestões, erros).
- **matriz_precedencia.yaml**: (Opcional) Define a hierarquia de precedência entre diferentes tipos de documentos normativos.
- **taxonomia_temas.yaml**: (Opcional) Taxonomia de temas e categorias para classificação de documentos.
- **politica_nao_resposta.md**: (Opcional) Política detalhada sobre quando e como o sistema deve recusar responder a uma pergunta.

## Logs de Auditoria

Os logs de auditoria são gravados em formato JSONL (JSON Lines), onde cada linha é um evento JSON independente. Isso facilita o processamento e análise posterior.

### Exemplo de Evento de Consulta

```json
{
  "timestamp": "2024-12-24T10:30:45.123456",
  "event_type": "answer",
  "question": "Qual o prazo para renovação?",
  "answer_preview": "O prazo para renovação de acreditação...",
  "confidence": "ALTA",
  "evidences_count": 3,
  "processing_time": 1.23
}
```

### Análise de Logs

Você pode usar ferramentas como `jq` para analisar os logs:

```bash
# Contar total de consultas
cat logs_auditoria/audit_*.jsonl | jq -s 'map(select(.event_type == "answer")) | length'

# Listar consultas com baixa confiança
cat logs_auditoria/audit_*.jsonl | jq 'select(.event_type == "answer" and .confidence == "BAIXA")'
```
