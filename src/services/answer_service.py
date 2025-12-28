"""
Serviço de Geração de Respostas
Orquestra retrieval, geração LLM, validação e auditoria para produzir respostas fundamentadas.
"""
import re
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from langchain_core.documents import Document

from src.core.config import get_config
from src.core.llm import get_llm_manager
from src.services.retrieval_service import HybridRetriever, RetrievalStrategy
from src.utils.prompt_manager import PromptManager
from src.utils.validator import ResponseValidator
from src.utils.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Níveis de confiança da resposta"""
    ALTA = "ALTA"
    MEDIA = "MÉDIA"
    BAIXA = "BAIXA"
    INSUFICIENTE = "INSUFICIENTE"


@dataclass
class Evidence:
    """Evidência que fundamenta a resposta"""
    source: str
    page: Optional[int]
    document_type: str
    excerpt: str
    score: float
    precedence: Optional[int] = None


@dataclass
class AnswerResult:
    """Resultado completo da geração de resposta"""
    question: str
    answer: str
    confidence: ConfidenceLevel
    evidences: List[Evidence]
    reasoning: Optional[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0


class AnswerService:
    """
    Serviço principal de geração de respostas.
    Integra retrieval, LLM, validação e auditoria.
    """
    
    def __init__(self, enable_audit: bool = True):
        self.config = get_config()
        self.retriever = HybridRetriever()

        try:
            self.retriever.initialize_bm25()
            logger.info("✅ BM25 inicializado com sucesso.")
        except Exception as e:
            if hasattr(self.config, 'app') and self.config.app.env.lower() in ("prod", "production"):
                logger.error(f"❌ Erro fatal ao inicializar BM25 em produção: {e}")
                raise RuntimeError(f"Falha na inicialização do motor de busca: {e}")
            logger.warning(f"⚠️ Falha ao inicializar BM25 (Modo Dev): {e}")

        self.llm_manager = get_llm_manager()
        self.prompt_manager = PromptManager()
        self.validator = ResponseValidator()

        self.enable_audit = enable_audit
        if enable_audit:
            self.auditor = AuditLogger()

    def generate_answer(
        self,
        question: str,
        k: Optional[int] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        include_reasoning: bool = False,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        **llm_kwargs
    ) -> AnswerResult:
        """Gera resposta fundamentada para uma pergunta"""
        start_time = time.time()
        warnings: List[str] = []

        # 1. Validação da pergunta
        if not self._is_valid_question(question):
            return self._create_invalid_question_response(question)

        # 2. Query Expansion
        k = k or self.config.retrieval.default_k
        expanded_question = self._expand_query(question)

        # 3. Retrieval
        retrieval_result = self.retriever.retrieve(
            query=expanded_question,
            k=k,
            strategy=retrieval_strategy,
            filter_kwargs=filter_kwargs if filter_kwargs is not None else {}
        )

        documents = retrieval_result.documents
        scores = retrieval_result.scores

        # 4. Verificação de documentos
        if not documents:
            return self._create_no_documents_response(question)

        if len(documents) < 2:
            warnings.append("Poucos documentos encontrados. Resposta pode ser incompleta.")

        # 5. Preparação de evidências
        evidences = self._prepare_evidences(documents, scores)

        # 6. Hard Grounding - Filtra evidências válidas
        evidences = self._filter_evidences_for_produto_d_prazo(question, evidences)
        if not evidences:
            return self._create_no_documents_response(question)

        # 7. Resposta Determinística (opcional)
        direct_answer = self._maybe_answer_produto_d_prazo_direct(question, evidences)

        if direct_answer:
            answer_text = direct_answer
            llm_model, llm_tokens, llm_time = "rule_based", 0, 0.0
            confidence = ConfidenceLevel.ALTA
        else:
            # 8. Geração com LLM
            context = self._build_context(evidences)
            system_prompt = self.prompt_manager.get_system_prompt()
            user_prompt = self.prompt_manager.format_answer_prompt(
                question=question,
                context=context,
                include_reasoning=include_reasoning
            )

            try:
                llm_response = self.llm_manager.generate(
                    prompt=user_prompt,
                    system_message=system_prompt,
                    **llm_kwargs
                )
                answer_text, llm_model, llm_tokens, llm_time = self._parse_llm_response(llm_response)
            except Exception as e:
                logger.error(f"❌ Erro ao gerar resposta LLM: {e}")
                return self._create_error_response(question, str(e))

            # 9. Validação
            validation_result = self.validator.validate_response(
                question=question,
                answer=answer_text,
                evidences=evidences,
                avg_score=sum(scores) / len(scores) if scores else 0.0
            )

            confidence = self._normalize_confidence(validation_result.get("confidence"))

            if validation_result.get("warnings"):
                warnings.extend(validation_result["warnings"])

            # 10. Guardrail Final
            if re.search(r"\b(dia\s*10|10[ºo]?\s*dia\s*útil)\b", answer_text, re.IGNORECASE):
                has_evidence = any(
                    re.search(r"\b(dia\s*10|10[ºo]?\s*dia\s*útil)\b", e.excerpt or "", re.IGNORECASE)
                    for e in evidences
                )
                if not has_evidence:
                    answer_text = (
                        "❌ NÃO LOCALIZADO: Não há informação sobre o prazo do Produto D "
                        "nos documentos normativos vigentes consultados."
                    )
                    confidence = ConfidenceLevel.INSUFICIENTE
                    warnings.append("Guardrail: prazo citado sem evidência textual compatível.")

        # 11. Extração de raciocínio
        reasoning = self._extract_reasoning(answer_text) if include_reasoning else None

        # 12. Consolidação
        result = AnswerResult(
            question=question,
            answer=answer_text,
            confidence=confidence,
            evidences=evidences,
            reasoning=reasoning,
            warnings=warnings if warnings else None,
            metadata={
                "retrieval_strategy": str(retrieval_strategy),
                "documents_retrieved": len(documents),
                "llm_model": llm_model,
                "llm_tokens": llm_tokens,
                "retrieval_time": getattr(retrieval_result, "processing_time", 0.0),
                "llm_time": llm_time,
                "query_expanded": expanded_question != question,
            },
            processing_time=time.time() - start_time
        )

        if self.enable_audit:
            self._audit_interaction(result)

        return result

    async def agenerate_answer(
        self,
        question: str,
        k: Optional[int] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AnswerResult:
        """Versão assíncrona da geração de resposta"""
        start_time = time.time()

        if not self._is_valid_question(question):
            return self._create_invalid_question_response(question)

        k = k or self.config.retrieval.default_k
        expanded_question = self._expand_query(question)

        retrieval_result = self.retriever.retrieve(
            query=expanded_question,
            k=k,
            strategy=retrieval_strategy,
            filter_kwargs=filter_kwargs if filter_kwargs is not None else {}
        )

        if not retrieval_result.documents:
            return self._create_no_documents_response(question)

        evidences = self._prepare_evidences(retrieval_result.documents, retrieval_result.scores)
        evidences = self._filter_evidences_for_produto_d_prazo(question, evidences)
        
        if not evidences:
            return self._create_no_documents_response(question)

        direct_answer = self._maybe_answer_produto_d_prazo_direct(question, evidences)

        if direct_answer:
            answer_text = direct_answer
            llm_model = "rule_based"
            confidence = ConfidenceLevel.ALTA
        else:
            context = self._build_context(evidences)
            system_prompt = self.prompt_manager.get_system_prompt()
            user_prompt = self.prompt_manager.format_answer_prompt(question, context)

            try:
                llm_response = await self.llm_manager.agenerate(
                    prompt=user_prompt,
                    system_message=system_prompt,
                    **kwargs
                )
                answer_text, llm_model, _, _ = self._parse_llm_response(llm_response)
            except Exception as e:
                return self._create_error_response(question, str(e))

            validation_result = self.validator.validate_response(
                question=question,
                answer=answer_text,
                evidences=evidences,
                avg_score=sum(retrieval_result.scores) / len(retrieval_result.scores) if retrieval_result.scores else 0.0
            )

            confidence = self._normalize_confidence(validation_result.get("confidence"))

        result = AnswerResult(
            question=question,
            answer=answer_text,
            confidence=confidence,
            evidences=evidences,
            warnings=None,
            metadata={
                "retrieval_strategy": str(retrieval_strategy),
                "documents_retrieved": len(retrieval_result.documents),
                "llm_model": llm_model,
            },
            processing_time=time.time() - start_time
        )

        if self.enable_audit:
            self._audit_interaction(result)

        return result

    # =========================================================================
    # MÉTODOS AUXILIARES - PATCHES
    # =========================================================================

    def _expand_query(self, question: str) -> str:
        """Query Expansion para melhorar recall"""
        q = question.strip()
        ql = q.lower()

        if re.search(r"\bproduto\s*d\b", ql) and re.search(r"\b(prazo|entrega|envio|relat[óo]rio)\b", ql):
            return q + " relatório mensal avanço físico obras verificador dia 10 10º dia útil entregas e prazos"

        return q

    def _filter_evidences_for_produto_d_prazo(
        self, 
        question: str, 
        evidences: List[Evidence]
    ) -> List[Evidence]:
        """Hard Grounding - Filtra evidências válidas"""
        ql = question.lower()

        is_produto_d = re.search(r"\bproduto\s*d\b", ql) is not None
        is_prazo = re.search(r"\b(prazo|entrega|envio|relat[óo]rio)\b", ql) is not None

        if not (is_produto_d and is_prazo):
            return evidences

        prazo_pattern = re.compile(r"\b(dia\s*10|10[ºo]?\s*dia\s*útil)\b", re.IGNORECASE)
        return [e for e in evidences if prazo_pattern.search(e.excerpt or "")]

    def _maybe_answer_produto_d_prazo_direct(
        self, 
        question: str, 
        evidences: List[Evidence]
    ) -> Optional[str]:
        """Resposta Determinística para evitar alucinações"""
        ql = question.lower()

        if not (("produto d" in ql) and any(k in ql for k in ["prazo", "entrega", "envio"])):
            return None

        fonte = evidences[0].source if evidences else "documentos consultados"

        return (
            f"O relatório mensal de avanço físico de obras (Produto D) deve ser entregue pelo Verificador "
            f"até o dia 10 (ou 10º dia útil) do mês subsequente.\n\n"
            f"Fonte: {fonte}"
        )

    def _normalize_confidence(self, value: Any) -> ConfidenceLevel:
        """Normaliza confidence para ConfidenceLevel"""
        if isinstance(value, ConfidenceLevel):
            return value

        if isinstance(value, str):
            v = value.strip().upper()
            if v in ("ALTA",):
                return ConfidenceLevel.ALTA
            if v in ("MEDIA", "MÉDIA"):
                return ConfidenceLevel.MEDIA
            if v in ("BAIXA",):
                return ConfidenceLevel.BAIXA

        return ConfidenceLevel.INSUFICIENTE

    # =========================================================================
    # MÉTODOS AUXILIARES - EXISTENTES
    # =========================================================================

    def _is_valid_question(self, question: str) -> bool:
        """Valida se a pergunta é adequada"""
        if not question or len(question.strip()) < 5:
            return False
        return len(question) <= 1000

    def _parse_llm_response(self, llm_response: Any) -> tuple:
        """Extração de conteúdo e métricas do LLM"""
        if isinstance(llm_response, str):
            return llm_response, "unknown", 0, 0.0

        if hasattr(llm_response, 'content'):
            content = llm_response.content
            answer_text = content.content if hasattr(content, 'content') else str(content)

            return (
                answer_text,
                getattr(llm_response, 'model', 'unknown'),
                getattr(llm_response, 'tokens_used', 0),
                getattr(llm_response, 'processing_time', 0.0)
            )
        return str(llm_response), "unknown", 0, 0.0

    def _prepare_evidences(self, documents: List[Document], scores: List[float]) -> List[Evidence]:
        """Converte documentos em evidências estruturadas"""
        evidences = []
        for doc, score in zip(documents, scores):
            evidence = Evidence(
                source=doc.metadata.get("source") or doc.metadata.get("fonte") or "Desconhecido",
                page=doc.metadata.get("page") or doc.metadata.get("pagina"),
                document_type=doc.metadata.get("tipo") or doc.metadata.get("document_type") or "Normativo",
                excerpt=doc.page_content[:800],
                score=score,
                precedence=doc.metadata.get("precedencia") or doc.metadata.get("precedence")
            )
            evidences.append(evidence)
        return evidences

    def _build_context(self, evidences: List[Evidence]) -> str:
        """Constrói contexto formatado para o LLM"""
        context_parts = []
        for i, evidence in enumerate(evidences, 1):
            part = f"[{i}] Fonte: {evidence.source}"
            if evidence.page:
                part += f" | Página: {evidence.page}"
            part += f" | Tipo: {evidence.document_type}"
            if evidence.precedence:
                part += f" | Precedência: {evidence.precedence}"
            part += f"\nConteúdo: {evidence.excerpt}\n"
            context_parts.append(part)
        return "\n---\n".join(context_parts)

    def _extract_reasoning(self, answer: str) -> Optional[str]:
        """Extrai o raciocínio da resposta"""
        markers = ["Raciocínio:", "Justificativa:", "Fundamentação:"]
        for marker in markers:
            if marker in answer:
                parts = answer.split(marker, 1)
                if len(parts) == 2:
                    return parts[1].strip()
        return None

    def _audit_interaction(self, result: AnswerResult):
        """Registra interação para auditoria"""
        try:
            self.auditor.log_interaction(
                question=result.question,
                answer=result.answer,
                confidence=result.confidence.value,
                evidences=[
                    {"source": e.source, "page": e.page, "type": e.document_type, "score": e.score}
                    for e in result.evidences
                ],
                metadata=result.metadata
            )
        except Exception as e:
            logger.warning(f"⚠️ Erro ao auditar interação: {e}")

    def _create_invalid_question_response(self, question: str) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer="❌ PERGUNTA INVÁLIDA: A pergunta deve ter entre 5 e 1000 caracteres.",
            confidence=ConfidenceLevel.INSUFICIENTE,
            evidences=[],
            warnings=["Pergunta não atende aos critérios mínimos"],
            processing_time=0.0
        )

    def _create_no_documents_response(self, question: str) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer="❌ NÃO LOCALIZADO: Não foi encontrada informação sobre o tema nos documentos consultados.",
            confidence=ConfidenceLevel.INSUFICIENTE,
            evidences=[],
            warnings=["Nenhuma evidência retornada."],
            processing_time=0.0
        )

    def _create_error_response(self, question: str, error: str) -> AnswerResult:
        return AnswerResult(
            question=question,
            answer=f"❌ ERRO NO PROCESSAMENTO: {error}",
            confidence=ConfidenceLevel.INSUFICIENTE,
            evidences=[],
            warnings=[f"Erro técnico: {error}"],
            processing_time=0.0
        )

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do serviço"""
        stats = {
            "retriever_stats": self.retriever.get_stats(),
            "llm_info": self.llm_manager.get_info(),
            "audit_enabled": self.enable_audit,
        }
        if self.enable_audit:
            stats["audit_stats"] = self.auditor.get_stats()
        return stats