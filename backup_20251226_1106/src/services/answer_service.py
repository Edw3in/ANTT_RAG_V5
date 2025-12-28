"""
Serviço de Geração de Respostas
Orquestra retrieval, geração LLM, validação e auditoria para produzir respostas fundamentadas.
"""

import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from langchain_core.documents import Document

from src.core.config import get_config
from src.core.llm import get_llm_manager, LLMResponse
from src.services.retrieval_service import HybridRetriever, RetrievalStrategy
from src.utils.prompt_manager import PromptManager
from src.utils.validator import ResponseValidator
from src.utils.audit_logger import AuditLogger


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
    Serviço principal de geração de respostas
    Integra retrieval, LLM, validação e auditoria
    """
    
    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        enable_audit: bool = True
    ):
        self.config = get_config()
        self.retriever = retriever or HybridRetriever()
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
        """
        Gera resposta fundamentada para uma pergunta
        """
        start_time = time.time()
        warnings = []
        
        # 1. Validação da pergunta
        if not self._is_valid_question(question):
            return self._create_invalid_question_response(question)
        
        # 2. Retrieval
        k = k or self.config.retrieval.default_k
        retrieval_result = self.retriever.retrieve(
            query=question,
            k=k,
            strategy=retrieval_strategy,
            filter_kwargs=filter_kwargs or {}
        )
        
        documents = retrieval_result.documents
        scores = retrieval_result.scores
        
        # 3. Verifica se há documentos suficientes
        if not documents:
            return self._create_no_documents_response(question)
        
        if len(documents) < 2:
            warnings.append("Poucos documentos encontrados. Resposta pode ser incompleta.")
        
        # 4. Prepara evidências
        evidences = self._prepare_evidences(documents, scores)
        
        # 5. Monta contexto para o LLM
        context = self._build_context(evidences)
        
        # 6. Carrega e formata prompt
        system_prompt = self.prompt_manager.get_system_prompt()
        user_prompt = self.prompt_manager.format_answer_prompt(
            question=question,
            context=context,
            include_reasoning=include_reasoning
        )
        
        # 7. Gera resposta com LLM
        try:
            llm_response = self.llm_manager.generate(
                prompt=user_prompt,
                system_message=system_prompt,
                **llm_kwargs
            )
            
            # CORREÇÃO: Extrai o conteúdo de forma segura
            if isinstance(llm_response, str):
                answer_text = llm_response
                llm_model = "unknown"
                llm_tokens = 0
                llm_time = 0.0
            elif hasattr(llm_response, 'content'):
                # Se content é string, usa direto
                if isinstance(llm_response.content, str):
                    answer_text = llm_response.content
                # Se content tem atributo content (nested), extrai
                elif hasattr(llm_response.content, 'content'):
                    answer_text = llm_response.content.content
                else:
                    answer_text = str(llm_response.content)
                
                llm_model = getattr(llm_response, 'model', 'unknown')
                llm_tokens = getattr(llm_response, 'tokens_used', 0)
                llm_time = getattr(llm_response, 'processing_time', 0.0)
            else:
                answer_text = str(llm_response)
                llm_model = "unknown"
                llm_tokens = 0
                llm_time = 0.0
                
        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(question, str(e))
        
        # 8. Valida resposta
        validation_result = self.validator.validate_response(
            question=question,
            answer=answer_text,
            evidences=evidences,
            avg_score=sum(scores) / len(scores) if scores else 0.0
        )
        
        confidence = validation_result["confidence"]
        if validation_result.get("warnings"):
            warnings.extend(validation_result["warnings"])
        
        # 9. Extrai raciocínio se solicitado
        reasoning = None
        if include_reasoning:
            reasoning = self._extract_reasoning(answer_text)
        
        # 10. Prepara resultado
        processing_time = time.time() - start_time
        
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
                "retrieval_time": retrieval_result.processing_time,
                "llm_time": llm_time,
            },
            processing_time=processing_time
        )
        
        # 11. Auditoria
        if self.enable_audit:
            self._audit_interaction(result)
        
        return result
    
    async def agenerate_answer(
        self,
        question: str,
        k: Optional[int] = None,
        retrieval_strategy: RetrievalStrategy = RetrievalStrategy.HYBRID_RERANK,
        **kwargs
    ) -> AnswerResult:
        """
        Versão assíncrona da geração de resposta
        """
        start_time = time.time()
        
        # Retrieval
        k = k or self.config.retrieval.default_k
        retrieval_result = self.retriever.retrieve(
            query=question,
            k=k,
            strategy=retrieval_strategy
        )
        
        if not retrieval_result.documents:
            return self._create_no_documents_response(question)
        
        # Prepara contexto
        evidences = self._prepare_evidences(
            retrieval_result.documents,
            retrieval_result.scores
        )
        context = self._build_context(evidences)
        
        # Prompt
        system_prompt = self.prompt_manager.get_system_prompt()
        user_prompt = self.prompt_manager.format_answer_prompt(question, context)
        
        # LLM assíncrono
        try:
            llm_response = await self.llm_manager.agenerate(
                prompt=user_prompt,
                system_message=system_prompt
            )
            
            # CORREÇÃO: Extrai o conteúdo de forma segura
            if isinstance(llm_response, str):
                answer_text = llm_response
                llm_model = "unknown"
            elif hasattr(llm_response, 'content'):
                if isinstance(llm_response.content, str):
                    answer_text = llm_response.content
                elif hasattr(llm_response.content, 'content'):
                    answer_text = llm_response.content.content
                else:
                    answer_text = str(llm_response.content)
                llm_model = getattr(llm_response, 'model', 'unknown')
            else:
                answer_text = str(llm_response)
                llm_model = "unknown"
                
        except Exception as e:
            print(f"❌ Erro ao gerar resposta: {e}")
            import traceback
            traceback.print_exc()
            return self._create_error_response(question, str(e))
        
        # Validação
        validation_result = self.validator.validate_response(
            question=question,
            answer=answer_text,
            evidences=evidences,
            avg_score=sum(retrieval_result.scores) / len(retrieval_result.scores)
        )
        
        processing_time = time.time() - start_time
        
        result = AnswerResult(
            question=question,
            answer=answer_text,
            confidence=validation_result["confidence"],
            evidences=evidences,
            warnings=validation_result.get("warnings"),
            metadata={
                "retrieval_strategy": str(retrieval_strategy),
                "documents_retrieved": len(retrieval_result.documents),
                "llm_model": llm_model,
            },
            processing_time=processing_time
        )
        
        if self.enable_audit:
            self._audit_interaction(result)
        
        return result
    
    def _is_valid_question(self, question: str) -> bool:
        """Valida se a pergunta é adequada"""
        if not question or len(question.strip()) < 5:
            return False
        
        if len(question) > 1000:
            return False
        
        return True
    
    def _prepare_evidences(
        self,
        documents: List[Document],
        scores: List[float]
    ) -> List[Evidence]:
        """Converte documentos em evidências estruturadas"""
        evidences = []
        
        for doc, score in zip(documents, scores):
            evidence = Evidence(
                source=doc.metadata.get("source", "Desconhecido"),
                page=doc.metadata.get("page"),
                document_type=doc.metadata.get("tipo", "Normativo"),
                excerpt=doc.page_content[:800],
                score=score,
                precedence=doc.metadata.get("precedencia")
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
        """Extrai raciocínio da resposta se presente"""
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
                    {
                        "source": e.source,
                        "page": e.page,
                        "type": e.document_type,
                        "score": e.score
                    }
                    for e in result.evidences
                ],
                metadata=result.metadata
            )
        except Exception as e:
            print(f"⚠️  Erro ao auditar interação: {e}")
    
    def _create_invalid_question_response(self, question: str) -> AnswerResult:
        """Cria resposta para pergunta inválida"""
        return AnswerResult(
            question=question,
            answer="❌ PERGUNTA INVÁLIDA: A pergunta deve ter entre 5 e 1000 caracteres.",
            confidence=ConfidenceLevel.INSUFICIENTE,
            evidences=[],
            warnings=["Pergunta não atende aos critérios mínimos"],
            processing_time=0.0
        )
    
    def _create_no_documents_response(self, question: str) -> AnswerResult:
        """Cria resposta quando não há documentos"""
        # Exemplo de regra segura
        if not evidences:
            return AnswerResponse(
                pergunta=question,
                resposta="❌ NÃO LOCALIZADO: Não foi encontrada informação sobre o tema nos documentos consultados.",
                confiabilidade="INSUFICIENTE",
                evidencias=[],
                avisos=["Nenhuma evidência retornada."]
            )   

        # Se há evidências, NUNCA escrever NÃO LOCALIZADO
        # Caso a definição não seja literal, sinalize como inferência/insuficiente
        if confiabilidade == "INSUFICIENTE":
            avisos = (avisos or [])
            if "Resposta indica informação insuficiente" not in avisos:
                avisos.append("Resposta baseada em trechos recuperados; definição pode não estar explícita.")

        return AnswerResponse(
            pergunta=question,
            resposta=resposta_final,            # sua resposta do LLM ou montagem do template
            confiabilidade=confiabilidade,
            evidencias=evidences,
            avisos=avisos
        )
    
    def _create_error_response(self, question: str, error: str) -> AnswerResult:
        """Cria resposta para erro no processamento"""
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


if __name__ == "__main__":
    print("🧪 Testando serviço de resposta...")
    
    service = AnswerService()
    
    question = "Qual o prazo para renovação de acreditação de verificador?"
    result = service.generate_answer(question, k=5)
    
    print(f"\n✅ Resposta gerada:")
    print(f"   Pergunta: {result.question}")
    print(f"   Resposta: {result.answer[:200]}...")
    print(f"   Confiança: {result.confidence}")
    print(f"   Evidências: {len(result.evidences)}")
    print(f"   Tempo: {result.processing_time:.2f}s")