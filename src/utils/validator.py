"""
Validador de Respostas
Valida qualidade, confiabilidade e conformidade das respostas geradas.
"""
from __future__ import annotations

import re
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    # Apenas para type hints (não executa em runtime)
    from src.services.answer_service import ConfidenceLevel, Evidence



@dataclass
class ValidationResult:
    """Resultado da validação"""
    is_valid: bool
    confidence: "ConfidenceLevel"
    warnings: List[str]
    scores: Dict[str, float]


class ResponseValidator:
    """
    Valida respostas do sistema RAG
    """
    
    def __init__(self):
        # Padrões de não resposta
        self.no_answer_patterns = [
            r"não localizado",
            r"não foi encontrad[oa]",
            r"insuficiente",
            r"não há informação",
            r"não consta",
        ]
        
        # Padrões de conflito
        self.conflict_patterns = [
            r"conflito normativo",
            r"interpretações conflitantes",
            r"divergência",
        ]
        
        # Marcadores de citação
        self.citation_pattern = r"\[(\d+)\]"
    
    def validate_response(
        self,
        question: str,
        answer: str,
        evidences: List["Evidence"],
        avg_score: float
    ) -> Dict[str, Any]:       
        from src.services.answer_service import ConfidenceLevel
        """
        Valida resposta completa e retorna nível de confiança
        """
        from src.services.answer_service import ConfidenceLevel, Evidence
        warnings = []
        scores = {}
        
        # 1. Verifica se é uma não-resposta
        if self._is_no_answer(answer):
            return {
                "is_valid": True,
                "confidence": ConfidenceLevel.INSUFICIENTE,
                "warnings": ["Resposta indica informação insuficiente"],
                "scores": {}
            }
        
        # 2. Valida citações
        citation_score = self._validate_citations(answer, len(evidences))
        scores["citations"] = citation_score
        
        if citation_score < 0.5:
            warnings.append("Poucas citações encontradas na resposta")
        
        # 3. Valida completude
        completeness_score = self._validate_completeness(answer, question)
        scores["completeness"] = completeness_score
        
        if completeness_score < 0.5:
            warnings.append("Resposta pode estar incompleta")
        
        # 4. Valida qualidade das evidências
        evidence_score = self._validate_evidence_quality(evidences, avg_score)
        scores["evidence_quality"] = evidence_score
        
        if evidence_score < 0.5:
            warnings.append("Qualidade das evidências é baixa")
        
        # 5. Detecta conflitos
        has_conflict = self._detect_conflicts(answer)
        if has_conflict:
            warnings.append("Possível conflito normativo detectado")
            return {
                "is_valid": True,
                "confidence": ConfidenceLevel.BAIXA,
                "warnings": warnings,
                "scores": scores
            }
        
        # 6. Calcula confiança geral
        confidence = self._calculate_confidence(scores, len(evidences))
        
        return {
            "is_valid": True,
            "confidence": confidence,
            "warnings": warnings if warnings else None,
            "scores": scores
        }
    
    def _is_no_answer(self, answer: str) -> bool:
        """Verifica se é uma resposta de não localização"""
        answer_lower = answer.lower()
        
        for pattern in self.no_answer_patterns:
            if re.search(pattern, answer_lower):
                return True
        
        return False
    
    def _detect_conflicts(self, answer: str) -> bool:
        """Detecta menções a conflitos normativos"""
        answer_lower = answer.lower()
        
        for pattern in self.conflict_patterns:
            if re.search(pattern, answer_lower):
                return True
        
        return False
    
    def _validate_citations(self, answer: str, num_evidences: int) -> float:
        """
        Valida presença e qualidade das citações
        Retorna score entre 0 e 1
        """
        citations = re.findall(self.citation_pattern, answer)
        
        if not citations:
            return 0.0
        
        # Verifica se citações são válidas
        valid_citations = [
            int(c) for c in citations 
            if c.isdigit() and 1 <= int(c) <= num_evidences
        ]
        
        if not valid_citations:
            return 0.0
        
        # Score baseado na proporção de evidências citadas
        unique_citations = len(set(valid_citations))
        citation_coverage = unique_citations / num_evidences if num_evidences > 0 else 0
        
        # Score baseado na frequência de citações no texto
        citation_density = len(valid_citations) / max(len(answer.split()), 1)
        citation_density = min(citation_density * 100, 1.0)  # Normaliza
        
        # Combina métricas
        score = (citation_coverage * 0.7) + (citation_density * 0.3)
        
        return min(score, 1.0)
    
    def _validate_completeness(self, answer: str, question: str) -> float:
        """
        Valida se a resposta parece completa
        Retorna score entre 0 e 1
        """
        # Tamanho mínimo esperado
        min_length = 100
        ideal_length = 500
        
        answer_length = len(answer)
        
        if answer_length < min_length:
            return 0.3
        
        if answer_length >= ideal_length:
            length_score = 1.0
        else:
            length_score = answer_length / ideal_length
        
        # Verifica estrutura (parágrafos, pontuação)
        has_structure = bool(re.search(r'\.\s+[A-Z]', answer))
        structure_score = 1.0 if has_structure else 0.7
        
        # Combina métricas
        score = (length_score * 0.6) + (structure_score * 0.4)
        
        return min(score, 1.0)
    
    def _validate_evidence_quality(
        self,
        evidences: List["Evidence"],
        avg_score: float
    ) -> float:
        """
        Valida qualidade das evidências
        Retorna score entre 0 e 1
        """
        if not evidences:
            return 0.0
        
        # Score baseado na quantidade de evidências
        num_evidences = len(evidences)
        quantity_score = min(num_evidences / 5, 1.0)  # Ideal: 5+ evidências
        
        # Score baseado na relevância média
        relevance_score = min(avg_score, 1.0)
        
        # Score baseado na precedência (se disponível)
        precedence_scores = [
            1.0 - (e.precedence / 100) 
            for e in evidences 
            if e.precedence is not None
        ]
        
        if precedence_scores:
            precedence_score = sum(precedence_scores) / len(precedence_scores)
        else:
            precedence_score = 0.5  # Neutro se não há info de precedência
        
        # Combina métricas
        score = (
            quantity_score * 0.3 +
            relevance_score * 0.5 +
            precedence_score * 0.2
        )
        
        return min(score, 1.0)
    
    def _calculate_confidence(
        self,
        scores: Dict[str, float],
        num_evidences: int
    ) -> "ConfidenceLevel":
        from src.services.answer_service import ConfidenceLevel
        """
        Calcula nível de confiança geral baseado nos scores
        """
        # Calcula score médio
        avg_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        # Ajusta baseado no número de evidências
        if num_evidences < 2:
            avg_score *= 0.7
        elif num_evidences >= 5:
            avg_score *= 1.1
        
        avg_score = min(avg_score, 1.0)
        
        # Mapeia para níveis de confiança
        if avg_score >= 0.8:
            return ConfidenceLevel.ALTA
        elif avg_score >= 0.6:
            return ConfidenceLevel.MEDIA
        elif avg_score >= 0.4:
            return ConfidenceLevel.BAIXA
        else:
            return ConfidenceLevel.INSUFICIENTE
    
    def validate_question(self, question: str) -> Dict[str, Any]:
        """
        Valida se uma pergunta é adequada
        """
        errors = []
        warnings = []
        
        # Verifica tamanho
        if len(question) < 5:
            errors.append("Pergunta muito curta (mínimo 5 caracteres)")
        
        if len(question) > 1000:
            errors.append("Pergunta muito longa (máximo 1000 caracteres)")
        
        # Verifica se tem conteúdo significativo
        if not re.search(r'[a-zA-Z]{3,}', question):
            errors.append("Pergunta não contém palavras significativas")
        
        # Verifica se parece uma pergunta
        question_indicators = ['?', 'qual', 'como', 'quando', 'onde', 'quem', 'por que', 'o que']
        has_question_indicator = any(
            indicator in question.lower() 
            for indicator in question_indicators
        )
        
        if not has_question_indicator:
            warnings.append("Texto não parece ser uma pergunta")
        
        is_valid = len(errors) == 0
        
        return {
            "is_valid": is_valid,
            "errors": errors if errors else None,
            "warnings": warnings if warnings else None
        }


if __name__ == "__main__":
    # Teste do validador
    print("🧪 Testando validador...")
    
    validator = ResponseValidator()
    
    # Teste 1: Valida pergunta
    question = "Qual o prazo para renovação?"
    q_validation = validator.validate_question(question)
    print(f"✅ Pergunta válida: {q_validation['is_valid']}")
    
    # Teste 2: Valida resposta
    answer = "O prazo é de 30 dias [1]. Conforme estabelecido na resolução [2], o processo deve ser iniciado com antecedência."
    
    from src.services.answer_service import Evidence
    evidences = [
        Evidence("Resolução 123", 1, "Resolução", "Texto exemplo", 0.9),
        Evidence("Portaria 456", 2, "Portaria", "Texto exemplo", 0.8),
    ]
    
    r_validation = validator.validate_response(question, answer, evidences, 0.85)
    print(f"✅ Resposta válida: {r_validation['is_valid']}")
    print(f"   Confiança: {r_validation['confidence']}")
    print(f"   Scores: {r_validation['scores']}")
