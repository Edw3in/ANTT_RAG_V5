# src/utils/text_cleaner.py
"""
Limpeza de Texto de PDFs
Corrige corrupções comuns de extração de PDFs com fontes subset
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional


_LETTERS = r"A-Za-zÀ-ÿ"


@dataclass
class PDFTextCleaner:
    """
    Limpa texto extraído de PDFs corrigindo corrupções comuns (subset fonts).
    A regra de ouro aqui: primeiro corrige padrões conhecidos (palavras),
    depois aplica correções contextuais (char entre letras).
    """

    # Correções de palavras completas (alta precisão)
    word_replacements: Dict[str, str] = field(default_factory=lambda: {
        "Lsico": "físico",
        "Gsico": "físico",
        "uGlizando": "utilizando",
        "uGlizar": "utilizar",
        "cerGficação": "certificação",
        "cerGficado": "certificado",
        "a7vidades": "atividades",
        "a7vidade": "atividade",
        "obje7vo": "objetivo",
        "obje7vos": "objetivos",
        "posi7vo": "positivo",
        "nega7vo": "negativo",
        "rela7vo": "relativo",
        "deGnitivo": "definitivo",
        "idenGficação": "identificação",
        "especLfico": "específico",
        "Gscalização": "fiscalização",
        "Gscalizador": "fiscalizador",
        "GscalGzação": "fiscalização",
        "GnanGciamento": "financiamento",
        "beneLcio": "benefício",
        "perLl": "perfil",
        "difLcil": "difícil",
        "fácGl": "fácil",
    })

    # Correções contextuais (média precisão) – somente entre letras
    # ATENÇÃO: aqui não entra “7 -> ti”, isso é via word_replacements.
    char_between_letters: Dict[str, str] = field(default_factory=lambda: {
        "G": "f",  # cerGficação -> certificação (quando não pegou palavra)
        "L": "f",  # Lsico -> físico (quando não pegou palavra)
        # Evite 7->t aqui como regra geral, pois gera "atvidades" se não houver word replacement.
        # Só use se você tiver evidência de outros casos reais e ainda assim com cuidado.
    })

    def clean(self, text: str) -> str:
        if not text:
            return ""

        text = self._fix_known_words(text)
        text = self._fix_char_between_letters(text)
        return text

    def _fix_known_words(self, text: str) -> str:
        # Substituições case-insensitive preservando capitalização inicial
        for wrong, correct in self.word_replacements.items():
            pattern = re.compile(re.escape(wrong), re.IGNORECASE)

            def repl(m):
                original = m.group(0)
                if original[:1].isupper():
                    return correct[:1].upper() + correct[1:]
                return correct

            text = pattern.sub(repl, text)

        return text

    def _fix_char_between_letters(self, text: str) -> str:
        for wrong, correct in self.char_between_letters.items():
            pattern = rf"(?<=[{_LETTERS}]){re.escape(wrong)}(?=[{_LETTERS}])"
            text = re.sub(pattern, correct, text)
        return text

    def get_statistics(self, text: str) -> Dict[str, int]:
        stats = {"total_words": len(text.split()), "corrupted_hits": 0}
        for wrong in self.word_replacements.keys():
            c = len(re.findall(re.escape(wrong), text, re.IGNORECASE))
            stats["corrupted_hits"] += c
        return stats


_cleaner_instance: Optional[PDFTextCleaner] = None


def get_text_cleaner() -> PDFTextCleaner:
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = PDFTextCleaner()
    return _cleaner_instance
