# src/utils/text_processor.py
"""
Processador de Texto
Limpeza, normalização e pré-processamento de texto extraído de documentos.
"""

import re
import unicodedata
from typing import Optional
from src.utils.text_cleaner import get_text_cleaner


class TextProcessor:
    def __init__(self):
        self.cleaner = get_text_cleaner()

        self.patterns = {
            "multiple_spaces": re.compile(r"[ \t]+"),
            "line_breaks": re.compile(r"\n{3,}"),
            "control_chars": re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]"),
            "hyphenation": re.compile(r"(\w+)-\s*\n\s*(\w+)"),
            "header_footer": re.compile(r"^(Página \d+|Page \d+|\d+/\d+)$", re.MULTILINE),
        }

    def clean_text(self, text: str, aggressive: bool = False) -> str:
        if not text:
            return ""

        # 1) Unicode
        text = unicodedata.normalize("NFKC", text)

        # 2) Remove controles
        text = self.patterns["control_chars"].sub("", text)

        # 3) Corrige hifenização quebrada
        text = self.patterns["hyphenation"].sub(r"\1\2", text)

        # 4) Remove header/footer simples
        text = self.patterns["header_footer"].sub("", text)

        # 5) Normaliza quebras de linha
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 6) Normaliza espaços por linha
        lines = [self.patterns["multiple_spaces"].sub(" ", ln).strip() for ln in text.split("\n")]
        text = "\n".join(lines)

        # 7) Remove quebras excessivas
        text = self.patterns["line_breaks"].sub("\n\n", text)

        # 8) Corrige corrupções típicas de PDF (sempre; não é OCR)
        text = self.cleaner.clean(text)

        # 9) Ajuste final
        return text.strip()
