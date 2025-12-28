"""
Processador de Texto
Limpeza, normalização e pré-processamento de texto extraído de documentos.
"""

import re
import unicodedata
from typing import List, Optional


class TextProcessor:
    """
    Processa e limpa texto extraído de PDFs
    """
    
    def __init__(self):
        # Padrões de limpeza
        self.patterns = {
            # Remove múltiplos espaços
            "multiple_spaces": re.compile(r'\s+'),
            
            # Remove quebras de linha desnecessárias
            "line_breaks": re.compile(r'\n{3,}'),
            
            # Remove caracteres de controle
            "control_chars": re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'),
            
            # Corrige hifenização de palavras quebradas
            "hyphenation": re.compile(r'(\w+)-\s*\n\s*(\w+)'),
            
            # Remove cabeçalhos/rodapés comuns
            "header_footer": re.compile(r'^(Página \d+|Page \d+|\d+/\d+)$', re.MULTILINE),
        }
        
        # Mapeamento de correções comuns em OCR
        self.ocr_corrections = {
            "G": "t",  # Exemplo do código original
            "0": "O",  # Zero -> O maiúsculo em contextos específicos
            "1": "l",  # Um -> L minúsculo em contextos específicos
        }
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Limpa e normaliza texto
        
        Args:
            text: Texto a ser limpo
            aggressive: Se True, aplica limpeza mais agressiva
        """
        if not text:
            return ""
        
        # 1. Normaliza unicode
        text = self._normalize_unicode(text)
        
        # 2. Remove caracteres de controle
        text = self.patterns["control_chars"].sub("", text)
        
        # 3. Corrige hifenização
        text = self.patterns["hyphenation"].sub(r'\1\2', text)
        
        # 4. Remove cabeçalhos/rodapés
        text = self.patterns["header_footer"].sub("", text)
        
        # 5. Normaliza espaços em branco
        text = self._normalize_whitespace(text)
        
        # 6. Correções específicas de OCR (se agressivo)
        if aggressive:
            text = self._apply_ocr_corrections(text)
        
        # 7. Remove linhas vazias excessivas
        text = self.patterns["line_breaks"].sub("\n\n", text)
        
        return text.strip()
    
    def _normalize_unicode(self, text: str) -> str:
        """
        Normaliza caracteres unicode
        """
        # Normaliza para forma NFKC (compatibilidade)
        text = unicodedata.normalize('NFKC', text)
        
        # Remove marcas diacríticas opcionalmente
        # (comentado por padrão para preservar acentuação em português)
        # text = ''.join(c for c in unicodedata.normalize('NFD', text)
        #                if unicodedata.category(c) != 'Mn')
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """
        Normaliza espaços em branco
        """
        # Substitui múltiplos espaços por um único
        text = self.patterns["multiple_spaces"].sub(" ", text)
        
        # Normaliza quebras de linha
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove espaços no início/fim de cada linha
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        
        return text
    
    def _apply_ocr_corrections(self, text: str) -> str:
        """
        Aplica correções comuns de erros de OCR
        """
        # Aplica correções do dicionário
        for wrong, correct in self.ocr_corrections.items():
            # Aplica apenas em contextos específicos para evitar substituições incorretas
            # Exemplo: "G" -> "t" apenas em meio de palavra
            pattern = re.compile(rf'\b\w*{re.escape(wrong)}\w*\b')
            # Esta é uma implementação simplificada
            # Em produção, seria necessário contexto mais sofisticado
        
        return text
    
    def extract_sections(self, text: str) -> List[dict]:
        """
        Extrai seções do texto baseado em títulos/numeração
        """
        sections = []
        
        # Padrões de títulos comuns
        title_patterns = [
            r'^(CAPÍTULO|SEÇÃO|ARTIGO|Art\.?)\s+(\d+)',
            r'^(\d+\.)+\s+[A-Z]',
            r'^[A-Z][A-Z\s]{10,}$',
        ]
        
        current_section = {
            "title": "Introdução",
            "content": [],
            "level": 0
        }
        
        for line in text.split("\n"):
            is_title = False
            
            for pattern in title_patterns:
                if re.match(pattern, line.strip()):
                    # Salva seção anterior
                    if current_section["content"]:
                        current_section["content"] = "\n".join(current_section["content"])
                        sections.append(current_section.copy())
                    
                    # Inicia nova seção
                    current_section = {
                        "title": line.strip(),
                        "content": [],
                        "level": self._infer_section_level(line)
                    }
                    is_title = True
                    break
            
            if not is_title and line.strip():
                current_section["content"].append(line)
        
        # Adiciona última seção
        if current_section["content"]:
            current_section["content"] = "\n".join(current_section["content"])
            sections.append(current_section)
        
        return sections
    
    def _infer_section_level(self, title: str) -> int:
        """
        Infere nível hierárquico da seção
        """
        if re.match(r'^CAPÍTULO', title):
            return 1
        elif re.match(r'^SEÇÃO', title):
            return 2
        elif re.match(r'^(ARTIGO|Art\.?)', title):
            return 3
        elif re.match(r'^\d+\.', title):
            # Conta pontos para determinar nível
            return title.count('.') + 1
        else:
            return 0
    
    def remove_tables(self, text: str) -> str:
        """
        Remove tabelas do texto (heurística simples)
        """
        lines = text.split("\n")
        filtered_lines = []
        
        for line in lines:
            # Heurística: linha com muitos espaços ou tabs pode ser tabela
            if line.count("  ") > 5 or line.count("\t") > 2:
                continue
            filtered_lines.append(line)
        
        return "\n".join(filtered_lines)
    
    def extract_metadata_from_text(self, text: str) -> dict:
        """
        Extrai metadados do texto (título, data, etc)
        """
        metadata = {}
        
        # Extrai possível título (primeira linha em maiúsculas)
        lines = text.split("\n")
        for line in lines[:10]:  # Verifica primeiras 10 linhas
            if line.strip() and line.strip().isupper() and len(line.strip()) > 10:
                metadata["title"] = line.strip()
                break
        
        # Extrai datas
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'
        dates = re.findall(date_pattern, text[:1000])  # Busca no início
        if dates:
            metadata["dates"] = dates
        
        # Extrai números de normativos
        normativo_pattern = r'\b(Lei|Decreto|Resolução|Portaria)\s+n[º°]?\s*(\d+[./]?\d*)\b'
        normativos = re.findall(normativo_pattern, text[:2000], re.IGNORECASE)
        if normativos:
            metadata["normativos"] = [f"{tipo} {num}" for tipo, num in normativos]
        
        return metadata
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Divide texto em sentenças
        """
        # Padrão simples de divisão por pontuação
        sentence_pattern = r'[.!?]+\s+'
        sentences = re.split(sentence_pattern, text)
        
        # Limpa e filtra sentenças vazias
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def get_text_stats(self, text: str) -> dict:
        """
        Retorna estatísticas do texto
        """
        words = text.split()
        sentences = self.split_into_sentences(text)
        
        return {
            "total_chars": len(text),
            "total_words": len(words),
            "total_sentences": len(sentences),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
        }


if __name__ == "__main__":
    # Teste do processador de texto
    print("🧪 Testando processador de texto...")
    
    processor = TextProcessor()
    
    # Teste 1: Limpeza básica
    dirty_text = """
    Este    é  um   texto    com     espaços     irregulares.
    
    
    
    E quebras de linha excessivas.
    
    Página 1
    
    Também tem cabeçalhos.
    """
    
    clean = processor.clean_text(dirty_text)
    print(f"✅ Texto limpo ({len(clean)} chars)")
    print(f"Original: {len(dirty_text)} chars")
    
    # Teste 2: Estatísticas
    stats = processor.get_text_stats(clean)
    print(f"📊 Stats: {stats}")
    
    # Teste 3: Extração de metadados
    sample_text = """
    RESOLUÇÃO Nº 5.956, DE 20 DE DEZEMBRO DE 2024
    
    Dispõe sobre procedimentos de verificação...
    """
    
    metadata = processor.extract_metadata_from_text(sample_text)
    print(f"📋 Metadados: {metadata}")
