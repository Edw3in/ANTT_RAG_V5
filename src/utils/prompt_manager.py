"""
Gerenciador de Prompts
Carrega, formata e gerencia templates de prompts do sistema.
"""

from pathlib import Path
from typing import Dict, Any, Optional
from string import Template

from src.core.config import get_config


class PromptManager:
    """
    Gerencia templates de prompts e formatação
    """
    
    def __init__(self, prompts_dir: Optional[Path] = None):
        self.config = get_config()
        self.prompts_dir = prompts_dir or self.config.paths.prompts_dir
        
        if not self.prompts_dir:
            self.prompts_dir = self.config.paths.base_dir / "prompts"
        
        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache de prompts carregados
        self._cache = {}
    
    def get_system_prompt(self) -> str:
        """
        Retorna o prompt de sistema base
        """
        return self.load_prompt("base_system.txt")
    
    def get_answer_prompt_template(self) -> str:
        """
        Retorna template para geração de respostas
        """
        template = """
Você deve responder à seguinte pergunta baseando-se EXCLUSIVAMENTE no contexto fornecido.

## CONTEXTO:
{context}

## PERGUNTA:
{question}

## INSTRUÇÕES:
1. Use APENAS informações presentes no contexto acima
2. Cite as fontes usando [número] conforme aparecem no contexto
3. Se a informação não estiver no contexto, responda com a política de não resposta
4. Seja preciso, objetivo e fundamentado
5. Estruture a resposta de forma clara e profissional

{additional_instructions}

## SUA RESPOSTA:
"""
        return template
    
    def format_answer_prompt(
        self,
        question: str,
        context: str,
        include_reasoning: bool = False,
        additional_instructions: str = ""
    ) -> str:
        """
        Formata prompt para geração de resposta
        """
        template = self.get_answer_prompt_template()
        
        if include_reasoning:
            additional_instructions += "\n6. Inclua uma seção 'Raciocínio:' explicando seu processo de análise"
        
        return template.format(
            context=context,
            question=question,
            additional_instructions=additional_instructions
        )
    
    def get_checklist_prompt_template(self) -> str:
        """
        Retorna template para geração de checklists
        """
        template = """
Com base no contexto fornecido, gere um checklist detalhado para: {objetivo}

## CONTEXTO:
{context}

## FORMATO ESPERADO:
- [ ] Item 1: Descrição clara
- [ ] Item 2: Descrição clara
...

## CHECKLIST:
"""
        return template
    
    def format_checklist_prompt(
        self,
        objetivo: str,
        context: str
    ) -> str:
        """
        Formata prompt para geração de checklist
        """
        template = self.get_checklist_prompt_template()
        return template.format(objetivo=objetivo, context=context)
    
    def get_conformidade_prompt_template(self) -> str:
        """
        Retorna template para análise de conformidade
        """
        template = """
Analise a conformidade do seguinte item com os requisitos normativos:

## ITEM A ANALISAR:
{item}

## REQUISITOS NORMATIVOS:
{context}

## ANÁLISE REQUERIDA:
1. Status de conformidade (Conforme / Não Conforme / Parcialmente Conforme)
2. Requisitos atendidos
3. Requisitos não atendidos (se houver)
4. Recomendações para adequação

## SUA ANÁLISE:
"""
        return template
    
    def format_conformidade_prompt(
        self,
        item: str,
        context: str
    ) -> str:
        """
        Formata prompt para análise de conformidade
        """
        template = self.get_conformidade_prompt_template()
        return template.format(item=item, context=context)
    
    def get_relatorio_prompt_template(self) -> str:
        """
        Retorna template para geração de relatórios
        """
        template = """
Gere um relatório técnico sobre: {titulo}

## DADOS PARA O RELATÓRIO:
{dados}

## ESTRUTURA DO RELATÓRIO:
1. Resumo Executivo
2. Introdução
3. Análise Detalhada
4. Conclusões
5. Recomendações

## RELATÓRIO:
"""
        return template
    
    def format_relatorio_prompt(
        self,
        titulo: str,
        dados: str
    ) -> str:
        """
        Formata prompt para geração de relatório
        """
        template = self.get_relatorio_prompt_template()
        return template.format(titulo=titulo, dados=dados)
    
    def load_prompt(self, filename: str) -> str:
        """
        Carrega prompt de arquivo com cache
        """
        if filename in self._cache:
            return self._cache[filename]
        
        prompt_path = self.prompts_dir / filename
        
        if not prompt_path.exists():
            # Retorna prompt padrão se arquivo não existir
            print(f"⚠️  Prompt não encontrado: {filename}, usando padrão")
            return self._get_default_prompt(filename)
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self._cache[filename] = content
            return content
        
        except Exception as e:
            print(f"❌ Erro ao carregar prompt {filename}: {e}")
            return self._get_default_prompt(filename)
    
    def save_prompt(self, filename: str, content: str):
        """
        Salva prompt em arquivo
        """
        prompt_path = self.prompts_dir / filename
        
        try:
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Atualiza cache
            self._cache[filename] = content
            print(f"✅ Prompt salvo: {filename}")
        
        except Exception as e:
            print(f"❌ Erro ao salvar prompt {filename}: {e}")
    
    def _get_default_prompt(self, filename: str) -> str:
        """
        Retorna prompt padrão baseado no nome do arquivo
        """
        if "system" in filename.lower():
            return self._get_default_system_prompt()
        elif "checklist" in filename.lower():
            return "Gere um checklist baseado nas informações fornecidas."
        elif "conformidade" in filename.lower():
            return "Analise a conformidade com os requisitos normativos."
        elif "relatorio" in filename.lower():
            return "Gere um relatório técnico detalhado."
        else:
            return "Responda baseando-se nas informações fornecidas."
    
    def _get_default_system_prompt(self) -> str:
        """
        Retorna prompt de sistema padrão
        """
        return """Você é um assistente técnico especializado da ANTT (Agência Nacional de Transportes Terrestres), responsável por responder consultas sobre normas, procedimentos e diretrizes relacionadas a verificadores e organismos de inspeção acreditados (OIA).

## RESTRIÇÕES ABSOLUTAS (INEGOCIÁVEIS):

1. **FONTE ÚNICA DE VERDADE:**
   - Use EXCLUSIVAMENTE o conteúdo dos documentos fornecidos no CONTEXTO.
   - NUNCA utilize conhecimento externo, informações gerais ou suposições.
   - Se a informação não estiver no CONTEXTO, você DEVE aplicar a Política de Não Resposta.

2. **CITAÇÃO OBRIGATÓRIA:**
   - TODA afirmação factual deve ter citação no formato: [n].
   - As citações devem corresponder aos documentos numerados no CONTEXTO.
   - Exemplo correto: "O prazo é de 30 dias [1]".

3. **POLÍTICA DE NÃO RESPOSTA:**
   Quando aplicável, retorne uma destas mensagens literais:
   a) "❌ NÃO LOCALIZADO: Não há informação sobre o tema nos documentos normativos vigentes consultados."
   b) "⚠️ INSUFICIENTE: Os trechos localizados são insuficientes para uma conclusão definitiva."
   c) "⚠️ CONFLITO NORMATIVO: Dispositivos [X] e [Y] apresentam interpretações conflitantes. Validação humana necessária."

4. **QUALIDADE DA RESPOSTA:**
   - Seja preciso, objetivo e fundamentado
   - Use linguagem técnica apropriada
   - Estruture respostas de forma clara e profissional
   - Priorize documentos com maior precedência normativa

5. **CONFORMIDADE:**
   - Todas as respostas devem ser auditáveis
   - Mantenha rastreabilidade das fontes
   - Indique nível de confiança quando apropriado
"""
    
    def list_prompts(self) -> list:
        """
        Lista todos os prompts disponíveis
        """
        if not self.prompts_dir.exists():
            return []
        
        return [p.name for p in self.prompts_dir.glob("*.txt")]
    
    def reload_cache(self):
        """
        Limpa o cache de prompts
        """
        self._cache.clear()
        print("🔄 Cache de prompts limpo")


if __name__ == "__main__":
    # Teste do gerenciador de prompts
    print("🧪 Testando gerenciador de prompts...")
    
    manager = PromptManager()
    
    # Teste 1: Carrega prompt de sistema
    system_prompt = manager.get_system_prompt()
    print(f"✅ System prompt carregado ({len(system_prompt)} chars)")
    
    # Teste 2: Formata prompt de resposta
    answer_prompt = manager.format_answer_prompt(
        question="Qual o prazo?",
        context="[1] O prazo é de 30 dias"
    )
    print(f"✅ Answer prompt formatado ({len(answer_prompt)} chars)")
    
    # Teste 3: Lista prompts
    prompts = manager.list_prompts()
    print(f"📋 Prompts disponíveis: {prompts}")
