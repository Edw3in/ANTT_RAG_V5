#!/usr/bin/env python3
"""
Script de Teste de Query
Testa o sistema de retrieval e geração de respostas
"""

import sys
from pathlib import Path

# Adiciona raiz do projeto ao PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.services import AnswerService, RetrievalStrategy


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Testa consultas no sistema ANTT RAG"
    )
    
    parser.add_argument(
        "pergunta",
        type=str,
        help="Pergunta a ser respondida"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Número de documentos a recuperar (padrão: 5)"
    )
    
    parser.add_argument(
        "--estrategia",
        choices=["vector", "bm25", "hybrid", "hybrid_rerank"],
        default="hybrid_rerank",
        help="Estratégia de retrieval (padrão: hybrid_rerank)"
    )
    
    parser.add_argument(
        "--raciocinio",
        action="store_true",
        help="Inclui raciocínio na resposta"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mostra informações detalhadas"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("  ANTT RAG - Teste de Query")
    print("="*80)
    
    try:
        # Mapeia estratégia
        strategy_map = {
            "vector": RetrievalStrategy.VECTOR_ONLY,
            "bm25": RetrievalStrategy.BM25_ONLY,
            "hybrid": RetrievalStrategy.HYBRID,
            "hybrid_rerank": RetrievalStrategy.HYBRID_RERANK,
        }
        
        strategy = strategy_map[args.estrategia]
        
        print(f"\n📝 Pergunta: {args.pergunta}")
        print(f"🔍 Estratégia: {args.estrategia}")
        print(f"📊 K: {args.k}")
        print(f"\n{'='*80}\n")
        
        # Inicializa serviço
        print("🔄 Inicializando sistema...")
        service = AnswerService()
        
        # Gera resposta
        print("🤔 Processando pergunta...\n")
        
        result = service.generate_answer(
            question=args.pergunta,
            k=args.k,
            retrieval_strategy=strategy,
            include_reasoning=args.raciocinio
        )
        
        # Mostra resposta
        print("="*80)
        print("  RESPOSTA")
        print("="*80)
        print(f"\n{result.answer}\n")
        
        # Confiabilidade
        confidence_icon = {
            "ALTA": "🟢",
            "MÉDIA": "🟡",
            "BAIXA": "🟠",
            "INSUFICIENTE": "🔴"
        }.get(result.confidence.value, "⚪")
        
        print("="*80)
        print(f"  CONFIABILIDADE: {confidence_icon} {result.confidence.value}")
        print("="*80)
        
        # Evidências
        if result.evidences:
            print(f"\n📚 EVIDÊNCIAS ({len(result.evidences)}):\n")
            
            for i, evidence in enumerate(result.evidences, 1):
                print(f"{i}. {evidence.source}")
                if evidence.page:
                    print(f"   Página: {evidence.page}")
                print(f"   Tipo: {evidence.document_type}")
                print(f"   Score: {evidence.score:.3f}")
                if evidence.precedence:
                    print(f"   Precedência: {evidence.precedence}")
                
                if args.verbose:
                    print(f"   Trecho: {evidence.excerpt[:200]}...")
                
                print()
        
        # Raciocínio
        if result.reasoning:
            print("="*80)
            print("  RACIOCÍNIO")
            print("="*80)
            print(f"\n{result.reasoning}\n")
        
        # Avisos
        if result.warnings:
            print("="*80)
            print("  ⚠️  AVISOS")
            print("="*80)
            for warning in result.warnings:
                print(f"   • {warning}")
            print()
        
        # Metadados
        if args.verbose and result.metadata:
            print("="*80)
            print("  METADADOS")
            print("="*80)
            for key, value in result.metadata.items():
                print(f"   {key}: {value}")
            print()
        
        # Tempo de processamento
        print("="*80)
        print(f"  ⏱️  Tempo de processamento: {result.processing_time:.2f}s")
        print("="*80)
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Erro durante consulta: {e}")
        
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
