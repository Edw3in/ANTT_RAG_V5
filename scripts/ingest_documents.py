#!/usr/bin/env python3
"""
Script de Ingestão de Documentos
Processa PDFs da pasta inbox e indexa no vectorstore
"""

import sys
from pathlib import Path

# Adiciona raiz do projeto ao PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from src.services import IngestService
from src.core import get_config


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Processa e indexa documentos PDF no sistema ANTT RAG"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força reprocessamento de documentos já indexados"
    )
    
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Mostra apenas estatísticas sem processar"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("  ANTT RAG - Ingestão de Documentos")
    print("="*60)
    
    try:
        # Inicializa serviço
        service = IngestService()
        
        # Mostra estatísticas se solicitado
        if args.stats:
            stats = service.get_stats()
            print("\n📊 Estatísticas:")
            print(f"   Arquivos na inbox: {stats['inbox_files']}")
            print(f"   Arquivos processados: {stats['processed_files']}")
            print(f"   Arquivos rejeitados: {stats['rejected_files']}")
            print(f"   Documentos indexados: {stats['total_documents_indexed']}")
            print(f"   Chunk size: {stats['chunk_size']}")
            print(f"   Chunk overlap: {stats['chunk_overlap']}")
            return 0
        
        # Verifica se há arquivos para processar
        config = get_config()
        inbox_files = list(config.paths.bcp_inbox.glob("*.pdf"))
        
        if not inbox_files:
            print("\n📭 Nenhum arquivo encontrado na inbox")
            print(f"   Pasta: {config.paths.bcp_inbox}")
            print("\nColoque arquivos PDF na pasta inbox e execute novamente.")
            return 0
        
        print(f"\n📥 Encontrados {len(inbox_files)} arquivo(s) para processar")
        
        if args.force:
            print("⚠️  Modo FORCE ativado - reprocessando todos os documentos")
        
        # Processa documentos
        print("\n🔄 Iniciando processamento...\n")
        
        result = service.ingest_all(force_reprocess=args.force)
        
        # Mostra resultados
        print("\n" + "="*60)
        print("  Resultado do Processamento")
        print("="*60)
        
        print(f"\n📊 Resumo:")
        print(f"   Total de arquivos: {result.total_files}")
        print(f"   ✅ Sucesso: {result.successful}")
        print(f"   ⏭️  Ignorados: {result.skipped}")
        print(f"   ❌ Erros: {result.errors}")
        print(f"   📦 Total de chunks: {result.total_chunks}")
        print(f"   ⏱️  Tempo: {result.processing_time:.2f}s")
        
        # Detalhes por arquivo
        if result.results:
            print(f"\n📄 Detalhes por arquivo:")
            for r in result.results:
                status_icon = {
                    "success": "✅",
                    "duplicate": "⏭️",
                    "skipped": "⏭️",
                    "error": "❌"
                }.get(r.status.value, "❓")
                
                print(f"\n   {status_icon} {r.filename}")
                print(f"      Status: {r.status.value}")
                print(f"      Chunks: {r.chunks_created}")
                print(f"      Páginas: {r.pages_processed}")
                print(f"      Tempo: {r.processing_time:.2f}s")
                
                if r.error_message:
                    print(f"      Erro: {r.error_message}")
        
        # Mensagem final
        if result.errors > 0:
            print(f"\n⚠️  {result.errors} arquivo(s) com erro")
            print(f"   Verifique a pasta 'rejeitado' para detalhes")
            return 1
        else:
            print(f"\n🎉 Processamento concluído com sucesso!")
            return 0
    
    except Exception as e:
        print(f"\n❌ Erro durante processamento: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
