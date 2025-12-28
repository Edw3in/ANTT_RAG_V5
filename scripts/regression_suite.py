# scripts/regression_suite.py
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
import requests

BASE_URL = "http://127.0.0.1:8001"

# Estratégias a testar (você pode reduzir se quiser)
STRATEGIES = ["bm25_only", "vector_only", "hybrid", "hybrid_rerank"]

# Regex-alvo para considerar "evidência-alvo presente"
# Ajuste conforme o padrão que você quer garantir em prazos, etc.
TARGET_REGEXES = {
    "prazo_dia10": re.compile(r"\bdia\s*10\b|\b10[ºo]\s*dia\s*útil\b", re.IGNORECASE),
    "prazo_dia05": re.compile(r"\bdia\s*05\b|\bdia\s*5\b", re.IGNORECASE),
    "sei": re.compile(r"\bSEI\b", re.IGNORECASE),
    "exif": re.compile(r"\bEXIF\b", re.IGNORECASE),
}

# 20 perguntas (exemplo inicial)
# Você pode/ deve substituir por perguntas reais de uso recorrente.
TESTS = [
    {"id": 1, "question": "Qual o prazo do Produto D?", "target": "prazo_dia10"},
    {"id": 2, "question": "Quando o verificador deve entregar o Produto D?", "target": "prazo_dia10"},
    {"id": 3, "question": "Qual o prazo de entrega do relatório mensal de avanço físico de obras?", "target": "prazo_dia10"},
    {"id": 4, "question": "Qual é o prazo da Concessionária para entregar o relatório mensal (Produto D)?", "target": "prazo_dia05"},
    {"id": 5, "question": "Quais são os prazos mensais para Concessionária e Verificador?", "target": "prazo_dia10"},
    {"id": 6, "question": "Qual o primeiro relatório mensal e sua data de entrega?", "target": "prazo_dia05"},
    {"id": 7, "question": "As comunicações formais devem ocorrer por qual meio?", "target": "sei"},
    {"id": 8, "question": "Como deve ser feita a comunicação de emergências?", "target": None},
    {"id": 9, "question": "Relatórios devem incluir fotos georreferenciadas? Quais requisitos?", "target": None},
    {"id": 10, "question": "É necessário manter metadados EXIF ativados nos relatórios?", "target": "exif"},
    {"id": 11, "question": "Qual o limite de tamanho no SEI e o que fazer se exceder?", "target": None},
    {"id": 12, "question": "O que é o Produto D?", "target": None},
    {"id": 13, "question": "O que é o Produto C e sua periodicidade?", "target": None},
    {"id": 14, "question": "O que é o Produto E no modelo double check?", "target": None},
    {"id": 15, "question": "Qual o prazo do Produto Z?", "target": None},  # esperado INSUFICIENTE
    {"id": 16, "question": "Qual a data de início das atividades em campo?", "target": None},
    {"id": 17, "question": "Quem é responsável por apresentar plano de trabalho detalhado?", "target": None},
    {"id": 18, "question": "Existe reunião mensal de acompanhamento? Qual formato?", "target": None},
    {"id": 19, "question": "Quais produtos são acompanhados inicialmente na medida cautelar?", "target": None},
    {"id": 20, "question": "Quais unidades da ANTT participam das manifestações pós-entrega do relatório?", "target": None},
]

def evidence_target_present(evidences, target_key):
    if not target_key:
        return None
    rx = TARGET_REGEXES.get(target_key)
    if not rx:
        return None
    for e in evidences or []:
        trecho = (e.get("trecho") or "")
        if rx.search(trecho):
            return True
    return False

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)

    csv_path = out_dir / f"regression_{ts}.csv"
    jsonl_path = out_dir / f"regression_{ts}.jsonl"

    fields = [
        "test_id", "strategy", "k", "question",
        "http_status",
        "confiabilidade",
        "tempo_processamento",
        "retrieval_time", "llm_time",
        "llm_model",
        "documents_retrieved",
        "evidence_target_present",
        "top_evidence_source", "top_evidence_page", "top_evidence_score"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, open(jsonl_path, "w", encoding="utf-8") as fjsonl:
        writer = csv.DictWriter(fcsv, fieldnames=fields)
        writer.writeheader()

        for t in TESTS:
            for strat in STRATEGIES:
                payload = {"pergunta": t["question"], "k": 5, "estrategia": strat}

                start = time.time()
                r = requests.post(f"{BASE_URL}/api/v1/answer", json=payload, timeout=120)
                elapsed = time.time() - start

                row = {
                    "test_id": t["id"],
                    "strategy": strat,
                    "k": 5,
                    "question": t["question"],
                    "http_status": r.status_code,
                    "confiabilidade": None,
                    "tempo_processamento": None,
                    "retrieval_time": None,
                    "llm_time": None,
                    "llm_model": None,
                    "documents_retrieved": None,
                    "evidence_target_present": None,
                    "top_evidence_source": None,
                    "top_evidence_page": None,
                    "top_evidence_score": None,
                }

                if r.status_code == 200:
                    data = r.json()
                    evids = data.get("evidencias") or []
                    meta = data.get("metadata") or {}

                    row["confiabilidade"] = data.get("confiabilidade")
                    row["tempo_processamento"] = data.get("tempo_processamento", elapsed)
                    row["retrieval_time"] = meta.get("retrieval_time")
                    row["llm_time"] = meta.get("llm_time")
                    row["llm_model"] = meta.get("llm_model")
                    row["documents_retrieved"] = meta.get("documents_retrieved")

                    row["evidence_target_present"] = evidence_target_present(evids, t.get("target"))

                    if evids:
                        # considera a 1ª evidência como “top” (se sua ordenação já reflete relevância)
                        top = evids[0]
                        row["top_evidence_source"] = top.get("fonte")
                        row["top_evidence_page"] = top.get("pagina")
                        row["top_evidence_score"] = top.get("score")

                    fjsonl.write(json.dumps({
                        "test": t,
                        "strategy": strat,
                        "payload": payload,
                        "response": data
                    }, ensure_ascii=False) + "\n")

                else:
                    try:
                        fjsonl.write(json.dumps({
                            "test": t, "strategy": strat, "payload": payload,
                            "error": r.text
                        }, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                writer.writerow(row)
                print(f"[{t['id']:02d}] {strat} -> {r.status_code} conf={row['confiabilidade']} target={row['evidence_target_present']}")

    print("\nArquivos gerados:")
    print(f"CSV : {csv_path}")
    print(f"JSONL: {jsonl_path}")

if __name__ == "__main__":
    main()
