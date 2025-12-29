import requests

BASE = "http://127.0.0.1:8001/api/v1/answer"

TESTS = [
    {
        "name": "Fora do domínio (deve falhar com INSUFICIENTE e evidencias vazias)",
        "q": "quem ganhou a copa do mundo?",
        "expect_conf": "INSUFICIENTE",
        "expect_evidences_empty": True,
    },
    {
        "name": "Domínio claro (deve responder com evidências)",
        "q": "Qual a periodicidade de apresentação do relatório mensal de auditoria de serviços operacionais?",
        "expect_conf": None,  # pode variar (MÉDIA/ALTA)
        "expect_evidences_empty": False,
    },
    {
        "name": "Domínio ambíguo (pode responder OU cair no gate, depende do corpus)",
        "q": "relatorio mensal apresentado ate o 5o dia util",
        "expect_conf": None,
        "expect_evidences_empty": None,  # aceitamos ambos, só vamos registrar
    },
]

def main():
    ok = 0
    fail = 0

    for t in TESTS:
        r = requests.post(BASE, json={"pergunta": t["q"]}, timeout=120)
        data = r.json()

        conf = data.get("confiabilidade")
        evid = data.get("evidencias") or []
        evid_empty = (len(evid) == 0)

        print("\n" + "=" * 80)
        print("TEST:", t["name"])
        print("Q:", t["q"])
        print("HTTP:", r.status_code)
        print("CONF:", conf)
        print("EVIDENCES:", len(evid))
        if data.get("avisos"):
            print("WARNINGS:", data["avisos"])

        # asserts “duros”
        local_fail = False
        if t["expect_conf"] is not None and conf != t["expect_conf"]:
            print("FAIL: confiabilidade esperada:", t["expect_conf"])
            local_fail = True

        if t["expect_evidences_empty"] is True and not evid_empty:
            print("FAIL: era esperado evidencias=[]")
            local_fail = True

        if t["expect_evidences_empty"] is False and evid_empty:
            print("FAIL: era esperado evidencias != []")
            local_fail = True

        if local_fail:
            fail += 1
        else:
            ok += 1

    print("\n" + "=" * 80)
    print(f"RESULTADO: OK={ok} FAIL={fail}")

if __name__ == "__main__":
    main()
