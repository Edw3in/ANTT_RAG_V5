import { useMemo, useState } from "react";

type Method = "hybrid" | "vector" | "bm25";

export default function ConsultaInteligente() {
  const [question, setQuestion] = useState("");
  const [useLLM, setUseLLM] = useState(true);
  const [improveQuery, setImproveQuery] = useState(false);
  const [checkHallucinations, setCheckHallucinations] = useState(false);

  const [method, setMethod] = useState<Method>("hybrid");
  const [k, setK] = useState<number>(5);

  const examples = useMemo(
    () => [
      "Quais os requisitos para acreditação de OIA?",
      "Como funciona o tacógrafo digital?",
      "Procedimentos para certificação de veículos",
      "Normas de segurança para transporte de cargas",
    ],
    []
  );

  async function onAsk() {
    if (!question.trim()) return;

    const payload = {
      query: question,
      use_llm: useLLM,
      improve_query: improveQuery,
      verify_hallucinations: checkHallucinations,
      method,
      k,
    };

    console.log("Enviando requisição:", payload);

    try {
      const r = await fetch("http://127.0.0.1:8001/api/v1/answer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await r.json();
      console.log("Resposta recebida:", data);
    } catch (error) {
      console.error("Erro ao consultar API:", error);
    }
  }

  return (
    <div style={{ padding: 24, fontFamily: "Segoe UI, Arial, sans-serif", backgroundColor: "#f5f5f5", minHeight: "100vh" }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 18 }}>
        <button 
          style={{ 
            border: "1px solid #ddd", 
            padding: "8px 12px", 
            borderRadius: 8, 
            background: "#fff",
            cursor: "pointer",
            fontSize: 14
          }}
        >
          ← Voltar
        </button>

        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 20, fontWeight: 700 }}>Consulta Inteligente</div>
          <div style={{ fontSize: 13, color: "#666" }}>Sistema RAG para normativos ANTT</div>
        </div>

        <div style={{ 
          border: "1px solid #bfe9c9", 
          background: "#eafff0", 
          color: "#1c7c3a", 
          padding: "6px 12px", 
          borderRadius: 999,
          fontSize: 13,
          fontWeight: 600
        }}>
          ● Sistema Online
        </div>
      </div>

      {/* Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 18 }}>
        {/* Left main */}
        <div style={{ border: "1px solid #eee", borderRadius: 14, padding: 18, background: "#fff", boxShadow: "0 1px 3px rgba(0,0,0,0.05)" }}>
          <div style={{ fontSize: 22, fontWeight: 700, marginBottom: 6 }}>Faça sua Consulta</div>
          <div style={{ fontSize: 13, color: "#666", marginBottom: 14 }}>
            Digite sua pergunta sobre normativos da ANTT e obtenha respostas precisas com fontes
          </div>

          <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>Pergunta</div>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ex.: Quais são os requisitos para acreditação de OIA?"
            style={{
              width: "100%",
              minHeight: 120,
              borderRadius: 12,
              border: "1px solid #ddd",
              padding: 12,
              outline: "none",
              resize: "vertical",
              fontFamily: "inherit",
              fontSize: 14
            }}
          />

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 12 }}>
            <div style={{ display: "flex", gap: 10 }}>
              <span style={{ 
                border: "1px solid #ddd", 
                borderRadius: 999, 
                padding: "6px 10px", 
                fontSize: 12,
                background: "#f9f9f9"
              }}>
                Método: {method}
              </span>
              <span style={{ 
                border: "1px solid #ddd", 
                borderRadius: 999, 
                padding: "6px 10px", 
                fontSize: 12,
                background: "#f9f9f9"
              }}>
                Documentos: {k}
              </span>
            </div>

            <button
              onClick={onAsk}
              disabled={!question.trim()}
              style={{
                border: "none",
                borderRadius: 12,
                padding: "10px 20px",
                background: question.trim() ? "#4f7cff" : "#ccc",
                color: "#fff",
                fontWeight: 700,
                cursor: question.trim() ? "pointer" : "not-allowed",
                fontSize: 14
              }}
            >
              🔍 Consultar
            </button>
          </div>
        </div>

        {/* Right column */}
        <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
          {/* Config card */}
          <div style={{ border: "1px solid #eee", borderRadius: 14, padding: 18, background: "#fff", boxShadow: "0 1px 3px rgba(0,0,0,0.05)" }}>
            <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>⚙️ Configurações</div>
            <div style={{ fontSize: 13, color: "#666", marginBottom: 14 }}>Ajuste os parâmetros da consulta</div>

            <Toggle label="Usar LLM" value={useLLM} onChange={setUseLLM} />
            <Toggle label="Aprimorar Query" value={improveQuery} onChange={setImproveQuery} />
            <Toggle label="Verificar Alucinações" value={checkHallucinations} onChange={setCheckHallucinations} />

            <div style={{ height: 1, background: "#eee", margin: "14px 0" }} />

            <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>Método de Busca</div>
            <select
              value={method}
              onChange={(e) => setMethod(e.target.value as Method)}
              style={{ 
                width: "100%", 
                padding: 10, 
                borderRadius: 12, 
                border: "1px solid #ddd", 
                background: "#fff",
                fontSize: 13,
                cursor: "pointer"
              }}
            >
              <option value="hybrid">Híbrido (Recomendado)</option>
              <option value="vector">Vetorial</option>
              <option value="bm25">BM25</option>
            </select>

            <div style={{ fontSize: 13, fontWeight: 700, marginTop: 14, marginBottom: 6 }}>Número de Documentos</div>
            <select
              value={k}
              onChange={(e) => setK(parseInt(e.target.value, 10))}
              style={{ 
                width: "100%", 
                padding: 10, 
                borderRadius: 12, 
                border: "1px solid #ddd", 
                background: "#fff",
                fontSize: 13,
                cursor: "pointer"
              }}
            >
              {[3, 5, 8, 10, 15, 20].map((n) => (
                <option key={n} value={n}>
                  {n} documentos
                </option>
              ))}
            </select>
          </div>

          {/* Examples card */}
          <div style={{ border: "1px solid #eee", borderRadius: 14, padding: 18, background: "#fff", boxShadow: "0 1px 3px rgba(0,0,0,0.05)" }}>
            <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 10 }}>⚡ Exemplos Rápidos</div>

            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {examples.map((ex) => (
                <button
                  key={ex}
                  onClick={() => setQuestion(ex)}
                  style={{
                    textAlign: "left",
                    border: "1px solid #ddd",
                    borderRadius: 12,
                    padding: "10px 12px",
                    background: "#fff",
                    cursor: "pointer",
                    fontSize: 13,
                    transition: "all 0.2s"
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "#f9f9f9";
                    e.currentTarget.style.borderColor = "#4f7cff";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "#fff";
                    e.currentTarget.style.borderColor = "#ddd";
                  }}
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Toggle(props: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  const { label, value, onChange } = props;
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
      <div style={{ fontSize: 13, fontWeight: 600 }}>{label}</div>
      <button
        onClick={() => onChange(!value)}
        style={{
          width: 46,
          height: 26,
          borderRadius: 999,
          border: "1px solid #ddd",
          background: value ? "#4f7cff" : "#f3f3f3",
          position: "relative",
          cursor: "pointer",
          transition: "background 0.2s"
        }}
        aria-label={label}
      >
        <span
          style={{
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "#fff",
            position: "absolute",
            top: 2,
            left: value ? 24 : 2,
            transition: "left 120ms ease",
            boxShadow: "0 1px 2px rgba(0,0,0,0.2)"
          }}
        />
      </button>
    </div>
  );
}