import React, { useEffect, useMemo, useState } from "react";
import { api } from "./lib/api";
import type { AnswerResponse, SystemHealth, SystemConfig, SystemGPU } from "./lib/api";
import { StatCard } from "./components/StatCard";
import { Chip } from "./components/Chip";
import { Cpu, Database, ShieldCheck, Sparkles } from "lucide-react";

export default function App() {
  // ---- Sidebar controls ----
  const [useLLM, setUseLLM] = useState(true);
  const [method, setMethod] = useState<"hybrid" | "vector">("hybrid");
  const [k, setK] = useState(5);
  const [improveQuery, setImproveQuery] = useState(false);
  const [checkHallucinations, setCheckHallucinations] = useState(false);

  // ---- Ask ----
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [answer, setAnswer] = useState<AnswerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // ---- System cards ----
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [cfg, setCfg] = useState<SystemConfig | null>(null);
  const [gpu, setGPU] = useState<SystemGPU | null>(null);

  const examples = useMemo(
    () => [
      "Quais os requisitos para acreditação de OIA?",
      "Como funciona o tacógrafo digital?",
      "Procedimentos para certificação de veículos",
      "Normas de segurança para transporte de cargas",
    ],
    []
  );

  async function refreshSystem() {
    try {
      setError(null);
      const [h, c, g] = await Promise.all([api.systemHealth(), api.systemConfig(), api.systemGPU()]);
      setHealth(h);
      setCfg(c);
      setGPU(g);
    } catch (e: any) {
      setError(e?.message ?? "Falha ao carregar status do sistema.");
    }
  }

  useEffect(() => {
    refreshSystem();
  }, []);

  async function onAsk() {
    const q = question.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setAnswer(null);

    try {
      const payload = {
        pergunta: q,
        use_llm: useLLM,
        improve_query: improveQuery,
        verify_hallucinations: checkHallucinations,
        method,
        k,
      };
      const data = await api.answer(payload);
      setAnswer(data);
    } catch (e: any) {
      setError(e?.message ?? "Falha ao consultar.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <div style={styles.brand}>
          <div style={styles.brandIcon}>
            <Sparkles size={18} />
          </div>
          <div>
            <div style={styles.brandTitle}>Consulta Inteligente</div>
            <div style={styles.brandSub}>Sistema RAG para normativos ANTT</div>
          </div>
        </div>

        <div style={styles.headerRight}>
          <span style={styles.badgeOnline}>{health?.status === "ok" ? "Sistema Online" : "Indisponível"}</span>
        </div>
      </header>

      <main style={styles.main}>
        {/* LEFT: Main */}
        <section style={styles.left}>
          <div style={styles.cardsGrid}>
            <StatCard
              title="Status API"
              value={health?.status ?? "—"}
              subtitle={
                health
                  ? `Serviço: ${health.service} • ${new Date(health.timestamp).toLocaleString()}`
                  : "Carregando…"
              }
              right={<ShieldCheck size={18} />}
            />
            <StatCard
              title="GPU / CUDA"
              value={gpu?.cuda_available ? "Ativo" : "Inativo"}
              subtitle={gpu ? `${gpu.device_name} • dispositivos: ${gpu.device_count}` : "Carregando…"}
              right={<Cpu size={18} />}
            />
            <StatCard
              title="LLM (Backend)"
              value={cfg?.models?.llm_provider ? cfg.models.llm_provider : "—"}
              subtitle={
                cfg
                  ? `Modelo: ${cfg.models.llm_model} • fallback: ${cfg.models.fallback_provider}/${cfg.models.fallback_model}`
                  : "Carregando…"
              }
              right={<Sparkles size={18} />}
            />
            <StatCard
              title="Vectorstore"
              value={cfg?.paths?.vectorstore_dir ? "OK" : "—"}
              subtitle={cfg ? cfg.paths.vectorstore_dir : "Carregando…"}
              right={<Database size={18} />}
            />
          </div>

          <div style={styles.panel}>
            <div style={styles.panelTitle}>Faça sua Consulta</div>
            <div style={styles.panelSub}>
              Digite sua pergunta sobre normativos da ANTT e obtenha respostas com evidências.
            </div>

            <textarea
              style={styles.textarea}
              placeholder="Ex.: Quais são os requisitos para acreditação de OIA?"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />

            <div style={styles.row}>
              <div style={styles.rowLeft}>
                <Chip label={`Método: ${method}`} />
                <Chip label={`Documentos: ${k}`} />
              </div>

              <button style={styles.btn} onClick={onAsk} disabled={loading}>
                {loading ? "Consultando..." : "Consultar"}
              </button>
            </div>

            {/* Result */}
            <div style={styles.result}>
              {error ? <div style={styles.errorBox}>{error}</div> : null}

              {answer ? (
                <>
                  <div style={styles.resultHeader}>
                    <div style={styles.resultTitle}>Resposta</div>
                    <div style={styles.resultMeta}>
                      <Chip label={`Confiabilidade: ${answer.confiabilidade ?? "—"}`} />
                      {typeof answer.tempo_processamento === "number" ? (
                        <Chip label={`Tempo: ${answer.tempo_processamento.toFixed(2)}s`} />
                      ) : null}
                    </div>
                  </div>

                  <div style={styles.answerBox}>{answer.resposta}</div>

                  <div style={styles.evidTitle}>Evidências</div>
                  {answer.evidencias && answer.evidencias.length > 0 ? (
                    <div style={styles.evidList}>
                      {answer.evidencias.map((ev, idx) => (
                        <div key={idx} style={styles.evidItem}>
                          <div style={styles.evidText}>{ev.trecho}</div>
                          <div style={styles.evidMeta}>
                            {ev.documento ? <Chip label={`Doc: ${ev.documento}`} /> : null}
                            {ev.pagina != null ? <Chip label={`Pág: ${ev.pagina}`} /> : null}
                            {ev.score != null ? <Chip label={`Score: ${ev.score.toFixed(3)}`} /> : null}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={styles.muted}>Nenhuma evidência retornada.</div>
                  )}

                  {answer.avisos && answer.avisos.length > 0 ? (
                    <>
                      <div style={styles.evidTitle}>Avisos</div>
                      <ul style={styles.ul}>
                        {answer.avisos.map((a, i) => (
                          <li key={i} style={styles.li}>
                            {a}
                          </li>
                        ))}
                      </ul>
                    </>
                  ) : null}
                </>
              ) : (
                <div style={styles.muted}>Sem consulta ainda.</div>
              )}
            </div>
          </div>
        </section>

        {/* RIGHT: Sidebar */}
        <aside style={styles.right}>
          <div style={styles.sideCard}>
            <div style={styles.sideTitle}>Configurações</div>
            <div style={styles.sideSub}>Ajuste os parâmetros da consulta</div>

            <div style={styles.toggleRow}>
              <div>Usar LLM</div>
              <input type="checkbox" checked={useLLM} onChange={(e) => setUseLLM(e.target.checked)} />
            </div>

            <div style={styles.toggleRow}>
              <div>Aprimorar Query</div>
              <input type="checkbox" checked={improveQuery} onChange={(e) => setImproveQuery(e.target.checked)} />
            </div>

            <div style={styles.toggleRow}>
              <div>Verificar Alucinações</div>
              <input
                type="checkbox"
                checked={checkHallucinations}
                onChange={(e) => setCheckHallucinations(e.target.checked)}
              />
            </div>

            <div style={styles.hr} />

            <div style={styles.field}>
              <div style={styles.label}>Método de Busca</div>
              <select value={method} onChange={(e) => setMethod(e.target.value as any)} style={styles.select}>
                <option value="hybrid">Híbrido (Recomendado)</option>
                <option value="vector">Vetorial</option>
              </select>
            </div>

            <div style={styles.field}>
              <div style={styles.label}>Número de Documentos (k)</div>
              <select value={k} onChange={(e) => setK(Number(e.target.value))} style={styles.select}>
                {[3, 5, 8, 10, 15, 20].map((n) => (
                  <option key={n} value={n}>
                    {n} documentos
                  </option>
                ))}
              </select>
            </div>

            <button style={styles.btnSecondary} onClick={refreshSystem}>
              Atualizar Status
            </button>
          </div>

          <div style={styles.sideCard}>
            <div style={styles.sideTitle}>Exemplos Rápidos</div>
            <div style={styles.sideSub}>Clique para preencher a pergunta</div>

            <div style={styles.examples}>
              {examples.map((ex) => (
                <button key={ex} style={styles.exampleBtn} onClick={() => setQuestion(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        </aside>
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  page: {
    minHeight: "100vh",
    background: "radial-gradient(1200px 600px at 20% 0%, rgba(70,110,255,0.25), transparent), #0b1020",
    color: "rgba(255,255,255,0.92)",
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
  },
  header: {
    height: 64,
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "0 20px",
    borderBottom: "1px solid rgba(255,255,255,0.08)",
  },
  brand: { display: "flex", alignItems: "center", gap: 12 },
  brandIcon: {
    width: 36,
    height: 36,
    borderRadius: 10,
    display: "grid",
    placeItems: "center",
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.10)",
  },
  brandTitle: { fontWeight: 800, letterSpacing: 0.2 },
  brandSub: { fontSize: 12, opacity: 0.75 },
  headerRight: { display: "flex", gap: 10, alignItems: "center" },
  badgeOnline: {
    padding: "6px 12px",
    borderRadius: 999,
    fontSize: 12,
    border: "1px solid rgba(255,255,255,0.14)",
    background: "rgba(0,200,140,0.18)",
  },
  main: {
    display: "grid",
    gridTemplateColumns: "1fr 340px",
    gap: 18,
    padding: 18,
    alignItems: "start",
  },
  left: { minWidth: 0 },
  right: { display: "flex", flexDirection: "column", gap: 14 },
  cardsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
    gap: 12,
    marginBottom: 14,
  },
  panel: {
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: 16,
    background: "rgba(255,255,255,0.03)",
    boxShadow: "0 10px 28px rgba(0,0,0,0.28)",
  },
  panelTitle: { fontSize: 18, fontWeight: 800 },
  panelSub: { fontSize: 13, opacity: 0.75, marginTop: 6, marginBottom: 12 },
  textarea: {
    width: "100%",
    minHeight: 120,
    resize: "vertical",
    borderRadius: 14,
    padding: 12,
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.25)",
    color: "rgba(255,255,255,0.92)",
    outline: "none",
    fontSize: 14,
    lineHeight: 1.4,
  },
  row: { display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 12, gap: 12 },
  rowLeft: { display: "flex", gap: 8, flexWrap: "wrap" },
  btn: {
    borderRadius: 12,
    padding: "10px 14px",
    border: "1px solid rgba(255,255,255,0.14)",
    background: "rgba(70,110,255,0.35)",
    color: "rgba(255,255,255,0.95)",
    fontWeight: 700,
    cursor: "pointer",
  },
  btnSecondary: {
    width: "100%",
    marginTop: 12,
    borderRadius: 12,
    padding: "10px 14px",
    border: "1px solid rgba(255,255,255,0.14)",
    background: "rgba(255,255,255,0.06)",
    color: "rgba(255,255,255,0.92)",
    fontWeight: 700,
    cursor: "pointer",
  },
  result: {
    marginTop: 16,
    borderTop: "1px solid rgba(255,255,255,0.08)",
    paddingTop: 14,
  },
  resultHeader: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" },
  resultTitle: { fontSize: 15, fontWeight: 800 },
  resultMeta: { display: "flex", gap: 8, flexWrap: "wrap" },
  answerBox: {
    marginTop: 10,
    padding: 12,
    borderRadius: 14,
    border: "1px solid rgba(255,255,255,0.10)",
    background: "rgba(0,0,0,0.22)",
    lineHeight: 1.45,
    whiteSpace: "pre-wrap",
  },
  evidTitle: { marginTop: 14, fontWeight: 800, fontSize: 13, opacity: 0.9 },
  evidList: { display: "flex", flexDirection: "column", gap: 10, marginTop: 10 },
  evidItem: {
    padding: 12,
    borderRadius: 14,
    border: "1px solid rgba(255,255,255,0.10)",
    background: "rgba(255,255,255,0.03)",
  },
  evidText: { fontSize: 13, lineHeight: 1.45, opacity: 0.92, whiteSpace: "pre-wrap" },
  evidMeta: { display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 },
  muted: { marginTop: 10, fontSize: 13, opacity: 0.7 },
  errorBox: {
    padding: 10,
    borderRadius: 12,
    border: "1px solid rgba(255,90,90,0.35)",
    background: "rgba(255,90,90,0.12)",
    marginBottom: 10,
    fontSize: 13,
  },
  sideCard: {
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: 14,
    background: "rgba(255,255,255,0.03)",
  },
  sideTitle: { fontWeight: 900, fontSize: 16 },
  sideSub: { fontSize: 12, opacity: 0.75, marginTop: 6, marginBottom: 12 },
  toggleRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 0",
    borderBottom: "1px dashed rgba(255,255,255,0.10)",
    fontSize: 13,
  },
  hr: { height: 1, background: "rgba(255,255,255,0.10)", margin: "12px 0" },
  field: { marginTop: 10 },
  label: { fontSize: 12, opacity: 0.8, marginBottom: 6 },
  select: {
    width: "100%",
    borderRadius: 12,
    padding: "10px 10px",
    border: "1px solid rgba(255,255,255,0.12)",
    background: "rgba(0,0,0,0.22)",
    color: "rgba(255,255,255,0.92)",
    outline: "none",
  },
  examples: { display: "flex", flexDirection: "column", gap: 8 },
  exampleBtn: {
    textAlign: "left",
    borderRadius: 12,
    padding: "10px 12px",
    border: "1px solid rgba(255,255,255,0.10)",
    background: "rgba(255,255,255,0.04)",
    color: "rgba(255,255,255,0.90)",
    cursor: "pointer",
    fontSize: 13,
  },
  ul: { marginTop: 10, paddingLeft: 18 },
  li: { fontSize: 13, opacity: 0.85, marginBottom: 6 },
};
