// src/lib/api.ts
export const API_BASE = "http://127.0.0.1:8001";

export type SystemHealth = { status: string; timestamp: string; service: string };

export type SystemConfig = {
  environment: string;
  debug: boolean;
  paths: { base_dir: string; data_dir: string; vectorstore_dir: string };
  models: {
    embedding: string;
    embedding_device: string;
    reranker_model: string;
    reranker_device: string;
    llm_provider: string;
    llm_model: string;
    ollama_base_url: string;
    fallback_enabled: boolean;
    fallback_provider: string;
    fallback_model: string;
  };
  retrieval: { use_hybrid: boolean; use_reranker: boolean; default_k: number; max_k: number };
  secrets_present: Record<string, boolean>;
};

export type SystemGPU = {
  torch_available: boolean;
  cuda_available: boolean;
  device_name: string;
  device_count: number;
};

export type AnswerEvidence = {
  trecho: string;
  fonte?: string;
  documento?: string;
  pagina?: number;
  score?: number;
};

export type AnswerResponse = {
  pergunta: string;
  resposta: string;
  confiabilidade?: string;
  evidencias?: AnswerEvidence[];
  avisos?: string[];
  tempo_processamento?: number;
};

export type AnswerRequest = {
  pergunta: string;
  use_llm: boolean;
  improve_query?: boolean;
  verify_hallucinations?: boolean;
  method: "hybrid" | "vector";
  k: number;
};

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
  return (await r.json()) as T;
}

async function postJSON<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${url} -> ${r.status}`);
  return (await r.json()) as T;
}

export const api = {
  systemHealth: () => getJSON<SystemHealth>(`${API_BASE}/system/health`),
  systemConfig: () => getJSON<SystemConfig>(`${API_BASE}/system/config`),
  systemGPU: () => getJSON<SystemGPU>(`${API_BASE}/system/gpu`),
  answer: (payload: AnswerRequest) => postJSON<AnswerResponse>(`${API_BASE}/api/v1/answer`, payload),
};
