export type AnswerResponse = {
  pergunta: string;
  resposta: string;
  confiabilidade: string;
  evidencias: Array<any>;
  avisos?: string[];
  tempo_processamento?: number;
};
