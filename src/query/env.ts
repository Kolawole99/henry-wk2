import { config } from 'dotenv';
import * as fs from 'fs';

config();

export interface EnvConfig {
  documentPath: string;
  vectorStorePath: string;
  embeddingModel: string;
  llmModel: string;
  retrievalK: number;
  chunkSize: number;
  chunkOverlap: number;
  openaiApiKey: string;
  openRouterApiKey: string;
}

export function loadAndValidateEnv(): EnvConfig {
  const requiredVars = [
    'DOCUMENT_PATH',
    'VECTOR_STORE_PATH',
    'EMBEDDING_MODEL',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP',
    'LLM_MODEL',
    'RETRIEVAL_K',
  ];

  const missing = requiredVars.filter(v => !process.env[v]);
  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }

  const chunkSize = Number.parseInt(process.env.CHUNK_SIZE!);
  const chunkOverlap = Number.parseInt(process.env.CHUNK_OVERLAP!);
  const retrievalK = Number.parseInt(process.env.RETRIEVAL_K!);

  if (Number.isNaN(chunkSize) || chunkSize <= 0) {
    throw new Error(`CHUNK_SIZE must be a positive number`);
  }

  if (Number.isNaN(chunkOverlap) || chunkOverlap < 0) {
    throw new Error(`CHUNK_OVERLAP must be a non-negative number`);
  }

  if (chunkOverlap >= chunkSize) {
    throw new Error(`CHUNK_OVERLAP must be less than CHUNK_SIZE`);
  }

  if (Number.isNaN(retrievalK) || retrievalK <= 0) {
    throw new Error(`RETRIEVAL_K must be a positive number`);
  }

  if (!fs.existsSync(process.env.DOCUMENT_PATH!)) {
    throw new Error(`DOCUMENT_PATH does not exist: ${process.env.DOCUMENT_PATH}`);
  }

  const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  if (!OPENROUTER_API_KEY && !OPENAI_API_KEY) {
    throw new Error('Either OPENROUTER_API_KEY or OPENAI_API_KEY must be set');
  }

  return {
    documentPath: process.env.DOCUMENT_PATH!,
    vectorStorePath: process.env.VECTOR_STORE_PATH!,
    embeddingModel: process.env.EMBEDDING_MODEL!,
    llmModel: process.env.LLM_MODEL!,
    retrievalK,
    chunkSize,
    chunkOverlap,
    openaiApiKey: process.env.OPENAI_API_KEY!,
    openRouterApiKey: process.env.OPENROUTER_API_KEY!,
  };
}
