import { config } from 'dotenv';
import * as fs from 'fs';

config();

export function validateEnvironmentVariables() {
  const requiredVars = [
    'DOCUMENT_PATH',
    'VECTOR_STORE_PATH',
    'EMBEDDING_MODEL',
    'CHUNK_SIZE',
    'CHUNK_OVERLAP'
  ];

  const missingVars = requiredVars.filter(varName => !process.env[varName]);
  if (missingVars.length > 0) {
    throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
  }

  const CHUNK_SIZE = Number.parseInt(process.env.CHUNK_SIZE!);
  const CHUNK_OVERLAP = Number.parseInt(process.env.CHUNK_OVERLAP!);

  if (Number.isNaN(CHUNK_SIZE) || CHUNK_SIZE <= 0) {
    throw new Error('CHUNK_SIZE must be a positive number');
  }

  if (Number.isNaN(CHUNK_OVERLAP) || CHUNK_OVERLAP < 0) {
    throw new Error('CHUNK_OVERLAP must be a non-negative number');
  }

  if (CHUNK_OVERLAP >= CHUNK_SIZE) {
    throw new Error('CHUNK_OVERLAP must be less than CHUNK_SIZE');
  }

  if (!fs.existsSync(process.env.DOCUMENT_PATH!)) {
    throw new Error(`DOCUMENT_PATH does not exist: ${process.env.DOCUMENT_PATH}`);
  }

  const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY;
  const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
  if (!OPENROUTER_API_KEY && !OPENAI_API_KEY) {
    throw new Error('Either OPENROUTER_API_KEY or OPENAI_API_KEY must be set');
  }

  console.log('✅ Environment variables validated successfully');

  return {
    DOCUMENT_PATH: process.env.DOCUMENT_PATH!,
    VECTOR_STORE_PATH: process.env.VECTOR_STORE_PATH!,
    EMBEDDING_MODEL: process.env.EMBEDDING_MODEL!,
    OPENROUTER_API_KEY: process.env.OPENROUTER_API_KEY!,
    OPENAI_API_KEY: process.env.OPENAI_API_KEY!,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
  };
}
