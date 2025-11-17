import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import * as fs from 'fs';
import { loadAndValidateEnv } from './env.js';
import { GetEmbeddingsClient } from '../utils/openai.js';

/**
 * Load vector store
 */
export async function loadVectorStore(env: ReturnType<typeof loadAndValidateEnv>) {
  if (!fs.existsSync(env.vectorStorePath)) {
    throw new Error(`Vector store not found at ${env.vectorStorePath}. Run 'npm run build:index' first.`);
  }

  const embeddings = GetEmbeddingsClient({
    modelName: env.embeddingModel,
    openaiApiKey: env.openaiApiKey,
    openRouterApiKey: env.openRouterApiKey,
  });

  const store = await HNSWLib.load(env.vectorStorePath, embeddings);

  console.log('Vector store loaded\n');
  
  return store;
}
