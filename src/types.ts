/**
 * Type definitions for the RAG-based FAQ Support Chatbot
 */

/**
 * Represents a text chunk with its content and metadata
 */
export interface ChunkData {
  content: string;
  metadata: {
    chunkIndex: number;
    source: string;
    startLine?: number;
    endLine?: number;
  };
}

/**
 * Represents a retrieved chunk with similarity score
 */
export interface RetrievedChunk {
  content: string;
  metadata: Record<string, any>;
  similarity_score?: number;
}

/**
 * Query response structure containing the answer and related chunks
 */
export interface QueryResponse {
  user_question: string;
  system_answer: string;
  chunks_related: RetrievedChunk[];
}

/**
 * Evaluation result from the evaluator agent
 */
export interface EvaluationResult {
  score: number; // 0-10
  reason: string;
  breakdown: {
    chunk_relevance: number; // 0-10
    answer_accuracy: number; // 0-10
    completeness: number; // 0-10
  };
}

/**
 * Configuration for text chunking
 */
export interface ChunkingConfig {
  chunkSize: number;
  chunkOverlap: number;
}

/**
 * Configuration for vector search
 */
export interface SearchConfig {
  k: number; // number of chunks to retrieve
  searchType?: 'similarity' | 'mmr'; // Maximum Marginal Relevance
}

/**
 * Application configuration
 */
export interface AppConfig {
  openaiApiKey: string;
  embeddingModel: string;
  llmModel: string;
  chunking: ChunkingConfig;
  search: SearchConfig;
  vectorStorePath: string;
  documentPath: string;
}

export interface EmbeddingClientArgs {
  openaiApiKey: string;
  modelName: string;
  openRouterApiKey?: string;
}
