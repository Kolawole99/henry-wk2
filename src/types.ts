/**
 * Type definitions for the RAG-based FAQ Support Chatbot
 */


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

export interface EmbeddingClientArgs {
  openaiApiKey: string;
  modelName: string;
  openRouterApiKey?: string;
}
