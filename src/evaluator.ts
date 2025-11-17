import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';
import { loadAndValidateEnv } from './query/env.js';
import type { QueryResponse, EvaluationResult } from './types.js';
import { GetOpenApiClient } from './utils/openai.js';
import { LoadPromptTemplate } from './utils/file.js';

/**
 * Evaluate RAG system response using modern LangChain LCEL (LangChain Expression Language)
 */
export async function evaluateResponse(
  response: QueryResponse,
  env: ReturnType<typeof loadAndValidateEnv>
): Promise<EvaluationResult> {
  const llm = GetOpenApiClient({
    openaiApiKey: env.openaiApiKey,
    openRouterApiKey: env.openRouterApiKey,
    modelName: env.llmModel,
    temperature: 0.2,
  });

  const promptTemplate = LoadPromptTemplate('../prompts/evaluation.md');
  const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);

  const chunksText = response.chunks_related
    .map((chunk, index) => `[Chunk ${index + 1}]\n${chunk.content}\n`)
    .join('\n');

  const evaluationChain = RunnableSequence.from([
    {
      question: () => response.user_question,
      answer: () => response.system_answer,
      chunks: () => chunksText,
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  const evaluationText = await evaluationChain.invoke({});

  const jsonMatch = evaluationText.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error('Failed to extract JSON from evaluation response');
  }

  const parsedEvaluation = JSON.parse(jsonMatch[0]);

  return {
    score: parsedEvaluation.overall_score,
    reason: parsedEvaluation.reason,
    breakdown: {
      chunk_relevance: parsedEvaluation.chunk_relevance,
      answer_accuracy: parsedEvaluation.answer_accuracy,
      completeness: parsedEvaluation.completeness,
    },
  };
}
