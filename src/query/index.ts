import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { loadAndValidateEnv } from './env.js';
import type { QueryResponse, LoggedQueryOutput } from '../types.js';
import { loadVectorStore } from './vector.js';
import { GetOpenApiClient } from '../utils/openai.js';
import { LoadPromptTemplate } from '../utils/file.js';
import { evaluateResponse } from '../evaluator.js';
import * as fs from 'fs';
import * as path from 'path';

const OUTPUT_DIR = './outputs';
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'sample_queries.json');

/**
 * Query RAG system using modern LangChain LCEL (LangChain Expression Language)
 */
export async function query(
  question: string,
  env: ReturnType<typeof loadAndValidateEnv>,
  enableEvaluation: boolean = true
): Promise<QueryResponse> {
  const vectorStore = await loadVectorStore(env);
  const retriever = vectorStore.asRetriever({
    k: env.retrievalK,
  });

  const llm = GetOpenApiClient({
    openaiApiKey: env.openaiApiKey,
    openRouterApiKey: env.openRouterApiKey,
    modelName: env.llmModel,
    temperature: 0.3,
  });

  const promptTemplate = LoadPromptTemplate('../prompts/query-qa.md');
  const prompt = ChatPromptTemplate.fromTemplate(promptTemplate);
  const retrievedDocs = await retriever.invoke(question);

  const ragChain = RunnableSequence.from([
    {
      context: async () => retrievedDocs.map((doc) => doc.pageContent).join('\n\n'),
      question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  const answer = await ragChain.invoke(question);
  const formattedChunks = retrievedDocs.map((doc: any) => ({
    content: doc.pageContent,
    metadata: doc.metadata ?? {},
  }));

  const response: QueryResponse = {
    user_question: question,
    system_answer: answer,
    chunks_related: formattedChunks,
  };

  // Run evaluation if enabled
  let evaluation;
  if (enableEvaluation) {
    try {
      evaluation = await evaluateResponse(response, env);
      console.log(`📊 Evaluation Score: ${evaluation.score}/10`);
    } catch (error) {
      console.error('Warning: Evaluation failed:', error);
    }
  }

  // Log output to file with evaluation
  logQueryOutput(response, evaluation);

  return response;
}

/**
 * Logs query output to the outputs folder
 */
function logQueryOutput(response: QueryResponse, evaluation?: any): void {
  try {
    // Ensure output directory exists
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    // Read existing queries or initialize empty array
    let queries: LoggedQueryOutput[] = [];
    if (fs.existsSync(OUTPUT_FILE)) {
      const existingData = fs.readFileSync(OUTPUT_FILE, 'utf-8');
      queries = JSON.parse(existingData);
    }

    // Create logged output with timestamp and evaluation
    const loggedOutput: LoggedQueryOutput = {
      ...response,
      timestamp: new Date().toISOString(),
      ...(evaluation && { evaluation }),
    };

    // Add new query response
    queries.push(loggedOutput);

    // Write updated queries to file
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(queries, null, 2));
    console.log(`✅ Logged to ${OUTPUT_FILE}`);
  } catch (error) {
    console.error('Warning: Failed to log query output:', error);
    // Don't throw - logging failure shouldn't break the query
  }
}

/**
 * CLI Main
 */
async function main() {
  const question = process.argv.slice(2).join(' ');
  if (!question) {
    console.error('Please pass a question.');
    process.exit(1);
  }

  const env = loadAndValidateEnv();

  const response = await query(question, env);

  console.log('\n💡 Answer:', response.system_answer);
  console.log('\n📚 Related Chunks:', response.chunks_related.length);
  console.log('\nJSON:\n', JSON.stringify(response, null, 2));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { loadVectorStore };
