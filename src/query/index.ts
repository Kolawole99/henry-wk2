import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence, RunnablePassthrough } from '@langchain/core/runnables';
import { loadAndValidateEnv } from './env.js';
import type { QueryResponse } from '../types.js';
import { loadVectorStore } from './vector.js';
import { GetOpenApiClient } from '../utils/openai.js';
import { LoadPromptTemplate } from '../utils/file.js';

/**
 * Query RAG system using modern LangChain LCEL (LangChain Expression Language)
 */
export async function query(question: string, env: ReturnType<typeof loadAndValidateEnv>): Promise<QueryResponse> {
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

  return {
    user_question: question,
    system_answer: answer,
    chunks_related: formattedChunks,
  };
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
