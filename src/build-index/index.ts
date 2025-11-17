/**
 * Build Index Pipeline
 *
 * This script loads the FAQ document, chunks it using LangChain's RecursiveCharacterTextSplitter,
 * generates embeddings using OpenAI, and stores them in an HNSW vector store for efficient retrieval.
 */

import { TextLoader } from "@langchain/classic/document_loaders/fs/text"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HNSWLib } from '@langchain/community/vectorstores/hnswlib';
import * as path from 'path';
import * as fs from 'fs';
import { validateEnvironmentVariables } from "./env.js";
import { GetEmbeddingsClient } from "../utils/openai.js";

/**
 * Main function to build the vector index
 */
async function buildIndex() {
  console.log('Starting indexing pipeline...\n');

  try {
    const {
      DOCUMENT_PATH,
      VECTOR_STORE_PATH, 
      EMBEDDING_MODEL, 
      CHUNK_SIZE, 
      CHUNK_OVERLAP,
      OPENROUTER_API_KEY,
      OPENAI_API_KEY,
    } = validateEnvironmentVariables();

    console.log('Loading document from:', DOCUMENT_PATH);

    const loader = new TextLoader(DOCUMENT_PATH);
    const docs = await loader.load();

    console.log('Splitting document into chunks with size:', CHUNK_SIZE, 'and overlap:', CHUNK_OVERLAP);

    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: CHUNK_SIZE,
      chunkOverlap: CHUNK_OVERLAP,
      separators: ['\n\n', '\n', '. ', ' ', ''], // Try to split on natural boundaries
    });

    const chunks = await textSplitter.splitDocuments(docs);
    if (chunks.length < 20) {
      console.warn(`Warning: Only ${chunks.length} chunks created. Requirement is 20+ chunks.`);
    }

    const chunksWithMetadata = chunks.map((chunk, index) => ({
      ...chunk,
      metadata: {
        ...chunk.metadata,
        chunkIndex: index,
        source: DOCUMENT_PATH,
      },
    }));

    const embeddings = GetEmbeddingsClient({
      modelName: EMBEDDING_MODEL,
      openaiApiKey: OPENAI_API_KEY,
      openRouterApiKey: OPENROUTER_API_KEY,
    });
    console.log('Creating HNSW vector store...');

    const vectorStore = await HNSWLib.fromDocuments(
      chunksWithMetadata,
      embeddings
    );

    console.log('Saving vector store to:', VECTOR_STORE_PATH);
    
    const dir = path.dirname(VECTOR_STORE_PATH);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    await vectorStore.save(VECTOR_STORE_PATH);
    console.log('Vector store saved successfully\n');

    const chunkInfo = {
      totalChunks: chunks.length,
      chunkSize: CHUNK_SIZE,
      chunkOverlap: CHUNK_OVERLAP,
      embeddingModel: EMBEDDING_MODEL,
      documentPath: DOCUMENT_PATH,
      chunks: chunks.map((chunk, index) => ({
        index,
        preview: chunk.pageContent.substring(0, 100) + '...',
        length: chunk.pageContent.length,
      })),
    };

    const chunkInfoPath = path.join(dir, 'chunk-info.json');
    fs.writeFileSync(chunkInfoPath, JSON.stringify(chunkInfo, null, 2));
    
    console.log('Chunk information saved to:', chunkInfoPath);

    // Summary
    console.log('\n✨ Indexing pipeline completed successfully!');
    console.log('─'.repeat(50));
    console.log(`📈 Summary:`);
    console.log(`   - Total chunks: ${chunks.length}`);
    console.log(`   - Embedding model: ${EMBEDDING_MODEL}`);
    console.log(`   - Vector store: ${VECTOR_STORE_PATH}`);
    console.log(`   - Ready for queries! ✅`);
    console.log('─'.repeat(50));

  } catch (error) {
    console.error('Error during indexing:', error);
    
    process.exit(1);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  buildIndex();
}

export { buildIndex };
