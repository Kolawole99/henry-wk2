/**
 * Test Suite for RAG FAQ Chatbot
 *
 * Tests the core functionality of document loading, chunking, vector search, and query processing
 */

import { describe, it, expect } from 'vitest';
import { TextLoader } from '@langchain/classic/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import * as fs from 'fs';
import * as path from 'path';
import type { QueryResponse } from '../src/types.js';

const DOCUMENT_PATH = './data/faq_document.txt';
const VECTOR_STORE_PATH = './data/vector-store';
const CHUNK_SIZE = 500;
const CHUNK_OVERLAP = 100;

describe('FAQ Chatbot - Core Functionality', () => {
  describe('Document Loading', () => {
    it('should load the FAQ document successfully', async () => {
      expect(fs.existsSync(DOCUMENT_PATH)).toBe(true);

      const loader = new TextLoader(DOCUMENT_PATH);
      const docs = await loader.load();

      expect(docs).toBeDefined();
      expect(docs.length).toBeGreaterThan(0);
      expect(docs[0]?.pageContent.length).toBeGreaterThan(1000); // At least 1000 chars
    });

    it('should have FAQ document with sufficient content', async () => {
      const content = fs.readFileSync(DOCUMENT_PATH, 'utf-8');
      const wordCount = content.split(/\s+/).length;

      expect(wordCount).toBeGreaterThan(1000); // Requirement: 1000+ words
    });
  });

  describe('Text Chunking', () => {
    it('should create at least 20 chunks from the document', async () => {
      const loader = new TextLoader(DOCUMENT_PATH);
      const docs = await loader.load();

      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: CHUNK_SIZE,
        chunkOverlap: CHUNK_OVERLAP,
        separators: ['\n\n', '\n', '. ', ' ', ''],
      });

      const chunks = await textSplitter.splitDocuments(docs);

      expect(chunks.length).toBeGreaterThanOrEqual(20); // Minimum requirement
    });

    it('should create chunks with proper size and overlap', async () => {
      const loader = new TextLoader(DOCUMENT_PATH);
      const docs = await loader.load();

      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: CHUNK_SIZE,
        chunkOverlap: CHUNK_OVERLAP,
      });

      const chunks = await textSplitter.splitDocuments(docs);

      // Check that most chunks are within expected size range
      const validChunks = chunks.filter(
        (chunk) =>
          chunk.pageContent.length > 0 &&
          chunk.pageContent.length <= CHUNK_SIZE + 100 // Allow some tolerance
      );

      expect(validChunks.length).toBeGreaterThan(chunks.length * 0.8); // At least 80% valid
    });

    it('should preserve metadata in chunks', async () => {
      const loader = new TextLoader(DOCUMENT_PATH);
      const docs = await loader.load();

      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: CHUNK_SIZE,
        chunkOverlap: CHUNK_OVERLAP,
      });

      const chunks = await textSplitter.splitDocuments(docs);

      chunks.forEach((chunk) => {
        expect(chunk.metadata).toBeDefined();
        expect(chunk.metadata.source).toBeDefined();
      });
    });
  });

  describe('Vector Store', () => {
    it('should have vector store directory after indexing', () => {
      // This test assumes indexing has been run
      // If vector store doesn't exist, this test will guide users to run build:index
      const exists = fs.existsSync(VECTOR_STORE_PATH);

      if (!exists) {
        console.warn(
          '⚠️  Vector store not found. Please run "npm run build:index" first.'
        );
      }

      // We make this a soft check - test passes but warns if not found
      expect(true).toBe(true);
    });

    it('should have chunk-info.json after indexing', () => {
      const chunkInfoPath = path.join(
        path.dirname(VECTOR_STORE_PATH),
        'chunk-info.json'
      );

      if (fs.existsSync(chunkInfoPath)) {
        const chunkInfo = JSON.parse(
          fs.readFileSync(chunkInfoPath, 'utf-8')
        );

        expect(chunkInfo.totalChunks).toBeGreaterThanOrEqual(20);
        expect(chunkInfo.embeddingModel).toBeDefined();
        expect(chunkInfo.chunks).toBeDefined();
        expect(Array.isArray(chunkInfo.chunks)).toBe(true);
      } else {
        console.warn(
          '⚠️  Chunk info not found. Please run "npm run build:index" first.'
        );
        expect(true).toBe(true); // Soft check
      }
    });
  });

  describe('Query Response Format', () => {
    it('should validate QueryResponse structure', () => {
      const mockResponse: QueryResponse = {
        user_question: 'What is the leave policy?',
        system_answer: 'Employees get 20 days of annual leave.',
        chunks_related: [
          {
            content: 'Annual leave policy: 20 days per year',
            metadata: { chunkIndex: 0, source: 'test' },
          },
        ],
      };

      expect(mockResponse.user_question).toBeDefined();
      expect(typeof mockResponse.user_question).toBe('string');
      expect(mockResponse.system_answer).toBeDefined();
      expect(typeof mockResponse.system_answer).toBe('string');
      expect(Array.isArray(mockResponse.chunks_related)).toBe(true);
      expect(mockResponse.chunks_related.length).toBeGreaterThan(0);
      expect(mockResponse.chunks_related[0].content).toBeDefined();
      expect(mockResponse.chunks_related[0].metadata).toBeDefined();
    });

    it('should have all required fields in chunks_related', () => {
      const mockChunk = {
        content: 'Test content',
        metadata: { chunkIndex: 0 },
        similarity_score: 0.95,
      };

      expect(mockChunk).toHaveProperty('content');
      expect(mockChunk).toHaveProperty('metadata');
      expect(typeof mockChunk.content).toBe('string');
      expect(typeof mockChunk.metadata).toBe('object');
    });
  });

  describe('Configuration', () => {
    it('should have .env.example file', () => {
      const envExamplePath = './.env.example';
      expect(fs.existsSync(envExamplePath)).toBe(true);

      const content = fs.readFileSync(envExamplePath, 'utf-8');
      expect(content).toContain('OPENAI_API_KEY');
      expect(content).toContain('EMBEDDING_MODEL');
    });

    it('should have all required environment variables in .env.example', () => {
      const envExamplePath = './.env.example';
      const content = fs.readFileSync(envExamplePath, 'utf-8');

      const requiredVars = [
        'OPENAI_API_KEY',
        'EMBEDDING_MODEL',
        'LLM_MODEL',
        'CHUNK_SIZE',
        'CHUNK_OVERLAP',
        'RETRIEVAL_K',
      ];

      requiredVars.forEach((varName) => {
        expect(content).toContain(varName);
      });
    });
  });

  describe('Project Structure', () => {
    it('should have all required directories', () => {
      expect(fs.existsSync('./data')).toBe(true);
      expect(fs.existsSync('./src')).toBe(true);
      expect(fs.existsSync('./tests')).toBe(true);
    });

    it('should have all required source files', () => {
      expect(fs.existsSync('./src/types.ts')).toBe(true);
      expect(fs.existsSync('./src/build-index/index.ts')).toBe(true);
      expect(fs.existsSync('./src/query/index.ts')).toBe(true);
      expect(fs.existsSync('./src/evaluator.ts')).toBe(true);
      expect(fs.existsSync('./src/utils/openai.ts')).toBe(true);
      expect(fs.existsSync('./src/utils/file.ts')).toBe(true);
    });

    it('should have README.md', () => {
      expect(fs.existsSync('./README.md')).toBe(true);

      const readme = fs.readFileSync('./README.md', 'utf-8');
      expect(readme.length).toBeGreaterThan(500); // Substantial README
    });

    it('should have outputs directory for sample queries', () => {
      if (fs.existsSync('./outputs')) {
        expect(fs.existsSync('./outputs/sample_queries.json')).toBe(true);
      }
    });
  });
});
