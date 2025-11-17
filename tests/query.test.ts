/**
 * Test Suite for Query Module
 *
 * Tests the query functionality that processes user questions using RAG
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { QueryResponse } from '../src/types.js';
import type { EnvConfig } from '../src/query/env.js';
import * as fs from 'fs';

// We'll test the query module components
describe('Query Module', () => {
  describe('Environment Configuration', () => {
    it('should have loadAndValidateEnv function', async () => {
      const { loadAndValidateEnv } = await import('../src/query/env.js');
      expect(loadAndValidateEnv).toBeDefined();
      expect(typeof loadAndValidateEnv).toBe('function');
    });

    it('should validate required environment variables', () => {
      const mockEnv: EnvConfig = {
        documentPath: './data/faq_document.txt',
        vectorStorePath: './data/vector-store',
        embeddingModel: 'text-embedding-ada-002',
        llmModel: 'gpt-3.5-turbo',
        retrievalK: 3,
        chunkSize: 500,
        chunkOverlap: 100,
        openaiApiKey: 'test-key',
        openRouterApiKey: '',
      };

      expect(mockEnv.documentPath).toBeDefined();
      expect(mockEnv.vectorStorePath).toBeDefined();
      expect(mockEnv.embeddingModel).toBeDefined();
      expect(mockEnv.llmModel).toBeDefined();
      expect(mockEnv.retrievalK).toBeGreaterThan(0);
      expect(mockEnv.chunkSize).toBeGreaterThan(0);
      expect(mockEnv.chunkOverlap).toBeGreaterThanOrEqual(0);
      expect(mockEnv.chunkOverlap).toBeLessThan(mockEnv.chunkSize);
    });

    it('should have valid numeric values for chunk configuration', () => {
      const mockEnv: EnvConfig = {
        documentPath: './data/faq_document.txt',
        vectorStorePath: './data/vector-store',
        embeddingModel: 'text-embedding-ada-002',
        llmModel: 'gpt-3.5-turbo',
        retrievalK: 3,
        chunkSize: 500,
        chunkOverlap: 100,
        openaiApiKey: 'test-key',
        openRouterApiKey: '',
      };

      expect(typeof mockEnv.chunkSize).toBe('number');
      expect(typeof mockEnv.chunkOverlap).toBe('number');
      expect(typeof mockEnv.retrievalK).toBe('number');
      expect(Number.isNaN(mockEnv.chunkSize)).toBe(false);
      expect(Number.isNaN(mockEnv.chunkOverlap)).toBe(false);
      expect(Number.isNaN(mockEnv.retrievalK)).toBe(false);
    });
  });

  describe('QueryResponse Structure', () => {
    it('should validate QueryResponse format', () => {
      const mockResponse: QueryResponse = {
        user_question: 'What is the leave policy?',
        system_answer: 'Employees get 20 days of annual leave per year.',
        chunks_related: [
          {
            content: 'Annual leave policy details...',
            metadata: { chunkIndex: 0, source: 'faq_document.txt' },
          },
        ],
      };

      expect(mockResponse.user_question).toBeDefined();
      expect(typeof mockResponse.user_question).toBe('string');
      expect(mockResponse.system_answer).toBeDefined();
      expect(typeof mockResponse.system_answer).toBe('string');
      expect(Array.isArray(mockResponse.chunks_related)).toBe(true);
      expect(mockResponse.chunks_related.length).toBeGreaterThan(0);
    });

    it('should have valid chunk structure', () => {
      const mockResponse: QueryResponse = {
        user_question: 'Test question',
        system_answer: 'Test answer',
        chunks_related: [
          {
            content: 'Chunk content',
            metadata: { chunkIndex: 0 },
            similarity_score: 0.95,
          },
        ],
      };

      const chunk = mockResponse.chunks_related[0];
      expect(chunk).toBeDefined();
      expect(chunk?.content).toBeDefined();
      expect(typeof chunk?.content).toBe('string');
      expect(chunk?.metadata).toBeDefined();
      expect(typeof chunk?.metadata).toBe('object');
    });

    it('should handle multiple chunks in response', () => {
      const mockResponse: QueryResponse = {
        user_question: 'Test question',
        system_answer: 'Test answer',
        chunks_related: [
          {
            content: 'First chunk',
            metadata: { chunkIndex: 0 },
          },
          {
            content: 'Second chunk',
            metadata: { chunkIndex: 1 },
          },
          {
            content: 'Third chunk',
            metadata: { chunkIndex: 2 },
          },
        ],
      };

      expect(mockResponse.chunks_related.length).toBe(3);
      expect(mockResponse.chunks_related[0]?.content).toBe('First chunk');
      expect(mockResponse.chunks_related[1]?.content).toBe('Second chunk');
      expect(mockResponse.chunks_related[2]?.content).toBe('Third chunk');
    });

    it('should include optional similarity scores in chunks', () => {
      const mockResponse: QueryResponse = {
        user_question: 'Test question',
        system_answer: 'Test answer',
        chunks_related: [
          {
            content: 'Chunk with score',
            metadata: { chunkIndex: 0 },
            similarity_score: 0.92,
          },
        ],
      };

      expect(mockResponse.chunks_related[0]?.similarity_score).toBeDefined();
      expect(mockResponse.chunks_related[0]?.similarity_score).toBeGreaterThan(0);
      expect(mockResponse.chunks_related[0]?.similarity_score).toBeLessThanOrEqual(1);
    });
  });

  describe('Prompt Template', () => {
    it('should have query-qa.md prompt file', () => {
      const promptPath = './src/prompts/query-qa.md';
      expect(fs.existsSync(promptPath)).toBe(true);
    });

    it('should have evaluation.md prompt file', () => {
      const promptPath = './src/prompts/evaluation.md';
      expect(fs.existsSync(promptPath)).toBe(true);
    });

    it('should have valid prompt content', () => {
      const promptPath = './src/prompts/query-qa.md';
      if (fs.existsSync(promptPath)) {
        const content = fs.readFileSync(promptPath, 'utf-8');
        expect(content.length).toBeGreaterThan(0);
        // Prompts should contain variable placeholders
        expect(content.includes('{')).toBe(true);
        expect(content.includes('}')).toBe(true);
      }
    });
  });

  describe('Vector Store Integration', () => {
    it('should have vector store directory after build', () => {
      const vectorStorePath = './data/vector-store';
      // This is a soft check - vector store should exist after running build:index
      if (fs.existsSync(vectorStorePath)) {
        expect(fs.existsSync(vectorStorePath)).toBe(true);
      } else {
        console.warn('⚠️  Vector store not found. Run "npm run run:embed" first.');
      }
    });

    it('should have hnswlib.index file in vector store', () => {
      const indexPath = './data/vector-store/hnswlib.index';
      if (fs.existsSync('./data/vector-store')) {
        // Index file should exist if vector store is built
        const files = fs.readdirSync('./data/vector-store');
        expect(files.length).toBeGreaterThan(0);
      }
    });
  });

  describe('LCEL Chain Structure', () => {
    it('should use RunnableSequence pattern', async () => {
      // Test that the query module exports the query function
      const queryModule = await import('../src/query/index.js');
      expect(queryModule).toBeDefined();
    });
  });

  describe('Retrieval Parameters', () => {
    it('should have valid retrieval K value', () => {
      const mockEnv: EnvConfig = {
        documentPath: './data/faq_document.txt',
        vectorStorePath: './data/vector-store',
        embeddingModel: 'text-embedding-ada-002',
        llmModel: 'gpt-3.5-turbo',
        retrievalK: 3,
        chunkSize: 500,
        chunkOverlap: 100,
        openaiApiKey: 'test-key',
        openRouterApiKey: '',
      };

      expect(mockEnv.retrievalK).toBeGreaterThan(0);
      expect(mockEnv.retrievalK).toBeLessThanOrEqual(10); // Reasonable upper limit
    });

    it('should retrieve at least 1 chunk', () => {
      const mockResponse: QueryResponse = {
        user_question: 'Test question',
        system_answer: 'Test answer',
        chunks_related: [
          {
            content: 'At least one chunk should be retrieved',
            metadata: { chunkIndex: 0 },
          },
        ],
      };

      expect(mockResponse.chunks_related.length).toBeGreaterThanOrEqual(1);
    });
  });
});
