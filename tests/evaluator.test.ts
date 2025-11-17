/**
 * Test Suite for Evaluator Module
 *
 * Tests the evaluation functionality that scores RAG responses
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { evaluateResponse } from '../src/evaluator.js';
import type { QueryResponse, EvaluationResult } from '../src/types.js';
import type { EnvConfig } from '../src/query/env.js';
import { RunnableLambda } from '@langchain/core/runnables';
import { AIMessage } from '@langchain/core/messages';

// Mock the dependencies
vi.mock('../src/utils/openai.js', () => ({
  GetOpenApiClient: vi.fn(() => {
    // Create a simple Runnable that returns an AIMessage
    return new RunnableLambda({
      func: async () => {
        const content = JSON.stringify({
          chunk_relevance: 9,
          answer_accuracy: 8,
          completeness: 9,
          overall_score: 8.7,
          reason: 'The response accurately addresses the question with relevant chunks.',
        });
        return new AIMessage(content);
      },
    });
  }),
}));

vi.mock('../src/utils/file.js', () => ({
  LoadPromptTemplate: vi.fn(() => 'Mock prompt template with {question}, {answer}, {chunks}'),
}));

describe('Evaluator Module', () => {
  let mockEnv: EnvConfig;
  let mockQueryResponse: QueryResponse;

  beforeEach(() => {
    mockEnv = {
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

    mockQueryResponse = {
      user_question: 'What is the company sick leave policy?',
      system_answer:
        'Employees are entitled to 10 days of paid sick leave per year. Sick leave does not roll over to the next year.',
      chunks_related: [
        {
          content:
            'Q: What is the sick leave policy?\nA: Employees are entitled to 10 days of paid sick leave per year.',
          metadata: { chunkIndex: 5, source: 'faq_document.txt' },
        },
      ],
    };
  });

  describe('evaluateResponse', () => {
    it('should return evaluation result with correct structure', async () => {
      const result = await evaluateResponse(mockQueryResponse, mockEnv);

      expect(result).toBeDefined();
      expect(result).toHaveProperty('score');
      expect(result).toHaveProperty('reason');
      expect(result).toHaveProperty('breakdown');
      expect(result.breakdown).toHaveProperty('chunk_relevance');
      expect(result.breakdown).toHaveProperty('answer_accuracy');
      expect(result.breakdown).toHaveProperty('completeness');
    });

    it('should return scores within valid range (0-10)', async () => {
      const result = await evaluateResponse(mockQueryResponse, mockEnv);

      expect(result.score).toBeGreaterThanOrEqual(0);
      expect(result.score).toBeLessThanOrEqual(10);
      expect(result.breakdown.chunk_relevance).toBeGreaterThanOrEqual(0);
      expect(result.breakdown.chunk_relevance).toBeLessThanOrEqual(10);
      expect(result.breakdown.answer_accuracy).toBeGreaterThanOrEqual(0);
      expect(result.breakdown.answer_accuracy).toBeLessThanOrEqual(10);
      expect(result.breakdown.completeness).toBeGreaterThanOrEqual(0);
      expect(result.breakdown.completeness).toBeLessThanOrEqual(10);
    });

    it('should have reason as a non-empty string', async () => {
      const result = await evaluateResponse(mockQueryResponse, mockEnv);

      expect(typeof result.reason).toBe('string');
      expect(result.reason.length).toBeGreaterThan(0);
    });

    it('should handle multiple chunks in response', async () => {
      const multiChunkResponse: QueryResponse = {
        ...mockQueryResponse,
        chunks_related: [
          {
            content: 'First chunk content',
            metadata: { chunkIndex: 0 },
          },
          {
            content: 'Second chunk content',
            metadata: { chunkIndex: 1 },
          },
          {
            content: 'Third chunk content',
            metadata: { chunkIndex: 2 },
          },
        ],
      };

      const result = await evaluateResponse(multiChunkResponse, mockEnv);

      expect(result).toBeDefined();
      expect(result.score).toBeGreaterThanOrEqual(0);
    });

    it('should process response with empty chunks gracefully', async () => {
      const emptyChunksResponse: QueryResponse = {
        ...mockQueryResponse,
        chunks_related: [],
      };

      const result = await evaluateResponse(emptyChunksResponse, mockEnv);

      expect(result).toBeDefined();
      // With no chunks, relevance should likely be low
      expect(result.breakdown.chunk_relevance).toBeDefined();
    });
  });

  describe('EvaluationResult Type', () => {
    it('should validate EvaluationResult structure', () => {
      const mockResult: EvaluationResult = {
        score: 8.5,
        reason: 'Good response with relevant chunks',
        breakdown: {
          chunk_relevance: 9,
          answer_accuracy: 8,
          completeness: 9,
        },
      };

      expect(mockResult.score).toBe(8.5);
      expect(mockResult.reason).toBe('Good response with relevant chunks');
      expect(mockResult.breakdown.chunk_relevance).toBe(9);
      expect(mockResult.breakdown.answer_accuracy).toBe(8);
      expect(mockResult.breakdown.completeness).toBe(9);
    });

    it('should handle minimum scores', () => {
      const mockResult: EvaluationResult = {
        score: 0,
        reason: 'Poor response with irrelevant chunks',
        breakdown: {
          chunk_relevance: 0,
          answer_accuracy: 0,
          completeness: 0,
        },
      };

      expect(mockResult.score).toBe(0);
      expect(mockResult.breakdown.chunk_relevance).toBe(0);
    });

    it('should handle maximum scores', () => {
      const mockResult: EvaluationResult = {
        score: 10,
        reason: 'Perfect response with highly relevant chunks',
        breakdown: {
          chunk_relevance: 10,
          answer_accuracy: 10,
          completeness: 10,
        },
      };

      expect(mockResult.score).toBe(10);
      expect(mockResult.breakdown.completeness).toBe(10);
    });
  });

  describe('Integration with Environment Config', () => {
    it('should work with OpenAI API configuration', async () => {
      const openaiEnv: EnvConfig = {
        ...mockEnv,
        openaiApiKey: 'test-openai-key',
        openRouterApiKey: '',
      };

      const result = await evaluateResponse(mockQueryResponse, openaiEnv);
      expect(result).toBeDefined();
    });

    it('should work with OpenRouter API configuration', async () => {
      const openRouterEnv: EnvConfig = {
        ...mockEnv,
        openaiApiKey: '',
        openRouterApiKey: 'test-openrouter-key',
      };

      const result = await evaluateResponse(mockQueryResponse, openRouterEnv);
      expect(result).toBeDefined();
    });

    it('should use correct model from env config', async () => {
      const customModelEnv: EnvConfig = {
        ...mockEnv,
        llmModel: 'gpt-4',
      };

      const result = await evaluateResponse(mockQueryResponse, customModelEnv);
      expect(result).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    it('should throw error when JSON parsing fails', async () => {
      const { GetOpenApiClient } = await import('../src/utils/openai.js');

      // Mock to return invalid JSON
      vi.mocked(GetOpenApiClient).mockReturnValueOnce(
        new RunnableLambda({
          func: async () => new AIMessage('This is not valid JSON'),
        }) as any
      );

      await expect(evaluateResponse(mockQueryResponse, mockEnv)).rejects.toThrow();
    });
  });
});
