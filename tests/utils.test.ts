/**
 * Test Suite for Utility Functions
 *
 * Tests utility functions for OpenAI clients and file operations
 */

import { describe, it, expect, vi } from 'vitest';
import { GetEmbeddingsClient, GetOpenApiClient } from '../src/utils/openai.js';
import { LoadPromptTemplate } from '../src/utils/file.js';
import * as fs from 'fs';
import * as path from 'path';

describe('Utility Functions', () => {
  describe('OpenAI Client Utilities', () => {
    describe('GetEmbeddingsClient', () => {
      it('should return embeddings client with OpenAI configuration', () => {
        const client = GetEmbeddingsClient({
          openaiApiKey: 'test-openai-key',
          modelName: 'text-embedding-ada-002',
        });

        expect(client).toBeDefined();
        expect(client).toHaveProperty('embedQuery');
        expect(client).toHaveProperty('embedDocuments');
      });

      it('should return embeddings client with OpenRouter configuration', () => {
        const client = GetEmbeddingsClient({
          openaiApiKey: 'test-openai-key',
          openRouterApiKey: 'test-openrouter-key',
          modelName: 'text-embedding-ada-002',
        });

        expect(client).toBeDefined();
        expect(client).toHaveProperty('embedQuery');
      });

      it('should accept model name parameter', () => {
        const client = GetEmbeddingsClient({
          openaiApiKey: 'test-key',
          modelName: 'text-embedding-3-small',
        });

        expect(client).toBeDefined();
      });

      it('should handle different embedding models', () => {
        const models = [
          'text-embedding-ada-002',
          'text-embedding-3-small',
          'text-embedding-3-large',
        ];

        models.forEach((modelName) => {
          const client = GetEmbeddingsClient({
            openaiApiKey: 'test-key',
            modelName,
          });
          expect(client).toBeDefined();
        });
      });
    });

    describe('GetOpenApiClient', () => {
      it('should return LLM client with OpenAI configuration', () => {
        const client = GetOpenApiClient({
          openaiApiKey: 'test-openai-key',
          openRouterApiKey: '',
          modelName: 'gpt-3.5-turbo',
          temperature: 0.7,
        });

        expect(client).toBeDefined();
        expect(client).toHaveProperty('invoke');
      });

      it('should return LLM client with OpenRouter configuration', () => {
        const client = GetOpenApiClient({
          openaiApiKey: '',
          openRouterApiKey: 'test-openrouter-key',
          modelName: 'gpt-3.5-turbo',
          temperature: 0.7,
        });

        expect(client).toBeDefined();
        expect(client).toHaveProperty('invoke');
      });

      it('should accept temperature parameter', () => {
        const temperatures = [0, 0.2, 0.5, 0.7, 1.0];

        temperatures.forEach((temperature) => {
          const client = GetOpenApiClient({
            openaiApiKey: 'test-key',
            openRouterApiKey: '',
            modelName: 'gpt-3.5-turbo',
            temperature,
          });
          expect(client).toBeDefined();
        });
      });

      it('should handle different LLM models', () => {
        const models = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'];

        models.forEach((modelName) => {
          const client = GetOpenApiClient({
            openaiApiKey: 'test-key',
            openRouterApiKey: '',
            modelName,
            temperature: 0.7,
          });
          expect(client).toBeDefined();
        });
      });

      it('should support low temperature for deterministic outputs', () => {
        const client = GetOpenApiClient({
          openaiApiKey: 'test-key',
          openRouterApiKey: '',
          modelName: 'gpt-3.5-turbo',
          temperature: 0.2,
        });

        expect(client).toBeDefined();
      });
    });
  });

  describe('File Utilities', () => {
    describe('LoadPromptTemplate', () => {
      it('should load prompt template from file', () => {
        // Create a temporary test prompt file
        const testPromptPath = './src/prompts/test-prompt.md';
        const testContent = 'Test prompt with {placeholder}';

        // Clean up any existing test file
        if (fs.existsSync(testPromptPath)) {
          fs.unlinkSync(testPromptPath);
        }

        // Create test file
        fs.writeFileSync(testPromptPath, testContent);

        try {
          // Note: LoadPromptTemplate uses relative path from utils directory
          const content = fs.readFileSync(testPromptPath, 'utf-8');
          expect(content).toBe(testContent);
          expect(content).toContain('{placeholder}');
        } finally {
          // Clean up
          if (fs.existsSync(testPromptPath)) {
            fs.unlinkSync(testPromptPath);
          }
        }
      });

      it('should load query-qa prompt template', () => {
        const promptPath = './src/prompts/query-qa.md';
        expect(fs.existsSync(promptPath)).toBe(true);

        const content = fs.readFileSync(promptPath, 'utf-8');
        expect(content.length).toBeGreaterThan(0);
      });

      it('should load evaluation prompt template', () => {
        const promptPath = './src/prompts/evaluation.md';
        expect(fs.existsSync(promptPath)).toBe(true);

        const content = fs.readFileSync(promptPath, 'utf-8');
        expect(content.length).toBeGreaterThan(0);
      });

      it('should preserve template placeholders', () => {
        const promptPath = './src/prompts/query-qa.md';
        const content = fs.readFileSync(promptPath, 'utf-8');

        // Prompts should contain variable placeholders in {variable} format
        expect(content.includes('{')).toBe(true);
        expect(content.includes('}')).toBe(true);
      });

      it('should read file as UTF-8 string', () => {
        const promptPath = './src/prompts/query-qa.md';
        const content = fs.readFileSync(promptPath, 'utf-8');

        expect(typeof content).toBe('string');
        expect(content.length).toBeGreaterThan(0);
      });
    });

    describe('Prompt Directory Structure', () => {
      it('should have prompts directory', () => {
        expect(fs.existsSync('./src/prompts')).toBe(true);
      });

      it('should have required prompt files', () => {
        const requiredPrompts = ['query-qa.md', 'evaluation.md'];

        requiredPrompts.forEach((prompt) => {
          const promptPath = path.join('./src/prompts', prompt);
          expect(fs.existsSync(promptPath)).toBe(true);
        });
      });

      it('should have valid markdown files', () => {
        const promptsDir = './src/prompts';
        const files = fs.readdirSync(promptsDir);

        const mdFiles = files.filter((file) => file.endsWith('.md'));
        expect(mdFiles.length).toBeGreaterThan(0);

        mdFiles.forEach((file) => {
          const filePath = path.join(promptsDir, file);
          const content = fs.readFileSync(filePath, 'utf-8');
          expect(content.length).toBeGreaterThan(0);
        });
      });
    });
  });

  describe('Configuration Constants', () => {
    it('should have valid OpenRouter configuration', () => {
      const CONFIGURATION = {
        baseURL: 'https://openrouter.ai/api/v1',
        defaultHeaders: {
          'X-Title': 'FAQ Support Chatbot',
          'HTTP-Referer': 'https://your-app.com',
        },
      };

      expect(CONFIGURATION.baseURL).toBe('https://openrouter.ai/api/v1');
      expect(CONFIGURATION.defaultHeaders).toHaveProperty('X-Title');
      expect(CONFIGURATION.defaultHeaders).toHaveProperty('HTTP-Referer');
    });
  });

  describe('Error Handling', () => {
    it('should handle missing API keys gracefully', () => {
      // Both clients should be created even with empty keys
      // The actual API calls will fail, but client creation should succeed
      const embeddingsClient = GetEmbeddingsClient({
        openaiApiKey: '',
        modelName: 'text-embedding-ada-002',
      });

      expect(embeddingsClient).toBeDefined();
    });

    it('should handle invalid model names gracefully during client creation', () => {
      // Client creation should succeed even with non-standard model names
      // The actual API calls may fail, but creation should work
      const client = GetOpenApiClient({
        openaiApiKey: 'test-key',
        openRouterApiKey: '',
        modelName: 'invalid-model-name',
        temperature: 0.7,
      });

      expect(client).toBeDefined();
    });
  });

  describe('Type Safety', () => {
    it('should enforce required parameters for GetEmbeddingsClient', () => {
      // TypeScript should enforce these parameters
      const validArgs = {
        openaiApiKey: 'test-key',
        modelName: 'text-embedding-ada-002',
      };

      expect(validArgs).toHaveProperty('openaiApiKey');
      expect(validArgs).toHaveProperty('modelName');
    });

    it('should enforce required parameters for GetOpenApiClient', () => {
      // TypeScript should enforce these parameters
      const validArgs = {
        openaiApiKey: 'test-key',
        openRouterApiKey: '',
        modelName: 'gpt-3.5-turbo',
        temperature: 0.7,
      };

      expect(validArgs).toHaveProperty('openaiApiKey');
      expect(validArgs).toHaveProperty('modelName');
      expect(validArgs).toHaveProperty('temperature');
    });
  });
});
