import { ChatOpenAI, OpenAIEmbeddings } from '@langchain/openai';
import type { EmbeddingClientArgs } from '../types.js';

const CONFIGURATION = {
    baseURL: 'https://openrouter.ai/api/v1',
    defaultHeaders: {
        'X-Title': 'FAQ Support Chatbot',
        'HTTP-Referer': 'https://your-app.com',
    },
}

export function GetEmbeddingsClient(args: EmbeddingClientArgs) {
    let embeddings;
    
    if (args.openRouterApiKey) {
        console.log('Using OpenRouter API for embeddings');
        
        embeddings = new OpenAIEmbeddings({
            apiKey: args.openRouterApiKey,
            modelName: args.modelName,
            configuration: CONFIGURATION,
        });
    } else {
        console.log('Using OpenAI API for embeddings');
        
        embeddings = new OpenAIEmbeddings({
            apiKey: args.openaiApiKey,
            modelName: args.modelName,
        });
    }

    return embeddings;
}

interface OpenApiClientArgs extends EmbeddingClientArgs {
    temperature: number;
}

export function GetOpenApiClient(args: OpenApiClientArgs) {
    let client;
    
    if (args.openRouterApiKey) {
        console.log('Using OpenRouter API for LLM');
        
        client = new ChatOpenAI({
            apiKey: args.openRouterApiKey,
            modelName: args.modelName,
            temperature: args.temperature,
            configuration: CONFIGURATION,
        });
    } else {
        console.log('Using OpenAI API for LLM');
        
        client = new ChatOpenAI({
            apiKey: args.openaiApiKey,
            modelName: args.modelName,
            temperature: args.temperature,
        });
    }

    return client;
}
