# RAG-Based FAQ Support Chatbot

An intelligent FAQ support chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions from HR documentation. Built with LangChain, OpenAI embeddings, and HNSW vector search.

## Quick Start

Get up and running in 5 minutes:

```bash
# 1. Install dependencies
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Build the vector index
npm run run:embed

# 4. Ask questions!
npm run run:query "What is the sick leave policy?"

# 5. Run sample queries (optional)
npm run run:sample

# 6. Run tests (optional)
npm test
```

### Common Commands

| Command | Description |
|---------|-------------|
| `npm run run:embed` | Build vector index from FAQ document |
| `npm run run:query "question"` | Ask a question |
| `npm run run:sample` | Run predefined sample queries and save to outputs/ |
| `npm test` | Run test suite |

## Overview

This system parses an HR SaaS FAQ document, chunks it intelligently, generates embeddings, and stores them in a vector database for efficient retrieval. When users ask questions, the system performs similarity search to find relevant chunks and uses an LLM to generate accurate, context-aware answers.

### Key Features

- **Document Chunking**: Uses RecursiveCharacterTextSplitter with overlap to preserve context
- **Vector Embeddings**: OpenAI text-embedding-3-small for high-quality, cost-effective embeddings
- **Vector Search**: HNSW (Hierarchical Navigable Small World) for fast approximate nearest neighbor search
- **RAG Pipeline**: Modern LangChain LCEL (LangChain Expression Language) for retrieval-augmented generation
- **JSON Output**: Structured responses with question, answer, and related chunks
- **Automatic Logging**: Every query is automatically logged to `outputs/sample_queries.json` with timestamp
- **Integrated Evaluation**: Each query is automatically evaluated and scored (0-10) with detailed reasoning
- **Type Safety**: Full TypeScript implementation with comprehensive type definitions

## Prerequisites

- **Node.js**: Version 18 or higher
- **Package Manager**: npm (Node Package Manager)
- **OpenAI API Key**: Get yours at https://platform.openai.com/api-keys

## Detailed Setup

### Installation

```bash
# Clone repository
git clone <repository-url>
cd henry-wk2

# Install dependencies
npm install
```

### Configuration

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Add your OpenAI API key to `.env`:

```env
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Building the Index

The vector index must be built before querying:

```bash
npm run run:embed
```

This process:
- Loads `data/faq_document.txt`
- Chunks text (500 chars, 100 overlap)
- Generates embeddings via OpenAI
- Stores vectors in `data/vector-store/`
- Creates `chunk-info.json` metadata

Expected output: 81 chunks created successfully

## Usage

### Query the Chatbot

Ask questions about the HR policies and procedures:

```bash
npm run run:query "How do I apply for leave?"
npm run run:query "What is our sick leave policy?"
npm run run:query "What is the company's sick leave policy?"
npm run run:query "How do I submit an expense reimbursement?"
```

### Example Query Output

When you run a query, the system automatically:
1. Retrieves relevant chunks from the vector store
2. Generates an answer using the LLM
3. Evaluates the response quality (0-10 score)
4. Logs everything to `outputs/sample_queries.json`

Example logged output:

```json
{
  "user_question": "What is the company's sick leave policy?",
  "system_answer": "The company provides 10 days of paid sick leave per year. Sick leave does not roll over to the next year. For absences of 3 consecutive days or more, a medical certificate is required. You should notify your manager as early as possible and log sick leave in the HR portal.",
  "chunks_related": [
    {
      "content": "Q: What is the sick leave policy?\nA: Employees are entitled to 10 days of paid sick leave per year...",
      "metadata": {
        "chunkIndex": 3,
        "source": "./data/faq_document.txt"
      }
    }
  ],
  "evaluation": {
    "score": 9,
    "reason": "The answer is highly accurate and directly addresses the question with all key details from the chunks.",
    "breakdown": {
      "chunk_relevance": 10,
      "answer_accuracy": 9,
      "completeness": 8
    }
  },
  "timestamp": "2025-11-17T10:30:45.123Z"
}
```

### Automatic Output Logging

All queries are **automatically logged** to `outputs/sample_queries.json`. Each entry includes:
- User question
- System answer
- Retrieved chunks with metadata
- Evaluation score and breakdown
- ISO timestamp

You can review all your queries by checking this file:

```bash
cat outputs/sample_queries.json
```

To disable automatic evaluation (faster queries, lower cost):

```typescript
import { query } from './src/query/index.js';
import { loadAndValidateEnv } from './src/query/env.js';

const env = loadAndValidateEnv();
const response = await query("Your question", env, false); // false = disable evaluation
```

### Integrated Evaluation

The evaluator runs **automatically** with every query and provides:
- **Overall score** (0-10): Comprehensive quality assessment
- **Breakdown scores**:
  - Chunk relevance (0-10): How relevant are the retrieved chunks?
  - Answer accuracy (0-10): How accurate is the generated answer?
  - Completeness (0-10): Does the answer fully address the question?
- **Reason**: Detailed explanation of the score

The evaluation is automatically included in the logged output at `outputs/sample_queries.json`.

The evaluation runs automatically with every query. To run sample queries with evaluations:

```bash
npm run run:sample
```

### Testing

Run the test suite (validates 20+ chunks, document loading, query format, etc.):

```bash
npm test
```

## Project Structure

```
henry-wk2/
├── data/
│   ├── faq_document.txt          # Source FAQ document (1000+ words)
│   └── vector-store/              # Persisted vector index (generated)
│       ├── docstore.json
│       ├── hnswlib.index
│       ├── args.json
│       └── chunk-info.json
├── src/
│   ├── types.ts                   # TypeScript type definitions
│   ├── build-index/
│   │   └── index.ts               # Indexing pipeline (chunking + embeddings)
│   ├── query/
│   │   └── index.ts               # Query pipeline (retrieval + generation + logging)
│   ├── evaluator.ts               # Evaluator agent (automatic scoring)
│   └── utils/
│       ├── openai.ts              # OpenAI client utilities
│       └── file.ts                # File loading utilities
├── tests/
│   └── test-core.test.ts          # Test suite
├── outputs/                        # Auto-generated query logs
│   └── sample_queries.json        # All queries with evaluations & timestamps
├── .env.example                    # Environment variable template
├── package.json                    # Dependencies and scripts
├── tsconfig.json                   # TypeScript configuration
└── README.md                       # This file
```

## Technical Decisions

### Chunking Strategy

**Choice**: RecursiveCharacterTextSplitter with 500 character chunks and 100 character overlap

**Rationale**:
- **Recursive splitting**: Splits on natural boundaries (paragraphs, sentences) to preserve semantic meaning
- **500 characters**: Balances context size with granularity (typically 1-2 Q&A pairs)
- **100 character overlap**: Prevents information loss at chunk boundaries, ensures questions split across chunks remain retrievable
- **Result**: Generated 81 meaningful chunks from the FAQ document

### Embedding Model

**Choice**: OpenAI `text-embedding-3-small`

**Rationale**:
- **Cost-effective**: ~5x cheaper than text-embedding-3-large
- **Performance**: 1536 dimensions, excellent for semantic search
- **Proven**: Industry-standard model with strong retrieval performance
- **API Integration**: Native LangChain support

### Vector Store

**Choice**: HNSWLib (Hierarchical Navigable Small World)

**Rationale**:
- **Fast ANN search**: Approximate Nearest Neighbor with O(log n) query time
- **No external dependencies**: Runs locally, no database server required
- **Persistable**: Save and load from disk
- **Production-ready**: Used in production systems at scale
- **LangChain integration**: First-class support

### Retrieval Method

**Choice**: Similarity search with k=3

**Rationale**:
- **k-NN approach**: Standard for RAG systems, retrieves most similar chunks
- **k=3**: Provides sufficient context without overwhelming the LLM with noise
- **Configurable**: Can adjust via RETRIEVAL_K environment variable
- **Alternative considered**: Hybrid search (vector + keyword) for edge cases

### LLM for Generation

**Choice**: GPT-3.5-turbo (default, configurable)

**Rationale**:
- **Fast**: Low latency for user-facing applications
- **Cost-effective**: Significantly cheaper than GPT-4
- **Sufficient for Q&A**: Excellent at summarizing and answering from provided context
- **Temperature 0.3**: Lower temperature for more factual, consistent responses

### RAG Implementation

**Choice**: LangChain Expression Language (LCEL) with RunnableSequence

**Rationale**:
- **Modern approach**: Replaces deprecated RetrievalQAChain
- **Composable**: Easy to customize and extend the RAG pipeline
- **Type-safe**: Full TypeScript support with proper type inference
- **Flexible**: Can easily add preprocessing, postprocessing, or multi-step reasoning
- **Production-ready**: Recommended by LangChain for new implementations

## Configuration

All configuration is managed through environment variables in `.env`:

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - | Your API key |
| `EMBEDDING_MODEL` | Model for embeddings | `text-embedding-3-small` | `text-embedding-3-large`, `text-embedding-ada-002` |
| `LLM_MODEL` | Model for answer generation | `gpt-3.5-turbo` | `gpt-4`, `gpt-4-turbo-preview` |
| `CHUNK_SIZE` | Characters per chunk | `500` | 200-1000 |
| `CHUNK_OVERLAP` | Overlap between chunks | `100` | 50-200 |
| `RETRIEVAL_K` | Number of chunks to retrieve | `3` | 1-10 |
| `DOCUMENT_PATH` | Path to FAQ document | `./data/faq_document.txt` | Any path |
| `VECTOR_STORE_PATH` | Path to vector store | `./data/vector-store` | Any path |

## API Cost Estimates

Based on OpenAI pricing (as of 2024):

**Initial Indexing** (one-time):
- ~25 chunks × 500 chars = 12,500 characters
- Embeddings: ~$0.0001 (negligible)

**Per Query**:
- Query embedding: ~$0.000001
- LLM generation (GPT-3.5): ~$0.0005 per query
- **Total**: ~$0.0005 per query

For 200 queries/day: ~$0.10/day or $3/month

## Known Limitations

1. **English Only**: FAQ document and queries are in English. Non-English queries may have reduced accuracy.

2. **No Conversation Memory**: Each query is independent. No conversation history or follow-up context.

3. **Static Knowledge Base**: Vector index must be rebuilt when FAQ document changes. No real-time updates.

4. **Similarity Score**: HNSW vector store doesn't return similarity scores by default. Scores are listed as `undefined` in output.

5. **Context Window**: With k=3 chunks at ~500 chars each, context is limited to ~1500 characters. Complex multi-topic questions may not retrieve all relevant information.

6. **No Hybrid Search**: Pure vector search. May miss exact keyword matches if semantic similarity is low.

7. **Answer Hallucination**: LLM may occasionally generate information not present in chunks, despite prompt instructions. Use evaluator to detect this.

## Extending the System

### Add More Documents

To add additional FAQ documents:

1. Add files to `data/` directory
2. Update `src/build-index.ts` to use `DirectoryLoader`:

```typescript
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { TextLoader } from 'langchain/document_loaders/fs/text';

const loader = new DirectoryLoader('data/', {
  '.txt': (path) => new TextLoader(path),
});
```

3. Rebuild the index: `npm run run:embed`

### Implement Web API

Create an Express/Hono server in `src/index.ts`:

```typescript
import { Hono } from 'hono';
import { query } from './query.js';

const app = new Hono();

app.post('/query', async (c) => {
  const { question } = await c.req.json();
  const response = await query(question);
  return c.json(response);
});

app.listen(3000);
```

### Add Conversation Memory

Use LangChain's `BufferMemory` or `ConversationChain`:

```typescript
import { BufferMemory } from 'langchain/memory';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const memory = new BufferMemory({
  memoryKey: 'chat_history',
  returnMessages: true,
});

const chain = ConversationalRetrievalQAChain.fromLLM(
  llm,
  retriever,
  { memory }
);
```

### Implement Hybrid Search

Combine vector search with keyword search:

```typescript
import { MultiQueryRetriever } from 'langchain/retrievers/multi_query';

const retriever = MultiQueryRetriever.fromLLM({
  llm,
  retriever: vectorStore.asRetriever(),
  queryCount: 3, // Generate 3 variations of the query
});
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Vector store not found** | Run `npm run run:embed` first |
| **OpenAI API key not set** | Check `.env` file has `OPENAI_API_KEY=sk-...` |
| **OpenAI rate limits** | Use paid account, reduce `RETRIEVAL_K`, or add retry logic |
| **Poor answer quality** | Increase `RETRIEVAL_K=5`, adjust chunking, switch to GPT-4, or use evaluator |
| **Irrelevant chunks** | Rebuild with different chunking params or try `text-embedding-3-large` |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `npm test`
5. Submit a pull request
