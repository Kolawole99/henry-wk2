import * as fs from 'fs';
import * as path from 'path';
import type { QueryResponse, LoggedQueryOutput } from '../types.js';

const OUTPUT_DIR = './outputs';
const OUTPUT_FILE = path.join(OUTPUT_DIR, 'sample_queries.json');

/**
 * Logs query output to the outputs folder
 */
export function logQueryOutput(response: QueryResponse, evaluation?: any): void {
  try {
    if (!fs.existsSync(OUTPUT_DIR)) {
      fs.mkdirSync(OUTPUT_DIR, { recursive: true });
    }

    let queries: LoggedQueryOutput[] = [];
    if (fs.existsSync(OUTPUT_FILE)) {
      const existingData = fs.readFileSync(OUTPUT_FILE, 'utf-8');
      queries = JSON.parse(existingData);
    }

    const loggedOutput: LoggedQueryOutput = {
      ...response,
      timestamp: new Date().toISOString(),
      ...(evaluation && { evaluation }),
    };

    queries.push(loggedOutput);

    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(queries, null, 2));
  } catch (error) {
    console.error('Warning: Failed to log query output:', error);
  }
}
