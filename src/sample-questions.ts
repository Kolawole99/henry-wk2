import { loadAndValidateEnv } from "./query/env.js";
import { query } from "./query/index.js";

const SAMPLE_QUESTIONS = [
  "What is the company's sick leave policy?",
  "How do I submit an expense reimbursement request?",
  "What professional development benefits does the company offer?",
  "How do I apply for leave?",
  "What is the process for performance reviews?",
  "How can I update my personal information in the HR system?",
];

async function executeSampleQuestions() {
  for (const question of SAMPLE_QUESTIONS) {
    const env = loadAndValidateEnv();
    
    await query(question, env);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) {
  await executeSampleQuestions();
}
