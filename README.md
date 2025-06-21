# AI-Based Code Evaluator: Architecture Overview

This document provides a full architecture diagram and breakdown of the code evaluator system using LangChain, Gemini, and FAISS. The system analyzes code submissions and provides feedback on syntax, logic, optimization, and progressive hints.

---

## 🔄 End-to-End Architecture Overview

```
           +-------------------+
           |   User Submission |
           +--------+----------+
                    |
                    v
           +--------+----------+
           |  process_code()   |  <-- Entry point
           +--------+----------+
                    |
                    v
+-------------------+-------------------+
|    Sequential Tool Evaluation Flow    |
+-------------------+-------------------+
                    |
   +----------------+----------------+
   |                                 |
   v                                 v
[Syntax Validator]         [Logic Consistency Checker]
  - ChatPromptTemplate       - ChatPromptTemplate
  - Gemini LLM               - Gemini LLM
  - JsonOutputParser         - JsonOutputParser
                    |                                 
                    v                                 
   +----------------+----------------+
   |                                 |
   v                                 v
[Optimization Advisor]      [Progressive Hinter]
  - ChatPromptTemplate        - Uses explanation first
  - Gemini LLM                - Then generates hints
  - JsonOutputParser          - Both use LLM chains
                    |                                 
                    v                                 
                 [Hint + Explanation Generator]
                 - RAG with FAISS
                 - HuggingFaceEmbeddings
                 - DocumentRetriever (LangChain)
                 - Gemini LLM
                    |
                    v
           +--------+----------+
           |  Aggregation &    |
           |  Final Response   |
           +--------+----------+
                    |
                    v
           +--------+----------+
           |     JSON Output    |
           +-------------------+
```

