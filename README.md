# AI Code Evaluator Architecture (with LangChain & MongoDB)

This architecture outlines a full-stack AI-powered code evaluator system. The diagram below illustrates components, their interactions, and data persistence in MongoDB.

---

## **Mermaid Architecture Diagram**

```mermaid
flowchart TD
    classDef inputLayer fill:#f9f,stroke:#333,stroke-width:2px;
    classDef apiLayer fill:#bbf,stroke:#333,stroke-width:2px;
    classDef agentLayer fill:#bfb,stroke:#333,stroke-width:2px;
    classDef sandboxLayer fill:#ffb,stroke:#333,stroke-width:2px;
    classDef dbLayer fill:#fbf,stroke:#333,stroke-width:2px;
    classDef outputLayer fill:#fbb,stroke:#333,stroke-width:2px;

    subgraph Input["User Input Layer"]
      A[Frontend UI / CLI]:::inputLayer
    end

    subgraph API["API Gateway"]
      B[FastAPI / analyze-code]:::apiLayer
    end

    subgraph Agents["LangChain Agent Orchestration"]
      C1[SyntaxValidatorAgent]:::agentLayer
      C2[LogicConsistencyAgent]:::agentLayer
      C3[OptimizationAdvisorAgent]:::agentLayer
      C4[ExplanationRAGAgent]:::agentLayer
      C5[ProgressiveHinterAgent]:::agentLayer
    end

    subgraph Sandbox["Execution Environment"]
      D[Docker Sandbox]:::sandboxLayer
    end

    subgraph Database["MongoDB Storage"]
      E1[(submissions)]:::dbLayer
      E2[(feedback_history)]:::dbLayer
    end

    subgraph Output["Response to User"]
      F[FullFeedbackResponse JSON]:::outputLayer
    end

    A -->|POST code + language| B
    B -->|run_manual_workflow| C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5
    C5 --> B
    B -->|store_submission| E1
    B -->|store_feedback| E2
    C2 -->|optional code run| D
    D -->|execution logs/results| B
    B -->|return feedback| F

    %% legend styling via classes
```

---

### **Component Descriptions**

1. **User Input Layer** (`Frontend UI / CLI`)

   * Collects code, language, and preferences

2. **API Gateway** (`FastAPI`)

   * Endpoint `/analyze-code`
   * Validates input, orchestrates workflow
   * Background tasks for MongoDB storage

3. **LangChain Agents**

   * **SyntaxValidatorAgent**: Checks syntax via `ChatPromptTemplate` → Gemini LLM → `JsonOutputParser`
   * **LogicConsistencyAgent**: Identifies logical flaws with LLM prompts
   * **OptimizationAdvisorAgent**: Suggests code improvements (Big-O rationale)
   * **ExplanationRAGAgent**: Uses FAISS retriever + embeddings to ground explanations
   * **ProgressiveHinterAgent**: Generates staged hints & final fix using LLM

4. **Execution Environment** (`Docker Sandbox`)

   * (Optional) Runs code against test cases, captures stdout/stderr, exit codes

5. **Database Layer** (`MongoDB`)

   * **submissions**: Stores session\_id, learner\_id, code, language, timestamp
   * **feedback\_history**: Stores full feedback payload (syntax, logic, optimizations, explanations, hints, final\_fix)

6. **Output Layer** (`JSON Response`)

   * Returns `FullFeedbackResponse` containing all feedback sections

---

Embed this Mermaid diagram and descriptions in your GitHub README for a clear, colorful architecture overview.
