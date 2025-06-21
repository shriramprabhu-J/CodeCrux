

## **Flow Summary (Input to Output):**

```
User → API Gateway → LangChain Agents:
  ├── CodeUnderstandingAgent
  ├── TestCaseGenerationAgent
  ├── CodeExecutionAgent
  ├── EvaluationAgent
  └── FeedbackAgent
→ Execution Sandbox
→ MongoDB (store everything)
→ API Response → User
```

---

Let me know if you'd like a **Mermaid.js diagram** or **PNG/SVG image** for this to embed in your README.
