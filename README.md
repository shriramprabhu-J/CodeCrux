# 🔁 Agentic Feedback Pipeline – Execution Flow Only

## 📘 Flow Overview (Agents: Syntax → Optimizer → Hint → Explainer)

```plaintext
[Frontend - React App]
     │
     ▼
Submit Code (/submit_code)
     │
     ▼
[Backend - FastAPI]
     │
     ▼
Store Submission (MongoDB)
     │
     ▼
Trigger Agentic Pipeline
     │
     ▼
┌──────────────────────────────┐
│   Syntax Checker Agent       │
│   ▸ Detect syntax issues     │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│   Optimizer Agent            │
│   ▸ Suggest code improvements│
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│   Hint Generator Agent       │
│   ▸ Provide learner-focused  │
│     hints without solutions  │
└──────────────────────────────┘
     │
     ▼
┌──────────────────────────────┐
│   Explainer Agent            │
│   ▸ Summarize code behavior  │
└──────────────────────────────┘
     │
     ▼
Store Final Feedback (MongoDB)
     │
     ▼
[Frontend Polls /feedback/:id]
     │
     ▼
Render Structured Feedback
```

---

Let me know if you’d like this exported as a PNG/SVG diagram.
