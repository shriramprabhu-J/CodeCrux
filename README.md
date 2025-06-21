# 🧠 Agentic Code Feedback System – Full Flow Diagram (With React + FastAPI + LangChain)

## 🎯 Objective:

A full-stack agentic system that enables learners to submit code, receive intelligent feedback using LLMs, and track submission history. The system uses specialized agents for syntax checking, optimization, hint generation, and explanation.

---

## 🖼️ High-Level Architecture:

```plaintext
┌─────────────┐       ┌────────────────────────┐       ┌────────────────────────────────┐
│  Frontend   │──────▶│ FastAPI Backend (API)  │──────▶│      Agentic Orchestration      │
│ (React App) │       └────────────────────────┘       └────────────────────────────────┘
│             │                                               │   │   │   │
│             │                                               ▼   ▼   ▼   ▼
│             │                                         Syntax  Opt  Hint  Explain
└─────────────┘                                          Agent Agent Agent Agent
```

---

## ⚙️ Detailed Agentic Flow:

### 1. Frontend (React)

* **Component:** CodeEditor.jsx
* **Actions:**

  * User inputs code
  * Clicks "Submit"
  * Sends to `/submit_code`
  * Polls `/feedback/:id` to retrieve agent outputs

---

### 2. Backend (FastAPI)

* Receives and stores submission
* Orchestrates multiple agents in sequence or parallel

---

## 🤖 Agent Roles and Interactions

### 🔎 Syntax Checker Agent

* **Purpose:** Identify basic syntax or formatting issues
* **LLM Prompt:**

  ```plaintext
  You are a syntax validator. Review this code and list all syntax or indentation errors.
  ```
* **Output:**

  ```json
  { "errors": ["Missing colon on line 2", "Indentation error at line 5"] }
  ```

### 🚀 Optimizer Agent

* **Purpose:** Improve code performance or structure
* **LLM Prompt:**

  ```plaintext
  You are an optimization expert. Suggest performance or readability improvements for this code.
  ```
* **Output:**

  ```json
  { "recommendations": ["Use list comprehension", "Avoid nested loops"] }
  ```

### 💡 Hint Generator Agent

* **Purpose:** Generate learning-focused hints (not answers)
* **LLM Prompt:**

  ```plaintext
  You are a coding tutor. Provide hints to help the student debug or understand this code.
  ```
* **Output:**

  ```json
  { "hints": ["Check variable scope near line 10"] }
  ```

### 🧠 Explainer Agent

* **Purpose:** Explain what the code is doing line-by-line
* **LLM Prompt:**

  ```plaintext
  You are a code explainer. Provide a simple explanation of what the code does.
  ```
* **Output:**

  ```json
  { "summary": "This code defines a function to calculate factorial using recursion." }
  ```

---

## 🔁 Execution Flow Summary

```plaintext
1. Frontend submits code
2. Backend stores code and triggers pipeline
3. Syntax Agent runs → catches errors
4. Optimizer Agent runs → gives suggestions
5. Hint Agent runs → gives debug clues
6. Explainer Agent runs → generates summary
7. Backend aggregates outputs and stores
8. Frontend polls and renders feedback
```

---

## 🗃️ Backend Data Storage (MongoDB)

* `submissions_col`: raw user submissions
* `feedback_col`: structured multi-agent outputs per submission

---

## 📦 Future Extensibility

* Add Critic Agent to evaluate optimizer
* Use streaming response to deliver agent results in real time
* Agent feedback scoring by other LLMs
* Per-agent user feedback rating

---

Let me know if you want this visualized as an actual flowchart!
