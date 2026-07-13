
# 🔍 PatchContext — RAG Pipeline over FastAPI Repository History

> *"Why was this designed this way?"* — Ask anything about FastAPI's design decisions,
> grounded in real commit history, pull requests, and issue threads.

---

## 📌 Project Overview

**PatchContext** is an AI-powered developer assistant that builds a
Retrieval-Augmented Generation (RAG) pipeline over the
[FastAPI GitHub repository](https://github.com/fastapi/fastapi).

It lets engineers ask questions about design decisions and get answers
grounded in actual developer discussions — with clickable citations to
commit SHAs, PR numbers, and issue IDs.

---

## 🎯 Key Features

- 🔍 **Semantic Search** over FastAPI issues, PRs, and commits
- 💬 **Natural Language Q&A** powered by Groq LLaMA 3.1
- 🔗 **Clickable Citations** — every answer cites Issue #, PR #, or Commit SHA
- 🔄 **MMR Retrieval** — Maximum Marginal Relevance for diverse results
- 🛡️ **Hallucination Guard** — NLI-based grounding check blocks fabricated answers
- 📊 **RAGAs Evaluation** — 10-question benchmark measures system performance
- 🎨 **Premium Dark UI** — Glassmorphic Streamlit dashboard

---

## 🏗️ Architecture


---

## 🛠️ Tech Stack

- **Python 3.11+**
- **Streamlit** (UI/Dashboard)
- **LangChain** (RAG Orchestration & Prompting)
- **FAISS** (Vector Indexing & Retrieval)
- **Sentence Transformers (`all-MiniLM-L6-v2`)** (Local Document Embeddings)
- **Groq Cloud API (`llama-3.1-8b-instant`)** (Inference Engine)
- **Pandas & Matplotlib** (Evaluation & Visualization)

---

## ⚙️ Architecture

```
User Query
    │
    ▼
RAG Pipeline (rag_pipeline.py)
    │
    ├─► MMR Retrieval (k=5, fetch_k=20, lambda_mult=0.7) ──► FAISS Vector Store
    │                                                              ▲
    │                                                              │ (Build / Load Cache)
    │                                                     GitHub REST API
    │                                                  (Issues, PRs, Commits)
    │
    ▼
ChatGroq (llama-3.1-8b-instant)
    │
    ▼
Raw Generated Answer
    │
    ▼
Grounding Check (hallucination_guard.py)
    │
    ├─► Pass (Score >= 0.15) ──► Badge: Safe 🟢
    ├─► Warning (0.05 - 0.15) ─► Badge: Warning 🟡
    └─► Fail (Score < 0.05) ──► Badge: Hallucinated 🔴
```

---

## 📂 Project Structure

```
PatchContext/
├── app.py                   # Streamlit main application and dashboard UI
├── rag_pipeline.py          # Vector store setup, MMR retriever, and QA chains
├── data_fetcher.py          # GitHub API integration (Issues, PRs, Commits)
├── hallucination_guard.py    # Text cleaning, normalization, and grounding score formula
├── evaluation.py            # Offline benchmark test suite and metrics plotting
├── requirements.txt         # Project dependencies
├── README.md                # System documentation
├── .gitignore               # Ignored files (secrets, caches, local indexes)
└── .env.example             # Environment variable configuration template
```

---

## 🔧 Installation & Setup

1. **Clone or Copy the Workspace**

2. **Set up virtual environment (Recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate   # On Windows
   source venv/bin/activate # On Unix/macOS
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root of the project (copying `.env.example`):
   ```bash
   GROQ_API_KEY=your_groq_key_here
   GITHUB_TOKEN=your_github_token_here  # Optional, but prevents rate limit restrictions
   ```

---

## 🖥️ Running the Application

To start the Streamlit dashboard:
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📈 Evaluation

To run the offline benchmark suite and measure grounding metrics:
1. Navigate to the **Pipeline Evaluation** tab in the Streamlit app.
2. Click **Run Offline Evaluation Suite**.
3. View the question-by-question breakdown table and generated grounding scores visualization.

---

## 👤 Author

- **Name**: Ashwani Kumar
- **Affiliation**: Celebal technologies Internship 2026
