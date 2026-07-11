# PatchContext: FastAPI Design Decision RAG Assistant

PatchContext is a production-grade Retrieval-Augmented Generation (RAG) assistant designed to help developers understand the architectural design decisions behind FastAPI. Instead of relying on general knowledge, it fetches, indexes, and queries closed issues, pull requests, and commits directly from the FastAPI repository, generating answers grounded entirely in historical evidence.

If the answer cannot be found in the repository history, PatchContext is guaranteed to return exactly:
`"I couldn't find this in repository history."`

---

## 🚀 Features

- **GitHub Data Fetching**: Retrieves issues, pull requests, and commits using the GitHub REST API.
- **MMR Retrieval**: Uses Maximal Marginal Relevance (MMR) retrieval to fetch highly diverse and relevant documents.
- **Strict Grounding Guard**: Measures the percentage of generated words present in source documents to detect and prevent hallucinations.
- **Visual Analytics**: Includes an offline evaluation pipeline measuring system latency, retrieval effectiveness, and grounding scores for 10 benchmark developer questions.
- **Premium Glassmorphic UI**: Beautiful dark-themed Streamlit web interface with interactive examples, execution metrics, and citation badges.

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
- **Affiliation**: Celebal Technology Internship 2026
