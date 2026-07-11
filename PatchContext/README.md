
# 🔍 PatchContext – Grounded RAG Assistant for FastAPI Repository History

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
```
GitHub API
      │
      ▼
Issues + PRs + Commits
      │
      ▼
RecursiveCharacterTextSplitter
      │
      ▼
Embeddings (all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Store
      │
      ▼
MMR Retriever
      │
      ▼
Groq Llama 3.1
      │
      ▼
Grounding Check
      │
      ▼
Final Answer
```


---

## 📁 Project Structure

PatchContext/
├── app.py                  ← Main Streamlit UI (dark theme)
├── data_fetcher.py         ← GitHub API data fetching
├── rag_pipeline.py         ← Embeddings, FAISS, MMR, Groq LLM
├── hallucination_guard.py  ← NLI grounding verification
├── evaluation.py           ← RAGAs 10-question benchmark
├── requirements.txt        ← All dependencies
├── .env                    ← API keys (never commit this)
├── .gitignore              ← Ignores .env and venv
└── README.md               ← This file

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | LangChain |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (free, local) |
| Vector DB | FAISS (Facebook AI Similarity Search) |
| Retrieval | MMR — Maximum Marginal Relevance |
| LLM | Groq `llama-3.1-8b-instant` (free, fastest) |
| UI | Streamlit (dark glassmorphic theme) |
| Data Source | GitHub REST API (fastapi/fastapi) |
| Hallucination | NLI keyword grounding check |
| Evaluation | RAGAs benchmark (10 questions) |

---

## ⚡ Setup Instructions

### Step 1 — Clone the repository
```bash
git clone https://github.com/ashwanichauhan7082/Celebal-Internship-Data-Science.git
cd Celebal-Internship-Data-Science/PatchContext
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Get your FREE Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free
3. Click **API Keys → Create API Key**
4. Copy your key

### Step 5 — Create `.env` file
```env
GROQ_API_KEY=your-groq-api-key
GITHUB_TOKEN=optional_github_token
```
### Step 6 — Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 💬 Example Questions

- *"Why was dependency injection designed this way in FastAPI?"*
- *"What motivated the APIRouter design decision?"*
- *"How is middleware handled in FastAPI?"*
- *"What are the most common issues reported?"*
- *"Why does FastAPI use Pydantic for validation?"*

---

## 📊 Evaluation Results

| Metric | Result |
|---------|--------|
| Benchmark Questions | 10 |
| Grounded Answers | 10 |
| Repository-supported Answers | 5 |
| Hallucinations | 0 |
| Overall Grounding Score | 100% |

> **Note:** The 50% answer rate is expected — the pipeline fetches a 
> sample of repository data. For unanswered questions, the system 
> correctly returns *"I couldn't find this in repository history"* 
> instead of hallucinating — validated by the hallucination guard.

---

## 🛡️ Hallucination Guard

The hallucination guard computes a grounding score by comparing the generated answer with the retrieved repository documents.
Grounding Score = overlapping words between answer and sources
─────────────────────────────────────────────
```
Grounding Score = Overlapping words between answer and retrieved sources
------------------------------------------------------
Total words in generated answer

Score > 0.15  → Grounded
Score ≤ 0.15  → Possible Hallucination
```

## 📸 Screenshots


### Home Dashboard


<img width="1366" height="642" alt="image" src="https://github.com/user-attachments/assets/a236dc6d-7582-4dcf-a06d-046905492eee" />


### Query Response


<img width="1366" height="637" alt="image" src="https://github.com/user-attachments/assets/cb0de0d1-0e86-4b5e-b303-19c7cf9580de" />


### Metrics Panel



<img width="988" height="225" alt="image" src="https://github.com/user-attachments/assets/15a0a65b-b1f5-49cf-8530-757344ba2ffc" />


### Repository References


<img width="1086" height="595" alt="image" src="https://github.com/user-attachments/assets/db8cc709-858d-4484-a037-9653324ea8f2" />


### Evaluation Dashboard


<img width="1352" height="674" alt="image" src="https://github.com/user-attachments/assets/60e11d9a-9797-47b7-a1a4-25f260475599" />


---

## 🔄 RAG Pipeline Details

| Setting | Value |
|---------|-------|
| Chunk size | 500 characters |
| Chunk overlap | 50 characters |
| Embedding model | all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| Retrieval type | MMR |
| k (results) | 5 |
| fetch_k | 20 |
| lambda_mult | 0.7 |
| LLM model | llama-3.1-8b-instant |
| Temperature | 0.2 |

---

## 🚀 Deployment

To deploy on Streamlit Cloud:

1. Push code to GitHub (without `.env`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `GROQ_API_KEY` in Streamlit secrets
5. Deploy!

---

## 👨‍💻 Author

**Ashwani Kumar**
- 🎓 B.Tech CSE — MMDU (2027 Batch)
- 💼 Data Scientist Intern — Celebal Technology
- 🏢 Celebal Technology — Microsoft Global AI Partner of the Year 2026
- 🔗 GitHub: [ashwanichauhan7082](https://github.com/ashwanichauhan7082)

---

## 🙏 Acknowledgements

- [FastAPI](https://github.com/fastapi/fastapi) — for the incredible open source repository
- [Celebal Technology](https://celebaltech.com) — for the internship opportunity
- [Groq](https://groq.com) — for the fastest free LLM API
- [LangChain](https://langchain.com) — for the RAG framework
- [HuggingFace](https://huggingface.co) — for free embedding models

## 📄 License

This project was developed as part of the Celebal Technologies Data Science Internship 2026 for educational and research purposes.

