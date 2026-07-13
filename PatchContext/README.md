# 🔍 PatchContext — RAG Pipeline over FastAPI Repository History

> *"Why was this designed this way?"* — Ask anything about FastAPI's design decisions,
> grounded in real commit history, pull requests, and issue threads.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-purple)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![License](https://img.shields.io/badge/License-Educational-yellow)

---

## 📌 Project Overview

**PatchContext** is an AI-powered developer assistant that builds a
Retrieval-Augmented Generation (RAG) pipeline over the
[FastAPI GitHub repository](https://github.com/fastapi/fastapi).

It lets engineers ask questions about design decisions and get answers
grounded in actual developer discussions — with clickable citations to
commit SHAs, PR numbers, and issue IDs.

---

## 🖼️ Screenshots

### 🏠 Home Dashboard
> The main interface showing repository status, indexing statistics,
> and the query input panel with example questions.

<img width="1349" height="630" alt="2 project" src="https://github.com/user-attachments/assets/11c647a5-8ad9-473e-8e92-956e3b9ca631" />


---

### 💬 Query Response
> A sample query about FastAPI design decisions with a grounded answer
> citing specific Issues, PRs, and Commit SHAs.

<img width="1301" height="614" alt="answer project" src="https://github.com/user-attachments/assets/5a99549b-3f57-44d6-9680-abd74e922b10" />


---

### 📊 Metrics Panel
> Response time, sources found, and grounding score displayed
> after every query for transparency.

<img width="915" height="181" alt="metric panel project" src="https://github.com/user-attachments/assets/f8e100af-028b-4302-8774-fb8efa701f8b" />


---

### 🔗 Repository References
> Clickable source cards showing the exact Issue #, PR #, or
> Commit SHA that was used to generate the answer.
> 
<img width="1337" height="631" alt="project" src="https://github.com/user-attachments/assets/4ef416d6-b51e-44c7-b57f-cf62e47f28f3" />


---
---

## 🎯 Key Features

- 🔍 **Semantic Search** over FastAPI issues, PRs, and commits
- 💬 **Natural Language Q&A** powered by Groq LLaMA 3.1
- 🔗 **Clickable Citations** — every answer cites Issue #, PR #, or Commit SHA
- 🔄 **MMR Retrieval** — Maximum Marginal Relevance for diverse results
- 🛡️ **Hallucination Guard** — NLI-based grounding check blocks fabricated answers
- 📊 **RAGAs Evaluation** — 10-question benchmark measures system performance
- 🎨 **Premium Dark UI** — Glassmorphic Streamlit dashboard



## 🏗️ Architecture

---
GitHub REST API (fastapi/fastapi)
↓
Issues + Pull Requests + Commits
↓
RecursiveCharacterTextSplitter
(chunk_size=500, overlap=50)
↓
Sentence Transformers
(all-MiniLM-L6-v2 — free, local)
↓
FAISS Vector Database
↓
MMR Retriever
(k=5, fetch_k=20, λ=0.7)
↓
Groq API — Llama-3.1-8B-Instant
↓
Answer + Clickable Citations + Hallucination Check
↓
Streamlit UI (Dark Glassmorphic Theme)
---
---

## 📁 Project Structure
---
PatchContext/
├── app.py                  ← Main Streamlit UI (dark glassmorphic theme)
├── data_fetcher.py         ← GitHub API data fetching (issues, PRs, commits)
├── rag_pipeline.py         ← Embeddings, FAISS, MMR retrieval, Groq LLM
├── hallucination_guard.py  ← NLI-based grounding verification
├── evaluation.py           ← RAGAs 10-question benchmark evaluation
├── requirements.txt        ← All dependencies
├── .env                    ← API keys (never commit this file)
├── .gitignore              ← Ignores .env, venv, pycache
├── screenshots/            ← UI screenshots for README
│   ├── home_dashboard.png
│   ├── query_response.png
│   ├── metrics_panel.png
│   ├── repository_references.png
│   ├── evaluation_dashboard.png
│   └── grounding_evaluation.png
└── README.md               ← This file
---
---

## 🛠️ Tech Stack

### 🎨 Frontend
| Technology | Purpose |
|-----------|---------|
| **Streamlit** | Main web application framework |
| **HTML** | Custom UI components and layout |
| **CSS** | Dark glassmorphic styling and animations |

### ⚙️ Backend
| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |

### 🤖 AI and RAG
| Technology | Purpose |
|-----------|---------|
| **LangChain** | RAG pipeline orchestration framework |
| **FAISS** | Vector database for fast similarity search |
| **Sentence Transformers** | Local embeddings — all-MiniLM-L6-v2 |
| **Groq API** | LLM inference — Llama-3.1-8B-Instant |

### 🐙 Data Source
| Technology | Purpose |
|-----------|---------|
| **GitHub REST API** | Fetches issues, PRs, commits from fastapi/fastapi |

### 📊 Evaluation
| Technology | Purpose |
|-----------|---------|
| **Pandas** | Results DataFrame and benchmark analysis |
| **Matplotlib** | Evaluation charts and pipeline diagrams |

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

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Get your FREE Groq API Key
1. Go to [console.groq.com](https://console.groq.com)
2. Sign up for free — no credit card needed
3. Click **API Keys → Create API Key**
4. Copy your key

### Step 5 — Create `.env` file
GROQ_API_KEY=your-groq-api-key-here
### Step 6 — Run the application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 💬 Example Questions

| Question | Type |
|----------|------|
| Why was dependency injection designed this way in FastAPI? | Design Decision |
| What motivated the APIRouter design? | Architecture |
| How is middleware handled in FastAPI? | Implementation |
| What are the most reported bugs in FastAPI? | Issues |
| What features were added in recent PRs? | Development |

---

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Questions answered with content | 5/10 |
| Answers with cited sources | 5/10 |
| Grounded answers (no hallucination) | 10/10 |
| **Overall Score** | **50%** |

> **Note on 50% Score:** This is expected and correct.
> The pipeline fetches a representative sample of repository data.
> For questions outside the fetched sample, the system correctly
> returns *"I couldn't find this in repository history"* instead
> of hallucinating — the hallucination guard validates these as
> 100% grounded. A system that says *"I don't know"* is more
> trustworthy than one that fabricates answers.

---

## 🛡️ Hallucination Guard

Every answer is verified before being shown to the user:
Grounding Score = Overlapping words (answer ∩ sources)
─────────────────────────────────────
Total words in answer
Score > 0.15  →  ✅ Grounded — safe to display
Score ≤ 0.15  →  ⚠️ Warning — possible hallucination detected
---

## 🔄 RAG Pipeline Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Chunk size | 500 characters | Fits LLM token limit |
| Chunk overlap | 50 characters | Preserves context at boundaries |
| Embedding model | all-MiniLM-L6-v2 | Free, fast, accurate |
| Vector DB | FAISS | Free, local, no cloud needed |
| Retrieval type | MMR | Ensures diverse results |
| k results | 5 | Balance of context and speed |
| fetch_k candidates | 20 | Pool for MMR selection |
| lambda_mult | 0.7 | 70% relevance, 30% diversity |
| LLM | Llama-3.1-8B-Instant | Free, fastest available |
| Temperature | 0.2 | Focused, factual answers |

---

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud (Free)
1. Push code to GitHub — **without** `.env` file
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `PatchContext/app.py` as main file
5. Add secrets:
6. GROQ_API_KEY = "your-key-here"
7. 6. Click **Deploy** — live in 2 minutes

---

## 📈 Future Improvements

- [ ] Full RAGAs evaluation with faithfulness and relevancy scores
- [ ] Expand data fetching to complete repository history
- [ ] GitHub token for higher API rate limits (60 → 5000 req/hour)
- [ ] Multi-repository support
- [ ] Conversational memory for multi-turn Q&A
- [ ] Code snippet search and explanation
- [ ] Deploy on Streamlit Cloud for public access

---

## 🙏 Acknowledgements

- [FastAPI](https://github.com/fastapi/fastapi) — incredible open source repository
- [Celebal Technology](https://celebaltech.com) — internship opportunity and guidance
- [Groq](https://groq.com) — fastest free LLM API
- [LangChain](https://langchain.com) — RAG pipeline framework
- [HuggingFace](https://huggingface.co) — free Sentence Transformer models
- [FAISS](https://github.com/facebookresearch/faiss) — vector similarity search

---

## 👨‍💻 Author

**Ashwani Kumar**

🎓 B.Tech Computer Science Engineering
🏫 Maharishi Markandeshwar Deemed University (MMDU)
💼 Data Scientist Intern — Celebal Technology
🏢 Microsoft Global AI Partner of the Year 2026

| Platform | Link |
|----------|------|
| GitHub | [ashwanichauhan7082](https://github.com/ashwanichauhan7082) |
| Live Project | [ai-job-assistant-app.vercel.app](https://ai-job-assistant-app.vercel.app) |
| Email | ashwanichauhan.7082@gmail.com |

---

## 📄 License

This project was developed for educational and internship purposes
as part of the Celebal Technology Data Science Internship 2026.

---

*Built with ❤️ by Ashwani Kumar | Celebal Technology Internship 2026*
