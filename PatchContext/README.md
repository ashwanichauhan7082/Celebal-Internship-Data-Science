PatchContext – FastAPI Design Decision RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) system that explains the architectural evolution of FastAPI using GitHub Issues, Pull Requests, and Commits as its only source of truth.

Project Overview

PatchContext is a repository-aware AI assistant that helps developers understand why FastAPI was designed in a particular way.

Unlike traditional chatbots that rely on pretrained knowledge, PatchContext retrieves relevant discussions directly from the FastAPI GitHub repository and generates grounded answers supported by repository evidence.

If no relevant evidence exists, the system safely responds:

"I couldn't find this in repository history."

This minimizes hallucinations and ensures trustworthy answers.

Features

GitHub Repository Mining
Fetches Issues, Pull Requests, and Commits using the GitHub API.
Retrieval-Augmented Generation (RAG)
Uses FAISS vector search with Maximal Marginal Relevance (MMR) retrieval.
Hallucination Detection
Calculates grounding scores by comparing generated answers against retrieved documents.
Interactive Dashboard
Clean Streamlit interface with:
Dark/Light mode
Suggested repository questions
Repository statistics
Confidence & Grounding metrics
Source references
Search history
Pipeline Evaluation
Offline benchmark evaluation
Grounding score visualization
Latency analysis
Retrieval performance metrics
Repository Grounding
Every answer is generated only from retrieved GitHub repository content.


Tech Stack

Frontend
Streamlit
HTML
CSS
Backend
Python
AI & RAG
LangChain
FAISS
Sentence Transformers
Groq API (Llama-3.1-8B-Instant)
Data Source
GitHub REST API
Evaluation
Pandas
Matplotlib


System Architecture

User Query
      │
      ▼
 Streamlit UI
      │
      ▼
 RAG Pipeline
      │
      ├── GitHub Repository Data
      │        ├── Issues
      │        ├── Pull Requests
      │        └── Commits
      │
      ▼
 FAISS Vector Database
      │
      ▼
 MMR Retriever
      │
      ▼
 Groq LLM
      │
      ▼
 Hallucination Guard
      │
      ▼
 Final Grounded Answer



Project Structure

PatchContext/
│
├── app.py
├── rag_pipeline.py
├── data_fetcher.py
├── hallucination_guard.py
├── evaluation.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
└── faiss_index/
Installation
Clone Repository
git clone <repository-url>
cd PatchContext
Create Virtual Environment
python -m venv venv

Windows

venv\Scripts\activate

Linux / macOS

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Environment Variables

Create a .env file:

GROQ_API_KEY=your_groq_api_key
GITHUB_TOKEN=your_github_token

Run the Project

streamlit run app.py

Visit:

http://localhost:8501

Evaluation

Run the evaluation module:

python evaluation.py

The evaluation reports:

Grounding Score
Latency
Confidence
Retrieved Sources
Benchmark Performance

Sample Questions
Why was APIRouter introduced?
How does dependency injection work?
Explain FastAPI middleware architecture.
Why are path parameters validated?
Which pull request introduced this feature?
What design decisions led to this implementation?

Results
Repository Indexed
429 Documents
Pull Requests Indexed
62
Issues Indexed
5
Grounded Responses
Yes
Hallucination Detection
Enabled
Offline Evaluation
Completed


Screenshots


Home Dashboard

<img width="1366" height="642" alt="image" src="https://github.com/user-attachments/assets/a236dc6d-7582-4dcf-a06d-046905492eee" />

Query Response


<img width="1366" height="637" alt="image" src="https://github.com/user-attachments/assets/cb0de0d1-0e86-4b5e-b303-19c7cf9580de" />

Metrics Panel


<img width="988" height="225" alt="image" src="https://github.com/user-attachments/assets/15a0a65b-b1f5-49cf-8530-757344ba2ffc" />

Repository References


<img width="1086" height="595" alt="image" src="https://github.com/user-attachments/assets/db8cc709-858d-4484-a037-9653324ea8f2" />

Evaluation Dashboard


<img width="1352" height="674" alt="image" src="https://github.com/user-attachments/assets/60e11d9a-9797-47b7-a1a4-25f260475599" />




Future Improvements

Multi-repository support
Repository comparison
Incremental indexing
User authentication
Conversation memory
Citation highlighting
Docker deployment

Author

Ashwani Kumar

B.Tech Computer Science Engineering

Celebal Technologies Internship 2026

License

This project was developed for educational and internship purposes.
