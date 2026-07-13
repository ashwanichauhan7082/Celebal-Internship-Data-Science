# 🎓 Celebal Technology — Data Science Internship

**Intern:** Ashwani Kumar

**College:** MMDU (Maharishi Markandeshwar Deemed University)

**Role:** Data Scientist Intern

**Program:** Data Science

**Duration:** May 2026 – July 2026


---

## 🏆 Internship Completed Successfully

All 8 weekly assignments and the final  project have been
completed and submitted. This repository serves as a complete
portfolio of the Data Science internship at Celebal Technology.

---

## 📁 Repository Structure

| File/Folder | Week | Topic |
|-------------|------|-------|
| `week1_Ashwani_kumar_MMDU.ipynb` | Week 1 | ML Foundations |
| `week2_Ashwani_kumar_MMDU.ipynb` | Week 2 | Classical ML Pipeline |
| `week3_Ashwani_kumar_MMDU.ipynb` | Week 3 | Customer Intelligence System |
| `week4_Ashwani_kumar_MMDU.ipynb` | Week 4 | Deep Learning — CIFAR-10 |
| `week5_Ashwani_kumar_MMDU.ipynb` | Week 5 | RNN, LSTM, GRU |
| `week6_Ashwani_kumar_MMDU.ipynb` | Week 6 | Autoencoder Image Denoising |
| `week7_Ashwani_kumar_MMDU.ipynb` | Week 7 | RAG Document QA System |
| `week8_Ashwani_kumar_MMDU.ipynb` | Week 8 | Agentic AI Pipeline |
| `PatchContext/` | Final Project | RAG Pipeline over FastAPI Repo |

---

## 📘 Week 1 — ML Foundations

**Topics Covered:**
- 🐍 Python — data types, control flow, exceptions, lambdas
- 🔢 NumPy — arrays, indexing, matrix operations
- 🐼 Pandas — DataFrames, filtering, groupby, missing data
- 📐 Linear Algebra — vectors, eigenvalues, SVD, PCA
- 📊 Statistics — hypothesis testing, error metrics, distributions
- 🎲 Probability — Bayes theorem, CLT, distribution testing

**Key Highlights:**
- Implemented MAE, MSE, RMSE, R² from scratch
- Built Naive Bayes spam classifier
- Applied ADF stationarity test and PSI model monitoring

---

## 📗 Week 2 — Classical Machine Learning Pipeline

**Dataset:** Tesla EV Deliveries and Production Data (2015–2025)
**Source:** [Kaggle](https://www.kaggle.com/datasets/nalisha/tesla-ea-deliveries-and-production-data20152025)

**Topics Covered:**
- 🧹 Data Cleaning and EDA
- 🔤 Encoding and Feature Engineering
- ⚖️ Feature Scaling — StandardScaler
- 📈 Linear, Ridge (L2) and Lasso (L1) Regression
- 🔁 Cross Validation and Bias Variance Tradeoff
- 🎛️ Hyperparameter Tuning — GridSearchCV
- 📅 Time Series — Rolling Statistics, ADF Test, Forecasting

**Key Results:**
- Built end-to-end ML pipeline on Tesla EV dataset
- Compared Linear, Ridge and Lasso on R² and MAE
- Implemented chronological split for time series

---

## 📕 Week 3 — Customer Intelligence System

**Dataset:** Unsupervised Learning on Country Data (167 countries)
**Source:** [Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)

**Topics Covered:**
- 🔵 Classification — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- 🤝 Ensemble Learning — Stacking with meta-learner
- 📊 Evaluation — Accuracy, Precision, Recall, F1, Confusion Matrix
- 🌟 Feature Importance — GDP per capita identified as top feature
- 🔵 Clustering — K-Means, DBSCAN, Hierarchical
- 🗺️ PCA Visualization of clusters

**Key Results:**
- XGBoost and Random Forest achieved F1 Score of 1.0
- Logistic Regression: Accuracy 0.9706, F1 0.9655
- K-Means K=3 Silhouette Score: 0.2833
- Countries segmented into Developed, Developing, Underdeveloped

---

## 📙 Week 4 — Intro to Deep Learning (CIFAR-10)

**Dataset:** CIFAR-10 — 60,000 color images, 10 classes

**Topics Covered:**
- 🧠 Perceptron, MLP, Forward Pass, Backpropagation
- ⚡ Activation Functions — Sigmoid, Tanh, ReLU
- 🔍 Convolution, Pooling, Stride, Padding
- 🏗️ CNN Architectures and Transfer Learning
- 📈 Data Augmentation and BatchNormalization
- 🛑 EarlyStopping

**Models Built:**
| Model | Highlights |
|-------|-----------|
| ANN Baseline | Flattens image — loses spatial structure |
| CNN Baseline | ~22% better than ANN |
| CNN + Larger Filters | More complex feature capture |
| CNN + 20 Epochs | Improved accuracy |
| CNN + EarlyStopping | Optimal stopping point |
| CNN + Augmentation | Better generalization |

---

## 📒 Week 5 — RNN, LSTM and GRU (Text Generation)

**Task:** Compare sequence models for next-word prediction

**Topics Covered:**
- 🔁 Vanilla RNN — vanishing gradient problem
- 🔒 LSTM — forget, input, output gates
- ⚡ GRU — reset and update gates
- 🔤 Tokenization and n-gram sequences
- ✍️ Custom text generation function

**Tasks Completed:**
- Custom corpus replacement
- Embedding dimension increase (32→100)
- Training to 200 epochs
- Hidden units increase (64→128)
- 10-word text generation

**Key Results:**
- LSTM and GRU achieved lower loss than vanilla RNN
- GRU faster than LSTM with comparable performance

---

## 📓 Week 6 — Autoencoder for Image Denoising

**Dataset:** MNIST Handwritten Digits (built into TensorFlow)

**Topics Covered:**
- 🗜️ Autoencoder — Encoder + Latent Space + Decoder
- 🔄 Variational Autoencoder (VAE) concepts
- 🎨 GAN — Generator and Discriminator
- ⚙️ Optimizers — SGD, Momentum, AdaGrad, RMSprop, Adam
- 🤖 Generative AI Introduction
- 🔗 RAG concepts and Prompt Engineering

**What Was Built:**
- Added Gaussian noise (factor=0.3) to MNIST images
- Convolutional Autoencoder — Conv2D + MaxPooling + UpSampling
- Trained noisy→clean, visualized Original→Noisy→Reconstructed

**Key Results:**
- MSE Noisy vs Original   : 0.0466
- MSE Reconstructed vs Original : 0.1140
- Model successfully reconstructed digit structure from noise

---

## 📔 Week 7 — RAG Document Question Answering

**Task:** Build a Retrieval-Augmented Generation pipeline on a PDF

**Topics Covered:**
- 🔗 RAG — Retrieval Augmented Generation
- 📄 PDF loading with PyPDFLoader
- ✂️ Text chunking (500 chars, 50 overlap)
- 🔢 Vector embeddings — Google embedding-001
- 🗄️ FAISS vector database
- 🔍 Similarity search retrieval
- 🤖 Gemini 2.5 Flash LLM
- 💬 Prompt Engineering

**Pipeline:**
PDF → Chunks → Embeddings → FAISS → Retriever → Gemini → Answer
**Key Results:**
- 8 chunks indexed from resume PDF
- 5 questions answered accurately with citations
- Custom prompt prevented hallucination
- Pipeline diagram visualized all 7 RAG stages

---

## 📃 Week 8 — Agentic AI Pipeline

**Task:** Build a Single Agent System with conditional tool routing

**Topics Covered:**
- 🤖 AI Agents — concept and architecture
- 🔧 Tool definition and registration
- 🔄 Conditional routing based on intent
- 📊 Structured JSON output format
- 🎯 ReAct — Reasoning + Acting paradigm
- 🔁 Interactive agent loop

**Agent Built:**
| Component | Details |
|-----------|---------|
| Tool 1 | Calculator — math expressions |
| Tool 2 | Keyword Extractor — text analysis |
| Routing | Intent-based conditional logic |
| Output | Structured JSON with type + result |
| Mode | Interactive real-time Q&A |

**Key Results:**
- Calculator correctly solved all math queries
- Keyword extractor identified top 5 keywords
- General knowledge routing for AI/DS topics
- Interactive mode tested with multiple queries

---

## 🔍 Final Project — PatchContext

**RAG Pipeline over FastAPI Repository History**

> *"Why was this designed this way?"* — Ask anything about FastAPI
> grounded in real commits, PRs, and issue threads.

**Source:** [FastAPI GitHub](https://github.com/fastapi/fastapi)
**Folder:** [PatchContext/](./PatchContext/)

**Features:**
- 🐙 Fetches live data from GitHub API — issues, PRs, commits
- 🔢 HuggingFace embeddings (all-MiniLM-L6-v2) — free, local
- 🗄️ FAISS vector database for fast retrieval
- 🔄 MMR retrieval — Maximum Marginal Relevance for diverse results
- 🔗 Clickable citations — Issue #, PR #, Commit SHA in every answer
- 🛡️ NLI-based hallucination guard
- 📊 RAGAs 10-question benchmark evaluation
- 🎨 Premium dark glassmorphic Streamlit UI

**Tech Stack:**
LangChain + FAISS + Groq LLaMA 3.1 + HuggingFace + Streamlit
**Evaluation Results:**
| Metric | Score |
|--------|-------|
| Questions answered | 5/10 |
| Answers with sources | 5/10 |
| Grounded (no hallucination) | 10/10 |
| Overall score | 50% |

---

## 🛠️ Tools and Libraries

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Scikit-learn](https://img.shields.io/badge/sklearn-1.3-red)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Keras](https://img.shields.io/badge/Keras-3.0-red)
![LangChain](https://img.shields.io/badge/LangChain-0.2-purple)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-blue)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)

---

## 🚀 Progress — All Complete

| Week | Topic | Status |
|------|-------|--------|
| Week 1 | ML Foundations | ✅ Completed |
| Week 2 | Classical ML Pipeline | ✅ Completed |
| Week 3 | Customer Intelligence System | ✅ Completed |
| Week 4 | Deep Learning — CIFAR-10 | ✅ Completed |
| Week 5 | RNN, LSTM, GRU | ✅ Completed |
| Week 6 | Autoencoder Image Denoising | ✅ Completed |
| Week 7 | RAG Document QA System | ✅ Completed |
| Week 8 | Agentic AI Pipeline | ✅ Completed |
| Final Project | PatchContext RAG Pipeline | ✅ Completed |

---

---

## 📌 How to Run

1. Clone this repository
2. Open any `.ipynb` in Google Colab
3. Upload required dataset (links in each notebook)
4. For Week 7 — add Gemini API key from aistudio.google.com
5. For PatchContext — see [PatchContext/README.md](./PatchContext/README.md)
6. Run all cells sequentially

---

## 👨‍💻 About

**Ashwani Kumar**
- 🎓 B.Tech CSE — MMDU (2027 Batch)
- 💼 Data Scientist Intern — Celebal Technology
- 🔗 GitHub: [ashwanichauhan7082](https://github.com/ashwanichauhan7082)
- 📧 ashwanichauhan.7082@gmail.com
