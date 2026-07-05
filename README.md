# 🎓 Celebal Technology — Data Science Internship

**Intern:** Ashwani Kumar
**College:** MMDU (Maharishi Markandeshwar Deemed University)
**Role:** Data Scientist Intern
**Program:** Data Science
**Duration:** May 2026 – July 2026

---

## 📁 Repository Structure

| File | Week | Topic |
|------|------|-------|
| `week1_Ashwani_kumar_MMDU.ipynb` | Week 1 | ML Foundations |
| `week2_Ashwani_kumar_MMDU.ipynb` | Week 2 | Classical ML Pipeline |
| `week3_Ashwani_kumar_MMDU.ipynb` | Week 3 | Customer Intelligence System |
| `week4_Ashwani_kumar_MMDU.ipynb` | Week 4 | Deep Learning — CIFAR-10 |
| `week5_Ashwani_kumar_MMDU.ipynb` | Week 5 | RNN, LSTM, GRU — Text Generation |
| `week6_Ashwani_kumar_MMDU.ipynb` | Week 6 | Autoencoder — Image Denoising |
| `week7_Ashwani_kumar_MMDU.ipynb` | Week 7 | RAG — Document Question Answering |

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
- 📅 Time Series — Rolling Statistics, ADF Test, Forecasting, Chronological Split

**Key Results:**
- Built end-to-end ML pipeline predicting Tesla delivery volumes
- Applied data leakage prevention via correct train/test scaling
- Compared Linear, Ridge and Lasso models on R² and MAE

---

## 📕 Week 3 — Customer Intelligence System

**Dataset:** Unsupervised Learning on Country Data (167 countries, 10 features)
**Source:** [Kaggle](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)

**Topics Covered:**
- 🔵 Classification — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost
- 🤝 Ensemble Learning — Random Forest, Gradient Boosting, XGBoost, Stacking
- 📊 Evaluation — Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- 🌟 Feature Importance — GDP per capita identified as key feature
- 🔵 Clustering — K-Means (Elbow Method), DBSCAN, Hierarchical
- 🗺️ PCA Visualization of clusters
- 💡 Cluster Profiling and Business Insights

**Key Results:**
- Ensemble models achieved F1 Score of 1.0
- Logistic Regression: Accuracy 0.9706, F1 0.9655
- K-Means K=3 Silhouette Score: 0.2833
- Countries segmented into Developed, Developing and Underdeveloped clusters

---

## 📙 Week 4 — Intro to Deep Learning (CIFAR-10)

**Dataset:** CIFAR-10 — 60,000 color images, 10 classes
**Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck

**Topics Covered:**
- 🧠 Perceptron, MLP, Forward Pass and Backpropagation
- ⚡ Activation Functions — Sigmoid, Tanh, ReLU Family
- 📉 Loss Functions — Sparse Categorical Cross Entropy
- 🔍 Convolution Layer, Pooling, Stride and Padding
- 🏗️ CNN Architectures and Transfer Learning concepts
- 📈 Data Augmentation — RandomFlip, RandomRotation, RandomZoom
- 🛑 EarlyStopping — prevent overfitting

**Models Built:**
| Model | Highlights |
|-------|-----------|
| ANN Baseline | Flattens image, ignores spatial structure |
| CNN Baseline | Significantly outperforms ANN |
| CNN + Larger Filters | Captures more complex features |
| CNN + 20 Epochs | Longer training, improved accuracy |
| CNN + EarlyStopping | Automatically finds optimal stopping point |
| CNN + Augmentation | Better generalization on unseen images |

**Key Results:**
- CNN significantly outperformed ANN on CIFAR-10
- Data augmentation improved model generalization
- BatchNormalization stabilized CNN training

---

## 📒 Week 5 — RNN, LSTM and GRU (Text Generation)

**Task:** Build and compare sequence models for next-word prediction and text generation

**Topics Covered:**
- 🔁 Vanilla RNN — baseline sequential model
- 🔒 LSTM — input, forget and output gates for long-term memory
- ⚡ GRU — reset and update gates, faster than LSTM
- 🔤 Tokenization and n-gram sequence creation
- ✍️ Custom text generation function
- 📉 Training loss comparison across all 3 models

**Tasks Completed:**
- Replaced corpus with custom paragraph
- Increased embedding dimension (32 → 100)
- Increased training epochs (100 → 200)
- Changed hidden units (64 → 128)
- Generated extended text sequences (10 words)

**Key Results:**
- LSTM and GRU achieved lower training loss than vanilla RNN
- GRU trained faster than LSTM with comparable performance
- Demonstrated why gated architectures handle long-term dependencies better

---

## 📓 Week 6 — Autoencoder for Image Denoising

**Dataset:** MNIST Handwritten Digits — 70,000 images, 28x28 pixels
**Source:** Built into TensorFlow/Keras (tf.keras.datasets.mnist)

**Topics Covered:**
- 🗜️ Autoencoder architecture — Encoder + Latent Space + Decoder
- 🔄 Variational Autoencoder (VAE) concepts
- 🎨 GAN — Generator, Discriminator, Adversarial Training
- ⚙️ Optimizers — SGD, Momentum, AdaGrad, RMSprop, Adam
- 🤖 Intro to Generative AI
- 🔗 RAG with LangChain concepts
- 💬 Prompt Engineering techniques

**What Was Built:**
- Added Gaussian noise (factor=0.3) to clean MNIST images
- Built Convolutional Autoencoder (Conv2D + MaxPooling + UpSampling)
- Trained on noisy input → clean output
- Visualized 3-row comparison: Original → Noisy → Reconstructed

**Key Results:**
- MSE (Noisy vs Original)         : 0.0466
- MSE (Reconstructed vs Original) : 0.1140
- Model successfully reconstructed recognizable digit structure from noisy images

---

## 📔 Week 7 — RAG Document Question Answering System

**Task:** Build a Retrieval-Augmented Generation (RAG) system to answer questions from custom documents

**Topics Covered:**
- 🔗 RAG — Retrieval Augmented Generation pipeline
- 📄 PDF document loading and text extraction
- ✂️ Text chunking — RecursiveCharacterTextSplitter
- 🔢 Vector embeddings — Google embedding-001
- 🗄️ Vector database — FAISS (Facebook AI Similarity Search)
- 🔍 Semantic similarity search and retrieval
- 🤖 LLM integration — Google Gemini 2.5 Flash
- 💬 Prompt Engineering for accurate Q&A
- 🏗️ End-to-end RAG pipeline with LangChain

**What Was Built:**
- Loaded PDF resume document using PyPDFLoader
- Split into 8 chunks with 500-char size and 50-char overlap
- Created vector embeddings using Google embedding-001
- Stored in FAISS vector database for similarity search
- Built RetrievalQA chain with Gemini 2.5 Flash
- Successfully answered 5 questions about the document
- Visualized complete RAG pipeline as diagram

**Key Results:**
- System correctly identified name, contact, skills, projects and education from document
- Top 3 most relevant chunks retrieved per question
- Custom prompt template prevented hallucination
- Pipeline diagram clearly shows all 7 RAG stages

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
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-teal)

---

## 📌 How to Run

1. Clone this repository
2. Open any `.ipynb` file in Google Colab
3. Upload the required dataset (links provided in each notebook)
4. For Week 7 — add your own Gemini API key from aistudio.google.com
5. Run all cells sequentially

---

## 🚀 Progress

| Week | Topic | Status |
|------|-------|--------|
| Week 1 | ML Foundations | ✅ Completed |
| Week 2 | Classical ML Pipeline | ✅ Completed |
| Week 3 | Customer Intelligence System | ✅ Completed |
| Week 4 | Deep Learning — CIFAR-10 | ✅ Completed |
| Week 5 | RNN, LSTM, GRU | ✅ Completed |
| Week 6 | Autoencoder — Image Denoising | ✅ Completed |
| Week 7 | RAG — Document QA System | ✅ Completed |
| Week 8 | AI Agents | 🔄 In Progress |

---

## 🚀 Personal Project

### AI Job Assistant
An AI-powered personal job agent built and deployed as a personal project.

**Features:**
- Finds 10–20 best matching jobs based on user skills and resume
- Ranks jobs by skill match score using semantic similarity
- Auto-customizes resume for each job application using LLMs
- Provides direct application links

**Live Demo:** https://ai-job-assistant-app.vercel.app
