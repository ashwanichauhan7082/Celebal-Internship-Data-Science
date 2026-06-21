# 🎓 Celebal Technology — Data Science Internship

**Intern:** Ashwani Kumar
**College:** MMDU (Maharishi Markandeshwar Deemed University)
**Role:** Data Scientist Intern
**Program:** Data Science
**Duration:** May 2026 – Present

---

## 📁 Repository Structure

| File | Week | Topic |
|------|------|-------|
| `week1_Ashwani_kumar_MMDU.ipynb` | Week 1 | ML Foundations |
| `week2_Ashwani_kumar_MMDU.ipynb` | Week 2 | Classical ML Pipeline |
| `week3_Ashwani_kumar_MMDU.ipynb` | Week 3 | Customer Intelligence System |
| `week4_Ashwani_kumar_MMDU.ipynb` | Week 4 | Deep Learning — CIFAR-10 |
| `week5_Ashwani_kumar_MMDU.ipynb` | Week 5 | RNN, LSTM, GRU — Text Generation |

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
- 🔤 Encoding (LabelEncoder) and Feature Engineering
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
- Ensemble models (Random Forest, XGBoost) achieved F1 Score of 1.0
- Logistic Regression: Accuracy 0.9706, F1 0.9655
- K-Means K=3 Silhouette Score: 0.2833
- Countries segmented into Developed, Developing and Underdeveloped clusters

---

## 📙 Week 4 — Intro to Deep Learning (CIFAR-10)

**Dataset:** CIFAR-10 — 60,000 color images, 10 classes

**Topics Covered:**
- 🧠 Perceptron, MLP, Forward Pass and Backpropagation
- ⚡ Activation Functions — Sigmoid, Tanh, ReLU Family
- 🔍 Convolution Layer, Pooling, Stride and Padding
- 🏗️ CNN Architectures and Transfer Learning concepts
- 📈 Data Augmentation — RandomFlip, RandomRotation, RandomZoom
- 🛑 EarlyStopping — prevent overfitting

**Models Built:**
| Model | Highlights |
|-------|-----------|
| ANN Baseline | Flattens image, ignores spatial structure |
| CNN Baseline | Uses convolution — significantly outperforms ANN |
| CNN + Larger Filters | Captures more complex features |
| CNN + 20 Epochs | Longer training, improved accuracy |
| CNN + EarlyStopping | Automatically finds optimal stopping point |
| CNN + Augmentation | Better generalization on unseen images |

**Key Results:**
- CNN significantly outperformed ANN on CIFAR-10 image classification
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

## 🛠️ Tools and Libraries

![Python](https://img.shields.io/badge/Python-3.10-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0-green)
![Scikit-learn](https://img.shields.io/badge/sklearn-1.3-red)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-yellow)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![Keras](https://img.shields.io/badge/Keras-3.0-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7-blue)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12-teal)

---

## 📌 How to Run

1. Clone this repository
2. Open any `.ipynb` file in Google Colab
3. Upload the required dataset (links provided in each notebook)
4. Run all cells sequentially

---

## 🚀 Progress

| Week | Topic | Status |
|------|-------|--------|
| Week 1 | ML Foundations | ✅ Completed |
| Week 2 | Classical ML Pipeline | ✅ Completed |
| Week 3 | Customer Intelligence System | ✅ Completed |
| Week 4 | Deep Learning — CIFAR-10 | ✅ Completed |
| Week 5 | RNN, LSTM, GRU | ✅ Completed |
| Week 6 | Autoencoders and GAN | 🔄 In Progress |
| Week 7 | RAG and LLMs | ⏳ Upcoming |
| Week 8 | AI Agents | ⏳ Upcoming |
