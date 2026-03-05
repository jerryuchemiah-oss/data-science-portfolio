# 🗂️ Data Science Portfolio — Uche Jeremiah Nzubechukwu

> Data Scientist | ML Engineer | NLP Specialist  
> 📧 jerryuchemiah@gmail.com | 📍 Ibadan, Nigeria (Remote-Ready)

---

## Projects Overview

| # | Project | Type | Key Skills |
|---|---------|------|-----------|
| 1 | [Stock Price Predictor](#1-stock-price-predictor) | Jupyter Notebook | LSTM, Time-Series, Feature Engineering |
| 2 | [NLP Sentiment Analysis](#2-nlp-sentiment-analysis) | Jupyter Notebook | TF-IDF, NLP, Model Comparison |
| 3 | [Customer Churn Predictor](#3-customer-churn-predictor) | Python Script | Random Forest, GBM, AUC-ROC |
| 4 | [ETL Data Pipeline](#4-etl-data-pipeline) | Python Script | ETL, SQLite, Data Quality |
| 5 | [AI Resume Screener](#5-ai-resume-screener) | Streamlit Web App | NLP, Cosine Similarity, UI |

---

## 1. Stock Price Predictor
**`project1_stock_predictor/`**  
LSTM-style deep learning model forecasting stock prices using 8 engineered features (RSI, MACD, Bollinger Bands, volatility). Includes 5-day forward forecast with BULLISH/BEARISH signal.

```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook stock_price_predictor.ipynb
```

---

## 2. NLP Sentiment Analysis
**`project2_sentiment_analysis/`**  
End-to-end NLP pipeline classifying text as Positive / Negative / Neutral. Compares Naive Bayes, Logistic Regression, and SVM with TF-IDF bigrams and 5-fold cross-validation.

```bash
pip install scikit-learn pandas numpy matplotlib
jupyter notebook sentiment_analysis.ipynb
```

---

## 3. Customer Churn Predictor
**`project3_churn_predictor/`**  
ML classification pipeline predicting telecom customer churn. Features ROC curves, feature importance, and a `predict_churn()` function ready for API deployment.

```bash
pip install numpy pandas scikit-learn matplotlib
python churn_predictor.py
```

---

## 4. ETL Data Pipeline
**`project4_etl_pipeline/`**  
Production-style ETL pipeline: Extract from 3 simulated sources → Transform (clean, validate, enrich) → Load into SQLite warehouse → Generate analytics dashboard. Full audit logging included.

```bash
pip install numpy pandas matplotlib
python etl_pipeline.py
```

---

## 5. AI Resume Screener ⭐
**`project5_resume_screener/`**  
Interactive Streamlit web app that scores resume-to-job-description fit using NLP (TF-IDF cosine similarity + keyword overlap + skill coverage). Returns actionable improvement tips.

```bash
pip install streamlit scikit-learn pandas numpy matplotlib
streamlit run resume_screener.py
```

---

## Tech Stack Summary

```
Languages   : Python, SQL, R
ML/AI       : Scikit-learn, TensorFlow (upgrade path), NLP, LSTM, BERT (upgrade path)
Data        : Pandas, NumPy, Matplotlib, Seaborn
Cloud       : AWS, Azure (referenced in ETL project)
Deployment  : Streamlit, FastAPI (upgrade path), Docker (upgrade path)
Database    : SQLite, PostgreSQL (upgrade path)
Tools       : Git, Jupyter, VS Code
```

---

*All projects use synthetic data and are designed to be extended with real data sources.*
