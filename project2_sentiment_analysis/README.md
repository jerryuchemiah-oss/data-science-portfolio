# 🧠 NLP Sentiment Analysis — IMDB Movie Reviews

**Author:** Uche Jeremiah Nzubechukwu  
**Stack:** Python · Pandas · Scikit-learn · TF-IDF · Matplotlib  
**Data:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — Kaggle  

## Overview
End-to-end NLP pipeline classifying 50,000 real IMDB movie reviews as Positive or Negative. Covers text preprocessing, TF-IDF feature extraction, 3-model comparison, error analysis, and a live inference function.

## Dataset
Download `IMDB Dataset.csv` from Kaggle and place it in this folder.  
**50,000 reviews · Perfectly balanced (25k pos / 25k neg) · Avg 233 words per review**

## Features
- HTML tag removal, URL stripping, custom stopwords, rule-based stemming
- TF-IDF with bigrams (up to 50,000 features, sublinear TF scaling)
- 3 models: Naive Bayes, Logistic Regression, Linear SVM (calibrated)
- 5-fold Stratified CV for robust evaluation
- ROC curves, confusion matrix, top predictive words chart
- **Error analysis** — examines false positives and false negatives with confidence scores
- `predict_sentiment(text)` → returns label, confidence %, and strength

## Quick Start
```bash
pip install numpy pandas scikit-learn matplotlib
# Download IMDB Dataset.csv from Kaggle first
jupyter notebook imdb_sentiment_analysis.ipynb
```

## Results
| Model | Test Accuracy | AUC-ROC | CV Accuracy |
|-------|--------------|---------|-------------|
| Naive Bayes | ~85% | ~0.93 | ~85% |
| Logistic Regression | ~89% | ~0.96 | ~89% |
| Linear SVM | ~90% | ~0.96 | ~90% |

## Key Insights
- **Logistic Regression & SVM** significantly outperform Naive Bayes on long reviews
- Most errors occur on **sarcastic or mixed-sentiment** reviews — a known limitation of bag-of-words models
- Top negative predictors: `worst`, `waste`, `awful`, `boring`, `terrible`
- Top positive predictors: `brilliant`, `wonderf`, `excel`, `perfect`, `masterpiece`

## Upgrade Path
- Fine-tune **BERT/RoBERTa** for ~93%+ accuracy and sarcasm handling
- Extend to **5-star rating prediction** (multi-class)
- Deploy as **FastAPI REST endpoint**
- Build real-time **Twitter/Reddit sentiment dashboard**
