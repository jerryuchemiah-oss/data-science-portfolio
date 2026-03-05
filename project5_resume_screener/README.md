# 🤖 AI Resume Screener — NLP Web App

**Author:** Uche Jeremiah Nzubechukwu  
**Stack:** Python · Streamlit · Scikit-learn · TF-IDF · Matplotlib  

## Overview
Interactive NLP-powered web app that analyzes how well a resume matches a job description. Returns an overall fit score, semantic similarity, keyword overlap, skill category coverage, and actionable improvement tips.

## Features
- **Overall Fit Score** (0–100%) with verdict (Strong / Good / Partial / Weak Match)
- **Semantic Similarity** using TF-IDF cosine similarity
- **Keyword Match** analysis — shows matching and missing terms
- **Skill Coverage** across 6 categories (ML/AI, Cloud, Databases, etc.)
- **Improvement Recommendations** — specific gaps to address
- **JSON Report Export**
- Sample resume + job description pre-loaded for instant demo

## Quick Start
```bash
pip install streamlit scikit-learn pandas numpy matplotlib
streamlit run resume_screener.py
```
Then open http://localhost:8501 in your browser.

## How It Works
```
Resume + JD → TF-IDF Vectorization → Cosine Similarity
                                   → Keyword Overlap Analysis
                                   → Skill Category Matching
                                   → Composite Score (weighted)
```

**Scoring Weights:**
| Component | Weight |
|-----------|--------|
| Semantic Similarity | 45% |
| Keyword Match | 30% |
| Skill Coverage | 25% |

## Upgrade Path
- BERT/sentence-transformers for deeper semantic matching
- OCR support for PDF resume uploads
- Database to track multiple candidates
- REST API: `POST /screen` → returns JSON score
