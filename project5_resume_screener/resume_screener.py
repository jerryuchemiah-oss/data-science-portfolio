"""
╔══════════════════════════════════════════════════════════════╗
║   AI Resume Screener — NLP-Powered Job Fit Analyzer          ║
║   Author: Uche Jeremiah Nzubechukwu                          ║
║   Stack:  Python · Streamlit · Scikit-learn · NLTK           ║
╚══════════════════════════════════════════════════════════════╝

Run:
    pip install streamlit scikit-learn pandas numpy matplotlib
    streamlit run resume_screener.py
"""

import re
import math
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

STOPWORDS = {
    'a','an','the','and','or','but','in','on','at','to','for','of','with',
    'by','from','up','about','into','through','during','is','are','was',
    'were','be','been','being','have','has','had','do','does','did','will',
    'would','could','should','may','might','shall','can','need','must',
    'i','me','my','we','our','you','your','he','his','she','her','they','their',
    'this','that','these','those','it','its','not','no','nor','so','yet',
    'both','either','neither','whether','as','than','such','very','just',
    'also','more','most','other','some','same','few','each','every',
}

TECH_CATEGORIES = {
    "Programming Languages": ['python', 'sql', 'r', 'java', 'scala', 'c++', 'javascript', 'go', 'rust'],
    "ML / AI": ['machine learning', 'deep learning', 'neural network', 'nlp', 'computer vision',
                'reinforcement learning', 'lstm', 'transformer', 'bert', 'gpt', 'llm', 'rag',
                'fine-tuning', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'xgboost'],
    "Data Engineering": ['etl', 'pipeline', 'spark', 'kafka', 'airflow', 'dbt', 'hadoop',
                         'data warehouse', 'data lake', 'snowflake', 'bigquery', 'redshift'],
    "Cloud & DevOps": ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'terraform',
                       'git', 'mlflow', 'sagemaker', 'databricks'],
    "Analytics & BI": ['pandas', 'numpy', 'matplotlib', 'seaborn', 'tableau', 'power bi',
                       'looker', 'a/b testing', 'statistics', 'hypothesis testing'],
    "Databases": ['postgresql', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'dynamodb'],
}

SAMPLE_JD = """Senior Data Scientist — Remote (Turing)

We are looking for a Senior Data Scientist with strong expertise in machine learning, 
deep learning, and NLP to join our growing AI team. You will work on complex data 
problems at scale and collaborate with engineers across multiple time zones.

Requirements:
- 3+ years experience in Python and SQL
- Strong knowledge of machine learning algorithms and statistical modeling
- Experience with NLP, text classification, and sentiment analysis
- Familiarity with TensorFlow, PyTorch, or Keras
- Experience with cloud platforms: AWS, Azure, or GCP
- ETL pipeline development and data engineering skills
- Strong communication skills and experience in remote/async environments
- Experience with Docker and CI/CD pipelines is a plus
- Familiarity with LLMs, RAG, or fine-tuning is highly desirable

Responsibilities:
- Build and deploy end-to-end machine learning models
- Conduct data analysis and generate actionable insights
- Collaborate with cross-functional teams in a remote-first environment
- Write clean, production-quality Python code
- Present findings to both technical and non-technical stakeholders
"""

SAMPLE_RESUME = """Uche Jeremiah Nzubechukwu — Data Scientist

Professional Summary:
Dynamic Data Scientist with 3+ years of experience developing Python-based machine 
learning models, conducting data analysis, and generating actionable insights. 
Holds MSc in Data Science (University of East London) and BSc in Computer Science.

Technical Skills:
Python, SQL, R, TensorFlow, Keras, PyTorch, Scikit-learn, NLP, BERT, Transformer, 
Deep Learning, LSTM, ETL, Pandas, NumPy, Matplotlib, AWS, Azure, Git, Docker

Experience:
- Freelance Data Scientist (2023–Present): Built ML pipelines, NLP sentiment analysis 
  models, ETL pipelines using AWS and SQL. Developed predictive models for forecasting.
- Data Analyst — Delta Broadcasting Service: Python and SQL automation, dashboards, 
  data visualization for non-technical stakeholders.
- AI & Embedded Systems Intern — LAUTECH: Machine learning for embedded systems, 
  Python, C++, cross-functional team collaboration.

Projects:
- Financial Market Prediction: LSTM deep learning model for stock price forecasting
- NLP Social Impact: BERT-based sentiment analysis for policy recommendations  
- ETL Data Integration: Multi-source pipeline with AWS/Azure cloud integration

Education:
- MSc Data Science, University of East London (2024)
- BSc Computer Science, Anchor University Lagos (2021)

Certifications: IBM Python for Data Science, IBM Data Analysis with Python
"""


# ─────────────────────────────────────────────────────────────────────────────
# NLP ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s/+#]', ' ', text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return ' '.join(tokens)

def extract_keywords(text: str, top_n: int = 30) -> list:
    clean = preprocess(text)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
    vectorizer.fit([clean])
    scores = dict(zip(vectorizer.get_feature_names_out(),
                      vectorizer.transform([clean]).toarray()[0]))
    sorted_kw = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kw[:top_n]]

def compute_match_score(resume: str, jd: str) -> dict:
    """Compute overall and section-level match scores."""
    r_clean = preprocess(resume)
    j_clean = preprocess(jd)

    # Cosine similarity (TF-IDF)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform([r_clean, j_clean])
    cosine = cosine_similarity(tfidf[0], tfidf[1])[0][0]

    # Keyword overlap
    r_words = set(r_clean.split())
    j_words = set(j_clean.split())
    overlap  = r_words & j_words
    keyword_score = len(overlap) / max(len(j_words), 1)

    # Tech skill coverage
    skill_hits, skill_miss, skill_coverage = {}, {}, {}
    for category, skills in TECH_CATEGORIES.items():
        hits = [s for s in skills if s in resume.lower() or s.replace(' ', '') in resume.lower()]
        miss = [s for s in skills if s not in hits]
        coverage = len(hits) / max(len(skills), 1)
        skill_hits[category] = hits
        skill_miss[category] = miss
        skill_coverage[category] = coverage

    avg_skill_coverage = np.mean(list(skill_coverage.values()))

    # Composite score
    overall = (cosine * 0.45 + keyword_score * 0.30 + avg_skill_coverage * 0.25) * 100
    overall = min(overall * 1.8, 100)  # scale to readable range

    return {
        'overall': round(overall, 1),
        'cosine': round(cosine * 100, 1),
        'keyword': round(keyword_score * 100, 1),
        'skill_coverage': round(avg_skill_coverage * 100, 1),
        'skill_hits': skill_hits,
        'skill_miss': skill_miss,
        'skill_coverage_by_cat': skill_coverage,
        'matched_keywords': list(overlap)[:20],
        'missing_keywords': list(j_words - r_words)[:20],
    }

def get_verdict(score: float) -> tuple:
    if score >= 80:   return "🟢 Strong Match",   "success", "Highly recommended for interview"
    elif score >= 60: return "🟡 Good Match",      "warning", "Consider for interview with minor gaps"
    elif score >= 40: return "🟠 Partial Match",   "warning", "Missing some key skills — upskilling recommended"
    else:             return "🔴 Weak Match",       "error",   "Significant skill gaps identified"

def get_recommendations(result: dict) -> list:
    recs = []
    for cat, miss in result['skill_miss'].items():
        if miss:
            recs.append(f"**{cat}**: Add {', '.join(miss[:3])} to your resume if you have experience")
    if result['cosine'] < 50:
        recs.append("**Language**: Mirror more of the job description's terminology in your resume")
    if result['keyword'] < 40:
        recs.append("**Keywords**: Include more exact phrases from the job description")
    return recs[:6]


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/AI-Resume%20Screener-1B3A6B?style=for-the-badge", use_column_width=True)
    st.markdown("## ⚙️ Settings")
    show_keywords = st.checkbox("Show keyword analysis", value=True)
    show_skills   = st.checkbox("Show skill breakdown", value=True)
    show_tips     = st.checkbox("Show improvement tips", value=True)
    st.markdown("---")
    st.markdown("**About**")
    st.caption("NLP-powered resume screener using TF-IDF cosine similarity, keyword overlap analysis, and technical skill coverage scoring.")
    st.caption("Built by Uche Jeremiah Nzubechukwu")

# Header
st.title("🤖 AI Resume Screener")
st.markdown("*Paste your resume and job description below to get an instant AI-powered fit score and improvement recommendations.*")
st.markdown("---")

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Resume")
    resume_text = st.text_area(
        label="Paste resume text here",
        value=SAMPLE_RESUME,
        height=380,
        placeholder="Paste your resume text here...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 💼 Job Description")
    jd_text = st.text_area(
        label="Paste job description here",
        value=SAMPLE_JD,
        height=380,
        placeholder="Paste the job description here...",
        label_visibility="collapsed"
    )

# Analyze button
st.markdown("")
_, btn_col, _ = st.columns([1.5, 1, 1.5])
with btn_col:
    analyze = st.button("🔍 Analyze Fit", type="primary", use_container_width=True)

if analyze:
    if not resume_text.strip() or not jd_text.strip():
        st.error("Please provide both a resume and job description.")
    else:
        with st.spinner("Analyzing with NLP..."):
            result  = compute_match_score(resume_text, jd_text)
            verdict, v_type, v_detail = get_verdict(result['overall'])

        st.markdown("---")
        st.markdown("## 📊 Analysis Results")

        # Score cards
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🎯 Overall Score",    f"{result['overall']}%")
        m2.metric("🔗 Semantic Match",   f"{result['cosine']}%")
        m3.metric("🔑 Keyword Match",    f"{result['keyword']}%")
        m4.metric("⚙️ Skill Coverage",  f"{result['skill_coverage']}%")

        # Verdict
        st.markdown("")
        if v_type == "success":
            st.success(f"{verdict} — {v_detail}")
        elif v_type == "warning":
            st.warning(f"{verdict} — {v_detail}")
        else:
            st.error(f"{verdict} — {v_detail}")

        # Score gauge chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#F8F9FA')

        # Score breakdown bar
        categories = ['Overall', 'Semantic\nMatch', 'Keyword\nMatch', 'Skill\nCoverage']
        scores     = [result['overall'], result['cosine'], result['keyword'], result['skill_coverage']]
        bar_colors = ['#1B3A6B' if s >= 60 else '#E74C3C' if s < 40 else '#F39C12' for s in scores]
        bars = axes[0].bar(categories, scores, color=bar_colors, edgecolor='white', linewidth=1.5, width=0.55)
        for bar, score in zip(bars, scores):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                         f'{score}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
        axes[0].set_ylim(0, 110)
        axes[0].axhline(60, color='green', linestyle='--', lw=1.2, alpha=0.7, label='Good threshold')
        axes[0].set_title('Score Breakdown', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Score (%)')
        axes[0].legend(fontsize=9)
        axes[0].set_facecolor('#F8F9FA')

        # Skill radar by category
        cats = list(result['skill_coverage_by_cat'].keys())
        vals = [v * 100 for v in result['skill_coverage_by_cat'].values()]
        y_pos = range(len(cats))
        h_colors = ['#27AE60' if v >= 60 else '#E74C3C' if v < 30 else '#F39C12' for v in vals]
        axes[1].barh(y_pos, vals, color=h_colors, edgecolor='white', linewidth=1)
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(cats, fontsize=10)
        axes[1].set_xlim(0, 110)
        axes[1].set_title('Skill Category Coverage', fontweight='bold', fontsize=12)
        axes[1].set_xlabel('Coverage (%)')
        axes[1].axvline(60, color='green', linestyle='--', lw=1.2, alpha=0.7)
        axes[1].set_facecolor('#F8F9FA')
        for i, v in enumerate(vals):
            axes[1].text(v + 1, i, f'{v:.0f}%', va='center', fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)

        # Skills detail
        if show_skills:
            st.markdown("### ⚙️ Skill Analysis")
            skill_cols = st.columns(3)
            for i, (cat, hits) in enumerate(result['skill_hits'].items()):
                with skill_cols[i % 3]:
                    miss = result['skill_miss'][cat]
                    cov  = result['skill_coverage_by_cat'][cat] * 100
                    icon = "🟢" if cov >= 60 else "🟡" if cov >= 30 else "🔴"
                    with st.expander(f"{icon} {cat} ({cov:.0f}%)"):
                        if hits:
                            st.markdown("**✅ Found:**  " + "  ·  ".join(hits))
                        if miss:
                            st.markdown("**❌ Missing:**  " + "  ·  ".join(miss))

        # Keyword analysis
        if show_keywords:
            st.markdown("### 🔑 Keyword Analysis")
            kw1, kw2 = st.columns(2)
            with kw1:
                st.markdown("**✅ Matching Keywords**")
                if result['matched_keywords']:
                    st.write("  ·  ".join(result['matched_keywords']))
                else:
                    st.caption("No direct keyword matches found")
            with kw2:
                st.markdown("**❌ Missing JD Keywords**")
                if result['missing_keywords']:
                    st.write("  ·  ".join(result['missing_keywords'][:15]))

        # Tips
        if show_tips:
            recs = get_recommendations(result)
            if recs:
                st.markdown("### 💡 Improvement Recommendations")
                for rec in recs:
                    st.markdown(f"- {rec}")

        # Export
        st.markdown("---")
        export_data = {
            'overall_score': result['overall'],
            'semantic_match': result['cosine'],
            'keyword_match': result['keyword'],
            'skill_coverage': result['skill_coverage'],
            'verdict': verdict,
        }
        st.download_button(
            label="⬇️ Download Report (JSON)",
            data=str(export_data),
            file_name="resume_analysis_report.json",
            mime="application/json"
        )

else:
    st.info("👆 Paste your resume and job description above, then click **Analyze Fit**.")
    st.caption("Sample data is pre-loaded — try clicking Analyze Fit to see a demo!")
