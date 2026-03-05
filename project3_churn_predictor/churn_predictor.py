"""
╔══════════════════════════════════════════════════════════════╗
║   Customer Churn Predictor — IBM Telco Dataset               ║
║   Author: Uche Jeremiah Nzubechukwu                          ║
║   Data:   IBM Telco Customer Churn (Kaggle)                  ║
║   Stack:  Python · Pandas · Scikit-learn · Matplotlib        ║
╚══════════════════════════════════════════════════════════════╝

Dataset:
    Download from Kaggle and place in the same folder:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
    Filename: WA_Fn-UseC_-Telco-Customer-Churn.csv

Run:
    python churn_predictor.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

DATASET_PATH = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

print("=" * 60)
print("  CUSTOMER CHURN PREDICTOR — IBM TELCO DATASET")
print("=" * 60)

try:
    df = pd.read_csv(DATASET_PATH)
    print(f"\n✅ Dataset loaded: {DATASET_PATH}")
except FileNotFoundError:
    print(f"\n❌ Dataset not found: '{DATASET_PATH}'")
    print("   Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
    print("   Place the CSV in the same folder as this script.")
    exit(1)

print(f"   Shape     : {df.shape[0]:,} customers x {df.shape[1]} features")
print(f"   Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")

# ── EDA ──────────────────────────────────────────────────────────────────────
print("\n📊 Running EDA...")
fig = plt.figure(figsize=(16, 12))
fig.suptitle('IBM Telco Customer Churn — Exploratory Data Analysis', fontsize=15, fontweight='bold')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
churn_counts = df['Churn'].value_counts()
ax1.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
        colors=['#27AE60', '#E74C3C'], startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax1.set_title('Churn Distribution', fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
ax2.bar(contract_churn.index, contract_churn.values * 100,
        color=['#E74C3C', '#F39C12', '#27AE60'], edgecolor='white', linewidth=1.5)
ax2.set_title('Churn Rate by Contract Type', fontweight='bold')
ax2.set_ylabel('Churn Rate (%)')
ax2.tick_params(axis='x', rotation=10)
for i, v in enumerate(contract_churn.values):
    ax2.text(i, v * 100 + 0.5, f'{v:.1%}', ha='center', fontweight='bold', fontsize=10)

ax3 = fig.add_subplot(gs[0, 2])
internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean())
ax3.bar(internet_churn.index, internet_churn.values * 100,
        color=['#3498DB', '#E74C3C', '#95A5A6'], edgecolor='white', linewidth=1.5)
ax3.set_title('Churn Rate by Internet Service', fontweight='bold')
ax3.set_ylabel('Churn Rate (%)')
for i, v in enumerate(internet_churn.values):
    ax3.text(i, v * 100 + 0.5, f'{v:.1%}', ha='center', fontweight='bold', fontsize=10)

ax4 = fig.add_subplot(gs[1, :2])
churned     = df[df['Churn'] == 'Yes']['tenure']
not_churned = df[df['Churn'] == 'No']['tenure']
ax4.hist(not_churned, bins=30, alpha=0.6, color='#27AE60', label='Retained', density=True)
ax4.hist(churned,     bins=30, alpha=0.6, color='#E74C3C', label='Churned',  density=True)
ax4.axvline(churned.mean(),     color='#E74C3C', linestyle='--', lw=2, label=f'Churn avg: {churned.mean():.0f}mo')
ax4.axvline(not_churned.mean(), color='#27AE60', linestyle='--', lw=2, label=f'Retain avg: {not_churned.mean():.0f}mo')
ax4.set_title('Tenure Distribution — Churned vs Retained', fontweight='bold')
ax4.set_xlabel('Tenure (months)')
ax4.set_ylabel('Density')
ax4.legend(fontsize=9)

ax5 = fig.add_subplot(gs[1, 2])
df_plot = df.copy()
df_plot['MonthlyCharges'] = pd.to_numeric(df_plot['MonthlyCharges'], errors='coerce')
bp = ax5.boxplot([df_plot[df_plot['Churn']=='No']['MonthlyCharges'].dropna(),
                  df_plot[df_plot['Churn']=='Yes']['MonthlyCharges'].dropna()],
                 labels=['Retained','Churned'], patch_artist=True, notch=True)
bp['boxes'][0].set_facecolor('#27AE60'); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('#E74C3C'); bp['boxes'][1].set_alpha(0.7)
ax5.set_title('Monthly Charges by Churn', fontweight='bold')
ax5.set_ylabel('Monthly Charges ($)')

ax6 = fig.add_subplot(gs[2, :])
payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x=='Yes').mean()).sort_values()
colors_pay = ['#27AE60' if v < 0.2 else '#F39C12' if v < 0.35 else '#E74C3C' for v in payment_churn.values]
ax6.barh(payment_churn.index, payment_churn.values * 100, color=colors_pay, edgecolor='white')
ax6.set_title('Churn Rate by Payment Method', fontweight='bold')
ax6.set_xlabel('Churn Rate (%)')
for i, v in enumerate(payment_churn.values):
    ax6.text(v * 100 + 0.2, i, f'{v:.1%}', va='center', fontweight='bold', fontsize=10)

plt.savefig('churn_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ EDA chart saved: churn_eda.png")

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
print("\n🔧 Preprocessing data...")
df_clean = df.copy()
df_clean.drop('customerID', axis=1, inplace=True)
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median(), inplace=True)
df_clean['Churn'] = (df_clean['Churn'] == 'Yes').astype(int)

for col in ['gender','Partner','Dependents','PhoneService','PaperlessBilling','MultipleLines']:
    df_clean[col] = (df_clean[col] == 'Yes').astype(int)

le = LabelEncoder()
for col in ['InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
            'TechSupport','StreamingTV','StreamingMovies','Contract','PaymentMethod']:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

# Feature engineering
df_clean['charges_per_month']   = df_clean['TotalCharges'] / (df_clean['tenure'] + 1)
df_clean['services_count']      = (df_clean[['OnlineSecurity','OnlineBackup','DeviceProtection',
                                              'TechSupport','StreamingTV','StreamingMovies']] > 0).sum(axis=1)
df_clean['is_new_customer']     = (df_clean['tenure'] <= 12).astype(int)
df_clean['high_value_customer'] = (df_clean['MonthlyCharges'] > df_clean['MonthlyCharges'].median()).astype(int)

FEATURES = [c for c in df_clean.columns if c != 'Churn']
X = df_clean[FEATURES]
y = df_clean['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"   ✅ Features: {len(FEATURES)} | Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── MODELS ───────────────────────────────────────────────────────────────────
print("\n🔧 Training models...")
models = {
    'Logistic Regression': Pipeline([('imputer', SimpleImputer(strategy='median')),
                                      ('scaler', StandardScaler()),
                                      ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))]),
    'Random Forest':       Pipeline([('imputer', SimpleImputer(strategy='median')),
                                      ('clf', RandomForestClassifier(n_estimators=200, max_depth=10,
                                                                      class_weight='balanced', random_state=42))]),
    'Gradient Boosting':   Pipeline([('imputer', SimpleImputer(strategy='median')),
                                      ('clf', GradientBoostingClassifier(n_estimators=200, learning_rate=0.07,
                                                                          max_depth=4, random_state=42))]),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)
    cv   = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='roc_auc').mean()
    results[name] = {'acc': acc, 'auc': auc, 'cv_auc': cv,
                     'y_pred': y_pred, 'y_proba': y_proba, 'model': model}
    print(f"   {name:<25} → Acc: {acc:.4f}  AUC: {auc:.4f}  CV-AUC: {cv:.4f}")

best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n🏆 Best: {best_name} (AUC={best['auc']:.4f})")

# ── RESULTS CHART ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle(f'IBM Telco Churn — Model Results ({best_name})', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'])
ConfusionMatrixDisplay(cm, display_labels=['Retained','Churned']).plot(ax=axes[0,0], colorbar=False, cmap='Reds')
axes[0,0].set_title(f'Confusion Matrix — {best_name}', fontweight='bold')

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    axes[0,1].plot(fpr, tpr, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
axes[0,1].plot([0,1],[0,1],'k--',lw=1)
axes[0,1].set_title('ROC Curves', fontweight='bold')
axes[0,1].set_xlabel('False Positive Rate'); axes[0,1].set_ylabel('True Positive Rate')
axes[0,1].legend(fontsize=9)

rf_clf = results['Random Forest']['model'].named_steps['clf']
feat_imp = pd.DataFrame({'feature': FEATURES, 'importance': rf_clf.feature_importances_})
feat_imp = feat_imp.sort_values('importance', ascending=True).tail(15)
axes[1,0].barh(feat_imp['feature'], feat_imp['importance'], color='#1B3A6B', alpha=0.85)
axes[1,0].set_title('Top 15 Feature Importances', fontweight='bold')
axes[1,0].set_xlabel('Importance')

names_ = list(results.keys())
x = np.arange(len(names_))
axes[1,1].bar(x-0.25, [results[n]['acc'] for n in names_],    0.25, label='Accuracy',   color='#1B3A6B', alpha=0.85)
axes[1,1].bar(x,      [results[n]['auc'] for n in names_],    0.25, label='AUC-ROC',    color='#27AE60', alpha=0.85)
axes[1,1].bar(x+0.25, [results[n]['cv_auc'] for n in names_], 0.25, label='CV AUC (5-fold)', color='#E74C3C', alpha=0.85)
axes[1,1].set_xticks(x); axes[1,1].set_xticklabels(names_, rotation=10, fontsize=9)
axes[1,1].set_ylim(0.6, 1.0); axes[1,1].set_title('Model Comparison', fontweight='bold')
axes[1,1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('churn_model_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 Results saved: churn_model_results.png")

print(f"\n📄 Classification Report — {best_name}")
print(classification_report(y_test, best['y_pred'], target_names=['Retained','Churned']))

# ── BUSINESS INSIGHTS ─────────────────────────────────────────────────────────
print("\n💡 Key Business Insights from IBM Telco Data")
print("=" * 55)
print(f"  Month-to-month contracts churn at : {(df[df['Contract']=='Month-to-month']['Churn']=='Yes').mean():.1%}")
print(f"  Fiber optic customers churn at    : {(df[df['InternetService']=='Fiber optic']['Churn']=='Yes').mean():.1%}")
print(f"  Electronic check users churn at   : {(df[df['PaymentMethod']=='Electronic check']['Churn']=='Yes').mean():.1%}")
print(f"  Customers in first 12 months      : {(df[df['tenure']<=12]['Churn']=='Yes').mean():.1%}")
print(f"\n  → Priority retention target: Month-to-month + Fiber optic + New customers")
