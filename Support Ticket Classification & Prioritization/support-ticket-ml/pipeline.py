"""
Support Ticket Classification & Priority System
End-to-End ML Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import re
import string
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────────────────────────
# STEP 1: DATA UNDERSTANDING
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — DATA UNDERSTANDING")
print("=" * 60)

df = pd.read_csv('data/customer_support_tickets.csv')

print(f"\n📦 Dataset Shape: {df.shape}")
print(f"\n📋 Columns:\n{df.columns.tolist()}")
print(f"\n🔍 Data Types:\n{df.dtypes}")
print(f"\n❌ Missing Values:\n{df.isnull().sum()}")
print(f"\n📊 Ticket Type Distribution:\n{df['Ticket Type'].value_counts()}")
print(f"\n🎯 Ticket Priority Distribution:\n{df['Ticket Priority'].value_counts()}")
print(f"\n📝 Sample Ticket Description:\n{df['Ticket Description'].iloc[0][:200]}")

# Visualization 1: Category distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Step 1 — Data Understanding', fontsize=14, fontweight='bold')

colors_cat = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
df['Ticket Type'].value_counts().plot(kind='bar', ax=axes[0], color=colors_cat,
                                       edgecolor='white', width=0.6)
axes[0].set_title('Ticket Category Distribution', fontweight='bold')
axes[0].set_xlabel('Category')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=30)
for p in axes[0].patches:
    axes[0].annotate(f'{int(p.get_height())}',
                     (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=9)

colors_pri = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71']
df['Ticket Priority'].value_counts().plot(kind='pie', ax=axes[1],
                                           autopct='%1.1f%%', startangle=140,
                                           colors=colors_pri,
                                           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
axes[1].set_title('Ticket Priority Distribution', fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('models/step1_data_understanding.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Step 1 chart saved.")

# ─────────────────────────────────────────────
# STEP 2: TEXT PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — TEXT PREPROCESSING")
print("=" * 60)

# Common English stopwords (built-in, no NLTK needed)
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up',
    'down','in','out','on','off','over','under','again','further','then',
    'once','here','there','when','where','why','how','all','both','each',
    'few','more','most','other','some','such','no','nor','not','only','own',
    'same','so','than','too','very','s','t','can','will','just','don',
    'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn',
    'didn','doesn','hadn','hasn','haven','isn','ma','mightn','mustn',
    'needn','shan','shouldn','wasn','weren','won','wouldn','also','would',
    'could','get','got','please','hi','hello','dear','sir','madam','team',
    'support','customer','service','product','purchased','help','need',
    'want','like','thank','thanks','regards','sincerely','best'
}

# Simple rule-based lemmatizer (no external library needed)
LEMMA_RULES = {
    'issues': 'issue', 'errors': 'error', 'problems': 'problem',
    'payments': 'payment', 'accounts': 'account', 'charges': 'charge',
    'refunds': 'refund', 'orders': 'order', 'subscriptions': 'subscription',
    'updates': 'update', 'crashes': 'crash', 'requests': 'request',
    'users': 'user', 'devices': 'device', 'services': 'service',
    'failed': 'fail', 'failing': 'fail', 'charged': 'charge',
    'billing': 'bill', 'billed': 'bill', 'updating': 'update',
    'updated': 'update', 'crashing': 'crash', 'crashed': 'crash',
    'requesting': 'request', 'requested': 'request', 'working': 'work',
    'worked': 'work', 'cancelling': 'cancel', 'cancelled': 'cancel',
    'cancellation': 'cancel', 'inquiries': 'inquiry', 'trying': 'try',
    'tried': 'try', 'unable': 'unable', 'experiencing': 'experience',
    'experienced': 'experience', 'resolving': 'resolve', 'resolved': 'resolve',
    'receiving': 'receive', 'received': 'receive', 'accessing': 'access',
    'accessed': 'access', 'contacting': 'contact', 'contacted': 'contact',
}

def simple_lemmatize(word):
    """Rule-based lemmatizer with common suffix stripping."""
    if word in LEMMA_RULES:
        return LEMMA_RULES[word]
    # Common suffix rules
    if len(word) > 5:
        if word.endswith('ing') and len(word) > 6:
            return word[:-3]
        if word.endswith('tion'):
            return word[:-4]
        if word.endswith('ness'):
            return word[:-4]
        if word.endswith('ment'):
            return word[:-4]
        if word.endswith('ies') and len(word) > 5:
            return word[:-3] + 'y'
        if word.endswith('es') and len(word) > 4:
            return word[:-2]
        if word.endswith('ed') and len(word) > 4:
            return word[:-2]
        if word.endswith('ly') and len(word) > 4:
            return word[:-2]
    return word

def preprocess_text(text):
    """
    Full text preprocessing pipeline:
    1. Lowercase        — normalizes vocabulary, reduces feature space
    2. Remove URLs      — URLs add noise, not useful for classification
    3. Remove numbers   — standalone numbers rarely carry semantic meaning
    4. Remove punctuation — punctuation doesn't help TF-IDF features
    5. Remove stopwords — common words dilute meaningful signal
    6. Lemmatization    — reduces inflected forms to base (work/working → work)
    7. Remove extra spaces — cleanup
    """
    if pd.isna(text):
        return ""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # 3. Remove template placeholders like {product_purchased}
    text = re.sub(r'\{.*?\}', '', text)
    # 4. Remove special characters / punctuation (keep spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # 5. Tokenize
    tokens = text.split()
    # 6. Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    # 7. Lemmatize
    tokens = [simple_lemmatize(t) for t in tokens]
    # 8. Remove extra spaces
    text = ' '.join(tokens)
    return text.strip()

# Combine Subject + Description for richer text signal
df['combined_text'] = df['Ticket Subject'].fillna('') + ' ' + df['Ticket Description'].fillna('')
df['clean_text'] = df['combined_text'].apply(preprocess_text)

print("\n📝 Preprocessing Example:")
print(f"  Original: {df['Ticket Description'].iloc[0][:150]}")
print(f"  Cleaned:  {df['clean_text'].iloc[0][:150]}")

# Visualization: Text length before/after cleaning
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Step 2 — Text Preprocessing: Text Length Distribution', fontsize=13, fontweight='bold')

df['raw_len'] = df['combined_text'].str.len()
df['clean_len'] = df['clean_text'].str.len()

axes[0].hist(df['raw_len'], bins=40, color='#4C72B0', edgecolor='white', alpha=0.8)
axes[0].set_title('Raw Text Length', fontweight='bold')
axes[0].set_xlabel('Character Count')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['raw_len'].mean(), color='red', linestyle='--', label=f"Mean: {df['raw_len'].mean():.0f}")
axes[0].legend()

axes[1].hist(df['clean_len'], bins=40, color='#55A868', edgecolor='white', alpha=0.8)
axes[1].set_title('Cleaned Text Length', fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')
axes[1].axvline(df['clean_len'].mean(), color='red', linestyle='--', label=f"Mean: {df['clean_len'].mean():.0f}")
axes[1].legend()

plt.tight_layout()
plt.savefig('models/step2_preprocessing.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Step 2 chart saved.")

# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING — TF-IDF
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — FEATURE ENGINEERING (TF-IDF)")
print("=" * 60)

"""
Why TF-IDF works well for ticket data:
- Term Frequency (TF): rewards words that appear frequently in one ticket
  (e.g., "refund" appearing 3x in a ticket signals a billing/refund issue)
- Inverse Document Frequency (IDF): penalizes words common across all tickets
  (e.g., "please", "help" appear everywhere → low IDF → low weight)
- Result: high weight for rare, discriminating words → perfect for multi-class classification

Why n-grams help:
- Unigrams capture single words: "payment", "error"
- Bigrams capture context: "payment failed", "not working", "account suspended"
  These bigrams are much more informative than individual words alone.
"""

X = df['clean_text']
y = df['Ticket Type']

# Encode labels
label_map = {label: idx for idx, label in enumerate(sorted(y.unique()))}
label_map_inv = {v: k for k, v in label_map.items()}
y_encoded = y.map(label_map)

print(f"\n🏷️  Label Mapping: {label_map}")
print(f"\n📐 Feature Engineering Config:")
print("  - Unigrams + Bigrams (ngram_range=(1,2))")
print("  - Max features: 15,000")
print("  - Min document frequency: 2 (removes very rare words)")
print("  - Sublinear TF scaling: True (dampens high-frequency terms)")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\n✂️  Train size: {len(X_train)} | Test size: {len(X_test)}")

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=15000,
    min_df=2,
    sublinear_tf=True,
    analyzer='word'
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"\n📦 TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
print(f"📦 TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}")

# Visualization: Top features per class
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
fig.suptitle('Step 3 — Top 10 TF-IDF Features Per Category', fontsize=13, fontweight='bold')

feature_names = tfidf.get_feature_names_out()
classes = sorted(y.unique())

# Fit a simple model to get coef for visualization
from sklearn.svm import LinearSVC as _LSVC
_viz_clf = _LSVC(max_iter=1000, C=1.0, class_weight='balanced')
_viz_clf.fit(X_train_tfidf, y_train)

for i, cls in enumerate(sorted(label_map.keys())):
    cls_idx = label_map[cls]
    coef = _viz_clf.coef_[cls_idx]
    top_idx = np.argsort(coef)[-10:][::-1]
    top_feats = [feature_names[j] for j in top_idx]
    top_vals = coef[top_idx]

    axes[i].barh(range(len(top_feats)), top_vals[::-1], color='#4C72B0', alpha=0.8)
    axes[i].set_yticks(range(len(top_feats)))
    axes[i].set_yticklabels(top_feats[::-1], fontsize=8)
    axes[i].set_title(cls.replace(' ', '\n'), fontweight='bold', fontsize=9)
    axes[i].set_xlabel('SVM Coefficient', fontsize=8)

plt.tight_layout()
plt.savefig('models/step3_tfidf_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Step 3 chart saved.")

# ─────────────────────────────────────────────
# STEP 4: MODEL — SVM (LINEAR KERNEL)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — MODEL: SVM (LINEAR KERNEL) + HYPERPARAMETER TUNING")
print("=" * 60)

"""
Why SVM with Linear Kernel?
- Sparse high-dimensional data (TF-IDF) is exactly where linear SVM excels
- Maximizes margin between classes → robust to outliers
- Efficient: LinearSVC scales linearly with samples (vs RBF SVM: O(n²))
- Strong proven baseline for NLP text classification tasks
"""

# Check class balance
print("\n📊 Class distribution in training set:")
for cls_idx, count in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {label_map_inv[cls_idx]}: {count}")

# Compute class weights for imbalance handling
class_weights = compute_class_weight('balanced',
                                      classes=np.unique(y_train),
                                      y=y_train)
class_weight_dict = {cls: w for cls, w in zip(np.unique(y_train), class_weights)}
print(f"\n⚖️  Class Weights: {class_weight_dict}")

# GridSearchCV for hyperparameter tuning
print("\n🔍 Running GridSearchCV for hyperparameter tuning...")
param_grid = {'C': [0.01, 0.1, 1.0, 5.0, 10.0]}
svm_base = LinearSVC(max_iter=2000, class_weight='balanced', random_state=42)
grid_search = GridSearchCV(svm_base, param_grid, cv=5, scoring='f1_macro',
                            n_jobs=-1, verbose=1)
grid_search.fit(X_train_tfidf, y_train)

print(f"\n🏆 Best C: {grid_search.best_params_['C']}")
print(f"🏆 Best CV F1 (macro): {grid_search.best_score_:.4f}")

best_svm = grid_search.best_estimator_

# Cross-validation scores
cv_scores = cross_val_score(best_svm, X_train_tfidf, y_train, cv=5,
                             scoring='f1_macro', n_jobs=-1)
print(f"\n📊 5-Fold CV F1 Scores: {cv_scores.round(4)}")
print(f"   Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# GridSearchCV visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Step 4 — SVM Hyperparameter Tuning', fontsize=13, fontweight='bold')

C_vals = param_grid['C']
mean_scores = grid_search.cv_results_['mean_test_score']
std_scores = grid_search.cv_results_['std_test_score']

axes[0].plot(C_vals, mean_scores, 'o-', color='#4C72B0', linewidth=2, markersize=8)
axes[0].fill_between(C_vals,
                      mean_scores - std_scores,
                      mean_scores + std_scores, alpha=0.2, color='#4C72B0')
axes[0].set_xscale('log')
axes[0].set_xlabel('Regularization Parameter C', fontsize=11)
axes[0].set_ylabel('CV F1 Score (Macro)', fontsize=11)
axes[0].set_title('GridSearchCV: C vs F1 Score', fontweight='bold')
axes[0].axvline(grid_search.best_params_['C'], color='red', linestyle='--',
                label=f"Best C={grid_search.best_params_['C']}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].bar(range(1, 6), cv_scores, color=['#4C72B0']*4 + ['#DD8452'],
             edgecolor='white', alpha=0.85)
axes[1].axhline(cv_scores.mean(), color='red', linestyle='--',
                label=f'Mean = {cv_scores.mean():.4f}')
axes[1].set_xlabel('Fold')
axes[1].set_ylabel('F1 Score (Macro)')
axes[1].set_title('5-Fold Cross Validation Scores', fontweight='bold')
axes[1].set_xticks(range(1, 6))
axes[1].legend()
axes[1].set_ylim(0.5, 1.0)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/step4_hyperparameter_tuning.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Step 4 chart saved.")

# ─────────────────────────────────────────────
# STEP 5: PRIORITY ASSIGNMENT (RULE-BASED)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — PRIORITY ASSIGNMENT LOGIC (RULE-BASED)")
print("=" * 60)

"""
Business Logic for Priority:
─────────────────────────────
HIGH PRIORITY (Score ≥ 6):
  Reason: These tickets directly impact revenue or service availability.
  A payment failure or crashed product means the customer is actively blocked
  and will churn if not resolved immediately.

MEDIUM PRIORITY (Score 3–5):
  Reason: Account issues need timely resolution but the customer can still
  use other parts of the product. Repeated issues signal frustration.

LOW PRIORITY (Score 0–2):
  Reason: Informational queries and feature requests don't block the customer.
  These can be batched and handled during low-traffic periods.
"""

# Keyword scoring system
HIGH_PRIORITY_KEYWORDS = {
    # Financial / payment urgency
    'refund': 3, 'payment failed': 5, 'charge': 3, 'overcharged': 5,
    'unauthorized charge': 6, 'double charged': 6, 'billing error': 4,
    # Technical urgency
    'not working': 4, 'crash': 4, 'crashed': 4, 'error': 2, 'bug': 2,
    'broken': 3, 'down': 3, 'outage': 5, 'data loss': 6,
    # Account security
    'hacked': 6, 'unauthorized access': 6, 'locked out': 4,
    'account suspended': 4, 'security breach': 6,
    # Emotional urgency
    'urgent': 4, 'immediately': 3, 'asap': 4, 'critical': 4,
    'emergency': 5, 'cannot': 2, 'unable': 2,
}

MEDIUM_PRIORITY_KEYWORDS = {
    'password': 2, 'login': 2, 'cannot login': 3, 'access': 2,
    'subscription': 2, 'cancel': 2, 'cancellation': 2,
    'account issue': 3, 'not receiving': 2, 'delay': 2, 'slow': 1,
    'complaint': 2, 'again': 2, 'still': 1, 'repeated': 3,
    'third time': 4, 'second time': 3, 'multiple times': 3,
}

LOW_PRIORITY_KEYWORDS = {
    'inquiry': 1, 'question': 1, 'information': 1, 'how to': 1,
    'feature request': 1, 'suggestion': 1, 'feedback': 1,
    'general': 1, 'curious': 1, 'wondering': 1,
}

# Category-based priority boost
CATEGORY_PRIORITY_BOOST = {
    'Technical issue': 3,
    'Billing inquiry': 3,
    'Refund request': 4,
    'Cancellation request': 2,
    'Product inquiry': 0,
}

def assign_priority(text, category):
    """
    Keyword-based scoring system.
    Returns: 'High', 'Medium', or 'Low' with score breakdown.
    """
    text_lower = str(text).lower()
    score = 0

    # Score from keywords
    for keyword, pts in HIGH_PRIORITY_KEYWORDS.items():
        if keyword in text_lower:
            score += pts

    for keyword, pts in MEDIUM_PRIORITY_KEYWORDS.items():
        if keyword in text_lower:
            score += pts

    # Subtract for low-priority signals
    for keyword, pts in LOW_PRIORITY_KEYWORDS.items():
        if keyword in text_lower:
            score -= pts

    # Category boost
    score += CATEGORY_PRIORITY_BOOST.get(category, 0)

    # Thresholds
    if score >= 6:
        return 'High'
    elif score >= 3:
        return 'Medium'
    else:
        return 'Low'

# Apply to dataset (use original combined text)
df['predicted_priority'] = df.apply(
    lambda row: assign_priority(row['combined_text'], row['Ticket Type']), axis=1
)

print("\n📊 Predicted Priority Distribution:")
print(df['predicted_priority'].value_counts())

# Visualization: Priority by category
pivot = df.groupby(['Ticket Type', 'predicted_priority']).size().unstack(fill_value=0)
pivot = pivot.reindex(columns=['High', 'Medium', 'Low'], fill_value=0)

fig, ax = plt.subplots(figsize=(11, 5))
pivot.plot(kind='bar', ax=ax,
           color=['#e74c3c', '#f39c12', '#2ecc71'],
           edgecolor='white', width=0.65)
ax.set_title('Step 5 — Predicted Priority by Ticket Category', fontsize=13, fontweight='bold')
ax.set_xlabel('Ticket Category')
ax.set_ylabel('Number of Tickets')
ax.tick_params(axis='x', rotation=25)
ax.legend(title='Priority', loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('models/step5_priority_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Step 5 chart saved.")

# ─────────────────────────────────────────────
# STEP 6: MODEL EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — MODEL EVALUATION")
print("=" * 60)

y_pred = best_svm.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print(f"\n📈 Overall Metrics:")
print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 Score:  {f1:.4f}")

print(f"\n📊 Classification Report:")
target_names = [label_map_inv[i] for i in sorted(label_map_inv.keys())]
report = classification_report(y_test, y_pred, target_names=target_names)
print(report)

# Per-class F1 analysis
report_dict = {}
lines = report.strip().split('\n')
for line in lines[2:-3]:
    parts = line.split()
    if len(parts) >= 5:
        cls = ' '.join(parts[:-4])
        prec_c = float(parts[-4])
        rec_c = float(parts[-3])
        f1_c = float(parts[-2])
        report_dict[cls] = {'precision': prec_c, 'recall': rec_c, 'f1': f1_c}

# Confusion matrix + metrics visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Step 6 — Model Evaluation', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100

sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names,
            ax=axes[0], linewidths=0.5, linecolor='white',
            annot_kws={'size': 9})
axes[0].set_title('Confusion Matrix (%)', fontweight='bold')
axes[0].set_xlabel('Predicted Label', fontsize=10)
axes[0].set_ylabel('True Label', fontsize=10)
axes[0].tick_params(axis='x', rotation=30)
axes[0].tick_params(axis='y', rotation=0)

# Per-class metrics bar chart
if report_dict:
    cls_names = list(report_dict.keys())
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(cls_names))
    width = 0.25
    colors = ['#4C72B0', '#DD8452', '#55A868']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [report_dict[c][metric] for c in cls_names]
        bars = axes[1].bar(x + i * width, vals, width, label=metric.capitalize(),
                           color=color, alpha=0.85, edgecolor='white')

    axes[1].set_title('Per-Class Performance Metrics', fontweight='bold')
    axes[1].set_xticks(x + width)
    short_names = [n.replace(' ', '\n') for n in cls_names]
    axes[1].set_xticklabels(short_names, fontsize=8)
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()
    axes[1].axhline(0.8, color='gray', linestyle=':', alpha=0.7)
    axes[1].grid(True, alpha=0.3, axis='y')

# Overall metrics summary box
overall_txt = (f"Overall Metrics\n{'─'*22}\n"
               f"Accuracy:   {acc:.4f}\n"
               f"Precision:  {prec:.4f}\n"
               f"Recall:     {rec:.4f}\n"
               f"F1 (macro): {f1:.4f}")
fig.text(0.01, 0.01, overall_txt, fontsize=9, family='monospace',
         verticalalignment='bottom',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('models/step6_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Step 6 chart saved.")

# ─────────────────────────────────────────────
# STEP 7: FINAL PREDICTION PIPELINE
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — FINAL PIPELINE: predict_ticket(text)")
print("=" * 60)

# Save model + vectorizer
with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
with open('models/svm_classifier.pkl', 'wb') as f:
    pickle.dump(best_svm, f)
with open('models/label_map.pkl', 'wb') as f:
    pickle.dump({'label_map': label_map, 'label_map_inv': label_map_inv}, f)

print("💾 Model, vectorizer, and label maps saved to models/")

def predict_ticket(text):
    """
    End-to-end ticket prediction function.
    
    Flow: Raw Text → Preprocess → TF-IDF → SVM → Category → Priority Rules → Output
    
    Args:
        text (str): Raw ticket description / subject text
    
    Returns:
        dict: {
            'category': predicted ticket category,
            'priority': assigned priority level,
            'confidence_scores': per-class decision scores
        }
    """
    # Step 1: Preprocess
    clean = preprocess_text(text)
    # Step 2: TF-IDF transform
    features = tfidf.transform([clean])
    # Step 3: SVM predict
    cat_idx = best_svm.predict(features)[0]
    category = label_map_inv[cat_idx]
    # Step 4: Decision scores (confidence proxy)
    scores = best_svm.decision_function(features)[0]
    score_dict = {label_map_inv[i]: round(float(s), 3) for i, s in enumerate(scores)}
    # Step 5: Priority
    priority = assign_priority(text, category)

    return {
        'category': category,
        'priority': priority,
        'decision_scores': score_dict
    }

# Test with sample tickets
test_cases = [
    "I was charged twice for the same subscription! This is unacceptable, I need an immediate refund!",
    "My app keeps crashing every time I open it. Nothing seems to work after the last update.",
    "I would like to cancel my subscription please.",
    "Can you tell me more about the features included in the premium plan?",
    "I can't login to my account. I've been locked out and tried resetting my password.",
]

print("\n🧪 Test Predictions:")
print("─" * 70)
for i, tc in enumerate(test_cases, 1):
    result = predict_ticket(tc)
    priority_emoji = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    print(f"\nTicket #{i}:")
    print(f"  Text:     {tc[:75]}...")
    print(f"  Category: {result['category']}")
    print(f"  Priority: {priority_emoji.get(result['priority'], '⚪')} {result['priority']}")

# Final pipeline flow visualization
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14)
ax.set_ylim(0, 4)
ax.axis('off')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')
ax.set_title('Step 7 — Final Pipeline: predict_ticket(text)', fontsize=13, fontweight='bold', pad=15)

steps = [
    ("📥 Input\nText", '#3498db'),
    ("🔧 Pre-\nprocess", '#9b59b6'),
    ("📊 TF-IDF\nVectorize", '#e67e22'),
    ("🤖 SVM\nClassify", '#e74c3c'),
    ("🏷️ Category\nOutput", '#27ae60'),
    ("⚡ Priority\nRules", '#f39c12'),
    ("📤 Final\nOutput", '#2c3e50'),
]

x_positions = np.linspace(1, 13, len(steps))

for i, ((label, color), x) in enumerate(zip(steps, x_positions)):
    # Draw box
    rect = plt.Rectangle((x - 0.75, 1.2), 1.5, 1.6, facecolor=color,
                          alpha=0.85, edgecolor='white', linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(x, 2.0, label, ha='center', va='center',
            fontsize=9, fontweight='bold', color='white', zorder=4)

    # Arrow
    if i < len(steps) - 1:
        ax.annotate('', xy=(x_positions[i+1] - 0.75, 2.0),
                    xytext=(x + 0.75, 2.0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2))

plt.tight_layout()
plt.savefig('models/step7_pipeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Step 7 pipeline chart saved.")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
📦 Dataset:     {df.shape[0]} tickets, {df['Ticket Type'].nunique()} categories
🔧 Preprocessing: Lowercase, remove URLs, punctuation, stopwords, lemmatize
📊 Features:    TF-IDF (unigrams + bigrams, max 15,000 features)
🤖 Model:       SVM Linear (Best C={grid_search.best_params_['C']})
📈 Accuracy:    {acc*100:.2f}%
📈 F1 (macro):  {f1:.4f}
⚡ Priority:    Rule-based keyword scoring (High/Medium/Low)
💾 Saved:       models/svm_classifier.pkl + tfidf_vectorizer.pkl
""")

# Save summary JSON for README generation
summary = {
    'dataset_size': df.shape[0],
    'num_categories': df['Ticket Type'].nunique(),
    'categories': df['Ticket Type'].unique().tolist(),
    'best_C': grid_search.best_params_['C'],
    'cv_f1_mean': round(cv_scores.mean(), 4),
    'test_accuracy': round(acc, 4),
    'test_f1_macro': round(f1, 4),
    'test_precision': round(prec, 4),
    'test_recall': round(rec, 4),
    'classification_report': report
}
with open('models/metrics_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("📝 Metrics summary saved to models/metrics_summary.json")
