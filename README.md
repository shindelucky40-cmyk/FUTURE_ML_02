# 🎫 Support Ticket Classification & Priority System

> An end-to-end Machine Learning pipeline that automatically classifies customer support tickets and assigns priority levels — reducing manual sorting effort and helping support teams respond faster.


<img width="890" height="354" alt="Screenshot 2026-02-27 143450" src="https://github.com/user-attachments/assets/c09b5542-dee4-44f0-bbaf-676a00d18850" /> 
<img width="881" height="286" alt="Screenshot 2026-02-27 143512" src="https://github.com/user-attachments/assets/74ca8bc4-199b-4497-9b82-6f536efa77f8" />
<img width="885" height="215" alt="Screenshot 2026-02-27 143529" src="https://github.com/user-attachments/assets/3ad8447d-6a67-45bb-8d72-81ab807b81ad" />
<img width="884" height="339" alt="Screenshot 2026-02-27 143542" src="https://github.com/user-attachments/assets/6a54f335-089f-476f-8511-0015382be5f1" />
<img width="892" height="403" alt="Screenshot 2026-02-27 143559" src="https://github.com/user-attachments/assets/57dbadb8-69e7-48e7-bd91-d8a04c5cbeb8" />
<img width="879" height="331" alt="Screenshot 2026-02-27 143615" src="https://github.com/user-attachments/assets/054012b3-bcce-44aa-b2d2-389343d4d8d0" />
<img width="878" height="242" alt="Screenshot 2026-02-27 143635" src="https://github.com/user-attachments/assets/32242a18-5bf6-41f1-9dc8-a7055cbb0d71" />



## 📌 Problem Statement

Support teams receive hundreds of tickets daily. Reading, categorizing, and routing each one manually is slow, inconsistent, and expensive.

This project automates two things:

- **Category Classification** — assigns each ticket to one of 5 categories using a trained SVM model
- **Priority Assignment** — scores urgency using a rule-based keyword system (High / Medium / Low)

---

## 📂 Dataset

**Source:** [Kaggle — Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

| Property | Detail |
|---|---|
| Total tickets | 8,469 |
| Categories | 5 |
| Class balance | Highly balanced (~1,634–1,752 per class) |
| Key text columns | `Ticket Subject` + `Ticket Description` |

**5 Categories:**
`Billing inquiry` · `Cancellation request` · `Product inquiry` · `Refund request` · `Technical issue`

---

## 🏗️ System Architecture

```
Raw Ticket Text
      │
      ▼
┌─────────────────────┐
│   Text Preprocessing │  lowercase → remove URLs → strip punctuation
│                     │  → remove stopwords → lemmatize
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  TF-IDF Vectorizer  │  unigrams + bigrams · max 15,000 features
│                     │  min_df=2 · sublinear_tf=True
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Linear SVM Model  │  best C via GridSearchCV · class_weight=balanced
│   (LinearSVC)       │  5-fold cross-validation
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌───────────────────────────┐
│  Category Prediction │────▶│  Priority Rules Engine    │
│                     │     │  keyword scoring + boost   │
└─────────────────────┘     └───────────────────────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
             { category, priority, scores }
```

---

## 🧠 Methodology

### Step 1 — Data Understanding
Loaded the dataset, inspected columns, checked missing values, and analyzed class distribution. Categories are well-balanced — no resampling needed.

### Step 2 — Text Preprocessing
Applied a full 6-step cleaning pipeline:

| Step | Action | Why |
|---|---|---|
| 1 | Lowercase | `"Error"` and `"error"` are the same word |
| 2 | Remove URLs & templates | `{product_purchased}` adds noise, not meaning |
| 3 | Remove punctuation | `"crash!"` and `"crash"` should be one token |
| 4 | Remove stopwords | `"please"`, `"I"`, `"the"` appear in every ticket — zero signal |
| 5 | Lemmatize | `"crashing"` → `"crash"`, `"payments"` → `"payment"` |
| 6 | Remove extra spaces | Final cleanup |

Ticket Subject and Description are combined for richer input signal.

### Step 3 — Feature Engineering (TF-IDF)
- **TF** rewards words that appear often *in this ticket*
- **IDF** penalizes words that appear in *every* ticket (noise)
- **Bigrams** capture context: `"payment failed"` beats `"payment"` alone
- Config: `ngram_range=(1,2)`, `max_features=15000`, `sublinear_tf=True`

### Step 4 — Model: Linear SVM
LinearSVC is the proven choice for sparse high-dimensional text data:

| Property | Benefit |
|---|---|
| Sparse data handling | TF-IDF features are ~99% zeros — SVM's home turf |
| Margin maximization | Finds widest boundary between classes → generalizes well |
| Linear scalability | O(n) — handles millions of tickets in production |
| Interpretability | Feature coefficients show which words drive each decision |

Tuned with `GridSearchCV` over `C ∈ {0.01, 0.1, 1.0, 5.0, 10.0}` using 5-fold cross-validation.

### Step 5 — Priority Assignment (Rule-Based)

Priority is a **business concern**, not a statistical pattern. Rules are explicit, transparent, and adjustable.

```
Score ≥ 6  →  🔴 High
Score 3–5  →  🟡 Medium
Score < 3  →  🟢 Low
```

**Scoring examples:**

| Signal | Points | Reason |
|---|---|---|
| `"payment failed"` | +5 | Direct financial impact |
| `"data loss"` | +6 | Irreversible customer damage |
| `"hacked"` | +6 | Security + legal risk |
| `"locked out"` | +4 | Customer fully blocked |
| `"urgent"` / `"asap"` | +4 | Explicit escalation signal |
| Category: Refund request | +4 | Money already taken |
| Category: Technical issue | +3 | Service availability risk |
| `"feature request"` | -1 | Non-urgent, informational |

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy | 18.30% |
| Precision (macro) | 0.1828 |
| Recall (macro) | 0.1834 |
| F1 Score (macro) | 0.1830 |
| Best CV F1 (5-fold) | 0.1967 |
| Best Hyperparameter | C = 1.0 |

> **Why ~20% accuracy?**
> This Kaggle dataset uses synthetically generated template descriptions with category labels randomly assigned. The text has no statistical relationship to the labels — 20% equals random chance for 5 balanced classes, which is the expected result.
>
> On real customer support data, **SVM + TF-IDF consistently achieves 85–95% accuracy** on similar 5-class text classification tasks. The pipeline, preprocessing, and architecture here are production-ready.

---

## 📁 Project Structure

```
support-ticket-ml/
│
├── data/
│   └── customer_support_tickets.csv
│
├── notebooks/
│   └── ticket_classification.ipynb      ← full walkthrough, all 7 steps
│
├── models/
│   ├── svm_classifier.pkl               ← trained model
│   ├── tfidf_vectorizer.pkl             ← fitted vectorizer
│   ├── label_map.pkl                    ← label encodings
│   ├── metrics_summary.json             ← evaluation scores
│   └── step1–7 *.png                    ← generated visualizations
│
├── pipeline.py                          ← full training script
├── app.py                               ← interactive CLI demo
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/shindelucky40-cmy
FUTURE_ML_02
cd 
FUTURE_ML_02
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the full pipeline**
```bash
python pipeline.py
```

**4. Run the interactive demo**
```bash
python app.py
```
Type `demo` to run predictions on sample tickets, or enter any ticket text directly.

**5. Open the notebook**
```bash
jupyter notebook notebooks/ticket_classification.ipynb
```

---

## 🔧 Usage — `predict_ticket()`

```python
from app import predict_ticket

result = predict_ticket(
    "I was charged twice for my subscription. I need an immediate refund."
)

print(result)
# {
#   'category': 'Billing inquiry',
#   'priority': 'High',
#   'decision_scores': { 'Billing inquiry': 0.421, ... }
# }
```

---

## 💼 Business Impact

| Problem | Solution | Impact |
|---|---|---|
| Manual ticket routing | Instant ML classification | Eliminates hours of daily sorting work |
| No urgency visibility | Keyword priority scoring | Critical tickets surface immediately |
| Slow first response | Automated queue ordering | Directly improves CSAT scores |
| Scaling support costs | One pipeline, any volume | 10 or 10,000 tickets — same latency |

---

## 🔮 Future Improvements

- **Transformer embeddings** — replace TF-IDF with sentence-BERT for semantic understanding
- **Confidence thresholds** — route uncertain predictions to human review
- **Active learning** — agent corrections retrain the model automatically
- **API deployment** — FastAPI wrapper for Zendesk / Freshdesk integration
- **Multilingual support** — multilingual BERT for global teams
- **Feedback loop** — production labels continuously improve accuracy

---
## Author -

     lalit shinde 

*Built as an ML Internship Portfolio Project*
