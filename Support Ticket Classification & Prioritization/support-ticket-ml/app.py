"""
app.py — Support Ticket Classification & Priority Demo
Run: python app.py
"""

import pickle
import re

# ── Load saved model artifacts ─────────────────────────────────────────────
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('models/svm_classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
with open('models/label_map.pkl', 'rb') as f:
    maps = pickle.load(f)

label_map_inv = maps['label_map_inv']

# ── Preprocessing (mirrors notebook Step 2) ────────────────────────────────
STOPWORDS = {
    'i','me','my','we','our','you','your','he','him','she','her','it','its',
    'they','them','what','which','who','this','that','these','those','am','is',
    'are','was','were','be','been','have','has','had','do','does','did','a','an',
    'the','and','but','if','or','as','at','by','for','with','to','from','in',
    'out','on','of','up','can','will','just','not','so','all','also','would',
    'could','get','please','hi','hello','thank','thanks','need','want','help',
    'support','customer','product','purchased',
}

LEMMA_RULES = {
    'issues': 'issue', 'errors': 'error', 'problems': 'problem',
    'payments': 'payment', 'charges': 'charge', 'refunds': 'refund',
    'crashes': 'crash', 'crashing': 'crash', 'crashed': 'crash',
    'billing': 'bill', 'billed': 'bill', 'failed': 'fail', 'failing': 'fail',
    'cancellation': 'cancel', 'cancelled': 'cancel', 'cancelling': 'cancel',
    'inquiries': 'inquiry',
}

def simple_lemmatize(word):
    if word in LEMMA_RULES:
        return LEMMA_RULES[word]
    if len(word) > 5:
        if word.endswith('ing') and len(word) > 6: return word[:-3]
        if word.endswith('tion'): return word[:-4]
        if word.endswith('ed') and len(word) > 4: return word[:-2]
        if word.endswith('es') and len(word) > 4: return word[:-2]
    return word

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [simple_lemmatize(t) for t in tokens]
    return ' '.join(tokens).strip()


# ── Priority Rules (mirrors notebook Step 5) ───────────────────────────────
HIGH_KW   = {
    'refund': 3, 'payment failed': 5, 'charge': 3, 'overcharged': 5,
    'not working': 4, 'crash': 4, 'error': 2, 'broken': 3, 'outage': 5,
    'hacked': 6, 'locked out': 4, 'urgent': 4, 'asap': 4, 'cannot': 2,
    'unable': 2, 'data loss': 6, 'security breach': 6,
}
MEDIUM_KW = {
    'password': 2, 'login': 2, 'access': 2, 'cancel': 2, 'subscription': 2,
    'delay': 2, 'complaint': 2, 'repeated': 3, 'still': 1,
}
LOW_KW    = {
    'inquiry': 1, 'question': 1, 'how to': 1, 'feature request': 1,
    'suggestion': 1, 'feedback': 1, 'wondering': 1,
}
CAT_BOOST = {
    'Technical issue': 3, 'Billing inquiry': 3,
    'Refund request': 4, 'Cancellation request': 2, 'Product inquiry': 0,
}

def assign_priority(text, category):
    tl = text.lower()
    score = sum(pts for kw, pts in HIGH_KW.items()   if kw in tl)
    score += sum(pts for kw, pts in MEDIUM_KW.items() if kw in tl)
    score -= sum(pts for kw, pts in LOW_KW.items()    if kw in tl)
    score += CAT_BOOST.get(category, 0)
    if score >= 6: return 'High'
    elif score >= 3: return 'Medium'
    else: return 'Low'


# ── Main predict function ───────────────────────────────────────────────────
def predict_ticket(text: str) -> dict:
    """
    Predict category and priority for a support ticket.

    Args:
        text: Raw ticket text (subject + description)

    Returns:
        dict with 'category', 'priority', 'decision_scores'
    """
    clean    = preprocess_text(text)
    features = tfidf.transform([clean])
    cat_idx  = classifier.predict(features)[0]
    category = label_map_inv[cat_idx]
    scores   = classifier.decision_function(features)[0]
    score_dict = {label_map_inv[i]: round(float(s), 3) for i, s in enumerate(scores)}
    priority = assign_priority(text, category)
    return {'category': category, 'priority': priority, 'decision_scores': score_dict}


# ── Interactive CLI Demo ────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════╗
║       🎫 Support Ticket Classifier — Demo CLI       ║
║       Type 'quit' to exit | 'demo' for examples     ║
╚══════════════════════════════════════════════════════╝
"""

DEMO_TICKETS = [
    "I was charged twice for the same subscription. I need an immediate refund.",
    "My app keeps crashing every time I open it. Nothing works after the update.",
    "I would like to cancel my subscription please.",
    "Can you tell me more about your premium plan features?",
    "I cannot login to my account — I have been locked out.",
    "I received a defective product. It is broken and not working at all.",
]


def print_result(text, result):
    pri_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    print(f"\n  📌 Input:    {text[:70]}{'...' if len(text)>70 else ''}")
    print(f"  🏷️  Category: {result['category']}")
    print(f"  {pri_color.get(result['priority'], '⚪')} Priority: {result['priority']}")
    print(f"  📊 Scores:   { {k: v for k,v in sorted(result['decision_scores'].items(), key=lambda x:-x[1])[:3]} }")
    print()


if __name__ == '__main__':
    print(BANNER)

    while True:
        try:
            user_input = input("Enter ticket text (or 'demo'/'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue
        elif user_input.lower() == 'quit':
            print("👋 Goodbye!")
            break
        elif user_input.lower() == 'demo':
            print("\n🧪 Running demo tickets...\n" + "─"*55)
            for ticket in DEMO_TICKETS:
                result = predict_ticket(ticket)
                print_result(ticket, result)
        else:
            result = predict_ticket(user_input)
            print_result(user_input, result)
