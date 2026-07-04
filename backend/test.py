import json
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from datetime import datetime
from dateutil import parser
import os
from google import genai

# -------------------- FILES --------------------
NEWS_FILE = "news.json"
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Get the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is missing! Make sure it's set in your .env file.")

# Initialize client
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------- HELPERS --------------------
def read_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_parse_date(published):
    try:
        dt = parser.parse(published)
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)
        return dt
    except:
        return datetime.utcnow()

def recency(published):
    dt = safe_parse_date(published)
    hrs = (datetime.utcnow() - dt).total_seconds() / 3600
    half_life = 48
    decay = np.log(2)/half_life
    return float(np.exp(-decay * max(0, hrs)))

def normalize(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        return arr
    if arr.max()-arr.min() == 0:
        return np.zeros_like(arr)
    return (arr-arr.min())/(arr.max()-arr.min())

def topic_relevance(X):
    if X.shape[0] <= 1:
        return np.zeros(X.shape[0])
    sim = cosine_similarity(X)
    scores = []
    for i in range(sim.shape[0]):
        sims = np.sort(sim[i])[::-1][1:6]
        scores.append(np.mean(sims) if len(sims)>0 else 0)
    return normalize(scores)

# -------------------- LOAD MODELS --------------------
model_linear = joblib.load("../models/model.pkl")
model_rf = joblib.load("../models/forestmodelreg.pkl")
model_xgb = joblib.load("../models/xgb_classifier.pkl")
model_light = joblib.load("../models/lightmodel.pkl")

vectorizer = joblib.load("../models/vectorizer.pkl")
vectorizerr = joblib.load("../models/vectorizerr.pkl")
vectorizer_xgb = joblib.load("../models/vectorizer_xgb.pkl")
vectorizer_light = joblib.load("../models/lightvectorizer.pkl")

# -------------------- KEYWORD SCORES --------------------
KEYWORD_WEIGHTS = {
    "crash": 6, "collapse": 6, "plunge": 5, "meltdown": 6,
    "bankruptcy": 6, "default": 6, "fraud": 6, "scandal": 5,
    "merger": 5, "acquisition": 5, "buyout": 5,
    "earnings": 2, "earnings beat": 5, "earnings miss": 4,
    "revenue": 2, "profit": 2, "loss": 2,
    "inflation": 3, "deflation": 3, "interest rate": 4,
    "rate hike": 4, "rate cut": 4, "recession": 4,
    "war": 5, "conflict": 4, "sanctions": 4,
    "ipo": 4, "listing": 3, "delisting": 4,
    "layoffs": 4, "job cuts": 4,
    "regulation": 3, "ban": 4, "investigation": 4,
    "shortage": 4, "supply chain": 4,
    "oil": 3, "gas": 3, "energy crisis": 5,
    "ai": 3, "cyberattack": 5, "data breach": 5,
    "liquidity": 4, "debt": 3,
    "panic": 5, "uncertainty": 3,
    "exports": 2, "imports": 2, "tariff": 4,
    "pandemic": 5, "earthquake": 5, "flood": 4,
    "upgrade": 3, "downgrade": 3,
    "strong demand": 4, "weak demand": 4
}

def compute_keyword_score(text, trend):
    text = text.lower()
    s = 0
    for k,v in KEYWORD_WEIGHTS.items():
        if k in text:
            s += v*trend[k]
    s = min(s,10)/10
    return np.sqrt(s)

def compute_trend(texts):
    freq = {}
    for t in texts:
        t = t.lower()
        for k in KEYWORD_WEIGHTS:
            if k in t:
                freq[k] = freq.get(k,0)+1
    total = sum(freq.values()) + 1
    return {k: 1 + np.log1p(freq.get(k,0))/np.log1p(total) for k in KEYWORD_WEIGHTS}

# -------------------- METRICS --------------------
def precision_at_k(pred_idx,k=5):
    return sum([1 for i in pred_idx[:k] if i<5])/k

def recall_at_k(pred_idx,k=5):
    return sum([1 for i in pred_idx[:k] if i<5])/5

def mrr(pred_idx):
    for rank,i in enumerate(pred_idx,1):
        if i<5:
            return 1.0/rank
    return 0.0

def f1_metric(pred_idx):
    y_true = [1 if i<5 else 0 for i in range(len(pred_idx))]
    y_pred = [1 if i in pred_idx[:5] else 0 for i in range(len(pred_idx))]
    return f1_score(y_true,y_pred)

# -------------------- LLM ANALYSIS --------------------
TOP_K_LLM = 20  # production uses top 20 news for LLM call

def gemini_analyze_news(news):
    try:
        prompt = "Return ONLY a JSON array with impact_score (0-1) and confidence (0-1) for each title.\n"
        for i,n in enumerate(news):
            prompt += f"{i}. {n.get('title','')}\n"
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        text = response.text.strip()
        start = text.find("[")
        end = text.rfind("]")+1
        if start==-1 or end==-1:
            return [{"impact_score":0.5,"confidence":0.5} for _ in news]
        data = json.loads(text[start:end])
        result = []
        for d in data:
            result.append({
                "impact_score": float(d.get("impact_score",0.5)),
                "confidence": float(d.get("confidence",0.5))
            })
        while len(result)<len(news):
            result.append({"impact_score":0.5,"confidence":0.5})
        return result
    except:
        return [{"impact_score":0.5,"confidence":0.5} for _ in news]

# -------------------- MAIN --------------------
news = read_json(NEWS_FILE)
texts = [n.get("title","")+" "+n.get("summary","") for n in news]

X1 = vectorizer.transform(texts)
X2 = vectorizerr.transform(texts)
X3 = vectorizer_xgb.transform(texts)
X4 = vectorizer_light.transform(texts)

# ML predictions
try:
    xgb_scores = model_xgb.predict_proba(X3)[:,1]
except:
    xgb_scores = model_xgb.predict(X3)

ml_scores = normalize(
    0.25*np.array(model_linear.predict(X1)).flatten() +
    0.25*np.array(model_rf.predict(X2)).flatten() +
    0.25*np.array(xgb_scores).flatten() +
    0.25*np.abs(np.array(model_light.predict(X4)).flatten())
)

# Other scores
topic_scores = topic_relevance(X1)
trend = compute_trend(texts)
ktv_scores = np.array([compute_keyword_score(t,trend) for t in texts])
recency_scores = np.array([recency(n.get("published","")) for n in news])

# -------------------- LLM TOP-K --------------------
ranked_ml_idx = np.argsort(ml_scores)[::-1][:TOP_K_LLM]
top_news_for_llm = [news[i] for i in ranked_ml_idx]
llm_results_top = gemini_analyze_news(top_news_for_llm)

# Map back to all news
llm_scores = np.array([0.5]*len(news))
for idx, res in zip(ranked_ml_idx, llm_results_top):
    llm_scores[idx] = res["impact_score"]*res["confidence"]

# -------------------- CONFIGS --------------------
configs = {
    "Chronological Baseline": np.array([safe_parse_date(n.get("published","")).timestamp() for n in news]),
    "ML Ensemble Only": ml_scores,
    "Hybrid ML + KTV": 0.8*ml_scores + 0.2*ktv_scores,
    "Proposed Hybrid (LLM + Ensemble + Recency + Topic)": (
        0.5*ml_scores + 0.15*ktv_scores + 0.15*llm_scores + 0.1*topic_scores + 0.1*recency_scores
    )
}

# -------------------- PRINT TABLE --------------------
print("\\begin{table}[htbp]")
print("\\caption{System Performance Metrics (N=%d)}" % len(news))
print("\\centering")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("Model Configuration & P@5 & R@5 & MRR & F1 \\\\")
print("\\midrule")

for name,scores in configs.items():
    ranked_idx = np.argsort(scores)[::-1]
    print(f"{name} & {precision_at_k(ranked_idx):.3f} & {recall_at_k(ranked_idx):.3f} & {mrr(ranked_idx):.3f} & {f1_metric(ranked_idx):.3f} \\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")