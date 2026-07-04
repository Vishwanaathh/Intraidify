import json
import os
import time
import hashlib
import yagmail
from threading import Thread
from datetime import datetime
from dateutil import parser
import feedparser
import joblib
import numpy as np

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from google import genai
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

USERS_FILE = "users.json"
NEWS_FILE = "news.json"

EMAIL_USER = "csfinancialservices4@gmail.com"
EMAIL_PASS = "ckvv hidk ikxq ugmf"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------- KEYWORDS --------------------

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

# -------------------- FILE --------------------

def read_json(file):
    try:
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def write_json(file, data):
    with open(file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def init_files():
    for f in [USERS_FILE, NEWS_FILE]:
        if not os.path.exists(f):
            write_json(f, [])

init_files()

# -------------------- DATE --------------------

def safe_parse_date(published):
    try:
        if not published:
            return datetime.utcnow()
        dt = parser.parse(published)
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    except:
        return datetime.utcnow()

def recency(published):
    try:
        dt = safe_parse_date(published)
        hrs = (datetime.utcnow() - dt).total_seconds() / 3600
        hrs = max(0, hrs)
        half_life = 48
        decay = np.log(2) / half_life
        return float(np.exp(-decay * hrs))
    except:
        return 0

# -------------------- GEMINI --------------------

def gemini_analyze_news(news):
    def fallback(n):
        return [{
            "impact_score": 0.5,
            "market_direction": "neutral",
            "event_type": "other",
            "confidence": 0.5
        } for _ in range(n)]

    try:
        prompt = """You are a financial analyst ranking news based on market impact Return ONLY a valid JSON array.
Each item must have:
impact_score (0-1), market_direction, event_type, confidence (0-1).
No explanation. Only JSON.
"""
        for i, item in enumerate(news):
            prompt += f"{i}. {item.get('title','')}\n"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        text = response.text.strip()

        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]

        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == -1:
            return fallback(len(news))

        json_text = text[start:end]
        data = json.loads(json_text)

        cleaned = []
        for item in data:
            cleaned.append({
                "impact_score": float(item.get("impact_score", 0.5)),
                "market_direction": item.get("market_direction", "neutral"),
                "event_type": item.get("event_type", "other"),
                "confidence": float(item.get("confidence", 0.5))
            })

        while len(cleaned) < len(news):
            cleaned.append({
                "impact_score": 0.5,
                "market_direction": "neutral",
                "event_type": "other",
                "confidence": 0.5
            })

        return cleaned[:len(news)]

    except:
        return fallback(len(news))

# -------------------- FASTAPI --------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- AUTH --------------------

@app.post("/signup")
async def signup(request: Request):
    data = await request.json()
    email = data.get("email")
    users = read_json(USERS_FILE)
    if email in users:
        return {"message": "User exists"}
    users.append(email)
    write_json(USERS_FILE, users)

    try:
        yagmail.SMTP(EMAIL_USER, EMAIL_PASS).send(
            to=email,
            subject="Welcome",
            contents=f"Welcome {email}"
        )
    except:
        pass

    return {"message": "ok"}

@app.get("/news")
def get_news():
    return read_json(NEWS_FILE)

# -------------------- EMAIL WATCHER --------------------

last_hash = None
last_sent_titles = set()

def get_file_hash():
    try:
        with open(NEWS_FILE, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

def should_send(data):
    global last_sent_titles
    titles = {n.get("title","") for n in data[:5]}
    if titles == last_sent_titles:
        return False
    last_sent_titles = titles
    return True

def send_email(data):
    try:
        users = read_json(USERS_FILE)
        if not users:
            return

        content = ""
        for n in data[:10]:
            content += f"{n.get('title','')} | Score: {round(n.get('score',0),3)}\n"

        yagmail.SMTP(EMAIL_USER, EMAIL_PASS).send(
            to=users,
            subject="Market Alert",
            contents=content
        )
    except:
        pass

def watch_news():
    global last_hash
    last_hash = get_file_hash()
    while True:
        try:
            time.sleep(5)
            new_hash = get_file_hash()
            if new_hash != last_hash:
                last_hash = new_hash
                data = read_json(NEWS_FILE)
                if data and should_send(data):
                    send_email(data)
        except:
            pass

# -------------------- MODELS --------------------

model_linear = joblib.load("../models/model.pkl")
model_rf = joblib.load("../models/forestmodelreg.pkl")
model_xgb = joblib.load("../models/xgb_classifier.pkl")
model_light = joblib.load("../models/lightmodel.pkl")

vectorizer = joblib.load("../models/vectorizer.pkl")
vectorizerr = joblib.load("../models/vectorizerr.pkl")
vectorizer_xgb = joblib.load("../models/vectorizer_xgb.pkl")
vectorizer_light = joblib.load("../models/lightvectorizer.pkl")

# -------------------- HELPERS --------------------

def normalize(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 0:
        return arr
    if arr.max() - arr.min() == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def softmax(a, b, T=0.7):
    m = max(a, b)
    ea = np.exp((a - m)/T)
    eb = np.exp((b - m)/T)
    s = ea + eb
    return ea/s, eb/s

def compute_trend(texts):
    freq = {}
    for t in texts:
        t = t.lower()
        for k in KEYWORD_WEIGHTS:
            if k in t:
                freq[k] = freq.get(k,0)+1
    total = sum(freq.values()) + 1
    return {
        k: 1 + np.log1p(freq.get(k,0)) / np.log1p(total)
        for k in KEYWORD_WEIGHTS
    }

def keyword_score(text, trend):
    text = text.lower()
    s = 0
    for k,v in KEYWORD_WEIGHTS.items():
        if k in text:
            s += v * trend[k]
    s = min(s,10)/10
    return np.sqrt(s)

def topic_relevance(X):
    try:
        if X.shape[0] <= 1:
            return np.zeros(X.shape[0])
        sim = cosine_similarity(X)
        scores = []
        for i in range(sim.shape[0]):
            sims = np.sort(sim[i])[::-1][1:6]
            scores.append(np.mean(sims) if len(sims) > 0 else 0)
        return normalize(scores)
    except:
        return np.zeros(X.shape[0])

def actionability_score(text):
    try:
        text = text.lower()
        
        IMPACT = [
    "acquisition", "merger", "ipo", "bankruptcy", "investigation", "contract", 
    "earnings", "revenue", "profit", "guidance", "forecast", "partnership",
    "divestiture", "buyback", "dividend", "restructuring", "lawsuit", "settlement", 
    "default", "liquidation", "joint venture", "licensing", "clinical trial", 
    "fda approval", "patent", "takeover", "tender offer", "spinoff"
]
        MACRO = [
    "inflation", "economy", "sentiment", "outlook", "analysts", "macro",
    "fed", "interest rates", "gdp", "unemployment", "treasury", "yield", 
    "central bank", "monetary policy", "fiscal", "recession", "stagnation", 
    "consumer price index", "cpi", "geopolitical", "trade war", "tariffs"
]
        EARLY = [
    "plans", "will", "expected", "upcoming", "set to", "announce",
    "considering", "exploring", "rumored", "potential", "aims", "seeks", 
    "prepares", "intends", "scheduled", "pending", "on track", "initial", 
    "proposes", "seeks approval", "preliminary", "eyes"
]      
        LATE = [
    "rose", "surged", "gained", "fell", "dropped", "plunged",
    "soared", "slumped", "tumbled", "rallied", "retreated", "spiked", 
    "closed", "ended", "climbed", "slipped", "dived", "crashed", 
    "jumped", "tanked", "stabilized", "rebounded"
]

        #VECTORIZE
        X = vectorizer.transform([text])
        X_dense = X.toarray()[0]
        vocab = vectorizer.vocabulary_

        def tfidf_score(word_list):
            s = 0
            for w in word_list:
                if w in vocab:
                    idx = vocab[w]
                    tfidf_val = X_dense[idx]
                    if tfidf_val > 0:
                        s += np.log1p(tfidf_val)
            return s

        #SEMANTIC ANCHORS
        ACTIONABLE_ANCHORS = ["company announces acquisition", "company reports earnings", "company raises guidance", "firm wins contract"]
        EARLY_ANCHORS = ["plans to announce", "expected to report", "upcoming ipo"]
        LATE_ANCHORS = ["shares surged today", "stock fell sharply"]

        anchor_vec = vectorizer.transform(ACTIONABLE_ANCHORS)
        early_vec = vectorizer.transform(EARLY_ANCHORS)
        late_vec = vectorizer.transform(LATE_ANCHORS)

        sim_actionable = np.mean(cosine_similarity(X, anchor_vec))
        sim_early = np.mean(cosine_similarity(X, early_vec))
        sim_late = np.mean(cosine_similarity(X, late_vec))

        action_factor = np.exp(sim_actionable)
        early_factor = np.exp(sim_early)
        late_factor = np.exp(sim_late)

        impact_score = action_factor * (tfidf_score(IMPACT) - tfidf_score(MACRO))
        timing_score = (early_factor * tfidf_score(EARLY) - late_factor * tfidf_score(LATE))

        return float(impact_score + timing_score)
    except:
        return 0.0

# -------------------- FETCH --------------------

def fetch_news():
    FEEDS = [
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://finance.yahoo.com/rss/topstories",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.investing.com/rss/news_25.rss",
        "https://www.investing.com/rss/news_1.rss",
        "https://www.investing.com/rss/news_301.rss",
        "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
        "https://www.ft.com/rss/home",
        "https://www.bloomberg.com/feed/podcast/etf-report.xml",
        "https://www.businessinsider.com/rss",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml"
    ]
    news=[]
    seen=set()
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for e in feed.entries[:10]:
                link = getattr(e, "link", None)
                if not link or link in seen:
                    continue
                seen.add(link)
                news.append({
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", ""),
                    "link": link,
                    "published": getattr(e, "published", "")
                })
        except:
            continue
    return news

# -------------------- RANKING --------------------

def rank_news(news):
    if not news:
        return []

    texts = [(n.get("title","") + " " + n.get("summary","")) for n in news]

    X1 = vectorizer.transform(texts)
    X2 = vectorizerr.transform(texts)
    X3 = vectorizer_xgb.transform(texts)
    X4 = vectorizer_light.transform(texts)

    try:
        xgb_scores = model_xgb.predict_proba(X3)[:,1]
    except:
        xgb_scores = model_xgb.predict(X3)

    ml = normalize(
        0.25*np.array(model_linear.predict(X1)).flatten() +
        0.25*np.array(model_rf.predict(X2)).flatten() +
        0.25*np.array(xgb_scores).flatten() +
        0.25*np.abs(np.array(model_light.predict(X4)).flatten())
    )

    topic_scores = topic_relevance(X1)
    
    # Calculate Actionability Scores
    act_scores_raw = [actionability_score(t) for t in texts]
    act_scores = normalize(act_scores_raw)

    combined = list(zip(news, texts, ml, topic_scores, act_scores))
    combined.sort(key=lambda x:x[2], reverse=True)

    news = [x[0] for x in combined]
    texts = [x[1] for x in combined]
    ml = [x[2] for x in combined]
    topic_scores = [x[3] for x in combined]
    act_scores = [x[4] for x in combined]

    top_k = min(10, len(news))
    llm = gemini_analyze_news(news[:top_k])
    trend = compute_trend(texts)

    for i in range(len(news)):
        m = ml[i]
        if i < top_k:
            impact = llm[i]["impact_score"]
            conf = llm[i]["confidence"]
        else:
            impact, conf = 0.5, 0.5

        kog = impact * conf
        w_llm, w_ml = softmax(kog, m)

        score = (
            w_ml * m +
            w_llm * kog +
            0.15 * keyword_score(texts[i], trend) +
            0.15 * recency(news[i].get("published","")) +
            0.15 * topic_scores[i] +
            0.20 * act_scores[i]  # Adding Actionability to the mix
        )

        news[i]["raw_score"] = float(score)

    scores = normalize([n["raw_score"] for n in news])
    for i in range(len(news)):
        news[i]["score"] = float(scores[i])

    return sorted(news, key=lambda x:x["score"], reverse=True)

# -------------------- BACKGROUND --------------------

def background_scraper():
    while True:
        try:
            news = fetch_news()
            ranked = rank_news(news)
            write_json(NEWS_FILE, ranked[:20])
            time.sleep(300)
        except:
            time.sleep(10)

# -------------------- START --------------------

@app.on_event("startup")
def start():
    Thread(target=background_scraper, daemon=True).start()
    Thread(target=watch_news, daemon=True).start()