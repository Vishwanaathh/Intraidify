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

# ------------------ Config ------------------
USERS_FILE = "users.json"
NEWS_FILE = "news.json"

EMAIL_USER = "vishwanaathh4@gmail.com"
EMAIL_PASS = "hiwr ageg pzsn cfop"

# ------------------ FastAPI Setup ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def init_files():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump([], f)
    if not os.path.exists(NEWS_FILE):
        with open(NEWS_FILE, "w") as f:
            json.dump([], f)

init_files()

# ------------------ User API ------------------
@app.post("/signup")
async def signup(request: Request):
    data = await request.json()
    email = data.get("email")
    with open(USERS_FILE, "r") as f:
        users = json.load(f)

    if email in users:
        return {"message": "User already exists"}

    users.append(email)

    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)
    yag=yagmail.SMTP(EMAIL_USER,EMAIL_PASS)
    to_s=email
    subjectt="Welcome to IntrAIdify!!"
    contentss=f"""
        <h2>Welcome to IntrAIdify!</h2>
        <p>Hi {email},</p>
        <p>Thank you for signing up! We're excited to have you on board.</p>
        <p>🚀 Explore the latest news updates and insights now.</p>
        """
    yag.send(
        to=to_s,
        subject=subjectt,
        contents=contentss

    )
    print("Welcome sent")

    return {"message": "Signup successful"}


@app.get("/news")
def get_news():
    if not os.path.exists(NEWS_FILE):
        return []

    with open(NEWS_FILE, "r") as f:
        return json.load(f)


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
    if not data:
        return False
    current_titles = {item["title"] for item in data[:5]}
    if current_titles == last_sent_titles:
        print("⚠️ Duplicate news → skip email")
        return False
    last_sent_titles = current_titles
    return True

def send_email(data):
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
        if not users:
            print("⚠️ No users to send email")
            return

        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        content = "<h2>📈 Market News Update</h2>"
        for item in data[:10]:
            # Format published time
            try:
                pub_dt = parser.parse(item.get("published",""))
                pub_str = pub_dt.strftime("%Y-%m-%d %H:%M UTC")
            except:
                pub_str = "Unknown"

            content += f"""
            <p>
                <b>{item['title']}</b><br>
                Score: {round(item.get('score', 0),3)}<br>
                Published: {pub_str}<br>
                <a href="{item['link']}">Read more</a>
            </p>
            <hr>
            """

        yag.send(
            to=users,
            subject="🚨 Market Alert",
            contents=content
        )
        print("📧 Emails sent!")
    except Exception as e:
        print("❌ Email error:", e)

def watch_news():
    global last_hash
    print("👀 Watching news.json for real changes...")
    last_hash = get_file_hash()
    while True:
        time.sleep(5)
        new_hash = get_file_hash()
        if new_hash == last_hash:
            continue
        last_hash = new_hash
        print("\n🔥 REAL CHANGE DETECTED")
        try:
            with open(NEWS_FILE, "r") as f:
                data = json.load(f)
            if should_send(data):
                send_email(data)
        except Exception as e:
            print("❌ Error:", e)


# Load your ML models & vectorizers here
model_linear = joblib.load("../models/model.pkl")
model_rf = joblib.load("../models/forestmodelreg.pkl")
model_xgb = joblib.load("../models/xgb_classifier.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")
vectorizerr = joblib.load("../models/vectorizerr.pkl")
vectorizer_xgb = joblib.load("../models/vectorizer_xgb.pkl")

KEYWORD_WEIGHTS = {
    "acquisition": 5, "merger": 5, "buyout": 5, "takeover": 5,
    "record profit": 5, "profit surge": 5, "earnings beat": 5,
    "guidance raised": 5,
    "growth": 2, "expansion": 2, "partnership": 3,
    "upgrade": 3, "bullish": 2, "rally": 3, "surge": 3,
    "bankruptcy": 6, "fraud": 6, "crash": 6,
    "plunge": 5, "collapse": 6,
    "sell-off": 4, "loss": 2, "earnings miss": 4,
    "downgrade": 3, "recession": 4, "layoffs": 4,
    "interest rate hike": 4, "inflation": 3,
    "fed": 3, "central bank": 3,
    "war": 5, "sanctions": 4, "trade war": 5,
    "dividend": 2, "earnings": 2,
    "volatility": 3, "market correction": 4,
    "short squeeze": 5
}

def normalize(arr):
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)

def keyword_score(text):
    text = text.lower()
    score = 0
    for word, weight in KEYWORD_WEIGHTS.items():
        if word in text:
            score += weight
    return score

def recency_score(published, decay=0.5):
    try:
        pub_date = parser.parse(published)
        now = datetime.utcnow()
        diff_hours = (now - pub_date).total_seconds() / 3600
        score = np.exp(-decay * diff_hours / 24)
        return score
    except:
        return 0

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
    ]
    news = []
    seen_links = set()
    for url in FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:15]:
            link = entry.link
            if link in seen_links:
                continue
            seen_links.add(link)
            news.append({
                "title": entry.title,
                "link": link,
                "summary": entry.get("summary",""),
                "published": entry.get("published","")
            })
    return news

def rank_news(news):
    texts = [item["title"] + " " + item["summary"] for item in news]
    X_linear = vectorizer.transform(texts)
    X_rf = vectorizerr.transform(texts)
    X_xgb = vectorizer_xgb.transform(texts)

    preds_linear = normalize(model_linear.predict(X_linear))
    preds_rf = normalize(model_rf.predict(X_rf))
    probs = model_xgb.predict_proba(X_xgb)
    preds_xgb = probs[:,1]

    for i in range(len(news)):
        base_score = 0.3*preds_linear[i]+0.3*preds_rf[i]+0.3*preds_xgb[i]
        boost = keyword_score(texts[i])
        recent = recency_score(news[i]["published"])
        news[i]["score"] = base_score + 0.3*boost + 0.4*recent
    return sorted(news, key=lambda x: x["score"], reverse=True)

def save_news(news):
    with open(NEWS_FILE, "w") as f:
        json.dump(news[:20], f, indent=4)

def background_scraper():
    while True:
        try:
            print("📰 Fetching news...")
            news = fetch_news()
            ranked = rank_news(news)
            save_news(ranked)
            print("✅ News updated")
        except Exception as e:
            print("❌ Scraper error:", e)
        time.sleep(300)  # every 5 minutes

# ------------------ Startup events ------------------
@app.on_event("startup")
def start_background_tasks():
    Thread(target=background_scraper, daemon=True).start()
    Thread(target=watch_news, daemon=True).start()
    print("🚀 Background scraper and watcher started")