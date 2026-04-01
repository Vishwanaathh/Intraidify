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
load_dotenv()

USERS_FILE = "users.json"
NEWS_FILE = "news.json"

EMAIL_USER = "csfinancialservices4@gmail.com"
EMAIL_PASS = "ckvv hidk ikxq ugmf"
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
client=genai.Client(api_key=GEMINI_API_KEY)

def gemini_score_news(news):
    try:
        prompt = """You are a financial analyst.

Score each news headline from 0 to 1 based on how much it can move the stock market.

Return ONLY a JSON list of scores in the same order.

News:
"""

        for i, item in enumerate(news):
            prompt += f"{i}. {item['title']}\n"

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        text = response.text.strip()

        scores = json.loads(text)

        if len(scores) != len(news):
            return [0.5] * len(news)

        return scores

    except Exception as e:
        print("gemini error", str(e))
        return [0.5] * len(news)

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

    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        yag.send(
            to=email,
            subject="Welcome to IntrAIdify!!",
            contents=f"""
            <h2>Welcome to IntrAIdify!</h2>
            <p>Hi {email},</p>
            <p>Thank you for signing up!</p>
            """
        )
        print("signup email sent to", email)
    except Exception as e:
        print("signup email error", str(e))

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
    except Exception as e:
        print("hash error", str(e))
        return None


def should_send(data):
    global last_sent_titles
    if not data:
        return False
    try:
        current_titles = {item.get("title", "") for item in data[:5]}
    except Exception as e:
        print("should_send error", str(e))
        return False
    if current_titles == last_sent_titles:
        return False
    last_sent_titles = current_titles
    return True


def send_email(data):
    try:
        with open(USERS_FILE, "r") as f:
            users = json.load(f)
        if not users:
            print("no users")
            return
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
        content = "<h2>Market News Update</h2>"
        for item in data[:10]:
            try:
                pub_dt = parser.parse(item.get("published", ""))
                pub_str = pub_dt.strftime("%Y-%m-%d %H:%M UTC")
            except:
                pub_str = "Unknown"
            content += f"""
            <p>
                <b>{item.get('title','')}</b><br>
                Score: {round(item.get('score', 0),3)}<br>
                Published: {pub_str}<br>
                <a href="{item.get('link','')}">Read more</a>
            </p>
            <hr>
            """
        yag.send(to=users, subject="Market Alert", contents=content)
        print("email sent", users)
    except Exception as e:
        print("email error", str(e))


def watch_news():
    global last_hash
    last_hash = get_file_hash()
    print("watching news")
    while True:
        time.sleep(5)
        new_hash = get_file_hash()
        if new_hash == last_hash:
            continue
        print("news changed")
        last_hash = new_hash
        try:
            with open(NEWS_FILE, "r") as f:
                data = json.load(f)
            if should_send(data):
                print("sending email")
                send_email(data)
            else:
                print("no new headlines")
        except Exception as e:
            print("watch error", str(e))


model_linear = joblib.load("../models/model.pkl")
model_rf = joblib.load("../models/forestmodelreg.pkl")
model_xgb = joblib.load("../models/xgb_classifier.pkl")
model_light=joblib.load("../models/lightmodel.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")
vectorizerr = joblib.load("../models/vectorizerr.pkl")
vectorizer_xgb = joblib.load("../models/vectorizer_xgb.pkl")
vectorizer_light=joblib.load("../models/lightvectorizer.pkl")


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
    if len(arr) == 0:
        return np.array([])
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
        return np.exp(-decay * diff_hours / 24)
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
        try:
            feed = feedparser.parse(url)
            if not feed.entries:
                continue
            for entry in feed.entries[:15]:
                link = entry.get("link", "")
                if not link or link in seen_links:
                    continue
                seen_links.add(link)
                news.append({
                    "title": entry.get("title", ""),
                    "link": link,
                    "summary": entry.get("summary", ""),
                    "published": entry.get("published", "")
                })
        except:
            continue

    return news

def rank_news(news):
    if not news:
        return []

    texts = []
    valid_news = []

    for item in news:
        text = (item.get("title", "") + " " + item.get("summary", "")).strip()
        if text:
            texts.append(text)
            valid_news.append(item)

    if not texts:
        return []

    try:
    
        X_linear = vectorizer.transform(texts)
        X_rf = vectorizerr.transform(texts)
        X_xgb = vectorizer_xgb.transform(texts)
        X_light = vectorizer_light.transform(texts)

        # Predictions
        preds_linear = normalize(model_linear.predict(X_linear))
        preds_rf = normalize(model_rf.predict(X_rf))

        probs = model_xgb.predict_proba(X_xgb)
        preds_xgb = probs[:, 1]  # already 0–1

        preds_light = normalize(np.abs(model_light.predict(X_light)))

    except Exception as e:
        print("ranking error", str(e))
        return []

    # Step 1: ML ensemble score
    ml_scores = []
    for i in range(len(valid_news)):
        score = (
            0.25 * preds_linear[i] +
            0.25 * preds_rf[i] +
            0.25 * preds_xgb[i] +
            0.25 * preds_light[i]
        )
        ml_scores.append(score)

    # Step 2: sort by ML score first
    combined = list(zip(valid_news, texts, ml_scores))
    combined.sort(key=lambda x: x[2], reverse=True)

    valid_news = [x[0] for x in combined]
    texts = [x[1] for x in combined]
    ml_scores = [x[2] for x in combined]

    
    top_k = min(10, len(valid_news))
    top_news = valid_news[:top_k]

    gemini_scores = gemini_score_news(top_news)

    # Step 4: Final scoring
    for i in range(len(valid_news)):
        boost = keyword_score(texts[i])
        recent = recency_score(valid_news[i].get("published", ""))

        if i < top_k:
            final_score = 0.7 * ml_scores[i] + 0.3 * gemini_scores[i]
        else:
            final_score = ml_scores[i]

        valid_news[i]["score"] = final_score + 0.3 * boost + 0.4 * recent

    
    return sorted(valid_news, key=lambda x: x["score"], reverse=True)


def save_news(news):
    with open(NEWS_FILE, "w") as f:
        json.dump(news[:20], f, indent=4)


def background_scraper():
    while True:
        try:
            news = fetch_news()
            if not news:
                print("no news fetched")
                time.sleep(60)
                continue

            ranked = rank_news(news)
            if not ranked:
                print("ranking failed")
                time.sleep(60)
                continue

            save_news(ranked)
            print("news updated")
        except Exception as e:
            print("scraper error", str(e))

        time.sleep(300)


@app.on_event("startup")
def start_background_tasks():
    Thread(target=background_scraper, daemon=True).start()
    Thread(target=watch_news, daemon=True).start()