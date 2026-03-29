import time
import os
import feedparser
import joblib
import json
import numpy as np
from datetime import datetime
from dateutil import parser  # pip install python-dateutil

# ------------------ Load models and vectorizers ------------------
model_linear = joblib.load("../models/model.pkl")
model_rf = joblib.load("../models/forestmodelreg.pkl")
model_xgb = joblib.load("../models/xgb_classifier.pkl")

vectorizer = joblib.load("../models/vectorizer.pkl")
vectorizerr = joblib.load("../models/vectorizerr.pkl")
vectorizer_xgb = joblib.load("../models/vectorizer_xgb.pkl")

# ------------------ Helper functions ------------------
def normalize(arr):
    arr = np.array(arr)
    min_val = arr.min()
    max_val = arr.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    return (arr - min_val) / (max_val - min_val)


def save_to_json(ranked_news, filename="news.json"):
    data = []
    for item in ranked_news[:20]:
        record = {
            "title": item["title"],
            "score": float(item["score"]),
            "link": item["link"],
            "published": item["published"]
        }
        data.append(record)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print("✅ JSON file updated")


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

    for idx, url in enumerate(FEEDS):
        feed = feedparser.parse(url)
        for entry in feed.entries[:15]:
            link = entry.link
            if link in seen_links:
                continue
            seen_links.add(link)
            record = {
                "title": entry.title,
                "link": link,
                "summary": entry.get("summary", ""),
                "published": entry.get("published", "")
            }
            news.append(record)
        print(f"Source {idx+1} fetched")

    print("All sources fetched")
    return news


# ------------------ Keyword scoring ------------------
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

def keyword_score(text):
    text = text.lower()
    score = 0
    for word, weight in KEYWORD_WEIGHTS.items():
        if word in text:
            score += weight
    return score


# ------------------ Recency scoring ------------------
def recency_score(published, decay=0.5):
    try:
        pub_date = parser.parse(published)
        now = datetime.utcnow()
        diff_hours = (now - pub_date).total_seconds() / 3600
        score = np.exp(-decay * diff_hours / 24)  # scaled in days
        return score
    except:
        return 0  # if date is missing or invalid


# ------------------ Ranking function ------------------
def rank_news(news):
    texts = [item["title"] + " " + item["summary"] for item in news]

    # Transform for ML models
    X_linear = vectorizer.transform(texts)
    X_rf = vectorizerr.transform(texts)
    X_xgb = vectorizer_xgb.transform(texts)

    # Predictions
    preds_linear = normalize(model_linear.predict(X_linear))
    preds_rf = normalize(model_rf.predict(X_rf))
    probs = model_xgb.predict_proba(X_xgb)
    preds_xgb = probs[:, 1]  # adjust for binary/multi-class

    for i in range(len(news)):
        base_score = (
            0.3 * preds_linear[i] +
            0.3 * preds_rf[i] +
            0.3 * preds_xgb[i]
        )
        boost = keyword_score(texts[i])
        recent_boost = recency_score(news[i]["published"])
        news[i]["score"] = base_score + 0.3 * boost + 0.4 * recent_boost

    return sorted(news, key=lambda x: x["score"], reverse=True)


# ------------------ Main loop ------------------
if __name__ == "__main__":
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')
            print("Fetching and ranking news...\n")

            news = fetch_news()
            ranked_news = rank_news(news)
            save_to_json(ranked_news)

            print("\nTOP RANKED NEWS:\n")
            for i, item in enumerate(ranked_news[:10]):
                # Format published date
                try:
                    pub_dt = parser.parse(item["published"])
                    pub_str = pub_dt.strftime("%Y-%m-%d %H:%M UTC")
                except:
                    pub_str = "Unknown"

                print(f"Rank {i+1}")
                print("Title:", item["title"])
                print("Score:", round(item["score"], 3))
                print("Published:", pub_str)
                print("Link:", item["link"])
                print("-" * 60)

            print("\nRefreshing in 300 seconds...\n")
            time.sleep(300)

        except Exception as e:
            print("Error:", e)
            time.sleep(60)