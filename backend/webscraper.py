import time
import os
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
model = joblib.load("../models/model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

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

    print("JSON file updated ✅")

def fetch_news():
    FEEDS = [
        "https://feeds.content.dowjones.io/public/rss/mw_topstories",
        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "https://www.investing.com/rss/news_25.rss",
        "https://finance.yahoo.com/rss/topstories",
    ]

    news = []
    seen_links = set()

    for idx, url in enumerate(FEEDS):
        feed = feedparser.parse(url)

        for entry in feed.entries[:15]:  
            link = entry.link

            # remove duplicates
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

        print(f"Source {idx} fetched")

    print("All sources fetched")
    return news


IMPORTANT_WORDS = [

    # 📈 Positive / Bullish Signals
    "acquisition", "acquire", "merger", "merge", "deal", "partnership",
    "collaboration", "joint venture", "takeover", "buyout",
    "expansion", "growth", "record profit", "profit surge",
    "earnings beat", "beat estimates", "strong earnings",
    "revenue growth", "guidance raised", "upgrade",
    "bullish", "rally", "surge", "soars", "jumps",
    "outperformance", "dividend increase", "stock split",
    "share buyback", "buyback", "new contract",
    "strategic investment", "funding", "investment round",
    "ipo success", "listing gains", "market expansion",

    # 📉 Negative / Bearish Signals
    "sell-off", "selloff", "decline", "drop", "plunge", "crash",
    "loss", "net loss", "earnings miss", "missed estimates",
    "guidance cut", "downgrade", "bearish",
    "recession", "slowdown", "bankruptcy", "default",
    "layoffs", "job cuts", "shutdown", "closure",
    "debt crisis", "liquidity crisis", "credit risk",
    "fraud", "scandal", "lawsuit", "investigation",
    "regulatory action", "fine", "penalty", "ban",

    # 🏦 Macro / Policy Impact
    "interest rate hike", "rate hike", "rate cut",
    "inflation", "deflation", "fed", "central bank",
    "monetary policy", "tightening", "stimulus",
    "quantitative easing", "bond yield", "treasury yield",

    # 🌍 Geopolitical / Risk
    "war", "conflict", "sanctions", "tariffs",
    "trade war", "geopolitical tension", "crisis",
    "oil prices", "commodity surge",

    # 🏢 Corporate Actions
    "dividend", "earnings", "quarter results",
    "annual report", "guidance", "forecast",
    "stake sale", "insider selling", "insider buying",
    "management change", "ceo resignation", "board change",

    # ⚡ Market Sentiment / Volatility
    "volatility", "market correction", "overvalued",
    "undervalued", "short squeeze", "options surge",
    "liquidation", "margin call"
]

def keyword_score(text):
    text = text.lower()
    score = 0
    for word in IMPORTANT_WORDS:
        if word in text:
            score += 1
    return score

def rank_news(news):
    texts = []
    for item in news:
        combined = item["title"] + " " + item["summary"]
        texts.append(combined)
    X = vectorizer.transform(texts)

    preds = model.predict(X)

    for i in range(len(news)):
        base_score = preds[i]
        boost = keyword_score(texts[i])
        news[i]["score"] = base_score + boost

    
    ranked_news = sorted(news, key=lambda x: x["score"], reverse=True)

    return ranked_news

if __name__ == "__main__":
    while True:
        try:
            os.system('cls' if os.name == 'nt' else 'clear')

            print("Fetching and ranking news...\n")

            news = fetch_news()
            ranked_news = rank_news(news)
            save_to_json(ranked_news)

            print("TOP RANKED NEWS:\n")

            for i, item in enumerate(ranked_news[:10]):
                print(f"Rank {i+1}")
                print("Title:", item["title"])
                print("Score:", item["score"])
                print("Link:", item["link"])
                print("-" * 60)

            print("\nRefreshing in 300 seconds...\n")

            time.sleep(300)   

        except Exception as e:
            print("Error:", e)
            time.sleep(300)