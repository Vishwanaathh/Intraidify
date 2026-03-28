import time
import os
import feedparser
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
model = joblib.load("../models/model.pkl")
modell=joblib.load("../models/forestmodelreg.pkl")
modelll=joblib.load("../models/xgb_classfier.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")
vectorizerr=joblib.load("../models/vectorizerr.pkl")
vectorizerrr=joblib.load("../models/vectorizer_xgb.pkl")

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
    "https://finance.yahoo.com/rss/topstories",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",

    # 📊 Market Specific
    "https://www.investing.com/rss/news_25.rss",
    "https://www.investing.com/rss/news_1.rss",   # general news
    "https://www.investing.com/rss/news_301.rss", # stock market

    # 🌍 Global Financial News
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "https://www.ft.com/rss/home",  # Financial Times
    "https://www.bloomberg.com/feed/podcast/etf-report.xml",
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


KEYWORD_WEIGHTS = {

    # 📈 VERY STRONG POSITIVE (market-moving)
    "acquisition": 5,
    "merger": 5,
    "buyout": 5,
    "takeover": 5,
    "strategic investment": 4,
    "funding round": 4,
    "ipo success": 4,
    "record profit": 5,
    "profit surge": 5,
    "earnings beat": 5,
    "beat estimates": 4,
    "guidance raised": 5,

    # 📈 MODERATE POSITIVE
    "growth": 2,
    "expansion": 2,
    "partnership": 3,
    "collaboration": 2,
    "joint venture": 3,
    "upgrade": 3,
    "bullish": 2,
    "rally": 3,
    "surge": 3,
    "soars": 3,
    "jumps": 2,
    "outperformance": 3,
    "dividend increase": 3,
    "stock split": 3,
    "share buyback": 4,
    "buyback": 3,
    "new contract": 3,
    "revenue growth": 3,
    "market expansion": 2,

    # 📉 VERY STRONG NEGATIVE
    "bankruptcy": 6,
    "default": 5,
    "fraud": 6,
    "scandal": 5,
    "crash": 6,
    "plunge": 5,
    "collapse": 6,
    "liquidity crisis": 5,
    "debt crisis": 5,

    # 📉 MODERATE NEGATIVE
    "sell-off": 4,
    "selloff": 4,
    "decline": 2,
    "drop": 2,
    "loss": 2,
    "net loss": 3,
    "earnings miss": 4,
    "missed estimates": 4,
    "guidance cut": 4,
    "downgrade": 3,
    "bearish": 2,
    "recession": 4,
    "slowdown": 3,
    "layoffs": 4,
    "job cuts": 4,
    "shutdown": 4,
    "closure": 3,
    "credit risk": 4,
    "lawsuit": 3,
    "investigation": 3,
    "regulatory action": 4,
    "fine": 3,
    "penalty": 3,
    "ban": 4,

    # 🏦 MACRO (context-heavy impact)
    "interest rate hike": 4,
    "rate hike": 4,
    "rate cut": 4,
    "inflation": 3,
    "deflation": 3,
    "fed": 3,
    "central bank": 3,
    "monetary policy": 3,
    "tightening": 4,
    "stimulus": 4,
    "quantitative easing": 4,
    "bond yield": 3,
    "treasury yield": 3,

    # 🌍 GEOPOLITICAL (high volatility impact)
    "war": 5,
    "conflict": 4,
    "sanctions": 4,
    "tariffs": 4,
    "trade war": 5,
    "geopolitical tension": 4,
    "crisis": 4,
    "oil prices": 4,
    "commodity surge": 4,

    # 🏢 CORPORATE ACTIONS
    "dividend": 2,
    "earnings": 2,
    "quarter results": 2,
    "annual report": 1,
    "guidance": 2,
    "forecast": 2,
    "stake sale": 3,
    "insider selling": 4,
    "insider buying": 4,
    "management change": 3,
    "ceo resignation": 4,
    "board change": 2,

    # ⚡ VOLATILITY / TRADING SIGNALS
    "volatility": 3,
    "market correction": 4,
    "overvalued": 3,
    "undervalued": 3,
    "short squeeze": 5,
    "options surge": 3,
    "liquidation": 4,
    "margin call": 5
}

def keyword_score(text):
    text = text.lower()
    score = 0

    for word, weight in KEYWORD_WEIGHTS.items():
        if word in text:
            score += weight

    return score

def rank_news(news):
    texts = []

    for item in news:
        combined = item["title"] + " " + item["summary"]
        texts.append(combined)

    # transforms
    X_linear = vectorizer.transform(texts)
    X_rf = vectorizerr.transform(texts)
    X_xgb = vectorizer_xgb.transform(texts)

    # predictions
    preds_linear = model.predict(X_linear)
    preds_rf = modell.predict(X_rf)

    probs = model_xgb.predict_proba(X_xgb)
    preds_xgb = probs[:, 2]   

    # normalize
    preds_linear = normalize(preds_linear)
    preds_rf = normalize(preds_rf)

    for i in range(len(news)):
        base_score = (
            0.3 * preds_linear[i] +
            0.3 * preds_rf[i] +
            0.4 * preds_xgb[i]
        )

        boost = keyword_score(texts[i])

        news[i]["score"] = base_score + 0.3 * boost

    return sorted(news, key=lambda x: x["score"], reverse=True)

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