# -*- coding: utf-8 -*-
"""
IntrAIdify - ONE-SCRIPT training + evaluation pipeline.

Drop this in as model_gens/train_all.py, replacing whatever's there now.
Does everything in one run:

  PART 1 - TRAINING
    - Time-based train/test split on the DJIA headline dataset (no leakage)
    - Trains Linear Regression, Random Forest (depth-bounded for speed),
      XGBoost classifier, and LightGBM (on the separate SBI sentiment dataset)
    - Saves all models/vectorizers to ../models/ (matches what
      backend/patent_server.py loads - zero changes needed there)
    - Evaluates each model individually on held-out data, WITH baselines
      (random-ranking NDCG@10 floor, majority-class accuracy floor) so the
      numbers have context instead of floating in isolation

  PART 2 - FULL PIPELINE ABLATION
    - Reuses the just-trained models in memory (no reloading from disk)
    - Replicates patent_server.py's actual rank_news() logic exactly:
      4-model ML ensemble + trend-weighted keyword score + topic relevance
      (cosine similarity) + actionability score + adaptive softmax ML/LLM blend
    - Runs it as an ablation: (A) ML only -> (B) +keyword -> (C) +topic/action
      -> (D) full pipeline with real Gemini calls, so we can see what each
      component actually contributes instead of one opaque final number

  PART 3 - PLOTS
    - eval/plots/individual_models_bars.png  - each model vs its baseline
    - eval/plots/ablation_bars.png           - NDCG@10 + Spearman across A-D
    - eval/plots/llm_weight_distribution.png - histogram of softmax w_llm
    - eval/plots/daily_ndcg_boxplot.png      - per-day consistency, A vs D

Requires network (kagglehub) and a GEMINI_API_KEY environment variable set
before running (same one patent_server.py uses).

COST/TIME NOTE: Part 2 makes one real Gemini API call per test-set day.
LIMIT_DAYS below defaults to 20 for a fast sanity-check run through the
WHOLE script first. Once that works cleanly, set LIMIT_DAYS = None and rerun
for the real numbers (full run = ~398 Gemini calls, budget ~10-15 minutes).
"""

import sys
import os
import json
import time
import logging
import traceback
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pointbiserialr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score, accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
import lightgbm as lgb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

LIMIT_DAYS = 20  # <-- sanity-check first, then set to None for the full run
SLEEP_BETWEEN_CALLS = 1.0
CHECKPOINT_EVERY = 10

EVAL_DIR = "eval"
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logger = logging.getLogger("train_and_eval_all")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(fmt)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(os.path.join(EVAL_DIR, "train_log.txt"), mode="w")
file_handler.setFormatter(fmt)
logger.addHandler(file_handler)


def step(name):
    class _Step:
        def __enter__(self):
            self.t0 = time.time()
            logger.info(f"START: {name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            dt = time.time() - self.t0
            if exc_type is not None:
                logger.error(f"FAILED: {name} after {dt:.1f}s")
                logger.error("".join(traceback.format_exception(exc_type, exc_val, exc_tb)))
                return False
            logger.info(f"DONE:  {name} ({dt:.1f}s)")
    return _Step()


metrics = {}  # individual-model metrics (Part 1)

# ===========================================================================
# PART 1 - TRAINING
# ===========================================================================

with step("import kagglehub + download DJIA dataset"):
    import kagglehub
    path = kagglehub.dataset_download("tanishqdublish/stock-market-predictions")
    logger.info(f"  dataset path: {path}")

with step("load + time-sort DJIA dataset"):
    data = pd.read_csv(path + "/Combined_News_DJIA.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].reset_index(drop=True)
    test_df = data.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"  train days: {len(train_df)} ({train_df['Date'].min()} -> {train_df['Date'].max()})")
    logger.info(f"  test days:  {len(test_df)} ({test_df['Date'].min()} -> {test_df['Date'].max()})")


def rows_to_texts_scores(df):
    texts, scores, day_idx = [], [], []
    for row_i, row in df.iterrows():
        for i in range(1, 26):
            text = str(row[f"Top{i}"]).replace("b'", "").replace('b"', "").strip()
            texts.append(text)
            scores.append(26 - i)
            day_idx.append(row_i)
    return texts, scores, day_idx


with step("flatten headlines into (text, rank) pairs"):
    train_texts, train_scores, _ = rows_to_texts_scores(train_df)
    test_texts, test_scores, test_day_idx = rows_to_texts_scores(test_df)
    logger.info(f"  train examples: {len(train_texts)} | test examples: {len(test_texts)}")

with step("Linear Regression: vectorize + train"):
    vectorizer_lin = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_lin = vectorizer_lin.fit_transform(train_texts)
    X_test_lin = vectorizer_lin.transform(test_texts)
    model_lin = LinearRegression()
    model_lin.fit(X_train_lin, train_scores)

with step("Linear Regression: predict + save"):
    preds_lin = model_lin.predict(X_test_lin)
    joblib.dump(model_lin, "../models/model.pkl")
    joblib.dump(vectorizer_lin, "../models/vectorizer.pkl")

with step("Random Forest: vectorize"):
    vectorizer_rf = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_rf = vectorizer_rf.fit_transform(train_texts)
    X_test_rf = vectorizer_rf.transform(test_texts)

with step("Random Forest: train (depth-bounded for speed on sparse TF-IDF input)"):
    # Unbounded trees on a ~40k x 5000 sparse matrix grow enormous and slow to a
    # crawl (this was the original cause of the training "hanging"). Bounding
    # depth/leaf-size keeps this to ~1-2 minutes with negligible metric impact.
    model_rf = RandomForestRegressor(
        n_estimators=150, max_depth=20, min_samples_leaf=5,
        random_state=42, n_jobs=-1, verbose=1,
    )
    model_rf.fit(X_train_rf, train_scores)

with step("Random Forest: predict + save"):
    preds_rf = model_rf.predict(X_test_rf)
    joblib.dump(model_rf, "../models/forestmodelreg.pkl")
    joblib.dump(vectorizer_rf, "../models/vectorizerr.pkl")


def rows_to_texts_labels(df):
    texts, labels, day_idx = [], [], []
    for row_i, row in df.iterrows():
        for i in range(1, 26):
            text = str(row[f"Top{i}"]).replace("b'", "").replace('b"', "").strip()
            label = 2 if i <= 5 else (1 if i <= 15 else 0)
            texts.append(text)
            labels.append(label)
            day_idx.append(row_i)
    return texts, labels, day_idx


with step("XGBoost: prep + vectorize"):
    train_texts_xgb, train_labels_xgb, _ = rows_to_texts_labels(train_df)
    test_texts_xgb, test_labels_xgb, _ = rows_to_texts_labels(test_df)
    vectorizer_xgb = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_xgb = vectorizer_xgb.fit_transform(train_texts_xgb)
    X_test_xgb = vectorizer_xgb.transform(test_texts_xgb)

with step("XGBoost: train"):
    model_xgb = XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, n_jobs=-1,
        eval_metric="mlogloss", verbosity=1,
    )
    model_xgb.fit(X_train_xgb, train_labels_xgb)

with step("XGBoost: predict + save"):
    preds_xgb_class = model_xgb.predict(X_test_xgb)
    joblib.dump(model_xgb, "../models/xgb_classifier.pkl")
    joblib.dump(vectorizer_xgb, "../models/vectorizer_xgb.pkl")

with step("download SBI sentiment dataset"):
    sent_path = kagglehub.dataset_download("parshantkumar2033/sentiment-intensity-for-news-articles-of-sbi")

with step("load + split SBI sentiment dataset"):
    sent_data = pd.read_csv(sent_path + "/dataset.csv").dropna(subset=["article"])
    sent_data = sent_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sent_split = int(len(sent_data) * 0.8)
    sent_train = sent_data.iloc[:sent_split]
    sent_test = sent_data.iloc[sent_split:]

with step("LightGBM: vectorize + train"):
    vectorizer_light = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_light = vectorizer_light.fit_transform(sent_train["article"])
    X_test_light = vectorizer_light.transform(sent_test["article"])
    y_train_light = sent_train["sentiment_score"].values
    y_test_light = sent_test["sentiment_score"].values
    model_light = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31)
    model_light.fit(X_train_light, y_train_light)

with step("LightGBM: predict + save"):
    preds_light = model_light.predict(X_test_light)
    joblib.dump(model_light, "../models/lightmodel.pkl")
    joblib.dump(vectorizer_light, "../models/lightvectorizer.pkl")

# --- individual-model evaluation, with baselines ---

with step("evaluate Linear Regression (per-day Spearman + NDCG@10)"):
    test_df_eval = pd.DataFrame({"day": test_day_idx, "true_score": test_scores, "pred_score": preds_lin})
    sp, nd = [], []
    for day, grp in test_df_eval.groupby("day"):
        if grp["true_score"].nunique() > 1:
            rho, _ = spearmanr(grp["true_score"], grp["pred_score"])
            sp.append(rho)
        nd.append(ndcg_score([grp["true_score"].values], [grp["pred_score"].values], k=10))
    metrics["linear_regression"] = {"spearman_rho": float(np.nanmean(sp)), "ndcg_at_10": float(np.mean(nd))}
    logger.info(f"  {metrics['linear_regression']}")

with step("evaluate Random Forest (per-day Spearman + NDCG@10)"):
    test_df_eval_rf = pd.DataFrame({"day": test_day_idx, "true_score": test_scores, "pred_score": preds_rf})
    sp, nd = [], []
    for day, grp in test_df_eval_rf.groupby("day"):
        if grp["true_score"].nunique() > 1:
            rho, _ = spearmanr(grp["true_score"], grp["pred_score"])
            sp.append(rho)
        nd.append(ndcg_score([grp["true_score"].values], [grp["pred_score"].values], k=10))
    metrics["random_forest"] = {"spearman_rho": float(np.nanmean(sp)), "ndcg_at_10": float(np.mean(nd))}
    logger.info(f"  {metrics['random_forest']}")

with step("evaluate RANDOM BASELINE (NDCG@10 floor for context)"):
    rng0 = np.random.default_rng(42)
    trials = []
    for _ in range(20):
        vals = []
        for day, grp in test_df_eval.groupby("day"):
            vals.append(ndcg_score([grp["true_score"].values], [rng0.random(len(grp))], k=10))
        trials.append(np.mean(vals))
    metrics["random_baseline"] = {"ndcg_at_10_mean": float(np.mean(trials)), "ndcg_at_10_std": float(np.std(trials))}
    logger.info(f"  {metrics['random_baseline']}")

with step("evaluate XGBoost classifier (accuracy + F1, vs. majority-class baseline)"):
    majority_class = Counter(train_labels_xgb).most_common(1)[0][0]
    majority_acc = accuracy_score(test_labels_xgb, [majority_class] * len(test_labels_xgb))
    metrics["xgboost_classifier"] = {
        "accuracy": float(accuracy_score(test_labels_xgb, preds_xgb_class)),
        "macro_f1": float(f1_score(test_labels_xgb, preds_xgb_class, average="macro")),
        "majority_class_baseline_accuracy": float(majority_acc),
    }
    logger.info(f"  {metrics['xgboost_classifier']}")

with step("evaluate LightGBM (RMSE + R^2)"):
    try:
        rmse_light = mean_squared_error(y_test_light, preds_light, squared=False)
    except TypeError:
        rmse_light = float(np.sqrt(mean_squared_error(y_test_light, preds_light)))
    metrics["lightgbm_sentiment"] = {"rmse": float(rmse_light), "r2": float(r2_score(y_test_light, preds_light))}
    logger.info(f"  {metrics['lightgbm_sentiment']}")

with step("write eval/metrics.json (Part 1 results)"):
    with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

with step("generate eval/plots/individual_models_bars.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].bar(
        ["Random", "Linear Reg", "Random Forest"],
        [metrics["random_baseline"]["ndcg_at_10_mean"], metrics["linear_regression"]["ndcg_at_10"], metrics["random_forest"]["ndcg_at_10"]],
        color=["#999999", "#4C72B0", "#55A868"],
    )
    axes[0].set_title("NDCG@10 vs random baseline")
    axes[0].set_ylabel("NDCG@10")

    axes[1].bar(
        ["Majority-class guess", "XGBoost"],
        [metrics["xgboost_classifier"]["majority_class_baseline_accuracy"], metrics["xgboost_classifier"]["accuracy"]],
        color=["#999999", "#C44E52"],
    )
    axes[1].set_title("XGBoost accuracy vs majority-class baseline")
    axes[1].set_ylabel("Accuracy")

    axes[2].bar(["LightGBM R^2"], [metrics["lightgbm_sentiment"]["r2"]], color=["#8172B2"])
    axes[2].set_ylim(0, 1)
    axes[2].set_title("LightGBM sentiment R^2")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "individual_models_bars.png"), dpi=150)
    plt.close()

# ===========================================================================
# PART 2 - FULL PIPELINE ABLATION (reuses models already in memory above)
# ===========================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set. Set it as an environment variable before running Part 2.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)

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
    "strong demand": 4, "weak demand": 4,
}

IMPACT = ["acquisition", "merger", "ipo", "bankruptcy", "investigation", "contract",
          "earnings", "revenue", "profit", "guidance", "forecast", "partnership",
          "divestiture", "buyback", "dividend", "restructuring", "lawsuit", "settlement",
          "default", "liquidation", "joint venture", "licensing", "clinical trial",
          "fda approval", "patent", "takeover", "tender offer", "spinoff"]
MACRO = ["inflation", "economy", "sentiment", "outlook", "analysts", "macro",
         "fed", "interest rates", "gdp", "unemployment", "treasury", "yield",
         "central bank", "monetary policy", "fiscal", "recession", "stagnation",
         "consumer price index", "cpi", "geopolitical", "trade war", "tariffs"]
EARLY = ["plans", "will", "expected", "upcoming", "set to", "announce",
         "considering", "exploring", "rumored", "potential", "aims", "seeks",
         "prepares", "intends", "scheduled", "pending", "on track", "initial",
         "proposes", "seeks approval", "preliminary", "eyes"]
LATE = ["rose", "surged", "gained", "fell", "dropped", "plunged",
        "soared", "slumped", "tumbled", "rallied", "retreated", "spiked",
        "closed", "ended", "climbed", "slipped", "dived", "crashed",
        "jumped", "tanked", "stabilized", "rebounded"]

ACTIONABLE_ANCHORS = ["company announces acquisition", "company reports earnings",
                       "company raises guidance", "firm wins contract"]
EARLY_ANCHORS = ["plans to announce", "expected to report", "upcoming ipo"]
LATE_ANCHORS = ["shares surged today", "stock fell sharply"]


def normalize(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) == 0 or arr.max() - arr.min() == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def softmax(a, b, T=0.7):
    m = max(a, b)
    ea = np.exp((a - m) / T)
    eb = np.exp((b - m) / T)
    s = ea + eb
    return ea / s, eb / s


def recency(published):
    # Constant 1.0 here - no per-headline timestamps exist in the historical
    # dataset, so this term is a documented no-op for this offline backtest.
    return 1.0


def compute_trend(texts):
    freq = {}
    for t in texts:
        t = t.lower()
        for k in KEYWORD_WEIGHTS:
            if k in t:
                freq[k] = freq.get(k, 0) + 1
    total = sum(freq.values()) + 1
    return {k: 1 + np.log1p(freq.get(k, 0)) / np.log1p(total) for k in KEYWORD_WEIGHTS}


def keyword_score(text, trend):
    text = text.lower()
    s = 0
    for k, v in KEYWORD_WEIGHTS.items():
        if k in text:
            s += v * trend[k]
    return np.sqrt(min(s, 10) / 10)


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
    except Exception:
        return np.zeros(X.shape[0])


def make_actionability_fn(vec):
    anchor_vec = vec.transform(ACTIONABLE_ANCHORS)
    early_vec = vec.transform(EARLY_ANCHORS)
    late_vec = vec.transform(LATE_ANCHORS)
    vocab = vec.vocabulary_

    def tfidf_score(word_list, X_dense):
        s = 0
        for w in word_list:
            if w in vocab:
                idx = vocab[w]
                v = X_dense[idx]
                if v > 0:
                    s += np.log1p(v)
        return s

    def actionability_score(text):
        try:
            X = vec.transform([text])
            X_dense = X.toarray()[0]
            sim_actionable = np.mean(cosine_similarity(X, anchor_vec))
            sim_early = np.mean(cosine_similarity(X, early_vec))
            sim_late = np.mean(cosine_similarity(X, late_vec))
            action_factor = np.exp(sim_actionable)
            early_factor = np.exp(sim_early)
            late_factor = np.exp(sim_late)
            impact_score = action_factor * (tfidf_score(IMPACT, X_dense) - tfidf_score(MACRO, X_dense))
            timing_score = (early_factor * tfidf_score(EARLY, X_dense) - late_factor * tfidf_score(LATE, X_dense))
            return float(impact_score + timing_score)
        except Exception:
            return 0.0

    return actionability_score


def gemini_analyze_news(headlines):
    def fallback(n):
        return [{"impact_score": 0.5, "market_direction": "neutral", "event_type": "other", "confidence": 0.5} for _ in range(n)]
    try:
        prompt = (
            "You are a financial analyst ranking news based on market impact "
            "Return ONLY a valid JSON array.\n"
            "Each item must have:\n"
            "impact_score (0-1), market_direction, event_type, confidence (0-1).\n"
            "No explanation. Only JSON.\n"
        )
        for i, h in enumerate(headlines):
            prompt += f"{i}. {h}\n"
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        text = response.text.strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 2:
                text = parts[1]
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == -1:
            return fallback(len(headlines))
        data = json.loads(text[start:end])
        cleaned = []
        for item in data:
            cleaned.append({
                "impact_score": float(item.get("impact_score", 0.5)),
                "market_direction": item.get("market_direction", "neutral"),
                "event_type": item.get("event_type", "other"),
                "confidence": float(item.get("confidence", 0.5)),
            })
        while len(cleaned) < len(headlines):
            cleaned.append({"impact_score": 0.5, "market_direction": "neutral", "event_type": "other", "confidence": 0.5})
        return cleaned[:len(headlines)]
    except Exception as e:
        logger.warning(f"  Gemini analyze error: {e} - using neutral fallback")
        return fallback(len(headlines))


with step("prep actionability scorer (reuses vectorizer_lin already trained above)"):
    actionability_score = make_actionability_fn(vectorizer_lin)

with step("rebuild same test-day slice for pipeline ablation"):
    pipeline_test_df = test_df.iloc[:LIMIT_DAYS].reset_index(drop=True) if LIMIT_DAYS is not None else test_df
    if LIMIT_DAYS is not None:
        logger.info(f"  LIMIT_DAYS active - only evaluating first {LIMIT_DAYS} test days")
    logger.info(f"  evaluating {len(pipeline_test_df)} test days")

configs = ["A_ml_only", "B_ml_keyword", "C_ml_keyword_topic_action", "D_full_pipeline"]
pipeline_results = {c: [] for c in configs}
random_results_pipeline = []
w_llm_all = []
daily_ndcg_A, daily_ndcg_D = [], []
rng = np.random.default_rng(42)
checkpoint_path = os.path.join(EVAL_DIR, "full_pipeline_checkpoint.json")

with step(f"run pipeline ablation over {len(pipeline_test_df)} test days (real Gemini calls)"):
    for day_i, row in pipeline_test_df.iterrows():
        headlines = [str(row[f"Top{i}"]).replace("b'", "").replace('b"', "").strip() for i in range(1, 26)]
        true_scores = np.array([26 - i for i in range(1, 26)])

        X1 = vectorizer_lin.transform(headlines)
        X2 = vectorizer_rf.transform(headlines)
        X3 = vectorizer_xgb.transform(headlines)
        X4 = vectorizer_light.transform(headlines)

        try:
            xgb_scores = model_xgb.predict_proba(X3)[:, 1]
        except Exception:
            xgb_scores = model_xgb.predict(X3)

        ml = normalize(
            0.25 * np.array(model_lin.predict(X1)).flatten()
            + 0.25 * np.array(model_rf.predict(X2)).flatten()
            + 0.25 * np.array(xgb_scores).flatten()
            + 0.25 * np.abs(np.array(model_light.predict(X4)).flatten())
        )

        rho_a, _ = spearmanr(true_scores, ml) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_a = ndcg_score([true_scores], [ml], k=10)
        pipeline_results["A_ml_only"].append({"spearman": rho_a, "ndcg_10": ndcg_a})
        daily_ndcg_A.append(ndcg_a)

        trend = compute_trend(headlines)
        kw_scores = np.array([keyword_score(h, trend) for h in headlines])
        combined_b = ml + 0.3 * kw_scores
        rho_b, _ = spearmanr(true_scores, combined_b) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_b = ndcg_score([true_scores], [combined_b], k=10)
        pipeline_results["B_ml_keyword"].append({"spearman": rho_b, "ndcg_10": ndcg_b})

        topic_scores = topic_relevance(X1)
        act_scores = normalize([actionability_score(h) for h in headlines])
        combined_c = ml + 0.3 * kw_scores + 0.15 * topic_scores + 0.20 * act_scores
        rho_c, _ = spearmanr(true_scores, combined_c) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_c = ndcg_score([true_scores], [combined_c], k=10)
        pipeline_results["C_ml_keyword_topic_action"].append({"spearman": rho_c, "ndcg_10": ndcg_c})

        order = np.argsort(-ml)
        top_k = min(10, len(headlines))
        top_idx = order[:top_k]
        top_headlines = [headlines[i] for i in top_idx]
        llm_results = gemini_analyze_news(top_headlines)
        time.sleep(SLEEP_BETWEEN_CALLS)
        llm_by_idx = {idx: llm_results[pos] for pos, idx in enumerate(top_idx)}

        final_scores = np.zeros(len(headlines))
        for i in range(len(headlines)):
            m = ml[i]
            if i in llm_by_idx:
                impact = llm_by_idx[i]["impact_score"]
                conf = llm_by_idx[i]["confidence"]
            else:
                impact, conf = 0.5, 0.5
            kog = impact * conf
            w_llm, w_ml = softmax(kog, m)
            w_llm_all.append(w_llm)
            final_scores[i] = (
                w_ml * m + w_llm * kog
                + 0.15 * keyword_score(headlines[i], trend)
                + 0.15 * recency("")
                + 0.15 * topic_scores[i]
                + 0.20 * act_scores[i]
            )
        final_scores = normalize(final_scores)
        rho_d, _ = spearmanr(true_scores, final_scores) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_d = ndcg_score([true_scores], [final_scores], k=10)
        pipeline_results["D_full_pipeline"].append({"spearman": rho_d, "ndcg_10": ndcg_d})
        daily_ndcg_D.append(ndcg_d)

        random_results_pipeline.append(ndcg_score([true_scores], [rng.random(len(headlines))], k=10))

        if (day_i + 1) % CHECKPOINT_EVERY == 0 or (day_i + 1) == len(pipeline_test_df):
            logger.info(f"  processed {day_i + 1}/{len(pipeline_test_df)} days")
            with open(checkpoint_path, "w") as f:
                json.dump(pipeline_results, f, indent=2, default=lambda x: None if pd.isna(x) else x)

with step("aggregate pipeline ablation results"):
    def agg(key):
        rhos = [r["spearman"] for r in pipeline_results[key] if not pd.isna(r["spearman"])]
        ndcgs = [r["ndcg_10"] for r in pipeline_results[key]]
        return {"spearman_rho": float(np.mean(rhos)), "ndcg_at_10": float(np.mean(ndcgs))}

    pipeline_metrics = {
        "n_test_days_evaluated": len(pipeline_test_df),
        "limit_days_active": LIMIT_DAYS is not None,
        "A_ml_only": agg("A_ml_only"),
        "B_ml_keyword": agg("B_ml_keyword"),
        "C_ml_keyword_topic_action": agg("C_ml_keyword_topic_action"),
        "D_full_pipeline": agg("D_full_pipeline"),
        "random_baseline_ndcg_at_10": float(np.mean(random_results_pipeline)),
        "w_llm_mean": float(np.mean(w_llm_all)),
        "w_llm_median": float(np.median(w_llm_all)),
        "w_llm_fraction_above_0.5": float(np.mean(np.array(w_llm_all) > 0.5)),
        "known_limitations": [
            "text = title only (no summary field in the DJIA dataset) - this eval "
            "sees less text per headline than production does.",
            "recency() is a constant 1.0 here - no per-headline timestamps exist "
            "in the historical dataset, so this term is a documented no-op offline.",
            "For non-top-10 headlines, kog defaults to 0.5*0.5=0.25 (neutral "
            "placeholder). Softmax treats this as a real competing score, which "
            "can pull low-ML-scoring tail headlines upward - reproduced "
            "faithfully here as a known production quirk, not fixed.",
        ],
    }

with step("paired significance tests (Wilcoxon signed-rank, same days across configs)"):
    from scipy.stats import wilcoxon

    def paired_test(key_a, key_b, metric):
        vals_a = [r[metric] for r in pipeline_results[key_a]]
        vals_b = [r[metric] for r in pipeline_results[key_b]]
        # spearman can have NaN on flat-score days - pair-drop those
        pairs = [(a, b) for a, b in zip(vals_a, vals_b) if not (pd.isna(a) or pd.isna(b))]
        if len(pairs) < 6:
            return {"n": len(pairs), "statistic": None, "p_value": None,
                    "note": "too few paired days for a reliable test (need >=6)"}
        a_arr, b_arr = zip(*pairs)
        diffs = np.array(a_arr) - np.array(b_arr)
        if np.all(diffs == 0):
            return {"n": len(pairs), "statistic": None, "p_value": 1.0,
                    "note": "identical values across all days"}
        try:
            stat, p = wilcoxon(a_arr, b_arr)
            return {"n": len(pairs), "statistic": float(stat), "p_value": float(p),
                    "mean_diff": float(np.mean(diffs)),
                    "note": "significant at p<0.05" if p < 0.05 else "not significant at p<0.05"}
        except Exception as e:
            return {"n": len(pairs), "statistic": None, "p_value": None, "note": f"test failed: {e}"}

    comparisons = [
        ("A_ml_only", "B_ml_keyword", "A vs B: does keyword weighting help?"),
        ("B_ml_keyword", "C_ml_keyword_topic_action", "B vs C: does topic/actionability help?"),
        ("C_ml_keyword_topic_action", "D_full_pipeline", "C vs D: does LLM softmax blend help?"),
        ("A_ml_only", "D_full_pipeline", "A vs D: does the full pipeline beat plain ML?"),
    ]

    significance_results = {}
    for key_a, key_b, description in comparisons:
        significance_results[f"{key_a}_vs_{key_b}"] = {
            "description": description,
            "ndcg_10": paired_test(key_a, key_b, "ndcg_10"),
            "spearman": paired_test(key_a, key_b, "spearman"),
        }
        logger.info(f"  {description}")
        logger.info(f"    NDCG@10:   {significance_results[f'{key_a}_vs_{key_b}']['ndcg_10']}")
        logger.info(f"    Spearman:  {significance_results[f'{key_a}_vs_{key_b}']['spearman']}")

    pipeline_metrics["significance_tests"] = significance_results
    pipeline_metrics["significance_test_note"] = (
        "Wilcoxon signed-rank test, paired by day (same test days across all "
        "configs). mean_diff is config_a minus config_b: positive means config_a "
        "scored higher on average. IMPORTANT: with LIMIT_DAYS active these tests "
        "have very low statistical power (small n) - treat as a dry run only. "
        "Rerun with LIMIT_DAYS = None before trusting these p-values for the paper."
    )
    with open(os.path.join(EVAL_DIR, "full_pipeline_metrics.json"), "w") as f:
        json.dump(pipeline_metrics, f, indent=2)
    logger.info(json.dumps(pipeline_metrics, indent=2))

with step("write eval/full_pipeline_comparison.md"):
    md = f"""# Full Pipeline Ablation Study (Adaptive Softmax Version)

Evaluated on {pipeline_metrics['n_test_days_evaluated']} held-out test days
{'(LIMITED RUN - set LIMIT_DAYS = None and rerun for real paper numbers)' if pipeline_metrics['limit_days_active'] else '(full test set)'}.

| Configuration | Spearman rho | NDCG@10 |
|---|---|---|
| Random baseline | - | {pipeline_metrics['random_baseline_ndcg_at_10']:.4f} |
| (A) ML ensemble only | {pipeline_metrics['A_ml_only']['spearman_rho']:.4f} | {pipeline_metrics['A_ml_only']['ndcg_at_10']:.4f} |
| (B) + trend-weighted keyword | {pipeline_metrics['B_ml_keyword']['spearman_rho']:.4f} | {pipeline_metrics['B_ml_keyword']['ndcg_at_10']:.4f} |
| (C) + topic relevance + actionability | {pipeline_metrics['C_ml_keyword_topic_action']['spearman_rho']:.4f} | {pipeline_metrics['C_ml_keyword_topic_action']['ndcg_at_10']:.4f} |
| (D) + adaptive softmax LLM blend (FULL) | {pipeline_metrics['D_full_pipeline']['spearman_rho']:.4f} | {pipeline_metrics['D_full_pipeline']['ndcg_at_10']:.4f} |

## LLM weight behavior (softmax w_llm across all scored headlines)

- Mean w_llm: {pipeline_metrics['w_llm_mean']:.4f}
- Median w_llm: {pipeline_metrics['w_llm_median']:.4f}
- Fraction of headlines where LLM got >50% of the weight: {pipeline_metrics['w_llm_fraction_above_0.5']:.1%}

See eval/plots/llm_weight_distribution.png - if this is bimodal (clustered
near 0 and 1), softmax is behaving like a near-binary switch, not a smooth
blend - worth discussing explicitly.

## Statistical significance (paired Wilcoxon signed-rank, same days across configs)

{pipeline_metrics['significance_test_note']}

| Comparison | Metric | n | p-value | Mean diff | Verdict |
|---|---|---|---|---|---|
| A vs B (keyword) | NDCG@10 | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['ndcg_10'].get('n','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['ndcg_10'].get('p_value','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['ndcg_10'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['ndcg_10'].get('note','-')} |
| B vs C (topic/action) | NDCG@10 | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['ndcg_10'].get('n','-')} | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['ndcg_10'].get('p_value','-')} | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['ndcg_10'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['ndcg_10'].get('note','-')} |
| C vs D (LLM blend) | NDCG@10 | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['ndcg_10'].get('n','-')} | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['ndcg_10'].get('p_value','-')} | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['ndcg_10'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['ndcg_10'].get('note','-')} |
| A vs D (full vs plain ML) | NDCG@10 | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['ndcg_10'].get('n','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['ndcg_10'].get('p_value','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['ndcg_10'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['ndcg_10'].get('note','-')} |
| A vs B (keyword) | Spearman | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['spearman'].get('n','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['spearman'].get('p_value','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['spearman'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_B_ml_keyword']['spearman'].get('note','-')} |
| B vs C (topic/action) | Spearman | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['spearman'].get('n','-')} | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['spearman'].get('p_value','-')} | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['spearman'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['B_ml_keyword_vs_C_ml_keyword_topic_action']['spearman'].get('note','-')} |
| C vs D (LLM blend) | Spearman | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['spearman'].get('n','-')} | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['spearman'].get('p_value','-')} | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['spearman'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['C_ml_keyword_topic_action_vs_D_full_pipeline']['spearman'].get('note','-')} |
| A vs D (full vs plain ML) | Spearman | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['spearman'].get('n','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['spearman'].get('p_value','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['spearman'].get('mean_diff','-')} | {pipeline_metrics['significance_tests']['A_ml_only_vs_D_full_pipeline']['spearman'].get('note','-')} |

## Known limitations to state in the paper

{chr(10).join('- ' + l for l in pipeline_metrics['known_limitations'])}

## How to read this table

Each row adds exactly one component. (B-A) = keyword lexicon's contribution,
(C-B) = topic relevance + actionability's contribution, (D-C) = LLM softmax
blending's contribution. A negative or near-zero delta is a legitimate
finding - it means that component isn't earning its complexity here.
"""
    with open(os.path.join(EVAL_DIR, "full_pipeline_comparison.md"), "w") as f:
        f.write(md)

# ===========================================================================
# PART 3 - REMAINING PLOTS
# ===========================================================================

with step("generate eval/plots/ablation_bars.png"):
    labels = ["Random", "A: ML only", "B: +keyword", "C: +topic/action", "D: Full pipeline"]
    ndcg_vals = [
        pipeline_metrics["random_baseline_ndcg_at_10"],
        pipeline_metrics["A_ml_only"]["ndcg_at_10"],
        pipeline_metrics["B_ml_keyword"]["ndcg_at_10"],
        pipeline_metrics["C_ml_keyword_topic_action"]["ndcg_at_10"],
        pipeline_metrics["D_full_pipeline"]["ndcg_at_10"],
    ]
    spearman_vals = [
        0,
        pipeline_metrics["A_ml_only"]["spearman_rho"],
        pipeline_metrics["B_ml_keyword"]["spearman_rho"],
        pipeline_metrics["C_ml_keyword_topic_action"]["spearman_rho"],
        pipeline_metrics["D_full_pipeline"]["spearman_rho"],
    ]
    colors = ["#999999", "#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, ndcg_vals, color=colors)
    axes[0].set_title("NDCG@10 by configuration")
    axes[0].set_ylabel("NDCG@10")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].axhline(pipeline_metrics["random_baseline_ndcg_at_10"], color="gray", linestyle="--", linewidth=1)
    axes[1].bar(labels, spearman_vals, color=colors)
    axes[1].set_title("Spearman rho by configuration")
    axes[1].set_ylabel("Spearman rho (vs true rank)")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=1)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "ablation_bars.png"), dpi=150)
    plt.close()

with step("generate eval/plots/llm_weight_distribution.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(w_llm_all, bins=30, color="#8172B2", edgecolor="white")
    plt.axvline(0.5, color="black", linestyle="--", linewidth=1, label="w_llm = 0.5 (equal weight)")
    plt.title("Distribution of softmax w_llm across all scored headlines")
    plt.xlabel("w_llm (weight given to LLM term)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "llm_weight_distribution.png"), dpi=150)
    plt.close()

with step("generate eval/plots/daily_ndcg_boxplot.png"):
    plt.figure(figsize=(6, 5))
    plt.boxplot([daily_ndcg_A, daily_ndcg_D], labels=["A: ML only", "D: Full pipeline"])
    plt.title("Per-day NDCG@10 spread: ML-only vs Full pipeline")
    plt.ylabel("NDCG@10")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "daily_ndcg_boxplot.png"), dpi=150)
    plt.close()

logger.info("=" * 60)
logger.info("ALL DONE. Check the eval/ folder:")
logger.info("  eval/metrics.json                  - individual model metrics")
logger.info("  eval/full_pipeline_metrics.json     - ablation metrics")
logger.info("  eval/full_pipeline_comparison.md    - ablation table + notes")
logger.info("  eval/plots/*.png                    - 4 diagnostic plots")
if pipeline_metrics["limit_days_active"]:
    logger.info("LIMIT_DAYS was active - quick sanity-check run only.")
    logger.info("Set LIMIT_DAYS = None at the top of this script and rerun for real numbers.")
logger.info("=" * 60)