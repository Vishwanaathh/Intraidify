# -*- coding: utf-8 -*-
"""
IntrAIdify - EVAL-ONLY script. Loads already-trained models from ../models/
and re-runs evaluation (Parts 1 and 2 from train_all.py) WITHOUT retraining
anything. Use this after train_all.py has been run at least once, whenever
you just want to re-test the pipeline without waiting for Random Forest/
XGBoost/LightGBM to retrain from scratch.

LLM PROVIDER: uses gemini-3.1-flash-lite. gemini-2.0-flash was retired
June 1, 2026. No new setup needed beyond your existing GEMINI_API_KEY.

Run this from inside model_gens/ (same place as train_all.py), with
../models/ already populated. Requires network (kagglehub, to rebuild the
identical time-based test split and get true labels) and a GEMINI_API_KEY
environment variable set for Part 2.

Produces the same outputs as train_all.py's Parts 1-3:
  eval/metrics.json                   - individual model metrics
  eval/full_pipeline_metrics.json     - ablation metrics (cumulative across all runs so far)
  eval/full_pipeline_comparison.md    - ablation table + notes (now leads with the CASCADE result)
  eval/full_pipeline_checkpoint.json  - which days are done + their results (DO NOT DELETE - this is how resuming works)
  eval/plots/*.png                    - 4 diagnostic plots

CASCADE METRIC (the actual research question): the system's design is a
two-stage cascade - the ML ensemble cheaply pre-filters to the top-10
headlines, and the LLM's real job is to re-rank WITHIN that pre-filtered
set, not to out-rank the ML ensemble across all 25 headlines (most of
which the LLM never even sees). Full-list Spearman/NDCG (configs A-D)
dilute the LLM's true contribution with 15 headlines it had no say in.
This script also computes and reports "cascade_top10": for each day,
it compares (a) the ML ensemble's own ordering of its own top-10 picks
against (b) the LLM-reranked ordering of that SAME top-10, both scored
against the true importance of just those 10 headlines. This isolates
exactly what the LLM re-ranking step is actually contributing.

BUG FIX (configuration E): the production formula (configuration D) has a
real bug - for the 15 headlines OUTSIDE the top-10, the LLM never scores
them, so impact/confidence default to 0.5/0.5 (kog=0.25). The production
code still runs softmax(kog=0.25, m) for these headlines, letting a
completely fake constant compete with the real ML score. If a headline's
real (normalized) ML score is below 0.25, the fake placeholder can
actually outweigh it, injecting noise into most of every batch.
Configuration E fixes this: non-top-10 headlines just use the ML score
directly (w_ml=1, no softmax), computed from the exact same LLM calls as
D, so it costs zero extra API quota within a single day's processing.
IMPORTANT: because D and E are computed together per-day, any day already
recorded in the checkpoint WITHOUT an "E" field will be automatically
reprocessed (this DOES cost a fresh LLM call for that day, since raw
per-headline LLM outputs aren't cached across runs - only the aggregated
day-level metrics are). This means the first run after this update will
effectively redo the whole checkpoint once to backfill E; after that,
resuming works with zero extra cost as before.

QUOTA NOTE: MAX_LLM_CALLS_PER_RUN below controls how many new LLM calls
happen per run, then the script stops and saves a checkpoint. Given the
one-time backfill above, budget for reprocessing your full existing
checkpoint (e.g. 398 days) rather than just new days.
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

from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import ndcg_score, accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MAX_LLM_CALLS_PER_RUN = 400  # gemini-2.5-flash-lite free tier: 1,000 req/day, comfortably
                              # covers the full ~398-day test set in one run.
SLEEP_BETWEEN_CALLS = 4.5    # gemini-2.5-flash-lite free tier caps at 15 requests/MINUTE
                              # (not just per day). 1.0s = 60 calls/min would blow through
                              # that and start hitting 429 errors partway through a long run.
                              # 4.5s gives ~13.3 calls/min, safely under the 15 RPM ceiling.
CHECKPOINT_EVERY = 5  # save progress every N days, in case it dies partway

EVAL_DIR = "eval"
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------

logger = logging.getLogger("eval_only")
logger.setLevel(logging.INFO)
logger.handlers.clear()
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(fmt)
logger.addHandler(console_handler)
file_handler = logging.FileHandler(os.path.join(EVAL_DIR, "eval_only_log.txt"), mode="w")
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


metrics = {}

# ===========================================================================
# LOAD PRE-TRAINED MODELS (no fitting - this is the whole point of this script)
# ===========================================================================

with step("load pre-trained models + vectorizers from ../models/"):
    model_lin = joblib.load("../models/model.pkl")
    vectorizer_lin = joblib.load("../models/vectorizer.pkl")
    model_rf = joblib.load("../models/forestmodelreg.pkl")
    vectorizer_rf = joblib.load("../models/vectorizerr.pkl")
    model_xgb = joblib.load("../models/xgb_classifier.pkl")
    vectorizer_xgb = joblib.load("../models/vectorizer_xgb.pkl")
    model_light = joblib.load("../models/lightmodel.pkl")
    vectorizer_light = joblib.load("../models/lightvectorizer.pkl")
    logger.info("  all 4 models + 4 vectorizers loaded successfully")

# ===========================================================================
# PART 1 - REBUILD TEST SPLIT + EVALUATE LOADED MODELS (no training)
# ===========================================================================

with step("import kagglehub + download DJIA dataset"):
    import kagglehub
    path = kagglehub.dataset_download("tanishqdublish/stock-market-predictions")

with step("load + time-sort DJIA dataset, rebuild identical split"):
    data = pd.read_csv(path + "/Combined_News_DJIA.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(data) * 0.8)
    train_df = data.iloc[:split_idx].reset_index(drop=True)  # needed only for majority-class baseline
    test_df = data.iloc[split_idx:].reset_index(drop=True)
    logger.info(f"  test days: {len(test_df)} ({test_df['Date'].min()} -> {test_df['Date'].max()})")


def rows_to_texts_scores(df):
    texts, scores, day_idx = [], [], []
    for row_i, row in df.iterrows():
        for i in range(1, 26):
            text = str(row[f"Top{i}"]).replace("b'", "").replace('b"', "").strip()
            texts.append(text)
            scores.append(26 - i)
            day_idx.append(row_i)
    return texts, scores, day_idx


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


with step("flatten test headlines + predict with loaded models (no fitting)"):
    test_texts, test_scores, test_day_idx = rows_to_texts_scores(test_df)
    _, train_labels_xgb, _ = rows_to_texts_labels(train_df)  # only for majority-class baseline
    test_texts_xgb, test_labels_xgb, _ = rows_to_texts_labels(test_df)

    X_test_lin = vectorizer_lin.transform(test_texts)
    preds_lin = model_lin.predict(X_test_lin)

    X_test_rf = vectorizer_rf.transform(test_texts)
    preds_rf = model_rf.predict(X_test_rf)

    X_test_xgb = vectorizer_xgb.transform(test_texts_xgb)
    preds_xgb_class = model_xgb.predict(X_test_xgb)

with step("download + split SBI sentiment dataset, predict with loaded LightGBM"):
    sent_path = kagglehub.dataset_download("parshantkumar2033/sentiment-intensity-for-news-articles-of-sbi")
    sent_data = pd.read_csv(sent_path + "/dataset.csv").dropna(subset=["article"])
    sent_data = sent_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
    sent_split = int(len(sent_data) * 0.8)
    sent_test = sent_data.iloc[sent_split:]
    X_test_light = vectorizer_light.transform(sent_test["article"])
    y_test_light = sent_test["sentiment_score"].values
    preds_light = model_light.predict(X_test_light)

with step("evaluate Linear Regression"):
    test_df_eval = pd.DataFrame({"day": test_day_idx, "true_score": test_scores, "pred_score": preds_lin})
    sp, nd = [], []
    for day, grp in test_df_eval.groupby("day"):
        if grp["true_score"].nunique() > 1:
            rho, _ = spearmanr(grp["true_score"], grp["pred_score"])
            sp.append(rho)
        nd.append(ndcg_score([grp["true_score"].values], [grp["pred_score"].values], k=10))
    metrics["linear_regression"] = {"spearman_rho": float(np.nanmean(sp)), "ndcg_at_10": float(np.mean(nd))}
    logger.info(f"  {metrics['linear_regression']}")

with step("evaluate Random Forest"):
    test_df_eval_rf = pd.DataFrame({"day": test_day_idx, "true_score": test_scores, "pred_score": preds_rf})
    sp, nd = [], []
    for day, grp in test_df_eval_rf.groupby("day"):
        if grp["true_score"].nunique() > 1:
            rho, _ = spearmanr(grp["true_score"], grp["pred_score"])
            sp.append(rho)
        nd.append(ndcg_score([grp["true_score"].values], [grp["pred_score"].values], k=10))
    metrics["random_forest"] = {"spearman_rho": float(np.nanmean(sp)), "ndcg_at_10": float(np.mean(nd))}
    logger.info(f"  {metrics['random_forest']}")

with step("evaluate random baseline (NDCG@10 floor)"):
    rng0 = np.random.default_rng(42)
    trials = []
    for _ in range(20):
        vals = []
        for day, grp in test_df_eval.groupby("day"):
            vals.append(ndcg_score([grp["true_score"].values], [rng0.random(len(grp))], k=10))
        trials.append(np.mean(vals))
    metrics["random_baseline"] = {"ndcg_at_10_mean": float(np.mean(trials)), "ndcg_at_10_std": float(np.std(trials))}
    logger.info(f"  {metrics['random_baseline']}")

with step("evaluate XGBoost classifier (vs. majority-class baseline)"):
    majority_class = Counter(train_labels_xgb).most_common(1)[0][0]
    majority_acc = accuracy_score(test_labels_xgb, [majority_class] * len(test_labels_xgb))
    metrics["xgboost_classifier"] = {
        "accuracy": float(accuracy_score(test_labels_xgb, preds_xgb_class)),
        "macro_f1": float(f1_score(test_labels_xgb, preds_xgb_class, average="macro")),
        "majority_class_baseline_accuracy": float(majority_acc),
    }
    logger.info(f"  {metrics['xgboost_classifier']}")

with step("evaluate LightGBM"):
    try:
        rmse_light = mean_squared_error(y_test_light, preds_light, squared=False)
    except TypeError:
        rmse_light = float(np.sqrt(mean_squared_error(y_test_light, preds_light)))
    metrics["lightgbm_sentiment"] = {"rmse": float(rmse_light), "r2": float(r2_score(y_test_light, preds_light))}
    logger.info(f"  {metrics['lightgbm_sentiment']}")

with step("write eval/metrics.json"):
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
# PART 2 - FULL PIPELINE ABLATION (uses the same loaded models, no reload)
# ===========================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not set. Set it as an environment variable before running Part 2.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
LLM_MODEL = "gemini-3.1-flash-lite"  # newer generation than 2.5-flash-lite; free tier is
                                      # similarly generous (~15 RPM). SLEEP_BETWEEN_CALLS
                                      # below is already set conservatively for this ceiling.

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
    return 1.0  # documented no-op offline - see paper Section III-H


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


def llm_analyze_news(headlines):
    def fallback(n):
        return [{"impact_score": 0.5, "market_direction": "neutral", "event_type": "other", "confidence": 0.5} for _ in range(n)]

    prompt = (
        "You are a financial analyst ranking news based on market impact. "
        "Respond with a JSON object of the form "
        '{"results": [{"impact_score": 0-1, "market_direction": "up/down/neutral", '
        '"event_type": "string", "confidence": 0-1}, ...]}, '
        "with exactly one entry per headline below, in the same order. "
        "No explanation, only the JSON object.\n\nHeadlines:\n"
    )
    for i, h in enumerate(headlines):
        prompt += f"{i}. {h}\n"

    max_retries = 4
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config={"response_mime_type": "application/json"},  # forces valid JSON back
            )
            text = response.text.strip()
            parsed = json.loads(text)
            data = parsed.get("results", parsed if isinstance(parsed, list) else [])

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
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "rate limit" in str(e).lower()
            if is_rate_limit and attempt < max_retries - 1:
                backoff = 10 * (attempt + 1)  # 10s, 20s, 30s
                logger.warning(f"  Rate limited (attempt {attempt+1}/{max_retries}), backing off {backoff}s: {e}")
                time.sleep(backoff)
                continue
            logger.warning(f"  LLM analyze error: {e} - using neutral fallback")
            return fallback(len(headlines))


with step("prep actionability scorer (reuses loaded vectorizer_lin)"):
    actionability_score = make_actionability_fn(vectorizer_lin)

with step("build day keys for the full test set (quota decides how many get processed today)"):
    # Use the actual date as the resume key, not a positional index - robust
    # to re-runs even if the underlying data pull order ever changes.
    pipeline_test_df = test_df.copy()
    pipeline_test_df["day_key"] = pipeline_test_df["Date"].dt.strftime("%Y-%m-%d")
    logger.info(f"  full test set: {len(pipeline_test_df)} days")

checkpoint_path = os.path.join(EVAL_DIR, "full_pipeline_checkpoint.json")

with step("load existing checkpoint (resume support)"):
    completed_days = {}  # day_key -> {"A": {...}, "B": {...}, "C": {...}, "D": {...}}
    all_w_llm = []
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as f:
            saved = json.load(f)
        completed_days = saved.get("completed_days", {})
        all_w_llm = saved.get("w_llm_all", [])
        logger.info(f"  resuming: {len(completed_days)}/{len(pipeline_test_df)} days already done")
    else:
        logger.info("  no checkpoint found - starting fresh")

configs = ["A_ml_only", "B_ml_keyword", "C_ml_keyword_topic_action", "D_full_pipeline"]
random_results_pipeline = []
rng = np.random.default_rng(42)

calls_made_this_run = 0
llm_looks_broken = False

with step(f"run pipeline ablation, up to {MAX_LLM_CALLS_PER_RUN} new LLM calls this run"):
    for day_i, row in pipeline_test_df.iterrows():
        day_key = row["day_key"]

        if day_key in completed_days and "E" in completed_days[day_key]:
            continue  # already done, with the bug-fix field present - skip, no API call spent

        if calls_made_this_run >= MAX_LLM_CALLS_PER_RUN:
            logger.info(f"  hit {MAX_LLM_CALLS_PER_RUN}-call quota for this run - stopping here")
            break

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

        trend = compute_trend(headlines)
        kw_scores = np.array([keyword_score(h, trend) for h in headlines])
        combined_b = ml + 0.3 * kw_scores
        rho_b, _ = spearmanr(true_scores, combined_b) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_b = ndcg_score([true_scores], [combined_b], k=10)

        topic_scores = topic_relevance(X1)
        act_scores = normalize([actionability_score(h) for h in headlines])
        combined_c = ml + 0.3 * kw_scores + 0.15 * topic_scores + 0.20 * act_scores
        rho_c, _ = spearmanr(true_scores, combined_c) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_c = ndcg_score([true_scores], [combined_c], k=10)

        order = np.argsort(-ml)
        top_k = min(10, len(headlines))
        top_idx = order[:top_k]
        top_headlines = [headlines[i] for i in top_idx]

        # --- this is the LLM call that counts against quota ---
        llm_results = llm_analyze_news(top_headlines)
        calls_made_this_run += 1
        time.sleep(SLEEP_BETWEEN_CALLS)

        # Fail fast: if the very first call this run looks like a fallback
        # (both values exactly 0.5), the key/model is probably broken - stop
        # immediately instead of burning the rest of today's quota on noise.
        if calls_made_this_run == 1:
            first = llm_results[0]
            if first["impact_score"] == 0.5 and first["confidence"] == 0.5:
                logger.error(
                    "  First LLM call returned fallback values (0.5/0.5) - "
                    "the API key or model is likely broken. Check GEMINI_API_KEY "
                    "and the LLM_MODEL setting availability before continuing. "
                    "Stopping now to avoid wasting today's quota."
                )
                llm_looks_broken = True
                break
            else:
                logger.info(f"  LLM call 1 succeeded: {first}")

        llm_by_idx = {idx: llm_results[pos] for pos, idx in enumerate(top_idx)}

        final_scores = np.zeros(len(headlines))       # D: production formula, bug included
        final_scores_fixed = np.zeros(len(headlines))  # E: bug fixed, same LLM calls, no extra cost
        day_w_llm = []
        for i in range(len(headlines)):
            m = ml[i]
            aux = (
                0.15 * keyword_score(headlines[i], trend)
                + 0.15 * recency("")
                + 0.15 * topic_scores[i]
                + 0.20 * act_scores[i]
            )
            if i in llm_by_idx:
                # Real LLM opinion exists for this headline - softmax blend is legitimate here.
                impact = llm_by_idx[i]["impact_score"]
                conf = llm_by_idx[i]["confidence"]
                kog = impact * conf
                w_llm, w_ml = softmax(kog, m)
                day_w_llm.append(w_llm)
                final_scores[i] = w_ml * m + w_llm * kog + aux
                final_scores_fixed[i] = w_ml * m + w_llm * kog + aux  # same - no bug here
            else:
                # No real LLM opinion. D (production, buggy) still runs softmax against a
                # fake kog=0.25 constant, letting it compete with m. E (fixed) just uses m,
                # since there is nothing real to blend for this headline.
                impact, conf = 0.5, 0.5
                kog = impact * conf
                w_llm, w_ml = softmax(kog, m)
                final_scores[i] = w_ml * m + w_llm * kog + aux       # D: bug reproduced faithfully
                final_scores_fixed[i] = m + aux                       # E: bug fixed, m used directly
        final_scores = normalize(final_scores)
        final_scores_fixed = normalize(final_scores_fixed)
        rho_d, _ = spearmanr(true_scores, final_scores) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_d = ndcg_score([true_scores], [final_scores], k=10)
        rho_e, _ = spearmanr(true_scores, final_scores_fixed) if len(set(true_scores)) > 1 else (np.nan, None)
        ndcg_e = ndcg_score([true_scores], [final_scores_fixed], k=10)

        # --- CASCADE-SPECIFIC METRIC: does the LLM improve ordering WITHIN ---
        # --- the top-10 the ML ensemble already selected? This is the real ---
        # --- research question - the LLM never sees or affects items      ---
        # --- outside top_idx, so full-list Spearman/NDCG (above) dilutes  ---
        # --- its true contribution with 15 headlines it had no say in.    ---
        top10_true = true_scores[top_idx]  # ground-truth importance of just these 10
        top10_ml_only = ml[top_idx]  # how the ML ensemble alone ordered them
        top10_final = final_scores[top_idx]  # how the LLM-blended pipeline ordered them

        if len(set(top10_true)) > 1:
            rho_top10_ml, _ = spearmanr(top10_true, top10_ml_only)
            rho_top10_llm, _ = spearmanr(top10_true, top10_final)
        else:
            rho_top10_ml, rho_top10_llm = np.nan, np.nan
        ndcg_top10_ml = ndcg_score([top10_true], [top10_ml_only], k=10)
        ndcg_top10_llm = ndcg_score([top10_true], [top10_final], k=10)

        random_score = ndcg_score([true_scores], [rng.random(len(headlines))], k=10)

        # Record this day's full result set into the resumable checkpoint
        completed_days[day_key] = {
            "cascade_top10": {
                "rho_ml_only": rho_top10_ml, "rho_llm_reranked": rho_top10_llm,
                "ndcg_ml_only": ndcg_top10_ml, "ndcg_llm_reranked": ndcg_top10_llm,
            },
            "A": {"spearman": rho_a, "ndcg_10": ndcg_a},
            "B": {"spearman": rho_b, "ndcg_10": ndcg_b},
            "C": {"spearman": rho_c, "ndcg_10": ndcg_c},
            "D": {"spearman": rho_d, "ndcg_10": ndcg_d},
            "E": {"spearman": rho_e, "ndcg_10": ndcg_e},  # bug-fixed version of D, same LLM calls
            "random_ndcg_10": random_score,
        }
        all_w_llm.extend(day_w_llm)

        if calls_made_this_run % CHECKPOINT_EVERY == 0:
            logger.info(f"  {calls_made_this_run}/{MAX_LLM_CALLS_PER_RUN} calls used this run "
                        f"({len(completed_days)}/{len(pipeline_test_df)} days total)")
            with open(checkpoint_path, "w") as f:
                json.dump({"completed_days": completed_days, "w_llm_all": all_w_llm}, f, indent=2,
                          default=lambda x: None if pd.isna(x) else x)

    # final checkpoint save regardless of how the loop ended
    with open(checkpoint_path, "w") as f:
        json.dump({"completed_days": completed_days, "w_llm_all": all_w_llm}, f, indent=2,
                  default=lambda x: None if pd.isna(x) else x)
    logger.info(f"  this run: {calls_made_this_run} new LLM calls made")
    logger.info(f"  cumulative: {len(completed_days)}/{len(pipeline_test_df)} days evaluated so far")

with step("aggregate pipeline ablation results (cumulative, all completed days)"):
    def agg(key):
        rhos = [d[key]["spearman"] for d in completed_days.values() if key in d and not pd.isna(d[key]["spearman"])]
        ndcgs = [d[key]["ndcg_10"] for d in completed_days.values() if key in d]
        return {"spearman_rho": float(np.mean(rhos)) if rhos else None,
                "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else None}

    daily_ndcg_A = [d["A"]["ndcg_10"] for d in completed_days.values()]
    daily_ndcg_D = [d["D"]["ndcg_10"] for d in completed_days.values()]
    random_results_pipeline = [d["random_ndcg_10"] for d in completed_days.values()]

    # E only exists for days evaluated AFTER this fix was added - older
    # checkpoint entries won't have it, so agg() above already skips those.
    n_fixed_days = sum(1 for d in completed_days.values() if "E" in d)

    # Cascade metric only exists for days evaluated AFTER this feature was added -
    # older checkpoint entries (from before this update) won't have it, so skip those.
    cascade_days = [d["cascade_top10"] for d in completed_days.values() if "cascade_top10" in d]
    n_cascade_days = len(cascade_days)

    def agg_cascade(field):
        vals = [d[field] for d in cascade_days if not pd.isna(d[field])]
        return float(np.mean(vals)) if vals else None

    pipeline_metrics = {
        "n_test_days_evaluated": len(completed_days),
        "n_test_days_total": len(pipeline_test_df),
        "n_cascade_days_evaluated": n_cascade_days,
        "n_fixed_days_evaluated": n_fixed_days,
        "llm_calls_made_this_run": calls_made_this_run,
        "llm_looked_broken": llm_looks_broken,
        "A_ml_only": agg("A"),
        "B_ml_keyword": agg("B"),
        "C_ml_keyword_topic_action": agg("C"),
        "D_full_pipeline": agg("D"),
        "E_full_pipeline_fixed": agg("E"),
        "random_baseline_ndcg_at_10": float(np.mean(random_results_pipeline)) if random_results_pipeline else None,
        "w_llm_mean": float(np.mean(all_w_llm)) if all_w_llm else None,
        "w_llm_median": float(np.median(all_w_llm)) if all_w_llm else None,
        "w_llm_fraction_above_0.5": float(np.mean(np.array(all_w_llm) > 0.5)) if all_w_llm else None,
        "cascade_top10": {
            "rho_ml_only": agg_cascade("rho_ml_only"),
            "rho_llm_reranked": agg_cascade("rho_llm_reranked"),
            "ndcg_ml_only": agg_cascade("ndcg_ml_only"),
            "ndcg_llm_reranked": agg_cascade("ndcg_llm_reranked"),
        },
    }

    if n_cascade_days > 0:
        logger.info(
            f"  CASCADE TOP-10 RESULT ({n_cascade_days} days): "
            f"ML-only rho={pipeline_metrics['cascade_top10']['rho_ml_only']:.4f}, "
            f"LLM-reranked rho={pipeline_metrics['cascade_top10']['rho_llm_reranked']:.4f} | "
            f"ML-only NDCG@10={pipeline_metrics['cascade_top10']['ndcg_ml_only']:.4f}, "
            f"LLM-reranked NDCG@10={pipeline_metrics['cascade_top10']['ndcg_llm_reranked']:.4f}"
        )

with step("paired significance tests (Wilcoxon signed-rank, same days across configs)"):
    def paired_test(key_a, key_b, metric):
        pairs = [(d[key_a][metric], d[key_b][metric]) for d in completed_days.values()
                 if key_a in d and key_b in d]
        pairs = [(a, b) for a, b in pairs if not (pd.isna(a) or pd.isna(b))]
        if len(pairs) < 6:
            return {"n": len(pairs), "statistic": None, "p_value": None,
                    "note": "too few paired days for a reliable test (need >=6)"}
        a_arr, b_arr = zip(*pairs)
        diffs = np.array(a_arr) - np.array(b_arr)
        if np.all(diffs == 0):
            return {"n": len(pairs), "statistic": None, "p_value": 1.0, "note": "identical values across all days"}
        try:
            stat, pv = wilcoxon(a_arr, b_arr)
            return {"n": len(pairs), "statistic": float(stat), "p_value": float(pv),
                    "mean_diff": float(np.mean(diffs)),
                    "note": "significant at p<0.05" if pv < 0.05 else "not significant at p<0.05"}
        except Exception as e:
            return {"n": len(pairs), "statistic": None, "p_value": None, "note": f"test failed: {e}"}

    comparisons = [
        ("A", "B", "A vs B: does keyword weighting help?"),
        ("B", "C", "B vs C: does topic/actionability help?"),
        ("C", "D", "C vs D: does LLM softmax blend help?"),
        ("A", "D", "A vs D: does the full pipeline beat plain ML?"),
        ("D", "E", "D vs E: does fixing the neutral-placeholder softmax bug change anything?"),
        ("A", "E", "A vs E: does the BUG-FIXED full pipeline beat plain ML?"),
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

    # --- THE KEY TEST: does LLM re-ranking beat ML-only ordering, WITHIN ---
    # --- the same 10 ML-selected candidates? This is the cascade's actual claim. ---
    def paired_test_cascade(field_a, field_b):
        pairs = [(d[field_a], d[field_b]) for d in cascade_days
                 if not (pd.isna(d[field_a]) or pd.isna(d[field_b]))]
        if len(pairs) < 6:
            return {"n": len(pairs), "statistic": None, "p_value": None,
                    "note": "too few paired days for a reliable test (need >=6)"}
        a_arr, b_arr = zip(*pairs)
        diffs = np.array(a_arr) - np.array(b_arr)
        if np.all(diffs == 0):
            return {"n": len(pairs), "statistic": None, "p_value": 1.0, "note": "identical values across all days"}
        try:
            stat, pv = wilcoxon(a_arr, b_arr)
            return {"n": len(pairs), "statistic": float(stat), "p_value": float(pv),
                    "mean_diff": float(np.mean(diffs)),
                    "note": "significant at p<0.05" if pv < 0.05 else "not significant at p<0.05"}
        except Exception as e:
            return {"n": len(pairs), "statistic": None, "p_value": None, "note": f"test failed: {e}"}

    cascade_sig = {
        "description": "Cascade test: LLM re-ranking of ML's own top-10 vs. ML's own ordering of that top-10",
        "spearman": paired_test_cascade("rho_ml_only", "rho_llm_reranked"),
        "ndcg_10": paired_test_cascade("ndcg_ml_only", "ndcg_llm_reranked"),
    }
    significance_results["cascade_ml_vs_llm_reranked"] = cascade_sig
    logger.info(f"  {cascade_sig['description']}")
    logger.info(f"    Spearman:  {cascade_sig['spearman']}")
    logger.info(f"    NDCG@10:   {cascade_sig['ndcg_10']}")

    pipeline_metrics["significance_tests"] = significance_results
    pipeline_metrics["significance_test_note"] = (
        "Wilcoxon signed-rank test, paired by day, computed over all days "
        "evaluated so far (cumulative across runs). With few days completed "
        "these tests have low statistical power - the more runs you "
        "accumulate (at 20 days/day), the more reliable these p-values "
        "become. Check n_test_days_evaluated vs n_test_days_total below."
    )
    pipeline_metrics["known_limitations"] = [
        "text = title only (no summary field in the DJIA dataset) - this eval "
        "sees less text per headline than production does.",
        "recency() is a constant 1.0 here - documented no-op offline.",
        "For non-top-10 headlines in configuration D (production formula), "
        "kog defaults to 0.5*0.5=0.25 (neutral placeholder), which softmax "
        "treats as a real competing score. Configuration E fixes this - see "
        "the D vs E comparison below - using the exact same LLM calls at no "
        "extra cost, since only the non-top-10 math changes.",
    ]

    with open(os.path.join(EVAL_DIR, "full_pipeline_metrics.json"), "w") as f:
        json.dump(pipeline_metrics, f, indent=2)
    logger.info(json.dumps(pipeline_metrics, indent=2))

with step("write eval/full_pipeline_comparison.md"):
    ct = pipeline_metrics["cascade_top10"]

    def fmt4(v):
        return f"{v:.4f}" if v is not None else "N/A"

    cascade_section = f"""## PRIMARY RESULT: Cascade Re-Ranking Quality (n = {pipeline_metrics['n_cascade_days_evaluated']} days)

This is the metric that actually reflects the system's design: does the LLM
improve the ordering of the SAME top-10 headlines the ML ensemble already
selected? (Not: does the full blended score beat ML on all 25 headlines -
that conflates the LLM's contribution with 15 headlines it never touched.)

| Ranking source | Spearman rho (within top-10) | NDCG@10 (within top-10) |
|---|---|---|
| ML ensemble's own ordering | {fmt4(ct['rho_ml_only'])} | {fmt4(ct['ndcg_ml_only'])} |
| LLM-reranked ordering | {fmt4(ct['rho_llm_reranked'])} | {fmt4(ct['ndcg_llm_reranked'])} |

Paired significance (Wilcoxon, same {pipeline_metrics['n_cascade_days_evaluated']} days):
- Spearman: {pipeline_metrics['significance_tests']['cascade_ml_vs_llm_reranked']['spearman']}
- NDCG@10: {pipeline_metrics['significance_tests']['cascade_ml_vs_llm_reranked']['ndcg_10']}

""" if pipeline_metrics['n_cascade_days_evaluated'] > 0 else (
        "## PRIMARY RESULT: Cascade Re-Ranking Quality\n\n"
        "No days evaluated yet under the cascade metric (added after earlier runs - "
        "old checkpoint days don't have this field). Rerun to start collecting it.\n\n"
    )

    ef = pipeline_metrics["E_full_pipeline_fixed"]
    bug_fix_section = f"""## BUG-FIX CHECK: Does the neutral-placeholder softmax bug matter? (n = {pipeline_metrics['n_fixed_days_evaluated']} days)

Configuration D is the production formula as coded (softmax runs against a
fake kog=0.25 for the 15 non-top-10 headlines every batch). Configuration E
is the same pipeline with that specific bug fixed (non-top-10 headlines use
the ML score directly, no fake softmax competition) - computed from the
exact same LLM calls as D, so this costs zero extra API quota.

| Configuration | Spearman rho | NDCG@10 |
|---|---|---|
| (D) Production formula (bug included) | {fmt4(pipeline_metrics['D_full_pipeline']['spearman_rho'])} | {fmt4(pipeline_metrics['D_full_pipeline']['ndcg_at_10'])} |
| (E) Bug fixed | {fmt4(ef['spearman_rho'])} | {fmt4(ef['ndcg_at_10'])} |

Paired significance (D vs E, same days):
- Spearman: {pipeline_metrics['significance_tests'].get('D_vs_E', {}).get('spearman', 'N/A')}
- NDCG@10: {pipeline_metrics['significance_tests'].get('D_vs_E', {}).get('ndcg_10', 'N/A')}

Paired significance (A vs E - does the BUG-FIXED pipeline beat plain ML?):
- Spearman: {pipeline_metrics['significance_tests'].get('A_vs_E', {}).get('spearman', 'N/A')}
- NDCG@10: {pipeline_metrics['significance_tests'].get('A_vs_E', {}).get('ndcg_10', 'N/A')}

"""

    md = f"""# Full Pipeline Ablation Study (Eval-Only Run)

Evaluated on {pipeline_metrics['n_test_days_evaluated']} of {pipeline_metrics['n_test_days_total']} held-out test days so far
({pipeline_metrics['llm_calls_made_this_run']} new LLM calls made this run
{'- LLM KEY LOOKS BROKEN, check GEMINI_API_KEY' if pipeline_metrics['llm_looked_broken'] else ''}).
{'**Not yet complete** - rerun this script (e.g. tomorrow, once quota resets) to cover more days.' if pipeline_metrics['n_test_days_evaluated'] < pipeline_metrics['n_test_days_total'] else '**Full test set complete.**'}

{cascade_section}
{bug_fix_section}
## Secondary: Full-List Ablation (for context only - see caveat above)

| Configuration | Spearman rho | NDCG@10 |
|---|---|---|
| Random baseline | - | {pipeline_metrics['random_baseline_ndcg_at_10']:.4f} |
| (A) ML ensemble only | {pipeline_metrics['A_ml_only']['spearman_rho']:.4f} | {pipeline_metrics['A_ml_only']['ndcg_at_10']:.4f} |
| (B) + trend-weighted keyword | {pipeline_metrics['B_ml_keyword']['spearman_rho']:.4f} | {pipeline_metrics['B_ml_keyword']['ndcg_at_10']:.4f} |
| (C) + topic relevance + actionability | {pipeline_metrics['C_ml_keyword_topic_action']['spearman_rho']:.4f} | {pipeline_metrics['C_ml_keyword_topic_action']['ndcg_at_10']:.4f} |
| (D) + adaptive softmax LLM blend (production, bug included) | {pipeline_metrics['D_full_pipeline']['spearman_rho']:.4f} | {pipeline_metrics['D_full_pipeline']['ndcg_at_10']:.4f} |
| (E) + adaptive softmax LLM blend (bug fixed) | {fmt4(ef['spearman_rho'])} | {fmt4(ef['ndcg_at_10'])} |

## LLM weight behavior

- Mean w_llm: {pipeline_metrics['w_llm_mean']:.4f}
- Median w_llm: {pipeline_metrics['w_llm_median']:.4f}
- Fraction of headlines where LLM got >50% of the weight: {pipeline_metrics['w_llm_fraction_above_0.5']:.1%}

## Known limitations

{chr(10).join('- ' + l for l in pipeline_metrics['known_limitations'])}
"""
    with open(os.path.join(EVAL_DIR, "full_pipeline_comparison.md"), "w") as f:
        f.write(md)

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
    plt.hist(all_w_llm, bins=30, color="#8172B2", edgecolor="white")
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
    try:
        # matplotlib >= 3.9 renamed the `labels` kwarg to `tick_labels`
        plt.boxplot([daily_ndcg_A, daily_ndcg_D], tick_labels=["A: ML only", "D: Full pipeline"])
    except TypeError:
        plt.boxplot([daily_ndcg_A, daily_ndcg_D], labels=["A: ML only", "D: Full pipeline"])
    plt.title("Per-day NDCG@10 spread: ML-only vs Full pipeline")
    plt.ylabel("NDCG@10")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "daily_ndcg_boxplot.png"), dpi=150)
    plt.close()

logger.info("=" * 60)
logger.info("ALL DONE (eval-only, no retraining occurred).")
logger.info(f"{calls_made_this_run} new LLM calls made this run.")
logger.info(f"Cumulative: {len(completed_days)}/{len(pipeline_test_df)} days evaluated.")
if llm_looks_broken:
    logger.info("LLM key looked broken - fix GEMINI_API_KEY and rerun before continuing.")
elif len(completed_days) < len(pipeline_test_df):
    logger.info(f"Not done yet - rerun this script once your quota resets to cover more days.")
else:
    logger.info("Full test set complete - these numbers are final.")
logger.info("=" * 60)