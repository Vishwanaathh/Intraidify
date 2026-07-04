# Comparison with Prior Research

**Important caveat (state this in the paper too):** these are different task
formulations on different datasets, so treat this as contextual positioning,
not a like-for-like benchmark.

| Study | Task | Reported Result |
|---|---|---|
| LSTM + Word2Vec embeddings (sentiment survey) | Headline/tweet sentiment classification | 92.65% accuracy, F1 0.9437, ROC-AUC 0.9570 |
| Financial news headline sentiment + stock data (IIETA 2024) | Combined sentiment + prediction system | 83% forecasting accuracy when models combined |
| FTSE100 news + Twitter sentiment study | Sentiment vs. next-day market returns | Reported evidence of correlation between sentiment and next-day returns (no volatility link found) |
| Stacking ensemble: XGBoost + RF + GBM + LR meta-learner | Binary FinBERT sentiment label prediction | Ensemble outperformed individual base learners |
| **This work (IntrAIdify)** | Headline importance ranking (Spearman/NDCG) + directional signal check | Spearman rho = 0.1391 (Linear), 0.1191 (RF); directional point-biserial r = 0.0778 (p=0.1213) |

## Baseline context (report these alongside the headline numbers, not instead of them)

| Metric | Model | Random-guess baseline | Lift over baseline |
|---|---|---|---|
| NDCG@10 | Linear Regression: 0.6601 | Random ranking: 0.6018 (+/- 0.0048) | +0.0582 |
| NDCG@10 | Random Forest: 0.6549 | Random ranking: 0.6018 (+/- 0.0048) | +0.0531 |
| Accuracy | XGBoost: 0.4158 | Majority-class guess: 0.4000 | +0.0158 |

## Honest framing for the paper

- Prior work mostly predicts market **direction/sentiment classification**
  with accuracy/F1. This project's core task is different: **ranking**
  headlines by importance, which is why Spearman rho and NDCG@10 are the
  primary metrics, not accuracy.
- The directional signal check is the closest analog to prior work's
  "does sentiment predict market movement" framing - report it plainly,
  including if it's weak. A weak/non-significant result here is consistent
  with mixed findings in the literature (e.g. the FTSE100 study found
  sentiment predicted returns but not volatility - signal strength varies
  by what you're trying to predict).
- Don't claim this system "beats" the cited studies - the tasks aren't the
  same. Position it as: an applied ensemble system for the DIFFERENT
  (and less-studied) problem of ranking news importance for an alert
  system, evaluated with ranking-appropriate metrics, with a supplementary
  directional check for continuity with the sentiment-prediction literature.
