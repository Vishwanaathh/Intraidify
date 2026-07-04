# Full Pipeline Ablation Study (Eval-Only Run)

Evaluated on 398 of 398 held-out test days so far
(338 new LLM calls made this run
).
**Full test set complete.**

## PRIMARY RESULT: Cascade Re-Ranking Quality (n = 378 days)

This is the metric that actually reflects the system's design: does the LLM
improve the ordering of the SAME top-10 headlines the ML ensemble already
selected? (Not: does the full blended score beat ML on all 25 headlines -
that conflates the LLM's contribution with 15 headlines it never touched.)

| Ranking source | Spearman rho (within top-10) | NDCG@10 (within top-10) |
|---|---|---|
| ML ensemble's own ordering | 0.0899 | 0.8606 |
| LLM-reranked ordering | 0.0745 | 0.8602 |

Paired significance (Wilcoxon, same 378 days):
- Spearman: {'n': 378, 'statistic': 32666.0, 'p_value': 0.3741485794234056, 'mean_diff': 0.015392015392015393, 'note': 'not significant at p<0.05'}
- NDCG@10: {'n': 378, 'statistic': 35412.0, 'p_value': 0.8494523250667765, 'mean_diff': 0.00036347104338691865, 'note': 'not significant at p<0.05'}


## Secondary: Full-List Ablation (for context only - see caveat above)

| Configuration | Spearman rho | NDCG@10 |
|---|---|---|
| Random baseline | - | 0.6064 |
| (A) ML ensemble only | 0.1454 | 0.6625 |
| (B) + trend-weighted keyword | 0.1275 | 0.6568 |
| (C) + topic relevance + actionability | 0.1227 | 0.6532 |
| (D) + adaptive softmax LLM blend (FULL) | 0.1236 | 0.6542 |

## LLM weight behavior

- Mean w_llm: 0.4186
- Median w_llm: 0.4294
- Fraction of headlines where LLM got >50% of the weight: 21.8%

## Known limitations

- text = title only (no summary field in the DJIA dataset) - this eval sees less text per headline than production does.
- recency() is a constant 1.0 here - documented no-op offline.
- For non-top-10 headlines, kog defaults to 0.5*0.5=0.25 (neutral placeholder), which softmax treats as a real competing score.
