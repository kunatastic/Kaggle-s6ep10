# Experiment 5: Encode every pair, plus triples

**Goal:** stop hand-picking the 45 pairs, and encode all of them plus the strongest triples.

## Run

```
uv run python src/train.py --model lgb --feats te --pairs all   # 21 singles + 210 pairs + 20 triples of the top 6
```

## Results

| Features | OOF AUC | Public | Private |
|---|---|---|---|
| 21 singles + 45 hand-picked pairs | 0.96031 | 0.95982 | 0.96073 |
| **21 singles + all 210 pairs + 20 triples** (251 groups, 279 features) | **0.96088** | **0.96027** | **0.96126** |

## Learnings

- **Encoding every pair beats hand-picking** (+0.0006 CV, +0.0005 on both boards, every fold improved). Pairs left out earlier, such as those with Flight Distance, the delays and the weaker ratings, still carried signal.
- Public 0.96027 ranks about 1,749 of 3,735. Tree models alone plateau around here (see experiment 8).
