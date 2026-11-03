# Experiment 2: Hand-made features vs target encoding

**Goal:** turn the EDA's interaction patterns into features that tree models can use.

## Run

```
uv run python src/train.py --model lgb --feats base   # rating stats, zero count, delay diff/max
uv run python src/train.py --model lgb --feats te     # + target encoding of 21 singles + 45 pairs
```

## Results

| Features | OOF AUC | Public | Private |
|---|---|---|---|
| Raw (experiment 1) | 0.95892 | 0.95836 | 0.95942 |
| + hand-made aggregates (`base`) | 0.95878 | | |
| + target encoding: 21 singles + 45 pairs of the top 10 (`te`) | **0.96031** | **0.95982** | **0.96073** |

## How target encoding works

Each value, or pair of values, is replaced by the satisfaction rate seen for it in the training data:

- `Online boarding = 5` → **0.88**, because 88% of those passengers were satisfied
- `boarding = 3 & wifi = 3` → **0.06**
- `boarding = 1 & wifi = 4` → **0.81**

One encoded feature states the rate of a whole cell of the interaction grid. A tree would otherwise need many splits to carve that cell out.

**Leakage guard:** the encoder is fitted only on each fold's training part, and sklearn's `TargetEncoder` cross-fits inside that part too. No passenger's encoding ever uses their own label.

## Learnings

- **Hand-made aggregates don't help trees** (−0.0001). Trees already approximate sums and zero counts through splits.
- **Target encoding helps on every fold** (+0.0014 CV, +0.0015 public), because it hands the model the non-additive pair patterns directly.
- LightGBM rejects JSON special characters such as `:` in feature names, so encoded columns are named `te_0…te_N`.
