# Experiment 9: Counts, more triples, route profiles and an original-data model

**Goal:** CPU-only feature ideas from a public post-competition write-up by a top-15% finisher: frequency counts for every group, more triples, Flight Distance treated as a route, and a model trained on the original data used as a feature.

## Run

```
uv run python src/train.py --model lgb --feats te --pairs all --triples 8 --counts --route --orig-pred \
    --lr 0.02 --leaves 127 --min-child 200 --colsample 0.3 --l2 5 --threads 10 --suffix _xfeat
uv run python src/blend.py "lgb_te_s42_allpairs*" "*_te_s42" "gen_*_s42"
```

| Flag | Adds |
|---|---|
| `--triples 8` | Triples among the top 8 columns: 56 instead of 20 |
| `--counts` | How often each value or combination appears in train + test, for every group (no labels involved) |
| `--route` | Per-Flight-Distance averages of the 13 ratings and age, plus the route's passenger count |
| `--orig-pred` | A LightGBM trained only on the 130K original passengers, predicting each competition row (original rows get cross-fitted values) |

## Results

| Model | OOF AUC | Public | Private |
|---|---|---|---|
| `orig_pred` on its own | 0.95305 | | |
| LightGBM t1 (experiment 8) | 0.96097 | | |
| **LightGBM with extended features** (618 features) | **0.96125** | | |
| **LR stack: all trees + transformers** | **0.96186** | **0.96176** | **0.96215** |

Extended-feature folds: 0.96143, 0.96035, 0.96184, 0.96148, 0.96121. Every fold beats t1.

## Learnings

- **+0.0003 for trees.** It ties the 5M transformer as the best single model.
- **The original data transfers as a model, not as rates.** A full model on the original 130K rows reaches 0.953 AUC on the competition data. It captures how the features combine, which per-value rates (experiment 4) can't.
- The stack now splits its weight between the new LightGBM (0.59) and the 5M transformer (0.46).
- Public 0.96176 is 0.00037 short of the top-10% cutoff (0.96213). Private 0.96215 is already above that number.
- Regularising the stack's logistic regression (C from 1 down to 0.001) makes no difference with about 10 members.
