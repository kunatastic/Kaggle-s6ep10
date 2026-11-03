# Experiment 8: LightGBM tuning on all-pairs features

**Goal:** check how much tree performance hyperparameters leave on the table.

## Run

```
uv run python src/train.py --model lgb --feats te --pairs all --lr 0.02 --leaves 127 --min-child 200 --colsample 0.3 --l2 5  --threads 6 --suffix _t1
uv run python src/train.py --model lgb --feats te --pairs all --lr 0.02 --leaves 31  --min-child 50  --colsample 0.3 --l2 1  --threads 6 --suffix _t2
uv run python src/train.py --model lgb --feats te --pairs all --lr 0.02 --leaves 255 --min-child 400 --colsample 0.2 --l2 10 --threads 6 --suffix _t3
```

## Results

| Trial | Leaves | Min child | Colsample | L2 | LR | OOF AUC |
|---|---|---|---|---|---|---|
| Default (experiment 5) | 63 | 100 | 0.5 | 1 | 0.03 | 0.96088 |
| **t1** | 127 | 200 | 0.3 | 5 | 0.02 | **0.96097** |
| t2 | 31 | 50 | 0.3 | 1 | 0.02 | 0.96086 |
| t3 | 255 | 400 | 0.2 | 10 | 0.02 | 0.96087 |

## Learnings

- **Tuning is saturated: ±0.0001.** Fewer columns per tree and more regularisation help a little. The features, not the settings, limit the trees.
- Run LightGBM on fewer cores (`--threads 6`) to keep the Mac responsive. At full speed with ~280 features it pushed several GB into swap.
