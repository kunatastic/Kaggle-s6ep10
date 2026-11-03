# Experiment 4: Using the original dataset (dropped)

**Goal:** use the 129,880-row original Airline Passenger Satisfaction dataset that the synthetic data was generated from.

Data: `kaggle datasets download -d arseniyshutko/binary-aviation-satisfaction-129k`, unzipped into `orig/`. Its columns match the competition's exactly.

## Run

```
uv run python src/train.py --model lgb --feats te --orig te     # satisfaction rates from the original data as features
uv run python src/train.py --model lgb --feats te --orig rows   # original rows added to each fold's training set
```

## Results

| Variant | OOF AUC | vs experiment 2 |
|---|---|---|
| Original-data rates per value and pair (`--orig te`) | 0.96028 | −0.0000 |
| Original rows in training (`--orig rows`, stopped after 3 folds) | ~0.9552 | −0.005 |

## Learnings

- **As per-value rates, it adds nothing.** Each original rate is strong on its own (boarding × wifi: 0.8948 AUC), but the competition's own encodings come from 560k rows per fold against 130k, so they already measure the same rates more precisely.
- **As training rows, it hurts.** Real passengers follow sharper rules than the noisy synthetic ones. The model learns patterns that don't hold, and early stopping fires after about 107 trees.
- Revisited in experiment 9: a full model trained on the original data, used as **one prediction feature**, does transfer (0.953 AUC on its own).
- Memory: encoding ~250 column groups as strings over 1.1M rows would need more than 10 GB, so groups are stored as integer codes. The results are identical.
