# Experiment 3: XGBoost, CatBoost and a stacked blend

**Goal:** check whether different tree libraries make different mistakes, so that a blend beats the best single model.

## Run

```
uv run python src/train.py --model xgb --feats te --lr 0.03
uv run python src/train.py --model cat --feats te --lr 0.06
uv run python src/blend.py "*_te_s42"     # logistic regression on out-of-fold logits
```

## Results

| Model (same target-encoded features) | OOF AUC |
|---|---|
| LightGBM | 0.96031 |
| XGBoost | 0.96021 |
| CatBoost | 0.96008 |
| Mean of the three | 0.96038 |
| **Logistic-regression stack** | **0.96039** |

Stack weights: LightGBM 0.57, CatBoost 0.30, XGBoost 0.14.

## How the stack works

1. Every model predicts each training passenger while that passenger is held out (out-of-fold, same 5 folds: `StratifiedKFold(5, shuffle=True, random_state=42)`).
2. A logistic regression learns one weight per model from those held-out predictions, as log-odds.
3. The stack is scored with its own 5-fold CV and applied to each model's test predictions.

## Learnings

- **The three libraries barely blend (+0.0001).** On the same features they make nearly the same mistakes. Diversity has to come from different features or different *kinds* of model, not different tree libraries.
