# Experiment 7: Scaling the transformer to 5M parameters

**Goal:** v1 was undertrained and too weak to matter in the stack. Give it more capacity and more training.

## Run

```
KAGGLE_SLUG=s6e10-gen5m-a uv run python src/make_kaggle_job.py "0,1,4" "--dim 192 --layers 6 --epochs 20 --orders 16"
KAGGLE_SLUG=s6e10-gen5m-b uv run python src/make_kaggle_job.py "2,3"   "--dim 192 --layers 6 --epochs 20 --orders 16"
uv run kaggle kernels push -p kaggle_jobs/s6e10-gen5m-a --accelerator NvidiaTeslaT4   # same for -b
uv run python src/gen_collect.py gen_5m_s42 "kaggle_out/s6e10-gen5m-[ab]/preds/gen"
uv run python src/blend.py "lgb_te_s42_allpairs*" "*_te_s42" "gen_*_s42"
uv run python src/stack_gbdt.py   # CatBoost stacker, for comparison
```

## Results

| Fold | Best tree (LightGBM t1) | Transformer 5M |
|---|---|---|
| 0 | 0.96122 | **0.96150** |
| 1 | 0.96002 | **0.96029** |
| 2 | 0.96159 | **0.96192** |
| 3 | 0.96117 | **0.96163** |
| 4 | 0.96093 | 0.96095 |

| Model | OOF AUC | Public | Private |
|---|---|---|---|
| Transformer 5M: 192 × 6 layers, 4.24M params, 20 epochs, 16 orders | **0.96125** | | |
| LR stack: trees + transformers | **0.96176** | **0.96161** | **0.96205** |
| CatBoost stacker on the same predictions + Class / trip type / loyalty | 0.96178 | 0.96148 | 0.96204 |

## Learnings

- **Biggest single gain since target encoding.** From 1.8M params and 8 epochs to 4.24M and 20 epochs: 0.95790 → 0.96125. It beats the best tree model on every fold, and loss kept falling through epoch 20.
- **The stack works once the new model is the stronger one.** The transformer takes 52% of the weight: +0.0007 CV and +0.0013 public over the previous stack.
- **The probe underestimates.** 4 orders on 20k rows read about 0.959; the full score with 16 orders on 140k rows is about 0.961.
- **The stacker choice doesn't matter.** CatBoost ties logistic regression on CV and does slightly worse on the leaderboard, so LR stays.
- A Kaggle bug worth remembering: in zsh, `"$ARGS"` passes the settings as one argument, so `make_kaggle_job.py` now splits them itself.
