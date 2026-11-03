# Experiment 10: Pre-trained and larger transformers (in progress)

**Goal:** push the transformer past the trees by pre-training on the original data, training longer and growing the model. Also test LightGBM with the transformer's prediction as a feature.

## Ideas

- **Pre-training** (`--pretrain 20`): 20 epochs on the 130K original passengers before fine-tuning on each fold. The winner found this "replica" starting point worth about +0.001. The Kaggle job attaches the original dataset automatically.
- **More epochs**: 30 instead of 20, since loss was still falling at epoch 20.
- **10M model**: 256 × 8 layers (8.86M params).
- **Transformer as a LightGBM feature** (`--extra gen_5m_s42`): LightGBM learns *where* to trust the transformer, rather than the stack's one fixed weight. It's safe because the transformer's predictions are out-of-fold on the same 5 folds.

## Run

```
KAGGLE_SLUG=s6e10-gen5m-pre-a uv run python src/make_kaggle_job.py "0,1,4" "--dim 192 --layers 6 --epochs 30 --orders 16 --pretrain 20"
KAGGLE_SLUG=s6e10-gen10m-b    uv run python src/make_kaggle_job.py "2,3"   "--dim 256 --layers 8 --epochs 30 --orders 16 --pretrain 20"
uv run python src/train.py --model lgb --feats te --pairs all --triples 8 --counts --route --orig-pred --extra gen_5m_s42 \
    --lr 0.02 --leaves 127 --min-child 200 --colsample 0.3 --l2 5 --threads 10 --suffix _xfeat_x5m
```

## Results so far

| Model | Fold 2 | Fold 3 | OOF |
|---|---|---|---|
| Transformer 5M, 20 epochs (experiment 7) | 0.96192 | 0.96163 | 0.96125 |
| **5M + pre-training, 30 epochs** | **0.96216** | **0.96191** | running |
| 10M + pre-training, 30 epochs | running | running | |
| LightGBM extended features + 5M as a feature | 0.96231 | 0.96190 | **0.96164** |

### Transformer as a LightGBM feature

| Fold | LightGBM extended (exp. 9) | Transformer 5M (exp. 7) | **LightGBM + 5M as a feature** |
|---|---|---|---|
| 0 | 0.96143 | 0.96150 | **0.96180** |
| 1 | 0.96035 | 0.96029 | **0.96077** |
| 2 | 0.96184 | 0.96192 | **0.96231** |
| 3 | 0.96148 | 0.96163 | **0.96190** |
| 4 | 0.96121 | 0.96095 | **0.96151** |
| OOF | 0.96125 | 0.96125 | **0.96164** |

- **Best single model so far (+0.0004 over either input, every fold).** LightGBM learns *where* to trust the transformer: it can lean on it in some segments and on the pair encodings in others, which a single stack weight can't.
- **The stack doesn't move (0.96186 → 0.96186).** The combined model already contains the transformer's signal, so stacking it next to the transformer adds nothing new; the weight just redistributes (transformer 0.36, combined LightGBM 0.33, plain extended LightGBM 0.40). Combining models inside one learner and combining them in the stack are two routes to the same gain, not two separate gains.
