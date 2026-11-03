# Experiment 10: Pre-trained and larger transformers

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
| **5M + pre-training, 30 epochs** | **0.96216** | **0.96191** | **0.96157** |
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

### Pre-trained 5M, all folds

| Fold | 5M, 20 epochs | **5M + pre-training, 30 epochs** |
|---|---|---|
| 0 | 0.96150 | **0.96172** |
| 1 | 0.96029 | **0.96069** |
| 2 | 0.96192 | **0.96216** |
| 3 | 0.96163 | **0.96191** |
| 4 | 0.96095 | **0.96139** |
| OOF | 0.96125 | **0.96157** |

| Submission | CV | Public | Private |
|---|---|---|---|
| Previous best stack (experiment 9) | 0.96186 | 0.96176 | 0.96215 |
| **Stack + pre-trained 5M** | **0.96206** | **0.96199** | **0.96239** |

- **Pre-training on the original data plus 30 epochs: +0.0003 single-model OOF, every fold better.** The pre-trained run starts from a model that already knows how real passengers' answers relate, then adapts to the synthetic data.
- **It's the stack's largest member** (weight 0.40). The plain 5M drops to 0.09, since the pre-trained version supersedes it.
- **Public 0.96199 is 0.00014 short of the top-10% cutoff (0.96213).** Private 0.96239 is above it.
- Commands: `KAGGLE_SLUG=s6e10-gen5m-pre-{a,b}`, collected with `uv run python src/gen_collect.py gen_5mpre_s42 "kaggle_out/s6e10-gen5m-pre-[ab]/preds/gen"`.

### Where 0.96239 ranks (full private leaderboard, 3,735 teams)

| | Private | Rank |
|---|---|---|
| Winner | 0.96318 | 1 |
| **Our best (stack + pre-trained 5M)** | **0.96239** | **~37–74** (38 teams tie at 0.96239) |
| Top-10% cutoff | 0.96237 | 373 |
| Our previous best (experiment 9) | 0.96215 | ~982 |

- **495 teams sit between 0.96233 and 0.96238**, mostly near-identical blends of shared public predictions. Clearing that cluster by 0.00002 jumps hundreds of places, which is why +0.00024 private took us from about rank 982 to the top ~2%.
- **The public board disagrees: 0.96199 against a 0.96213 cutoff.** The public split is only ~20% of the test set and is noisy at this level; the private board (~80%) is the final standing, and our private score has run above CV on every submission.
- These are late submissions, so the ranks are where they *would* place.

### 10M + pre-training (folds 2 + 3 so far)

| Fold | Pre-trained 5M | 10M | Average of both |
|---|---|---|---|
| 2 | 0.96216 | 0.96210 | **0.96237** |
| 3 | 0.96191 | 0.96222 | **0.96233** |

- 10M alone is only about +0.0001 better than 5M (lower training loss, 1.1698 vs 1.1746, but similar held-out scores), so size has hit diminishing returns.
- **Averaged with the pre-trained 5M it gains +0.0002 to +0.0004 per fold.** The two runs correlate 0.995 but make partly different mistakes.
- LightGBM with the *pre-trained* 5M as a feature: OOF 0.96167, the same as with the plain 5M (0.96164), and the stack doesn't move (0.96205).

### 10M, all folds, and the final stack

| Fold | Pre-trained 5M | **10M pre-trained** |
|---|---|---|
| 0 | 0.96172 | **0.96182** |
| 1 | 0.96069 | **0.96078** |
| 2 | 0.96216 | 0.96210 |
| 3 | 0.96191 | **0.96222** |
| 4 | 0.96139 | **0.96151** |
| OOF | 0.96157 | **0.96168** |

| Submission | CV | Public | Private |
|---|---|---|---|
| Stack + pre-trained 5M | 0.96206 | 0.96199 | 0.96239 |
| **Stack + pre-trained 5M + 10M** | **0.96216** | **0.96208** | **0.96248** |

- **Private 0.96248 ranks about 13 462 of 3,735 (top-10% cutoff 0.96237).** Public 0.96208 ranks about , just 0.00005 under its cutoff of 0.96213.
- The 10M takes the largest stack weight (0.31), with the pre-trained 5M at 0.16. Two transformer runs together add +0.0001 CV over one.
- The probe AUC in the logs varies about 0.004 between folds (fold 4 read 0.9638, folds 0–1 about 0.960). That's which 20k passengers the probe holds, not overfitting: it's held-out data, it never selects the epoch, and CV has tracked the leaderboard on every submission.

### Stopped here

A second pre-trained 5M seed (seed 7) was started to close the last 0.00005 on the public board, then cancelled after 3 epochs; no more runs were needed. The likely next steps, if resumed:
- 2–3 more transformer seeds (averaging the 5M and 10M runs gave +0.0002 to +0.0004 per fold), cheapest on rented RTX 4090s;
- longer pre-training on the original data (we used 20 epochs; the winner found 60 clearly better than 8);
- the label placed at a random position instead of first, as a diverse stack member.
