import glob
import sys

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import pandas as pd

tag = sys.argv[1] if len(sys.argv) > 1 else "gen_s42"
src = sys.argv[2] if len(sys.argv) > 2 else "kaggle_out/*/preds/gen"
y = pd.read_csv("train.csv", usecols=["satisfaction"]).satisfaction.astype(int).values
oof, tests, seen = np.full(len(y), np.nan), [], set()
for f in sorted(glob.glob(f"{src}/f*_val.npy")):
    fold = f.rsplit("/f", 1)[1].split("_")[0]
    if fold in seen:
        continue
    seen.add(fold)
    idx, val = np.load(f)
    oof[idx.astype(int)] = val
    tests.append(np.load(f.replace("_val", "_test")))
    print(f"fold {fold}: AUC {roc_auc_score(y[idx.astype(int)], val):.5f}")
assert len(seen) == 5 and not np.isnan(oof).any(), f"have folds {sorted(seen)}"
print(f"{tag} OOF AUC: {roc_auc_score(y, oof):.5f}")
np.save(f"preds/{tag}_oof.npy", expit(oof))
np.save(f"preds/{tag}_test.npy", expit(np.mean(tests, axis=0)))
