import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

y = pd.read_csv("train.csv", usecols=["satisfaction"]).satisfaction.astype(int).values
ids = pd.read_csv("test.csv", usecols=["id"]).id
pattern = sys.argv[1] if len(sys.argv) > 1 else "*"
tags = sorted(p.name[:-8] for p in Path("preds").glob(f"{pattern}_oof.npy"))


def load(tag, kind):
    return logit(np.clip(np.load(f"preds/{tag}_{kind}.npy"), 1e-6, 1 - 1e-6))


O = np.column_stack([load(t, "oof") for t in tags])
T = np.column_stack([load(t, "test") for t in tags])
for t, col in zip(tags, O.T):
    print(f"{roc_auc_score(y, col):.5f}  {t}")
print(f"{roc_auc_score(y, O.mean(axis=1)):.5f}  mean of {len(tags)}")

lr = LogisticRegression(C=1.0, max_iter=1000)
stack_oof = cross_val_predict(lr, O, y, cv=5, method="predict_proba")[:, 1]
print(f"{roc_auc_score(y, stack_oof):.5f}  stacked (LR, 5-fold)")
w = lr.fit(O, y).coef_[0]
print("weights:", dict(zip(tags, w.round(3))))
pd.DataFrame({"id": ids, "satisfaction": lr.predict_proba(T)[:, 1]}).to_csv("sub_blend.csv", index=False)
print("wrote sub_blend.csv")
