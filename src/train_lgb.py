import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
y = tr.pop("satisfaction").astype(int).values
X, T = tr.drop(columns="id"), te.drop(columns="id")
for c in X.select_dtypes(exclude="number"):
    X[c] = X[c].astype("category")
    T[c] = pd.Categorical(T[c], categories=X[c].cat.categories)

oof, pred = np.zeros(len(X)), np.zeros(len(T))
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y)):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.03, num_leaves=63, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.7, verbose=-1)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], eval_metric="auc",
          callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    pred += m.predict_proba(T)[:, 1] / 5
    print(f"fold {f}: {roc_auc_score(y[b], oof[b]):.5f} @ {m.best_iteration_}")

print(f"OOF AUC: {roc_auc_score(y, oof):.5f}")
np.save("oof_lgb.npy", oof)
pd.DataFrame({"id": te.id, "satisfaction": pred}).to_csv("sub_lgb.csv", index=False)
