import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TAGS = ["gen_5m_s42", "lgb_te_s42_allpairs_t1", "lgb_te_s42_allpairs", "lgb_te_s42_allpairs_t3", "lgb_te_s42", "cat_te_s42", "xgb_te_s42"]
SEG = ["Class", "Type of Travel", "Customer Type"]
tr = pd.read_csv("train.csv", usecols=SEG + ["satisfaction"])
te = pd.read_csv("test.csv", usecols=["id"] + SEG)
y = tr.satisfaction.astype(int).values
L = lambda t, k: logit(np.clip(np.load(f"preds/{t}_{k}.npy"), 1e-6, 1 - 1e-6))
O = pd.concat([pd.DataFrame({t: L(t, "oof") for t in TAGS}), tr[SEG]], axis=1)
T = pd.concat([pd.DataFrame({t: L(t, "test") for t in TAGS}), te[SEG]], axis=1)
oof, pred = np.zeros(len(O)), np.zeros(len(T))
for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(O, y):
    m = CatBoostClassifier(iterations=3000, learning_rate=0.05, depth=4, eval_metric="AUC", od_type="Iter",
                           od_wait=200, verbose=0, cat_features=SEG, thread_count=6)
    m.fit(O.iloc[a], y[a], eval_set=(O.iloc[b], y[b]))
    oof[b] = m.predict_proba(O.iloc[b])[:, 1]
    pred += m.predict_proba(T)[:, 1] / 5
print(f"CatBoost stacker (preds + segments): {roc_auc_score(y, oof):.5f}")
pd.DataFrame({"id": te.id, "satisfaction": pred}).to_csv("sub_stack_cat.csv", index=False)
