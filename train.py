import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

p = argparse.ArgumentParser()
p.add_argument("--model", choices=["lgb", "xgb", "cat"], default="lgb")
p.add_argument("--feats", choices=["raw", "base", "te"], default="te")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--lr", type=float, default=0.03)
args = p.parse_args()
tag = f"{args.model}_{args.feats}_s{args.seed}"

RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
           "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
DEP, ARR = "Departure Delay in Minutes", "Arrival Delay in Minutes"
# ponytail: fixed pair list from single-feature AUC ranking — search pairs by CV gain if this plateaus
PAIR_COLS = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
             "Inflight entertainment", "Seat comfort", "Age", "Leg room service", "Ease of Online booking"]

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
y = tr.pop("satisfaction").astype(int).values
df = pd.concat([tr, te], ignore_index=True).drop(columns="id")
raw_cols = list(df.columns)

if args.feats != "raw":
    r = df[RATINGS].replace(0, np.nan)
    df["n_zero"] = (df[RATINGS] == 0).sum(axis=1)
    df["r_mean"] = r.mean(axis=1)
    df["r_std"] = r.std(axis=1)
    df["r_min"] = r.min(axis=1)
    df["r_max"] = r.max(axis=1)
    df["delay_diff"] = df[ARR] - df[DEP]
    df["delay_max"] = df[[DEP, ARR]].max(axis=1)

te_cols = []
if args.feats == "te":
    as_str = df[raw_cols].astype(str)
    for a, b in itertools.combinations(PAIR_COLS, 2):
        as_str[f"{a}|{b}"] = as_str[a] + "_" + as_str[b]
    te_cols = list(as_str.columns)

for c in CATS:
    df[c] = df[c].astype("category")
X_all, T_all = df.iloc[: len(tr)].reset_index(drop=True), df.iloc[len(tr):].reset_index(drop=True)
if te_cols:
    S_all, ST_all = as_str.iloc[: len(tr)].reset_index(drop=True), as_str.iloc[len(tr):].reset_index(drop=True)


def add_te(Xa, Xb, Xt, a_idx, b_idx):
    enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=args.seed)
    names = [f"te_{i}" for i in range(len(te_cols))]
    ea = enc.fit_transform(S_all.iloc[a_idx], y[a_idx])
    eb, et = enc.transform(S_all.iloc[b_idx]), enc.transform(ST_all)
    return [pd.concat([X, pd.DataFrame(e, columns=names, index=X.index)], axis=1) for X, e in ((Xa, ea), (Xb, eb), (Xt, et))]


def fit_predict(Xa, ya, Xb, yb, Xt):
    if args.model == "lgb":
        import lightgbm as lgb
        m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=args.lr, num_leaves=63, min_child_samples=100,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0,
                               random_state=args.seed, verbose=-1)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
        it = m.best_iteration_
    elif args.model == "xgb":
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=20000, learning_rate=args.lr, max_depth=7, min_child_weight=5,
                              subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0, tree_method="hist",
                              enable_categorical=True, eval_metric="auc", early_stopping_rounds=200,
                              random_state=args.seed)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        it = m.best_iteration
    else:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=20000, learning_rate=args.lr, depth=7, eval_metric="AUC",
                               od_type="Iter", od_wait=200, random_seed=args.seed, verbose=0,
                               cat_features=[c for c in CATS if c in Xa.columns])
        Xa, Xb, Xt = (X.astype({c: str for c in CATS}) for X in (Xa, Xb, Xt))
        m.fit(Xa, ya, eval_set=(Xb, yb))
        it = m.get_best_iteration()
    return m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], it


oof, pred = np.zeros(len(X_all)), np.zeros(len(T_all))
t0 = time.time()
for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(X_all, y)):
    Xa, Xb, Xt = X_all.iloc[a], X_all.iloc[b], T_all
    if te_cols:
        Xa, Xb, Xt = add_te(Xa, Xb, Xt, a, b)
    oof[b], pt, it = fit_predict(Xa, y[a], Xb, y[b], Xt)
    pred += pt / 5
    print(f"[{tag}] fold {f}: {roc_auc_score(y[b], oof[b]):.5f} @ {it}  ({time.time() - t0:.0f}s)", flush=True)

score = roc_auc_score(y, oof)
print(f"[{tag}] OOF AUC: {score:.5f}  ({time.time() - t0:.0f}s, {X_all.shape[1] + len(te_cols)} feats)")
Path("preds").mkdir(exist_ok=True)
np.save(f"preds/{tag}_oof.npy", oof)
np.save(f"preds/{tag}_test.npy", pred)
