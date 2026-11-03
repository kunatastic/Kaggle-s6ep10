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
p.add_argument("--orig", choices=["none", "te", "rows", "both"], default="none")
p.add_argument("--pairs", choices=["top", "all"], default="top")
args = p.parse_args()
tag = f"{args.model}_{args.feats}_s{args.seed}" + ("" if args.orig == "none" else f"_orig{args.orig}") \
    + ("" if args.pairs == "top" else "_allpairs")
use_orig_te, use_orig_rows = args.orig in ("te", "both"), args.orig in ("rows", "both")

RATINGS = ["Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
           "Gate location", "Food and drink", "Online boarding", "Seat comfort", "Inflight entertainment",
           "On-board service", "Leg room service", "Baggage handling", "Checkin service", "Cleanliness"]
CATS = ["Gender", "Customer Type", "Type of Travel", "Class"]
DEP, ARR = "Departure Delay in Minutes", "Arrival Delay in Minutes"
PAIR_COLS = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
             "Inflight entertainment", "Seat comfort", "Age", "Leg room service", "Ease of Online booking"]

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
y = tr.pop("satisfaction").astype(int).values
orig = pd.read_csv("orig/data.csv")
y_orig = orig.pop("satisfaction").astype(int).values
n_tr, n_te = len(tr), len(te)
df = pd.concat([tr, te], ignore_index=True).drop(columns="id")
df = pd.concat([df, orig[df.columns]], ignore_index=True)
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
    # integer codes, not strings: 200+ string columns over 1.1M rows don't fit in 16 GB
    groups = [(c,) for c in raw_cols]
    if args.pairs == "all":
        groups += list(itertools.combinations(raw_cols, 2)) + list(itertools.combinations(PAIR_COLS[:6], 3))
    else:
        groups += list(itertools.combinations(PAIR_COLS, 2))
    single = {c: pd.factorize(df[c], use_na_sentinel=False)[0].astype(np.int64) for c in raw_cols}
    codes = np.empty((len(df), len(groups)), np.int32)
    for i, g in enumerate(groups):
        k = single[g[0]]
        for c in g[1:]:
            k = k * (single[c].max() + 1) + single[c]
        codes[:, i] = pd.factorize(k)[0]
    te_cols = ["|".join(g) for g in groups]
    as_str = pd.DataFrame(codes, columns=[f"c{i}" for i in range(len(groups))])

if use_orig_te:
    assert te_cols, "--orig te needs --feats te"
    o_rows = np.arange(n_tr + n_te, len(df))

    def orig_rates(src, dst, m=10):
        ys = y_orig[src - n_tr - n_te]
        prior, out = ys.mean(), np.empty((len(dst), codes.shape[1]), np.float32)
        for i in range(codes.shape[1]):
            c = codes[:, i]
            s = np.bincount(c[src], weights=ys, minlength=c.max() + 1)
            n = np.bincount(c[src], minlength=c.max() + 1)
            out[:, i] = ((s + m * prior) / (n + m))[c[dst]]
        return pd.DataFrame(out, index=dst, columns=[f"o_{i}" for i in range(codes.shape[1])])

    # original rows get cross-fitted rates so their own labels never feed their features
    parts = [orig_rates(o_rows, np.arange(n_tr + n_te))]
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(o_rows, y_orig):
        parts.append(orig_rates(o_rows[a], o_rows[b]))
    df = pd.concat([df, pd.concat(parts).sort_index()], axis=1)

for c in CATS:
    df[c] = df[c].astype("category")
split = lambda d: [d.iloc[i:j].reset_index(drop=True) for i, j in ((0, n_tr), (n_tr, n_tr + n_te), (n_tr + n_te, len(d)))]
X_all, T_all, X_orig = split(df)
if te_cols:
    S_all, ST_all, S_orig = split(as_str)


def add_te(Xs, a_idx):
    enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=args.seed)
    names = [f"te_{i}" for i in range(len(te_cols))]
    encoded = [enc.fit_transform(Xs[0][1], y[a_idx]).astype(np.float32)] + [enc.transform(S).astype(np.float32) for _, S in Xs[1:]]
    return [pd.concat([X, pd.DataFrame(e, columns=names, index=X.index)], axis=1) for (X, _), e in zip(Xs, encoded)]


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
    Xa, Xb, Xt, Xo = X_all.iloc[a], X_all.iloc[b], T_all, X_orig
    if te_cols:
        Xa, Xb, Xt, Xo = add_te([(Xa, S_all.iloc[a]), (Xb, S_all.iloc[b]), (Xt, ST_all), (Xo, S_orig)], a)
    ya = y[a]
    if use_orig_rows:
        Xa, ya = pd.concat([Xa, Xo], ignore_index=True), np.concatenate([ya, y_orig])
    oof[b], pt, it = fit_predict(Xa, ya, Xb, y[b], Xt)
    pred += pt / 5
    print(f"[{tag}] fold {f}: {roc_auc_score(y[b], oof[b]):.5f} @ {it}  ({time.time() - t0:.0f}s)", flush=True)

score = roc_auc_score(y, oof)
print(f"[{tag}] OOF AUC: {score:.5f}  ({time.time() - t0:.0f}s, {X_all.shape[1] + len(te_cols)} feats)")
Path("preds").mkdir(exist_ok=True)
np.save(f"preds/{tag}_oof.npy", oof)
np.save(f"preds/{tag}_test.npy", pred)
