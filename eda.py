import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
y = tr.satisfaction.astype(int)
X = tr.drop(columns=["id", "satisfaction"])
num = X.select_dtypes("number").columns
cats = X.select_dtypes(exclude="number").columns

print(f"train {tr.shape}, test {te.shape}, positive rate {y.mean():.4f}")
print("missing (train):", tr.isna().sum()[lambda s: s > 0].to_dict())
print("missing (test): ", te.isna().sum()[lambda s: s > 0].to_dict())
print("duplicate feature rows:", X.duplicated().sum())
for c in cats:
    print(tr.groupby(c).satisfaction.agg(["mean", "size"]).round(3), "\n")

print("single-feature AUC:")
print(pd.Series({c: abs(roc_auc_score(y, X[c].fillna(-1)) - 0.5) + 0.5 for c in num})
      .sort_values(ascending=False).round(3).to_string())
print("max train/test mean drift (std units):",
      round(max(abs((tr[c].mean() - te[c].mean()) / tr[c].std()) for c in num), 4))

X = X.astype({c: "category" for c in cats})
a, b, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
m = HistGradientBoostingClassifier(max_iter=2000, max_leaf_nodes=63, categorical_features="from_dtype",
                                   n_iter_no_change=50, random_state=0).fit(a, ya)
p = m.predict_proba(b)[:, 1]
print(f"HGB holdout AUC {roc_auc_score(yb, p):.5f}, acc {accuracy_score(yb, p > 0.5):.5f}")
