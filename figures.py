from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

RATINGS = ["Online boarding", "Inflight entertainment", "Seat comfort", "On-board service", "Cleanliness",
           "Leg room service", "Inflight wifi service", "Baggage handling", "Checkin service", "Food and drink",
           "Ease of Online booking", "Departure/Arrival time convenient", "Gate location"]
tr = pd.read_csv("train.csv")
tr["y"] = tr.satisfaction.astype(int)
base = tr.y.mean()


def pct_axis(ax, axis="y"):
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


# 1. Satisfaction rate by rating value, every rating
rate = pd.DataFrame({r: tr.groupby(r).y.mean() for r in RATINGS}).T.reindex(columns=range(6))
share = pd.DataFrame({r: tr[r].value_counts(normalize=True) for r in RATINGS}).T.reindex(columns=range(6))
fig, ax = plt.subplots(figsize=(8, 6.2))
im = ax.imshow(rate.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
ax.grid(False)
ax.set_xticks(range(6), [f"{v}" for v in range(6)])
ax.set_yticks(range(len(RATINGS)), RATINGS)
ax.set_xlabel("Rating given")
for i in range(len(RATINGS)):
    for j in range(6):
        v = rate.iat[i, j]
        if np.isnan(v):
            ax.text(j, i, "none", ha="center", va="center", color=INK2, fontsize=8)
        else:
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.55 else INK)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.outline.set_visible(False)
cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_title("Share satisfied at each rating value")
fig.tight_layout()
fig.savefig(OUT / "rating_heatmap.png", dpi=150)
plt.close(fig)

# 2. Segments: Class x Type of Travel x Customer Type
seg = tr.groupby(["Type of Travel", "Customer Type", "Class"]).y.agg(["mean", "size"]).reset_index()
seg = seg[seg["size"] >= 500]
seg["label"] = seg["Type of Travel"].str.replace(" travel", "").str.replace(" Travel", "") + " · " + \
    seg["Customer Type"].str.replace(" Customer", "").str.capitalize()
classes = ["Business", "Eco Plus", "Eco"]
labels = list(dict.fromkeys(seg.sort_values("mean", ascending=False).label))
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.26
for k, (cls, col) in enumerate(zip(classes, [BLUE, ORANGE, AQUA])):
    d = seg[seg.Class == cls].set_index("label").reindex(labels)
    x = np.arange(len(labels)) + (k - 1) * w
    ax.bar(x, d["mean"], width=w - 0.02, color=col, label=cls, zorder=2)
    for xi, v in zip(x, d["mean"]):
        if not np.isnan(v):
            ax.text(xi, v + 0.015, f"{v:.0%}", ha="center", fontsize=7.5, color=INK2)
ax.axhline(base, color=INK2, linewidth=1, zorder=1)
ax.text(len(labels) - 0.5, base + 0.02, f"overall {base:.0%}", ha="right", fontsize=8, color=INK2)
ax.set_xticks(range(len(labels)), labels)
ax.set_ylim(0, 1)
pct_axis(ax)
ax.grid(axis="x", visible=False)
ax.legend(title="Class", frameon=False, loc="upper right", bbox_to_anchor=(1, 0.92))
ax.set_title("Share satisfied by trip type, loyalty and class")
fig.tight_layout()
fig.savefig(OUT / "segments.png", dpi=150)
plt.close(fig)

# 3. Age and flight distance, split by type of travel
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, (col, bins, xl) in zip(axes, [("Age", np.arange(5, 90, 5), "Age"),
                                     ("Flight Distance", np.arange(0, 5250, 250), "Flight distance (miles)")]):
    for tt, c in zip(["Business travel", "Personal Travel"], [BLUE, ORANGE]):
        d = tr[tr["Type of Travel"] == tt]
        g = d.groupby(pd.cut(d[col], bins), observed=True).y.agg(["mean", "size"])
        g = g[g["size"] >= 500]
        mids = [iv.mid for iv in g.index]
        ax.plot(mids, g["mean"], color=c, label=tt.replace(" Travel", " travel"))
        ax.text(mids[-1], g["mean"].iloc[-1] + 0.03, tt.split()[0].lower(), color=INK2, fontsize=8, ha="right")
    ax.set_xlabel(xl)
    pct_axis(ax)
    ax.set_ylim(0, 1)
axes[0].set_title("Share satisfied by age and distance")
axes[0].legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(OUT / "age_distance.png", dpi=150)
plt.close(fig)

# 4. Delays
fig, ax = plt.subplots(figsize=(7, 3.4))
dbins = [-1, 0, 5, 15, 30, 60, 120, 500]
dlab = ["0", "1–5", "6–15", "16–30", "31–60", "61–120", "120+"]
for col, c, name in [("Departure Delay in Minutes", BLUE, "departure"), ("Arrival Delay in Minutes", ORANGE, "arrival")]:
    g = tr.groupby(pd.cut(tr[col], dbins, labels=dlab), observed=True).y.agg(["mean", "size"])
    ax.plot(range(len(g)), g["mean"], color=c, marker="o", markersize=5, label=f"{name} delay")
ax.set_xticks(range(len(dlab)), dlab)
ax.set_xlabel("Delay (minutes)")
pct_axis(ax)
ax.set_ylim(0, 0.6)
ax.legend(frameon=False)
zero = (tr["Departure Delay in Minutes"] == 0).mean()
ax.set_title(f"Delays matter only when long, and {zero:.0%} of flights have none")
fig.tight_layout()
fig.savefig(OUT / "delays.png", dpi=150)
plt.close(fig)

# 5. Online boarding x Inflight wifi
pv = tr.pivot_table(index="Online boarding", columns="Inflight wifi service", values="y", aggfunc="mean")
fig, ax = plt.subplots(figsize=(5.6, 4.6))
im = ax.imshow(pv.values, cmap=SEQ, vmin=0, vmax=1, origin="lower")
ax.grid(False)
ax.set_xticks(range(pv.shape[1]), pv.columns)
ax.set_yticks(range(pv.shape[0]), pv.index)
ax.set_xlabel("Inflight wifi rating")
ax.set_ylabel("Online boarding rating")
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        v = pv.iat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.55 else INK)
ax.set_title("Top two ratings combined")
fig.tight_layout()
fig.savefig(OUT / "boarding_wifi.png", dpi=150)
plt.close(fig)

# Numbers quoted in NOTES.md
print("rating=5 vs rating=1 lift:")
print((rate[5] - rate[1]).sort_values(ascending=False).round(3).to_string())
print("share of zeros:", share[0].round(4).to_dict())
print(seg.sort_values("mean").to_string())
print("personal travel, age<25 / 40-60:",
      tr[(tr["Type of Travel"] == "Personal Travel")].groupby(pd.cut(tr.Age, [0, 25, 40, 60, 90]), observed=True).y.mean().round(3).to_dict())
print("business travel by age:",
      tr[(tr["Type of Travel"] == "Business travel")].groupby(pd.cut(tr.Age, [0, 25, 40, 60, 90]), observed=True).y.mean().round(3).to_dict())
print("wifi=5 rate:", tr[tr["Inflight wifi service"] == 5].y.mean().round(3), "share", (tr["Inflight wifi service"] == 5).mean().round(3))
print("wifi=0 rate:", tr[tr["Inflight wifi service"] == 0].y.mean().round(3))
nz = (tr[RATINGS] == 0).sum(axis=1)
print("n_zero dist / rate:", tr.groupby(nz).y.agg(["mean", "size"]).to_string())
print("dep/arr delay corr:", tr[["Departure Delay in Minutes", "Arrival Delay in Minutes"]].corr().iat[0, 1].round(3))
