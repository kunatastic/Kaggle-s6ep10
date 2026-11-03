import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
STEPS = {
    "data": ("#fde7d9", "#eb6834"),
    "feat": ("#cde2fb", "#2a78d6"),
    "model": ("#cde2fb", "#2a78d6"),
    "stack": ("#2a78d6", "#1c5cab"),
    "result": ("#d3f2e6", "#1baf7a"),
    "gen": ("#e6e2fa", "#6250d6"),
    "dropped": ("#efeeeb", "#b9b8b3"),
}

fig, ax = plt.subplots(figsize=(14, 4.6))
fig.patch.set_facecolor(SURFACE)
ax.set_xlim(0, 14)
ax.set_ylim(0, 4.6)
ax.axis("off")


def box(x, y, w, h, kind, title, lines, round_=0.08, light_text=False):
    fill, edge = STEPS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                                facecolor=fill, edgecolor=edge, linewidth=1.6))
    ink = "white" if light_text else INK
    sub = "white" if light_text else (MUTED if kind == "dropped" else INK2)
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=11, fontweight="bold",
            color=MUTED if kind == "dropped" else ink)
    for i, line in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.68 - i * 0.3, line, ha="center", va="center", fontsize=9.5, color=sub)
    return x, y, w, h


def arrow(x1, y1, x2, y2, label=None, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color, linewidth=1.4, connectionstyle="arc3,rad=0"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label, ha="center", fontsize=8.5, color=INK2)


box(0.2, 2.5, 2.0, 1.5, "data", "Data", ["700K train", "300K test", "21 columns"])
box(2.7, 2.5, 2.8, 1.5, "feat", "Features", ["fold-safe target encoding +", "counts: 21 singles, 210 pairs,", "56 triples · routes · orig_pred"])
box(6.0, 2.5, 2.6, 1.5, "model", "Tree models", ["LightGBM  CV 0.96164", "(+ transformer as feature)", "XGBoost · CatBoost"])
box(9.1, 2.0, 2.1, 1.5, "stack", "LR stack", ["logistic regression", "on out-of-fold logits"], light_text=True)
box(11.7, 2.0, 2.1, 1.5, "result", "Best submission", ["public 0.96208", "private 0.96248", "(≈ rank 13 private)"], round_=0.5)
box(2.7, 0.35, 5.9, 1.3, "gen", "Generative transformers  ·  5M + 10M  ·  OOF 0.96168",
    ["label first, then 21 value tokens in random order; pre-trained on original data", "score = which label makes the row more likely  ·  Kaggle T4 ×2"])
box(0.2, 0.35, 2.0, 1.3, "dropped", "Tried, dropped", ["130K original rows", "rows −0.005 · rates ±0"])

arrow(2.2, 3.25, 2.7, 3.25)
arrow(5.5, 3.25, 6.0, 3.25)
arrow(8.6, 3.25, 9.1, 3.05)
arrow(1.2, 2.5, 3.2, 1.65, color=STEPS["gen"][1])
arrow(8.6, 1.0, 9.1, 2.35, color=STEPS["gen"][1])
ax.text(8.98, 1.45, "same 5 folds", ha="left", fontsize=8.5, color=INK2)
arrow(0.7, 2.5, 0.7, 1.65, color=STEPS["dropped"][1])
arrow(11.2, 2.75, 11.7, 2.75)
ax.text(12.75, 1.7, "top-10% cutoffs: private 0.96237 · public 0.96213", ha="center", fontsize=9, color=INK2)

fig.tight_layout()
fig.savefig("docs/pipeline.png", dpi=150, facecolor=SURFACE)
