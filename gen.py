"""Label-first generative transformer: score each row by which label makes its features more likely."""
import argparse
import glob
import math
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

p = argparse.ArgumentParser()
p.add_argument("--folds", default="0", help="comma-separated folds; split across GPUs when there are several")
p.add_argument("--device", default=None)
p.add_argument("--out", default="preds/gen")
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--orders", type=int, default=8, help="random column orders averaged when scoring")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)
folds = [int(f) for f in args.folds.split(",")]
n_gpu = torch.cuda.device_count()
if args.device is None and n_gpu > 1 and len(folds) > 1:
    rest = [a for a in sys.argv[1:] if not a.startswith("--folds") and a != args.folds]
    procs = [subprocess.Popen([sys.executable, sys.argv[0], *rest, "--folds", ",".join(map(str, folds[g::n_gpu])),
                               "--device", f"cuda:{g}"]) for g in range(min(n_gpu, len(folds)))]
    sys.exit(max(pr.wait() for pr in procs))
dev = args.device or ("cuda" if n_gpu else "mps" if torch.backends.mps.is_available() else "cpu")
amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if dev.startswith("cuda") else nullcontext
scaler = torch.amp.GradScaler("cuda", enabled=dev.startswith("cuda"))

data = next(iter(glob.glob("/kaggle/input/**/playground-series-s6e10/", recursive=True)), "./")
tr = pd.read_csv(data + "train.csv")
te = pd.read_csv(data + "test.csv")
y = tr.pop("satisfaction").astype(int).values
feats = pd.concat([tr, te], ignore_index=True).drop(columns="id")
n_col = feats.shape[1]

codes, offsets, V = [], [], 0
for c in feats.columns:
    k = pd.factorize(feats[c], use_na_sentinel=False)[0]
    codes.append(k + V)
    offsets.append((V, V + k.max() + 1))
    V += k.max() + 1
tokens = torch.tensor(np.stack(codes, axis=1), dtype=torch.long, device=dev)
col_bias = torch.full((n_col, V), float("-inf"), device=dev)
for i, (a, b) in enumerate(offsets):
    col_bias[i, a:b] = 0.0
X, XT = tokens[: len(tr)], tokens[len(tr):]
yt = torch.tensor(y, device=dev)
prior_logit = math.log(y.mean() / (1 - y.mean()))


class Gen(nn.Module):
    def __init__(self, d, layers):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.label = nn.Embedding(2, d)
        self.next_col = nn.Embedding(n_col, d)
        self.pos = nn.Embedding(n_col + 1, d)
        layer = nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, V)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(n_col))

    def log_probs(self, toks, label, order):
        """Per-feature log P(value | label, earlier values) for rows `toks` read in column `order`."""
        seq = toks.gather(1, order)
        h = torch.cat([self.label(label)[:, None], self.tok(seq[:, :-1])], 1)
        h = h + self.next_col(order) + self.pos.weight[None, :n_col]
        h = self.norm(self.body(h, mask=self.mask, is_causal=True))
        logits = self.out(h) + col_bias[order]
        return logits.log_softmax(-1).gather(-1, seq[..., None])[..., 0]


def random_orders(n):
    return torch.rand(n, n_col, device=dev).argsort(1)


@torch.no_grad()
def score(model, toks, orders, batch=2048):
    model.eval()
    out = []
    for i in range(0, len(toks), batch):
        t = toks[i: i + batch]
        lp = torch.zeros(2, orders, len(t), device=dev)
        for k in range(orders):
            o = random_orders(len(t))
            for lab in (0, 1):
                with amp():
                    lp[lab, k] = model.log_probs(t, torch.full((len(t),), lab, device=dev), o).float().sum(1)
        out.append((lp[1].logsumexp(0) - lp[0].logsumexp(0) + prior_logit).cpu())
    model.train()
    return torch.cat(out).numpy()


def run_fold(fold):
    a_idx, b_idx = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))[fold]
    a_t, b_t = torch.tensor(a_idx, device=dev), torch.tensor(b_idx, device=dev)
    probe = b_idx[rng.choice(len(b_idx), 20000, replace=False)]

    model = Gen(args.dim, args.layers).to(dev)
    steps = args.epochs * math.ceil(len(a_idx) / args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps, pct_start=0.05)
    print(f"fold {fold}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, vocab {V}, {steps} steps", flush=True)

    t0 = time.time()
    for ep in range(args.epochs):
        perm = a_t[torch.randperm(len(a_t), device=dev)]
        total = 0.0
        for i in range(0, len(perm), args.batch):
            rows = perm[i: i + args.batch]
            with amp():
                loss = -model.log_probs(X[rows], yt[rows], random_orders(len(rows))).float().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(rows)
        auc = roc_auc_score(y[probe], score(model, X[torch.tensor(probe, device=dev)], 4))
        print(f"epoch {ep}: loss {total / len(perm):.4f}  probe AUC {auc:.5f}  ({time.time() - t0:.0f}s)", flush=True)

    val = score(model, X[b_t], args.orders)
    print(f"fold {fold} AUC: {roc_auc_score(y[b_idx], val):.5f}  ({time.time() - t0:.0f}s)", flush=True)
    test = score(model, XT, args.orders)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.save(f"{args.out}/f{fold}_val.npy", np.stack([b_idx, val]))
    np.save(f"{args.out}/f{fold}_test.npy", test)
    print(f"done ({time.time() - t0:.0f}s)")


for fold in folds:
    run_fold(fold)
