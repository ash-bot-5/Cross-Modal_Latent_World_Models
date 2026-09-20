"""
Nonlinear (single-hidden-layer MLP) probe on the fine-tuned CLIP image embeddings --
tests whether linear_probe_global_table_image.py's weak-to-moderate linear separability
(val AUC 0.69-0.85) reflects a genuine representational limit or just a linear-readout
limit, following the Alain & Bengio (2016) "one step past linear" precedent.

Reuses the EXACT cached embeddings and identity-disjoint splits already produced by
linear_probe_global_table_image.py -- no re-encoding, no re-deriving splits, so the
linear-vs-MLP comparison is apples-to-apples on identical train/val data:
  - Embeddings: linear_probe_global_table_image_{ckpt}_embeddings.npz (key "embeddings")
  - Labels/identities: linear_probe_image_data.npz (ab_label, peg_label, pair_id,
    peg_block_id)
  - Splits: not separately cached -- re-derived here via the identical seeded
    random.Random(42/123).sample(...) logic from linear_probe_global_table_image.py,
    so train1_mask/val1_mask/train2_mask/val2_mask come out bit-identical.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mlp_probe_global_table_image.py
"""

import os
import json
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(OUT_DIR, "nonlinear_probe_image_embeddings_histograms")
os.makedirs(HIST_DIR, exist_ok=True)
DEVICE = "cpu"  # trivial compute size (few thousand x 768) -- no GPU needed

CHECKPOINTS = ["pretrained", "epoch_000.pt", "epoch_001.pt", "epoch_008.pt"]
PROBES = ["probe_ab", "probe_peg"]

# Must match linear_probe_global_table_image.py exactly, so splits are bit-identical.
SPLIT_SEED_PROBE1 = 42
SPLIT_SEED_PROBE2 = 123
VAL_PAIR_FRAC = 0.2
VAL_PEG_FRAC = 0.25

# Expected split sizes from the linear probe run, asserted below as a correctness check.
EXPECTED_SPLIT_SIZES = {
    "probe_ab": (9565, 2502),
    "probe_peg": (9127, 2940),
}

SEEDS = [0, 1, 2, 3, 4]
HIDDEN_DIM = 128
MAX_EPOCHS = 200
PATIENCE = 20
LR = 1e-3

DATA_PATH = os.path.join(OUT_DIR, "linear_probe_image_data.npz")
LINEAR_RESULTS_PATH = os.path.join(OUT_DIR, "linear_probe_global_table_image_results.json")


def embedding_path(ckpt_name: str) -> str:
    return os.path.join(OUT_DIR, f"linear_probe_global_table_image_{ckpt_name}_embeddings.npz")


# ---------------------------------------------------------------------------
# 1. Load cached labels/identities + per-checkpoint embeddings (no re-encoding)
# ---------------------------------------------------------------------------
print(f"Loading cached labels/identities from {DATA_PATH}")
d = np.load(DATA_PATH, allow_pickle=False)
ab_label = d["ab_label"].astype(bool)
peg_label = d["peg_label"].astype(bool)
pair_id_arr = d["pair_id"]
peg_block_id_arr = d["peg_block_id"]

embeddings_by_ckpt = {}
for ckpt_name in CHECKPOINTS:
    p = embedding_path(ckpt_name)
    print(f"Loading cached embeddings from {p}")
    embeddings_by_ckpt[ckpt_name] = np.load(p)["embeddings"].astype(np.float32)

# ---------------------------------------------------------------------------
# 2. Identity-disjoint splits -- inline-copied verbatim from
#    linear_probe_global_table_image.py so masks are bit-identical.
# ---------------------------------------------------------------------------
unique_pairs = sorted(set(pair_id_arr.tolist()))
n_val_pairs = max(1, round(VAL_PAIR_FRAC * len(unique_pairs)))
val_pairs = set(random.Random(SPLIT_SEED_PROBE1).sample(unique_pairs, n_val_pairs))
val1_mask = np.isin(pair_id_arr, list(val_pairs))
train1_mask = ~val1_mask

unique_peg_blocks = sorted(set(peg_block_id_arr.tolist()))
n_val_peg_blocks = max(1, round(VAL_PEG_FRAC * len(unique_peg_blocks)))
val_peg_blocks = set(random.Random(SPLIT_SEED_PROBE2).sample(unique_peg_blocks, n_val_peg_blocks))
val2_mask = np.isin(peg_block_id_arr, list(val_peg_blocks))
train2_mask = ~val2_mask

SPLITS = {
    "probe_ab": (train1_mask, val1_mask, ab_label),
    "probe_peg": (train2_mask, val2_mask, peg_label),
}

for probe_name, (train_mask, val_mask, _) in SPLITS.items():
    n_train, n_val = int(train_mask.sum()), int(val_mask.sum())
    exp_train, exp_val = EXPECTED_SPLIT_SIZES[probe_name]
    assert (n_train, n_val) == (exp_train, exp_val), (
        f"{probe_name} split size mismatch vs. linear probe run: "
        f"got ({n_train}, {n_val}), expected ({exp_train}, {exp_val}) -- "
        f"splits are not reproducing the linear probe's masks."
    )
    print(f"[{probe_name}] split verified: n_train={n_train}, n_val={n_val} (matches linear probe run)")
print()


# ---------------------------------------------------------------------------
# 3. MLP probe architecture + single-seed training loop
# ---------------------------------------------------------------------------
class MLPProbe(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # logits


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_single_seed(X_train, y_train, X_val, y_val, seed: int):
    set_seed(seed)

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val)
    y_val_np = y_val.astype(np.float32)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = MLPProbe(in_dim=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_auc = -1.0
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = criterion(logits, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t).numpy()
        val_auc = roc_auc_score(y_val_np, val_logits)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train_t).numpy()
        val_logits = model(X_val_t).numpy()

    train_acc = float(((train_logits > 0) == y_train).mean())
    val_acc = float(((val_logits > 0) == y_val).mean())
    val_auc = float(roc_auc_score(y_val_np, val_logits))

    return {
        "seed": seed,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_logits": val_logits,
    }


# ---------------------------------------------------------------------------
# 4. Run grid: 4 checkpoints x 2 probes x 5 seeds
# ---------------------------------------------------------------------------
per_seed_results = []  # flat list, one row per (checkpoint, probe, seed)
summary_rows = []      # one row per (checkpoint, probe), mean +/- std over seeds

for ckpt_name in CHECKPOINTS:
    X = embeddings_by_ckpt[ckpt_name]
    print(f"=== {ckpt_name} ===")

    for probe_name in PROBES:
        train_mask, val_mask, y = SPLITS[probe_name]
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]

        seed_runs = [run_single_seed(X_train, y_train, X_val, y_val, seed) for seed in SEEDS]

        for r in seed_runs:
            per_seed_results.append({
                "checkpoint": ckpt_name,
                "probe": probe_name,
                "seed": r["seed"],
                "train_acc": r["train_acc"],
                "val_acc": r["val_acc"],
                "val_auc": r["val_auc"],
            })

        train_accs = np.array([r["train_acc"] for r in seed_runs])
        val_accs = np.array([r["val_acc"] for r in seed_runs])
        val_aucs = np.array([r["val_auc"] for r in seed_runs])

        summary_rows.append({
            "checkpoint": ckpt_name,
            "probe": probe_name,
            "n_train": int(train_mask.sum()),
            "n_val": int(val_mask.sum()),
            "train_acc_mean": float(train_accs.mean()),
            "train_acc_std": float(train_accs.std()),
            "val_acc_mean": float(val_accs.mean()),
            "val_acc_std": float(val_accs.std()),
            "val_auc_mean": float(val_aucs.mean()),
            "val_auc_std": float(val_aucs.std()),
        })

        print(f"  [{probe_name}] train_acc={train_accs.mean():.3f}+/-{train_accs.std():.3f}  "
              f"val_acc={val_accs.mean():.3f}+/-{val_accs.std():.3f}  "
              f"val_auc={val_aucs.mean():.3f}+/-{val_aucs.std():.3f}")

        # --- best-seed histogram (mirrors the linear probe's margin histogram) ---
        best_run = max(seed_runs, key=lambda r: r["val_auc"])
        val_logits = best_run["val_logits"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(val_logits[y_val], bins=40, alpha=0.6, label=f"touching (n={int(y_val.sum())})", color="tab:blue")
        ax.hist(val_logits[~y_val], bins=40, alpha=0.6, label=f"not touching (n={int((~y_val).sum())})", color="tab:orange")
        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("MLP logit (pre-sigmoid)")
        ax.set_ylabel("count")
        ax.set_title(
            f"{ckpt_name} -- {probe_name} (MLP, best of {len(SEEDS)} seeds = seed {best_run['seed']})\n"
            f"val_acc={best_run['val_acc']:.3f}, val_auc={best_run['val_auc']:.3f}"
        )
        ax.legend()
        plt.tight_layout()
        hist_path = os.path.join(HIST_DIR, f"{ckpt_name}_{probe_name}_mlp_margin_histogram.png")
        plt.savefig(hist_path, dpi=150)
        plt.close()

    print()

# ---------------------------------------------------------------------------
# 5. Summary tables
# ---------------------------------------------------------------------------
print("=" * 100)
print(f"{'checkpoint':<14} {'probe':<10} {'n_train':>8} {'n_val':>6} "
      f"{'train_acc':>16} {'val_acc':>16} {'val_auc':>16}")
print("-" * 100)
for r in summary_rows:
    print(
        f"{r['checkpoint']:<14} {r['probe']:<10} {r['n_train']:>8} {r['n_val']:>6} "
        f"{r['train_acc_mean']:.3f}+/-{r['train_acc_std']:.3f}   "
        f"{r['val_acc_mean']:.3f}+/-{r['val_acc_std']:.3f}   "
        f"{r['val_auc_mean']:.3f}+/-{r['val_auc_std']:.3f}"
    )
print("=" * 100)

# --- linear-vs-mlp comparison ---
with open(LINEAR_RESULTS_PATH) as f:
    linear_results = json.load(f)
linear_auc_by_key = {(r["checkpoint"], r["probe"]): r["val_auc"] for r in linear_results}

print("\n" + "=" * 70)
print(f"{'checkpoint':<14} {'probe':<10} {'linear_val_auc':>15} {'mlp_val_auc':>13} {'delta':>8}")
print("-" * 70)
comparison_rows = []
for r in summary_rows:
    key = (r["checkpoint"], r["probe"])
    linear_auc = linear_auc_by_key[key]
    mlp_auc = r["val_auc_mean"]
    delta = mlp_auc - linear_auc
    comparison_rows.append({
        "checkpoint": r["checkpoint"],
        "probe": r["probe"],
        "linear_val_auc": linear_auc,
        "mlp_val_auc": mlp_auc,
        "delta": delta,
    })
    print(f"{r['checkpoint']:<14} {r['probe']:<10} {linear_auc:>15.3f} {mlp_auc:>13.3f} {delta:>+8.3f}")
print("=" * 70)

results_path = os.path.join(OUT_DIR, "mlp_probe_global_table_image_results.json")
with open(results_path, "w") as f:
    json.dump({
        "per_seed": per_seed_results,
        "summary": summary_rows,
        "linear_vs_mlp": comparison_rows,
    }, f, indent=2)
print(f"\nResults saved to {results_path}")
