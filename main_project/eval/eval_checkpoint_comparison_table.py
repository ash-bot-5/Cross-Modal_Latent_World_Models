"""
Compare 5 checkpoints on train vs. held-out InfoNCE ranking accuracy:
  - the old model (pre negative-indexing-bug-fix)
  - non-upweighted retrain: best.pt (lowest val loss) and latest.pt (final epoch)
  - upweighted retrain: best.pt and latest.pt

Datasets are built once (checkpoint-independent) and every checkpoint is scored
against the same train sample and the same held-out set, reusing the exact
scoring formula from eval_train_vs_heldout_ranking.py:
  z_pred = model(z_t, actions[t:t+HORIZON]) scored against 1 positive
  (z_sem_target) + 224 hard negatives (LangTableDataset's own z_hard_neg,
  which is now correctly t+HORIZON-indexed for all 5 checkpoints alike).

Held-out set = the actual ~750-episode validation split used during training
(val_files from checkpoints_upweighted/train_val_split.json — identical split
in both retraining runs), not a hand-picked subset.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_checkpoint_comparison_table.py
"""

import os
import csv
import json
import random
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langtable_dataset import LangTableDataset
from training.dynamics_model import LatentDynamicsModel, InfoNCELoss

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
HORIZON   = 8
N_TRAIN_SAMPLES = 1000
DEVICE    = "cpu"  # eval is fast enough on CPU

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

N_CANDIDATES = 225  # 1 positive + 224 hard negatives

CHECKPOINTS = [
    ("old_buggy", os.path.join(REPO_ROOT, "OLD", "checkpoints_OLD", "epoch_199.pt")),
    ("non-upweighted_best", os.path.join(REPO_ROOT, "checkpoints", "best.pt")),
    ("non-upweighted_latest", os.path.join(REPO_ROOT, "checkpoints", "latest.pt")),
    ("upweighted_best", os.path.join(REPO_ROOT, "checkpoints_upweighted", "best.pt")),
    ("upweighted_latest", os.path.join(REPO_ROOT, "checkpoints_upweighted", "latest.pt")),
]
for _, ckpt_path in CHECKPOINTS:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint_comparison_table.csv")

temperature = InfoNCELoss().temperature


# ---------------------------------------------------------------------------
# 1. Datasets — built once, reused for every checkpoint
# ---------------------------------------------------------------------------
with open(os.path.join(REPO_ROOT, "checkpoints_upweighted", "train_val_split.json")) as f:
    split = json.load(f)
val_files = split["val_files"]
print(f"val_files: {len(val_files)} (seed={split['seed']}, n_val={split['n_val']})")

print(f"Loading train dataset from {TRAIN_DIR} ...")
train_dataset = LangTableDataset(TRAIN_DIR, device=DEVICE)
print(f"  {len(train_dataset)} valid (episode, t) samples")

print(f"Loading held-out (validation) dataset ({len(val_files)} episodes) ...")
heldout_dataset = LangTableDataset(TRAIN_DIR, device=DEVICE, file_list=val_files)
print(f"  {len(heldout_dataset)} valid (episode, t) samples")

# Held-out actions must be normalized with the TRAINING set's min/max stats,
# not this subset's own stats (LangTableDataset always computes its own
# action_stats from whatever's in file_list at __init__ time).
heldout_dataset.action_stats = train_dataset.action_stats

train_sample_idx = random.sample(range(len(train_dataset)), N_TRAIN_SAMPLES)
heldout_all_idx = range(len(heldout_dataset))


# ---------------------------------------------------------------------------
# 2. Scoring (reused formula from eval_train_vs_heldout_ranking.py)
# ---------------------------------------------------------------------------
def score(z_pred: torch.Tensor, z_pos: torch.Tensor, z_hard_neg: torch.Tensor) -> dict:
    pos_sim_raw  = (z_pred * z_pos).sum(dim=-1)             # scalar, pre-temperature
    hard_sim_raw = (z_pred * z_hard_neg).sum(dim=-1)         # (224,), pre-temperature
    logits = torch.cat([pos_sim_raw.unsqueeze(0), hard_sim_raw]) / temperature  # (225,)
    rank = int((logits > logits[0]).sum().item()) + 1  # 1 = best; ties don't inflate rank
    return {"top1": int(torch.argmax(logits).item() == 0), "rank": rank}


def eval_point(model: torch.nn.Module, dataset: LangTableDataset, flat_idx: int) -> dict:
    sample = dataset[flat_idx]
    z_t          = sample["z_t"].unsqueeze(0)
    actions      = sample["actions"].unsqueeze(0)
    z_sem_target = sample["z_sem_target"]
    z_hard_neg   = sample["z_hard_neg"]

    with torch.no_grad():
        z_pred = model(z_t, actions).squeeze(0)

    return score(z_pred, z_sem_target, z_hard_neg)


def aggregate(results: list[dict]) -> dict:
    n = len(results)
    ranks = [r["rank"] for r in results]
    return {
        "top1": sum(r["top1"] for r in results) / n,
        "mean_rank": sum(ranks) / n,
        "median_rank": statistics.median(ranks),
        "n": n,
    }


# ---------------------------------------------------------------------------
# 3. Per-checkpoint evaluation
# ---------------------------------------------------------------------------
rows = []
for label, ckpt_path in CHECKPOINTS:
    print(f"\n=== {label} ===")
    print(f"Checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    epoch = ckpt["epoch"]
    print(f"Model loaded (epoch {epoch}, step {ckpt.get('step', '?')})")

    train_results = [eval_point(model, train_dataset, i) for i in train_sample_idx]
    heldout_results = [eval_point(model, heldout_dataset, i) for i in heldout_all_idx]

    train_agg = aggregate(train_results)
    heldout_agg = aggregate(heldout_results)

    print(f"  train_top1={train_agg['top1']:.4f}  heldout_top1={heldout_agg['top1']:.4f}  "
          f"heldout_mean_rank={heldout_agg['mean_rank']:.2f}")

    rows.append({
        "run": label,
        "checkpoint_path": ckpt_path,
        "epoch": epoch,
        "train_top1": train_agg["top1"],
        "heldout_top1_pooled": heldout_agg["top1"],
        "heldout_mean_rank_pooled": heldout_agg["mean_rank"],
        "n_train_samples": train_agg["n"],
        "n_heldout_samples": heldout_agg["n"],
    })


# ---------------------------------------------------------------------------
# 4. Write CSV + print summary table
# ---------------------------------------------------------------------------
fieldnames = [
    "run", "checkpoint_path", "epoch", "train_top1", "heldout_top1_pooled",
    "heldout_mean_rank_pooled", "n_train_samples", "n_heldout_samples",
]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

chance_top1 = 1 / N_CANDIDATES
chance_mean_rank = (N_CANDIDATES + 1) / 2
print(f"\nChance level (225-way): top-1={chance_top1:.2%}  mean rank={chance_mean_rank:.0f}")

header = f"{'run':<24} {'epoch':>5} {'train_top1':>11} {'heldout_top1':>13} {'heldout_mean_rank':>18}"
print("\n" + header)
print("-" * len(header))
for r in rows:
    print(
        f"{r['run']:<24} {r['epoch']:>5} {r['train_top1']:>11.2%} "
        f"{r['heldout_top1_pooled']:>13.2%} {r['heldout_mean_rank_pooled']:>18.2f}"
    )

print(f"\nWrote {OUT_CSV}")
