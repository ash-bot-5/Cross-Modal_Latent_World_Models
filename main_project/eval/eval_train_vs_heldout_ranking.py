"""
InfoNCE ranking accuracy: train sample vs. 3 held-out episodes, both compared
under two hard-negative conventions:

  neg@t      - hard negatives from neg_indices[t]        (matches the literal
                training loss, as LangTableDataset.__getitem__ implements it
                today)
  neg@t+H    - hard negatives from neg_indices[t+HORIZON]  (conceptually
                correct: guaranteed false at the same timestep as the
                z_sem_target the model is actually predicting)

For each sampled (episode, t): z_pred = model(z_t, actions[t:t+H]) is scored
against 1 positive (z_sem[t+H]) + 224 hard negatives, forming a 225-way
softmax exactly as InfoNCELoss.forward does (same temperature). Records
top-1/top-5/top-10 correctness, rank of the true positive, and raw
(pre-temperature) cosine similarities.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_train_vs_heldout_ranking.py
"""

import os
import random
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langtable_dataset import LangTableDataset
from training.dynamics_model import LatentDynamicsModel, InfoNCELoss

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(REPO_ROOT, "checkpoints", "latest.pt")
TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
HELD_DIR  = "/home/ashwink/3_expert_trajs/"
HORIZON   = 8
N_TRAIN_SAMPLES = 1000
DEVICE    = "cpu"  # eval is fast enough on CPU; avoids competing with training GPU

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

N_CANDIDATES = 225  # 1 positive + 224 hard negatives


# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
print(f"Checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
model = LatentDynamicsModel().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded (epoch {ckpt['epoch']}, step {ckpt.get('step', '?')})")

temperature = InfoNCELoss().temperature

# ---------------------------------------------------------------------------
# 2. Load datasets
# ---------------------------------------------------------------------------
print(f"Loading train dataset from {TRAIN_DIR} ...")
train_dataset = LangTableDataset(TRAIN_DIR, device=DEVICE)
print(f"  {len(train_dataset)} valid (episode, t) samples")

print(f"Loading held-out dataset from {HELD_DIR} ...")
heldout_dataset = LangTableDataset(HELD_DIR, device=DEVICE)
print(f"  {len(heldout_dataset)} valid (episode, t) samples")

# Held-out actions must be normalized with the TRAINING set's min/max stats,
# not the 3-episode subset's own stats (LangTableDataset always computes its
# own action_stats from whatever's in data_dir at __init__ time).
heldout_dataset.action_stats = train_dataset.action_stats

# Map held-out ep_idx -> episode stem, via the same sorted-glob order LangTableDataset uses.
heldout_stems = sorted(f[:-4] for f in os.listdir(HELD_DIR) if f.endswith(".pkl"))
print(f"  held-out episodes (ep_idx order): {heldout_stems}")


# ---------------------------------------------------------------------------
# 3. Per-sample evaluation
# ---------------------------------------------------------------------------
def gather_hard_neg_at(dataset: LangTableDataset, ep_idx: int, t: int) -> torch.Tensor:
    """Same gather LangTableDataset.__getitem__ does for z_hard_neg, at an arbitrary timestep."""
    idx = dataset.neg_indices[ep_idx][t].astype(np.int64)  # (224,)
    return dataset.unique_zsem[ep_idx][idx]                # (224, 768)


def score(z_pred: torch.Tensor, z_pos: torch.Tensor, z_hard_neg: torch.Tensor) -> dict:
    """Exact InfoNCELoss.forward formula, kept unbatched to expose per-sample rank/top-k."""
    pos_sim_raw  = (z_pred * z_pos).sum(dim=-1)              # scalar, pre-temperature
    hard_sim_raw = (z_pred * z_hard_neg).sum(dim=-1)          # (224,), pre-temperature

    logits = torch.cat([pos_sim_raw.unsqueeze(0), hard_sim_raw]) / temperature  # (225,)
    rank = int((logits > logits[0]).sum().item()) + 1  # 1 = best; ties don't inflate rank

    return {
        "top1": int(torch.argmax(logits).item() == 0),
        "top5": int(rank <= 5),
        "top10": int(rank <= 10),
        "rank": rank,
        "pos_sim": pos_sim_raw.item(),
        "hard_sim": hard_sim_raw.mean().item(),
    }


def eval_point(dataset: LangTableDataset, flat_idx: int) -> dict:
    ep_idx, t = dataset.indices[flat_idx]

    sample = dataset[flat_idx]
    z_t          = sample["z_t"].unsqueeze(0)
    actions      = sample["actions"].unsqueeze(0)
    z_sem_target = sample["z_sem_target"]
    z_hard_neg_t = sample["z_hard_neg"]                       # neg@t (dataset's own convention)
    z_hard_neg_h = gather_hard_neg_at(dataset, ep_idx, t + HORIZON)  # neg@t+HORIZON

    with torch.no_grad():
        z_pred = model(z_t, actions).squeeze(0)

    return {
        "ep_idx": ep_idx,
        "t": t,
        "neg@t":   score(z_pred, z_sem_target, z_hard_neg_t),
        "neg@t+H": score(z_pred, z_sem_target, z_hard_neg_h),
    }


# ---------------------------------------------------------------------------
# 4. Run over train sample + all held-out points
# ---------------------------------------------------------------------------
train_sample_idx = random.sample(range(len(train_dataset)), N_TRAIN_SAMPLES)
print(f"\nEvaluating {len(train_sample_idx)} train samples ...")
train_results = [eval_point(train_dataset, i) for i in train_sample_idx]

print(f"Evaluating {len(heldout_dataset)} held-out samples (all valid t) ...")
heldout_results = [eval_point(heldout_dataset, i) for i in range(len(heldout_dataset))]


# ---------------------------------------------------------------------------
# 5. Aggregate + report
# ---------------------------------------------------------------------------
def aggregate(results: list[dict], variant: str) -> dict:
    pts = [r[variant] for r in results]
    n = len(pts)
    return {
        "n": n,
        "top1":   sum(p["top1"] for p in pts) / n,
        "top5":   sum(p["top5"] for p in pts) / n,
        "top10":  sum(p["top10"] for p in pts) / n,
        "mean_rank":   sum(p["rank"] for p in pts) / n,
        "median_rank": statistics.median(p["rank"] for p in pts),
        "mean_pos_sim":  sum(p["pos_sim"] for p in pts) / n,
        "mean_hard_sim": sum(p["hard_sim"] for p in pts) / n,
    }


rows = []
for variant in ["neg@t", "neg@t+H"]:
    rows.append((f"Train (n={N_TRAIN_SAMPLES}) [{variant}]", aggregate(train_results, variant)))
for variant in ["neg@t", "neg@t+H"]:
    rows.append((f"Held-out pooled (n={len(heldout_results)}) [{variant}]", aggregate(heldout_results, variant)))
for ep_idx, stem in enumerate(heldout_stems):
    ep_results = [r for r in heldout_results if r["ep_idx"] == ep_idx]
    for variant in ["neg@t", "neg@t+H"]:
        rows.append((f"  {stem} (n={len(ep_results)}) [{variant}]", aggregate(ep_results, variant)))

# ---------------------------------------------------------------------------
# 6. Print summary table
# ---------------------------------------------------------------------------
chance_top1 = 1 / N_CANDIDATES
chance_top5 = 5 / N_CANDIDATES
chance_top10 = 10 / N_CANDIDATES
chance_mean_rank = (N_CANDIDATES + 1) / 2
print(
    f"\nChance level (225-way): top-1={chance_top1:.2%}  top-5={chance_top5:.2%}  "
    f"top-10={chance_top10:.2%}  mean rank={chance_mean_rank:.0f}"
)

header = f"{'group':<42} {'n':>5} {'top1':>7} {'top5':>7} {'top10':>7} {'mean_rk':>8} {'med_rk':>7} {'pos_sim':>8} {'hard_sim':>9}"
print("\n" + header)
print("-" * len(header))
for name, m in rows:
    print(
        f"{name:<42} {m['n']:>5} {m['top1']:>7.2%} {m['top5']:>7.2%} {m['top10']:>7.2%} "
        f"{m['mean_rank']:>8.1f} {m['median_rank']:>7.1f} {m['mean_pos_sim']:>8.4f} {m['mean_hard_sim']:>9.4f}"
    )
