"""
Quantify how much better/worse the CLIP-LoRA jointly fine-tuned dynamics model (epoch 8) is
compared to the upweighted-baseline FiLM-MLP model, in terms that matter for planning: MPPI
selects actions by maximizing cos(z_pred, z_star), so what matters is (1) whether z_pred reliably
ranks the true goal above decoys at dataset scale, and (2) whether cos(z_pred, z_star) moves
monotonically as the robot makes real progress toward the goal.

Both models are evaluated on the exact same 750-episode held-out validation split (confirmed
identical across both training runs' train_val_split.json, seed=0).

Metric 1 — held-out ranking accuracy (449-way: 1 positive + 448 hard negatives, the actual
InfoNCE training objective, using the current cached negative count):
  z_pred = model(z_t, actions[t:t+HORIZON]), scored against z_sem[t+HORIZON] (positive) and
  448 hard negatives gathered via neg_indices[t+HORIZON] -> unique_zsem, pooled across every
  (episode, t) in the held-out set.

Metric 2 — progress-curve correlation (reuses build_probe_episode_records / encode_probe_episodes
/ compute_margin_probe / compute_progress_curve_correlation from dynamics_model_clip_lora_finetune.py
unmodified — those functions are generic over clip_model/preprocess, frozen or LoRA alike):
  within-episode contrastive margin, and Pearson correlation of signed cosine-centroid progress
  vs. ground-truth task-relevant block distance, pooled over the same 750 held-out episodes.

z_t source differs per model:
  - upweighted: read straight from the cached frozen-CLIP _zobs.npz (that's what it was trained on).
  - clip_lora_epoch8: recomputed on-the-fly through the LoRA-loaded CLIP visual encoder (this
    model jointly fine-tuned the image encoder, so the old frozen cache would not reflect it).

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_model_comparison_clip_lora_vs_upweighted.py
"""

import os
import sys
import csv
import pickle
import statistics

import numpy as np
import torch
import open_clip

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # main_project/
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))  # so dynamics_model_clip_lora_finetune.py's
                                                          # internal `from dynamics_model import ...`
                                                          # resolves when imported from here

from langtable_dataset import train_val_split
from training.dynamics_model import LatentDynamicsModel, InfoNCELoss
from training.clip_lora import apply_lora_to_visual_encoder, load_lora_state_dict
from training.dynamics_model_clip_lora_finetune import (
    build_probe_episode_records,
    encode_probe_episodes,
    compute_margin_probe,
    compute_progress_curve_correlation,
)

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
HORIZON = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_VAL_EPISODES = 750
VAL_SEED = 0
NEG_COUNT = 448  # current cached negative count (post 224->448 upgrade)
N_CANDIDATES = NEG_COUNT + 1
ENCODE_BATCH_SIZE = 64

UPWEIGHTED_CKPT = os.path.join(REPO_ROOT, "checkpoints_upweighted", "best.pt")
CLIP_LORA_CKPT = os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_008.pt")
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_comparison_clip_lora_vs_upweighted.csv")

for _p in (UPWEIGHTED_CKPT, CLIP_LORA_CKPT):
    if not os.path.exists(_p):
        raise FileNotFoundError(f"No checkpoint at {_p}")

torch.manual_seed(0)

temperature = InfoNCELoss().temperature


# ---------------------------------------------------------------------------
# 1. Held-out split + action stats (from the TRAIN split only, matching what both
#    models were actually trained on)
# ---------------------------------------------------------------------------
print(f"Computing train/val split (seed={VAL_SEED}, n_val={N_VAL_EPISODES}) ...")
train_files, val_files = train_val_split(TRAIN_DIR, n_val=N_VAL_EPISODES, seed=VAL_SEED)
val_stems = [f[:-4] for f in val_files]  # strip ".pkl"
print(f"  {len(train_files)} train / {len(val_files)} val episodes")

print("Computing action stats from train split ...")
all_actions = []
for pkl_file in train_files:
    with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
        ep = pickle.load(f)
    all_actions.append(np.array(ep["actions"], dtype=np.float32))
all_actions = np.concatenate(all_actions, axis=0)
lo = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)
hi = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)
print(f"  action shape: {all_actions.shape}  lo={lo.tolist()}  hi={hi.tolist()}\n")

print(f"Building probe records for {len(val_stems)} held-out episodes ...")
records = build_probe_episode_records(TRAIN_DIR, val_stems)


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
def load_cached_zobs_as_encoded(records: dict, data_dir: str) -> dict:
    """Same return shape as encode_probe_episodes, but reads the pre-cached frozen-CLIP
    _zobs.npz directly instead of re-running CLIP — correct for the upweighted baseline,
    which was trained/evaluated against exactly this frozen embedding."""
    encoded = {}
    for stem, rec in records.items():
        z_obs = np.load(os.path.join(data_dir, stem + "_zobs.npz"))["z_obs"].astype(np.float32)
        encoded[stem] = {"Z": z_obs, "touching": rec["touching"], "dist_ab": rec["dist_ab"]}
    return encoded


def score(z_pred: torch.Tensor, z_pos: torch.Tensor, z_hard_neg: torch.Tensor) -> dict:
    pos_sim_raw = (z_pred * z_pos).sum(dim=-1)
    hard_sim_raw = (z_pred * z_hard_neg).sum(dim=-1)
    logits = torch.cat([pos_sim_raw.unsqueeze(0), hard_sim_raw]) / temperature  # (449,)
    rank = int((logits > logits[0]).sum().item()) + 1  # 1 = best; ties don't inflate rank
    return {"top1": int(torch.argmax(logits).item() == 0), "rank": rank}


def ranking_accuracy(mlp: torch.nn.Module, encoded: dict) -> dict:
    """449-way held-out InfoNCE ranking accuracy, pooled over every (episode, t)."""
    results = []
    for stem, enc in encoded.items():
        with open(os.path.join(TRAIN_DIR, stem + ".pkl"), "rb") as f:
            ep = pickle.load(f)
        actions_raw = ep["actions"]
        z_sem = np.load(os.path.join(TRAIN_DIR, stem + "_zsem.npz"))["z_sem"]
        zhard = np.load(os.path.join(TRAIN_DIR, stem + "_zhard.npz"))
        unique_zsem = zhard["unique_zsem"]
        neg_indices = zhard["neg_indices"]

        z_obs = enc["Z"]
        T = z_obs.shape[0]

        with torch.no_grad():
            for t in range(T - HORIZON):
                z_t = torch.tensor(z_obs[t], dtype=torch.float32, device=DEVICE).unsqueeze(0)
                chunk = torch.tensor(np.array(actions_raw[t : t + HORIZON], dtype=np.float32))
                actions_norm = ((chunk - lo) / (hi - lo) * 2.0 - 1.0).unsqueeze(0).to(DEVICE)

                z_pred = mlp(z_t, actions_norm).squeeze(0).cpu()

                z_pos = torch.tensor(z_sem[t + HORIZON], dtype=torch.float32)
                z_neg = torch.tensor(unique_zsem[neg_indices[t + HORIZON]], dtype=torch.float32)

                results.append(score(z_pred, z_pos, z_neg))

    ranks = [r["rank"] for r in results]
    return {
        "top1": sum(r["top1"] for r in results) / len(results),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": statistics.median(ranks),
        "n": len(results),
    }


# ---------------------------------------------------------------------------
# 3. upweighted (baseline)
# ---------------------------------------------------------------------------
print("=== upweighted (baseline) ===")
print(f"Checkpoint: {UPWEIGHTED_CKPT}")
ckpt_up = torch.load(UPWEIGHTED_CKPT, map_location=DEVICE)
mlp_up = LatentDynamicsModel().to(DEVICE)
mlp_up.load_state_dict(ckpt_up["model_state_dict"])
mlp_up.eval()
print(f"Model loaded (epoch {ckpt_up['epoch']}, step {ckpt_up.get('step', '?')})")

encoded_up = load_cached_zobs_as_encoded(records, TRAIN_DIR)
margin_up, per_ep_centroids_up = compute_margin_probe(encoded_up)
progress_corr_up = compute_progress_curve_correlation(encoded_up, per_ep_centroids_up)
print("Scoring held-out ranking accuracy ...")
rank_up = ranking_accuracy(mlp_up, encoded_up)
print(f"  top1={rank_up['top1']:.4f}  mean_rank={rank_up['mean_rank']:.2f}  "
      f"margin={margin_up['mean']:.4f}  progress_corr={progress_corr_up:.4f}\n")

# ---------------------------------------------------------------------------
# 4. clip_lora_epoch8
# ---------------------------------------------------------------------------
print("=== clip_lora_epoch8 ===")
print(f"Checkpoint: {CLIP_LORA_CKPT}")
print("Loading OpenCLIP ViT-L/14 (datacomp_xl_s13b_b90k) + LoRA adapter ...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
)
apply_lora_to_visual_encoder(clip_model, r=16, alpha=32, dropout=0.2)
clip_model = clip_model.to(DEVICE)

ckpt_lora = torch.load(CLIP_LORA_CKPT, map_location=DEVICE)
load_lora_state_dict(clip_model, ckpt_lora["lora_state_dict"])
clip_model.eval()

mlp_lora = LatentDynamicsModel().to(DEVICE)
mlp_lora.load_state_dict(ckpt_lora["mlp_state_dict"])
mlp_lora.eval()
print(f"Model loaded (epoch {ckpt_lora['epoch']}, step {ckpt_lora.get('step', '?')})")

print(f"Encoding {len(val_stems)} held-out episodes' frames through the LoRA CLIP visual "
      f"encoder on {DEVICE} (this is the slow part) ...")
encoded_lora = encode_probe_episodes(clip_model, records, preprocess, device=DEVICE, batch_size=ENCODE_BATCH_SIZE)

margin_lora, per_ep_centroids_lora = compute_margin_probe(encoded_lora)
progress_corr_lora = compute_progress_curve_correlation(encoded_lora, per_ep_centroids_lora)
print("Scoring held-out ranking accuracy ...")
rank_lora = ranking_accuracy(mlp_lora, encoded_lora)
print(f"  top1={rank_lora['top1']:.4f}  mean_rank={rank_lora['mean_rank']:.2f}  "
      f"margin={margin_lora['mean']:.4f}  progress_corr={progress_corr_lora:.4f}\n")

# ---------------------------------------------------------------------------
# 5. Write CSV + print summary table
# ---------------------------------------------------------------------------
rows = [
    {
        "model": "upweighted_best",
        "heldout_top1": rank_up["top1"],
        "heldout_mean_rank": rank_up["mean_rank"],
        "heldout_median_rank": rank_up["median_rank"],
        "n_heldout_samples": rank_up["n"],
        "margin_mean": margin_up["mean"],
        "margin_std": margin_up["std"],
        "progress_pearson_corr": progress_corr_up,
        "n_qualifying_episodes": margin_up["n_qualifying"],
    },
    {
        "model": "clip_lora_epoch8",
        "heldout_top1": rank_lora["top1"],
        "heldout_mean_rank": rank_lora["mean_rank"],
        "heldout_median_rank": rank_lora["median_rank"],
        "n_heldout_samples": rank_lora["n"],
        "margin_mean": margin_lora["mean"],
        "margin_std": margin_lora["std"],
        "progress_pearson_corr": progress_corr_lora,
        "n_qualifying_episodes": margin_lora["n_qualifying"],
    },
]

fieldnames = [
    "model", "heldout_top1", "heldout_mean_rank", "heldout_median_rank", "n_heldout_samples",
    "margin_mean", "margin_std", "progress_pearson_corr", "n_qualifying_episodes",
]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

chance_top1 = 1 / N_CANDIDATES
chance_mean_rank = (N_CANDIDATES + 1) / 2
print(f"Chance level ({N_CANDIDATES}-way): top-1={chance_top1:.2%}  mean rank={chance_mean_rank:.0f}")

header = (f"{'model':<20} {'top1':>8} {'mean_rank':>10} {'median_rank':>12} "
          f"{'margin':>9} {'progress_corr':>14}")
print("\n" + header)
print("-" * len(header))
for r in rows:
    print(
        f"{r['model']:<20} {r['heldout_top1']:>8.2%} {r['heldout_mean_rank']:>10.2f} "
        f"{r['heldout_median_rank']:>12.1f} {r['margin_mean']:>9.4f} {r['progress_pearson_corr']:>14.4f}"
    )

print(f"\nWrote {OUT_CSV}")
