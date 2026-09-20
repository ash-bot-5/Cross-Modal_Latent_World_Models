"""
Same comparison as eval_model_comparison_clip_lora_vs_upweighted.py, but:
  - runs on a random 100-episode subsample of the 750-episode held-out set (deterministic,
    seed=0), instead of all 750 — much faster, at the cost of noisier per-row estimates.
  - covers all 4 models in a single run: upweighted_best, plus 3 CLIP-LoRA checkpoints
    (epoch 0, 8, 13), instead of one hardcoded CLIP-LoRA checkpoint.

Writes to a separate CSV (model_comparison_clip_lora_vs_upweighted_100ep.csv) rather than the
750-episode results file — different sample size, not meant to be mixed with those rows.

See eval_model_comparison_clip_lora_vs_upweighted.py's docstring for the metric definitions
(held-out 449-way InfoNCE ranking accuracy; margin + progress-curve Pearson correlation).

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_model_comparison_clip_lora_vs_upweighted_100ep.py
"""

import os
import sys
import csv
import random
import pickle
import statistics

import numpy as np
import torch
import open_clip
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)  # flush every print() immediately, even when piped
                                              # through tee/redirected to a file (Python otherwise
                                              # block-buffers non-tty stdout, which made an earlier
                                              # run of the sibling script look "stuck" for minutes)

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
    compute_margin_probe,
    compute_progress_curve_correlation,
    load_episode_frames_mmap,
)

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
HORIZON = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_VAL_EPISODES = 750
VAL_SEED = 0
N_HELDOUT_SUBSAMPLE = 100
SUBSAMPLE_SEED = 0
NEG_COUNT = 448  # current cached negative count (post 224->448 upgrade)
N_CANDIDATES = NEG_COUNT + 1
ENCODE_BATCH_SIZE = 64

UPWEIGHTED_CKPT = os.path.join(REPO_ROOT, "checkpoints_upweighted", "best.pt")
CLIP_LORA_CHECKPOINTS = [
    ("clip_lora_epoch0", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_000.pt")),
    ("clip_lora_epoch8", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_008.pt")),
    ("clip_lora_epoch13", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_013.pt")),
]
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_comparison_clip_lora_vs_upweighted_100ep.csv")

for _p in [UPWEIGHTED_CKPT] + [p for _, p in CLIP_LORA_CHECKPOINTS]:
    if not os.path.exists(_p):
        raise FileNotFoundError(f"No checkpoint at {_p}")

torch.manual_seed(0)

temperature = InfoNCELoss().temperature


# ---------------------------------------------------------------------------
# 1. Held-out split + 100-episode subsample + action stats (from the TRAIN split only,
#    matching what both models were actually trained on)
# ---------------------------------------------------------------------------
print(f"Computing train/val split (seed={VAL_SEED}, n_val={N_VAL_EPISODES}) ...")
train_files, val_files = train_val_split(TRAIN_DIR, n_val=N_VAL_EPISODES, seed=VAL_SEED)
val_stems = [f[:-4] for f in val_files]  # strip ".pkl"
print(f"  {len(train_files)} train / {len(val_files)} val episodes")

val_stems = sorted(random.Random(SUBSAMPLE_SEED).sample(val_stems, N_HELDOUT_SUBSAMPLE))
print(f"  subsampled to {len(val_stems)} held-out episodes (seed={SUBSAMPLE_SEED})")

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


def encode_probe_episodes_verbose(clip_model, records: dict, preprocess, device: str, batch_size: int = 64) -> dict:
    """Same as dynamics_model_clip_lora_finetune.encode_probe_episodes, but prints a
    [i/n] line per episode so progress is visible in the terminal during this run's
    slowest phase, instead of one opaque function call."""
    was_training = clip_model.training
    clip_model.eval()
    encoded = {}
    n = len(records)
    with torch.no_grad():
        for i, (stem, rec) in enumerate(records.items(), start=1):
            frames = load_episode_frames_mmap(rec["frames_path"])
            T = len(frames)
            imgs = torch.stack([preprocess(Image.fromarray(np.asarray(frames[t]))) for t in range(T)])
            z_parts = []
            for start in range(0, T, batch_size):
                batch = imgs[start : start + batch_size].to(device)
                z = clip_model.encode_image(batch, normalize=True)
                z_parts.append(z.cpu().float().numpy())
            Z = np.concatenate(z_parts, axis=0)
            encoded[stem] = {"Z": Z, "touching": rec["touching"], "dist_ab": rec["dist_ab"]}
            print(f"  [{i}/{n}] encoded {stem} (T={T})")
    if was_training:
        clip_model.train()
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
    n = len(encoded)
    for i, (stem, enc) in enumerate(encoded.items(), start=1):
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

        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] scored {stem} ({len(results)} samples so far)")

    ranks = [r["rank"] for r in results]
    return {
        "top1": sum(r["top1"] for r in results) / len(results),
        "mean_rank": sum(ranks) / len(ranks),
        "median_rank": statistics.median(ranks),
        "n": len(results),
    }


def evaluate_and_row(label: str, mlp: torch.nn.Module, encoded: dict) -> dict:
    margin, per_ep_centroids = compute_margin_probe(encoded)
    progress_corr = compute_progress_curve_correlation(encoded, per_ep_centroids)
    print("Scoring held-out ranking accuracy ...")
    rank = ranking_accuracy(mlp, encoded)
    print(f"  top1={rank['top1']:.4f}  mean_rank={rank['mean_rank']:.2f}  "
          f"margin={margin['mean']:.4f}  progress_corr={progress_corr:.4f}\n")
    return {
        "model": label,
        "heldout_top1": rank["top1"],
        "heldout_mean_rank": rank["mean_rank"],
        "heldout_median_rank": rank["median_rank"],
        "n_heldout_samples": rank["n"],
        "margin_mean": margin["mean"],
        "margin_std": margin["std"],
        "progress_pearson_corr": progress_corr,
        "n_qualifying_episodes": margin["n_qualifying"],
    }


rows = []

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
rows.append(evaluate_and_row("upweighted_best", mlp_up, encoded_up))

# ---------------------------------------------------------------------------
# 4. clip_lora epoch 0, 8, 13 — each rebuilt from scratch (no LoRA-state leakage between epochs)
# ---------------------------------------------------------------------------
for label, ckpt_path in CLIP_LORA_CHECKPOINTS:
    print(f"=== {label} ===")
    print(f"Checkpoint: {ckpt_path}")
    print("Loading OpenCLIP ViT-L/14 (datacomp_xl_s13b_b90k) + LoRA adapter ...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    apply_lora_to_visual_encoder(clip_model, r=16, alpha=32, dropout=0.2)
    clip_model = clip_model.to(DEVICE)

    ckpt_lora = torch.load(ckpt_path, map_location=DEVICE)
    load_lora_state_dict(clip_model, ckpt_lora["lora_state_dict"])
    clip_model.eval()

    mlp_lora = LatentDynamicsModel().to(DEVICE)
    mlp_lora.load_state_dict(ckpt_lora["mlp_state_dict"])
    mlp_lora.eval()
    print(f"Model loaded (epoch {ckpt_lora['epoch']}, step {ckpt_lora.get('step', '?')})")

    print(f"Encoding {len(val_stems)} held-out episodes' frames through the LoRA CLIP visual "
          f"encoder on {DEVICE} ...")
    encoded_lora = encode_probe_episodes_verbose(clip_model, records, preprocess, device=DEVICE, batch_size=ENCODE_BATCH_SIZE)

    rows.append(evaluate_and_row(label, mlp_lora, encoded_lora))

# ---------------------------------------------------------------------------
# 5. Write CSV + print summary table
# ---------------------------------------------------------------------------
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
