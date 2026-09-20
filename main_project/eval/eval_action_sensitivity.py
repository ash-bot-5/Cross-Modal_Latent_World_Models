"""
Action-sensitivity diagnostic: for a fixed state z_t and fixed goal z_star, does varying the
candidate action chunk actually move cos(z_pred, z_star)? This is the property MPPI planning
literally depends on (it selects among candidate action chunks by this exact score) — none of the
other comparison metrics (held-out ranking accuracy, margin, progress correlation) test it, since
they either hold the action fixed (real, expert-taken) or don't condition on actions at all.

For every (episode, t) in the same 100-episode held-out subsample used by
eval_model_comparison_clip_lora_vs_upweighted_100ep.py (reused verbatim for direct comparability):
  - z_star = z_sem[t + HORIZON]  (same positive target ranking_accuracy scores against elsewhere)
  - real action: actions[t:t+HORIZON], normalized with train-split lo/hi (same convention used
    everywhere else in this project)
  - K=32 random action chunks, sampled uniformly in the same normalized [-1, 1] range the model
    was trained on (the model's full in-distribution action range — not a tighter
    Gaussian-around-nominal like MPPI itself samples, since this measures the model's general
    action-sensitivity ceiling independent of any one planner's sampling sigma)
  - one batched MLP forward over all K+1 chunks (real + random) against the same z_t

Reports per (episode, t):
  - candidate_std: std of cos_sim across the K random candidates — near 0 means MPPI's
    softmax-weighted selection has no signal to exploit at that state, regardless of how good the
    model looks on ranking/margin/progress metrics.
  - real_action_rank: 1 + count of random candidates scoring higher than the real (expert) action
    — tests whether the model favors the actual expert action over noise, not just whether it
    responds to *something*.

Aggregated per model into mean/median candidate_std and mean real-action percentile.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_action_sensitivity.py
"""

import os
import sys
import csv
import random
import pickle
import statistics

import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)  # flush every print() immediately, even when piped
                                              # through tee/redirected to a file (Python otherwise
                                              # block-buffers non-tty stdout)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # main_project/
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "training"))  # so dynamics_model_clip_lora_finetune.py's
                                                          # internal `from dynamics_model import ...`
                                                          # resolves when imported from here

from langtable_dataset import train_val_split
from training.dynamics_model import LatentDynamicsModel
from training.clip_lora import apply_lora_to_visual_encoder, load_lora_state_dict
from training.dynamics_model_clip_lora_finetune import (
    build_probe_episode_records,
    load_episode_frames_mmap,
)

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
HORIZON = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_VAL_EPISODES = 750
VAL_SEED = 0
N_HELDOUT_SUBSAMPLE = 100
SUBSAMPLE_SEED = 0
ENCODE_BATCH_SIZE = 64
K_CANDIDATES = 32

UPWEIGHTED_CKPT = os.path.join(REPO_ROOT, "checkpoints_upweighted", "best.pt")
CLIP_LORA_CHECKPOINTS = [
    ("clip_lora_epoch0", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_000.pt")),
    ("clip_lora_epoch8", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_008.pt")),
    ("clip_lora_epoch13", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_013.pt")),
]
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "action_sensitivity_comparison.csv")

for _p in [UPWEIGHTED_CKPT] + [p for _, p in CLIP_LORA_CHECKPOINTS]:
    if not os.path.exists(_p):
        raise FileNotFoundError(f"No checkpoint at {_p}")

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Held-out split + 100-episode subsample + action stats — identical to
#    eval_model_comparison_clip_lora_vs_upweighted_100ep.py, so results line up 1:1
#    (same n_heldout_samples / n_probe_points expected: 5211)
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
        encoded[stem] = {"Z": z_obs}
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
            encoded[stem] = {"Z": Z}
            print(f"  [{i}/{n}] encoded {stem} (T={T})")
    if was_training:
        clip_model.train()
    return encoded


def action_sensitivity(mlp: torch.nn.Module, encoded: dict) -> dict:
    """For every (episode, t): K random action chunks + the real one, one batched MLP forward,
    scored against z_sem[t+HORIZON]. Returns per-probe-point candidate_std and real_action_rank,
    aggregated to mean/median."""
    candidate_stds = []
    real_ranks = []
    n = len(encoded)

    for i, (stem, enc) in enumerate(encoded.items(), start=1):
        with open(os.path.join(TRAIN_DIR, stem + ".pkl"), "rb") as f:
            ep = pickle.load(f)
        actions_raw = ep["actions"]
        z_sem = np.load(os.path.join(TRAIN_DIR, stem + "_zsem.npz"))["z_sem"]

        z_obs = enc["Z"]
        T = z_obs.shape[0]

        with torch.no_grad():
            for t in range(T - HORIZON):
                z_t = torch.tensor(z_obs[t], dtype=torch.float32, device=DEVICE)
                z_t_batch = z_t.unsqueeze(0).repeat(K_CANDIDATES + 1, 1)  # (K+1, 768)

                real_chunk = torch.tensor(np.array(actions_raw[t : t + HORIZON], dtype=np.float32))
                real_norm = ((real_chunk - lo) / (hi - lo) * 2.0 - 1.0)  # (HORIZON, 2)
                random_actions = torch.empty(K_CANDIDATES, HORIZON, 2).uniform_(-1.0, 1.0)
                actions_batch = torch.cat([real_norm.unsqueeze(0), random_actions], dim=0).to(DEVICE)  # (K+1, H, 2)

                z_pred_batch = mlp(z_t_batch, actions_batch)  # (K+1, 768)

                z_star = torch.tensor(z_sem[t + HORIZON], dtype=torch.float32, device=DEVICE)
                cos_sim = F.cosine_similarity(z_pred_batch, z_star.unsqueeze(0).expand(K_CANDIDATES + 1, -1), dim=-1)

                real_sim = cos_sim[0].item()
                random_sims = cos_sim[1:].cpu().numpy()

                candidate_stds.append(float(random_sims.std()))
                real_ranks.append(1 + int((random_sims > real_sim).sum()))

        if i % 10 == 0 or i == n:
            print(f"  [{i}/{n}] probed {stem} ({len(candidate_stds)} probe points so far)")

    real_percentiles = [1.0 - (r - 1) / K_CANDIDATES for r in real_ranks]
    return {
        "mean_candidate_std": statistics.mean(candidate_stds),
        "median_candidate_std": statistics.median(candidate_stds),
        "mean_real_action_percentile": statistics.mean(real_percentiles),
        "n": len(candidate_stds),
    }


def evaluate_and_row(label: str, mlp: torch.nn.Module, encoded: dict) -> dict:
    print("Probing action sensitivity ...")
    result = action_sensitivity(mlp, encoded)
    print(f"  mean_candidate_std={result['mean_candidate_std']:.4f}  "
          f"mean_real_action_percentile={result['mean_real_action_percentile']:.2%}\n")
    return {
        "model": label,
        "mean_candidate_std": result["mean_candidate_std"],
        "median_candidate_std": result["median_candidate_std"],
        "mean_real_action_percentile": result["mean_real_action_percentile"],
        "n_probe_points": result["n"],
        "k_candidates": K_CANDIDATES,
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
    "model", "mean_candidate_std", "median_candidate_std", "mean_real_action_percentile",
    "n_probe_points", "k_candidates",
]
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

header = f"{'model':<20} {'mean_std':>10} {'median_std':>11} {'real_pctile':>12} {'n':>7}"
print("\n" + header)
print("-" * len(header))
for r in rows:
    print(
        f"{r['model']:<20} {r['mean_candidate_std']:>10.4f} {r['median_candidate_std']:>11.4f} "
        f"{r['mean_real_action_percentile']:>12.2%} {r['n_probe_points']:>7}"
    )

print(f"\nWrote {OUT_CSV}")
