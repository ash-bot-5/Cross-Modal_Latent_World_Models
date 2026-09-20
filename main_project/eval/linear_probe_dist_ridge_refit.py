"""
Narrow refit of ONLY the 2 distance-regression task-relevant probes (dist_peg_start, dist_ab)
from linear_probe_MLP_output.py, using RidgeCV instead of a bare LinearRegression.

Motivation: linear_probe_MLP_output.py's fit_regression_probe() fits an unregularized
LinearRegression on the full 768-dim z_pred against only ~2878 training rows -- at that
feature/sample ratio (~0.27), OLS overfits (train_r2 ~0.45-0.77, val_r2 from -0.66 to +0.11 in
every epoch, both checkpoint dirs; confirmed against results.json). The t+8 indexing itself
(verified directly in sample_ground_truth_data()) is correct -- this is a probe-capacity problem,
not a data pipeline bug. RidgeCV's cross-validated L2 penalty should make val_r2 a more
meaningful signal of the linear structure z_pred actually captures.

This script deliberately does NOT re-run the full linear_probe_MLP_output.py pipeline, which
would also touch the 2 classification line charts (peg_touch_start, ab_touch) and all
general_state/ bar charts. Instead it:
  - reuses the cached ground_truth_data.npz (checkpoint-independent, force_recompute=False)
  - reuses the cached zpred_cache/{tag}_epoch{epoch:03d}.npz per checkpoint (force_recompute=
    False) -- only computes+caches a fresh one for an epoch that doesn't have a cache yet (e.g.
    a checkpoint dir that has grown a new epoch_*.pt since the last full run)
  - refits ONLY dist_peg_start and dist_ab with RidgeCV, and merges just those rows into
    results.json via the same (checkpoint_dir_tag, epoch, probe, block, filter) key used by
    linear_probe_MLP_output.py's merge_results -- so every peg_touch_start/ab_touch/
    peg_touch_block row is left byte-identical
  - re-renders ONLY dist_peg_start_val_r2_vs_epoch.png and dist_ab_val_r2_vs_epoch.png

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/linear_probe_dist_ridge_refit.py
"""

import os
import re
import json
import glob
import pickle
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Config -- verbatim from linear_probe_MLP_output.py
# ---------------------------------------------------------------------------
HORIZON = 8
TOUCH_THRESH = 0.05
ACTION_NORM_CONST = 0.035

DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_CKPT_DIRS = [
    os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_finetuned_wider_hl"),
    os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data"),
]
CKPT_DIR_LABELS = {
    "checkpoints_clip_full_finetuned_wider_hl": "expert-only_model",
    "checkpoints_clip_full_fine-tuned_with_subopt_data": "expert+subopt-model",
}
DEFAULT_OUT_DIR = os.path.join(MAIN_PROJECT_DIR, "linear_probe_MLP_output_charts")

RIDGE_ALPHAS = np.logspace(-3, 4, 30)

TAB10 = matplotlib.colormaps["tab10"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dirs", nargs="+", default=DEFAULT_CKPT_DIRS,
                    help="One or more checkpoint directories containing epoch_*.pt files.")
    p.add_argument("--n-episodes", type=int, default=600)
    p.add_argument("--timesteps-per-episode", type=int, default=6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def ckpt_dir_tag(ckpt_dir: str) -> str:
    base = os.path.basename(os.path.normpath(ckpt_dir))
    return CKPT_DIR_LABELS.get(base, base)


# ---------------------------------------------------------------------------
# Inline model classes -- verbatim from linear_probe_MLP_output.py
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * hidden + beta


class ActionEncoderConv(nn.Module):
    def __init__(self, action_dim: int = 2, horizon: int = 8,
                 conv_channels: int = 64, kernel_size: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(action_dim, conv_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.act = nn.Mish()
        self.out = nn.Linear(conv_channels * horizon, out_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv(actions.transpose(1, 2)))
        return self.out(h.flatten(1))


class LatentDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = ActionEncoderConv()
        self.fc1 = nn.Linear(768, 1024);  self.ln1 = nn.LayerNorm(1024)
        self.film1 = FiLMLayer(256, 1024); self.act1 = nn.Mish()
        self.fc2 = nn.Linear(1024, 1024);  self.ln2 = nn.LayerNorm(1024)
        self.film2 = FiLMLayer(256, 1024); self.act2 = nn.Mish()
        self.fc_out = nn.Linear(1024, 768); self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        global_cond = self.action_encoder(actions)
        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)


def get_block_names(data_dir: str) -> list[str]:
    pkl_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    with open(os.path.join(data_dir, pkl_files[0]), "rb") as f:
        ep = pickle.load(f)
    names = sorted(k for k in ep["block_states"][0].keys() if k != "peg")
    assert len(names) == 8, f"expected 8 block names, got {len(names)}: {names}"
    return names


# ---------------------------------------------------------------------------
# Ground-truth sampling -- verbatim from linear_probe_MLP_output.py
# ---------------------------------------------------------------------------

def sample_ground_truth_data(data_dir: str, block_names: list[str], n_episodes: int,
                              timesteps_per_episode: int, val_frac: float, seed: int,
                              cache_path: str, force_recompute: bool = False) -> dict:
    if os.path.exists(cache_path) and not force_recompute:
        print(f"Loading cached ground-truth data from {cache_path}")
        d = np.load(cache_path, allow_pickle=False)
        return {
            "frames": d["frames"],
            "actions_windows": d["actions_windows"],
            "episode_id": d["episode_id"],
            "start_block": d["start_block"],
            "peg_touch_start": d["peg_touch_start"].astype(bool),
            "ab_touch": d["ab_touch"].astype(bool),
            "dist_peg_start": d["dist_peg_start"].astype(np.float32),
            "dist_ab": d["dist_ab"].astype(np.float32),
            "peg_touch_block": d["peg_touch_block"].astype(bool),
            "is_val": d["is_val"].astype(bool),
        }

    all_files = sorted(glob.glob(os.path.join(data_dir, "*.pkl")))
    shuffled = random.Random(seed).sample(all_files, len(all_files))
    candidate_files = shuffled[:n_episodes]

    n_val_episodes = max(1, round(val_frac * len(candidate_files)))
    val_episode_files = set(random.Random(seed + 1).sample(candidate_files, n_val_episodes))

    frames, actions_windows = [], []
    episode_id, start_block_arr = [], []
    peg_touch_start, ab_touch = [], []
    dist_peg_start, dist_ab = [], []
    peg_touch_block, is_val = [], []

    n_used_episodes, n_skipped_short, n_skipped_missing_qa = 0, 0, 0

    for pkl_path in candidate_files:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        T = len(ep["frames"])
        if T <= HORIZON:
            n_skipped_short += 1
            continue
        if "peg_touching_starting_block_qa" not in ep:
            n_skipped_missing_qa += 1
            continue

        meta = ep["metadata"]
        start_block = meta["start_block"]
        block_states = ep["block_states"]
        peg_positions = ep["actual_peg_positions"]
        actions = ep["actions"]
        task_relevant_qa = ep["task_relevant_qa"]
        peg_qa = ep["peg_touching_starting_block_qa"]
        dist_btwn = ep["dist_btwn_relevant_blocks"]

        stem = os.path.basename(pkl_path)[:-4]
        ep_is_val = pkl_path in val_episode_files

        n_valid_t = T - HORIZON
        n_pick = min(timesteps_per_episode, n_valid_t)
        ts = sorted(set(np.linspace(0, n_valid_t - 1, n_pick).round().astype(int).tolist()))

        for t in ts:
            tf = t + HORIZON
            frame = ep["frames"][t]
            actions_window = np.stack(actions[t:tf], axis=0).astype(np.float32)  # (8,2)

            bpos = block_states[tf]
            peg = peg_positions[tf]

            touch_row = np.array(
                [np.linalg.norm(peg - bpos[b]) < TOUCH_THRESH for b in block_names],
                dtype=bool,
            )

            frames.append(frame)
            actions_windows.append(actions_window)
            episode_id.append(stem)
            start_block_arr.append(start_block)
            peg_touch_start.append(bool(peg_qa[tf]["answer"]))
            ab_touch.append(bool(task_relevant_qa[tf][1]["answer"]))
            dist_peg_start.append(float(np.linalg.norm(peg - bpos[start_block])))
            dist_ab.append(float(dist_btwn[tf]))
            peg_touch_block.append(touch_row)
            is_val.append(ep_is_val)

        n_used_episodes += 1
        if n_used_episodes % 100 == 0:
            print(f"  [{n_used_episodes}/{len(candidate_files)}] episodes processed, "
                  f"{len(frames)} rows so far")

    print(f"Done sampling: {n_used_episodes} episodes used, {len(frames)} rows total "
          f"({n_skipped_short} skipped too-short, {n_skipped_missing_qa} skipped missing QA).")

    data = {
        "frames": np.stack(frames, axis=0),
        "actions_windows": np.stack(actions_windows, axis=0),
        "episode_id": np.array(episode_id, dtype="<U64"),
        "start_block": np.array(start_block_arr, dtype="<U32"),
        "peg_touch_start": np.array(peg_touch_start, dtype=bool),
        "ab_touch": np.array(ab_touch, dtype=bool),
        "dist_peg_start": np.array(dist_peg_start, dtype=np.float32),
        "dist_ab": np.array(dist_ab, dtype=np.float32),
        "peg_touch_block": np.stack(peg_touch_block, axis=0),
        "is_val": np.array(is_val, dtype=bool),
    }
    print(f"Caching ground-truth data to {cache_path} ...")
    np.savez(cache_path, **data)
    return data


# ---------------------------------------------------------------------------
# Per-checkpoint z_pred computation -- verbatim from linear_probe_MLP_output.py
# ---------------------------------------------------------------------------

def compute_zpred_for_checkpoint(ckpt_path: str, frames: np.ndarray, actions_windows: np.ndarray,
                                  batch_size: int, cache_path: str,
                                  force_recompute: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(cache_path) and not force_recompute:
        print(f"    Loading cached z_pred from {cache_path}")
        d = np.load(cache_path)
        return d["z_pred"], d["z_obs"]

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()

    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    z_pred_chunks, z_obs_chunks = [], []
    N = len(frames)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_actions = actions_windows[i:i + batch_size]

            imgs = torch.stack(
                [clip_preprocess(Image.fromarray(f)) for f in batch_frames]
            ).to(DEVICE)
            z_obs = F.normalize(clip_model.encode_image(imgs), dim=-1)  # (b,768)

            actions_norm = torch.from_numpy(batch_actions).to(DEVICE) / ACTION_NORM_CONST
            z_pred = model(z_obs, actions_norm)  # (b,768)

            z_obs_chunks.append(z_obs.cpu().float().numpy())
            z_pred_chunks.append(z_pred.cpu().float().numpy())

    z_obs_full = np.concatenate(z_obs_chunks, axis=0)
    z_pred_full = np.concatenate(z_pred_chunks, axis=0)

    np.savez(cache_path, z_pred=z_pred_full, z_obs=z_obs_full)

    del clip_model, model
    torch.cuda.empty_cache()

    return z_pred_full, z_obs_full


# ---------------------------------------------------------------------------
# New: RidgeCV regression probe (same schema as linear_probe_MLP_output.py's
# fit_regression_probe, but StandardScaler + RidgeCV instead of bare LinearRegression --
# standardizing first is necessary since Ridge's L2 penalty, unlike OLS, is not
# scale-invariant across the 768 CLIP dims).
# ---------------------------------------------------------------------------

def fit_regression_probe_ridge(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                                val_mask: np.ndarray, meta: dict) -> dict:
    result = dict(meta)
    result["probe_type"] = "regression"
    result["regressor"] = "ridge_cv"
    result["n_train"] = int(train_mask.sum())
    result["n_val"] = int(val_mask.sum())

    if train_mask.sum() < 2 or val_mask.sum() < 2:
        result["skipped"] = True
        return result

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    result["label_mean_val"] = float(y_val.mean())
    result["label_std_val"] = float(y_val.std())

    reg = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
    reg.fit(X_train, y_train)
    result["ridge_alpha"] = float(reg.named_steps["ridgecv"].alpha_)

    result["train_r2"] = float(reg.score(X_train, y_train))
    result["val_r2"] = float(reg.score(X_val, y_val))

    pred_val = reg.predict(X_val)
    result["val_mae"] = float(mean_absolute_error(y_val, pred_val))
    result["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, pred_val)))

    if np.std(pred_val) > 1e-8 and np.std(y_val) > 1e-8:
        r, _ = pearsonr(y_val, pred_val)
        result["pearson_r"] = float(r)
    else:
        result["pearson_r"] = None

    mean_baseline_pred = np.full_like(y_val, y_train.mean())
    result["mean_baseline_rmse"] = float(np.sqrt(mean_squared_error(y_val, mean_baseline_pred)))

    result["skipped"] = False
    return result


# ---------------------------------------------------------------------------
# Results merging + plotting -- verbatim from linear_probe_MLP_output.py
# ---------------------------------------------------------------------------

def load_existing_results(results_path: str) -> list[dict]:
    if os.path.exists(results_path):
        with open(results_path) as f:
            return json.load(f)
    return []


def result_key(r: dict):
    return (r.get("checkpoint_dir_tag"), r.get("epoch"), r.get("probe"), r.get("block"), r.get("filter"))


def merge_results(existing: list[dict], new: list[dict]) -> list[dict]:
    """Keyed by (checkpoint_dir_tag, epoch, probe, block, filter) -- writing new dist_peg_start/
    dist_ab rows for a given (tag, epoch) replaces ONLY those rows, leaving every
    peg_touch_start/ab_touch/peg_touch_block row (everything backing the 2 classification charts
    and all general_state/ bar charts) untouched."""
    merged = {result_key(r): r for r in existing}
    for r in new:
        merged[result_key(r)] = r
    return list(merged.values())


def plot_task_relevant_line_chart(results: list[dict], probe_name: str, metric_key: str,
                                   title: str, out_path: str):
    by_tag = {}
    for r in results:
        if r.get("probe") != probe_name or r.get("skipped"):
            continue
        by_tag.setdefault(r["checkpoint_dir_tag"], []).append((r["epoch"], r.get(metric_key)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (tag, pairs) in enumerate(sorted(by_tag.items())):
        pairs = [(e, v) for e, v in sorted(pairs) if v is not None]
        if not pairs:
            continue
        epochs, vals = zip(*pairs)
        ax.plot(epochs, vals, marker="o", label=tag, color=TAB10(i))

    if "auc" in metric_key:
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric_key)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main -- only the 2 distance regression probes, only the 2 corresponding charts
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    task_relevant_dir = os.path.join(args.out_dir, "task_relevant")
    zpred_cache_dir = os.path.join(args.out_dir, "zpred_cache")
    os.makedirs(task_relevant_dir, exist_ok=True)
    os.makedirs(zpred_cache_dir, exist_ok=True)

    print("Building block-name list ...")
    block_names = get_block_names(DATA_DIR)
    print(f"  {block_names}\n")

    gt_cache_path = os.path.join(args.out_dir, "ground_truth_data.npz")
    print("Loading ground-truth data (checkpoint-independent, cached) ...")
    gt = sample_ground_truth_data(
        DATA_DIR, block_names, args.n_episodes, args.timesteps_per_episode,
        args.val_frac, args.seed, gt_cache_path, force_recompute=False,
    )
    train_mask = ~gt["is_val"]
    val_mask = gt["is_val"]
    print(f"Total rows: {len(gt['frames'])}  (train={train_mask.sum()}, val={val_mask.sum()})\n")

    new_results = []

    for ckpt_dir in args.ckpt_dirs:
        tag = ckpt_dir_tag(ckpt_dir)
        epoch_paths = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")))
        print(f"=== {tag} ({ckpt_dir}): {len(epoch_paths)} checkpoints found ===")

        for ckpt_path in epoch_paths:
            m = re.match(r"epoch_(\d+)\.pt$", os.path.basename(ckpt_path))
            epoch = int(m.group(1))
            print(f"  -- epoch {epoch} --")

            zpred_cache_path = os.path.join(zpred_cache_dir, f"{tag}_epoch{epoch:03d}.npz")
            z_pred = compute_zpred_for_checkpoint(
                ckpt_path, gt["frames"], gt["actions_windows"], args.batch_size,
                zpred_cache_path, force_recompute=False,
            )[0]

            base_meta = {"checkpoint_dir_tag": tag, "checkpoint_dir": ckpt_dir, "epoch": epoch}

            r = fit_regression_probe_ridge(
                z_pred, gt["dist_peg_start"], train_mask, val_mask,
                {**base_meta, "probe": "dist_peg_start"},
            )
            new_results.append(r)
            print(f"    dist_peg_start: val_r2={r.get('val_r2')}  alpha={r.get('ridge_alpha')}")

            r = fit_regression_probe_ridge(
                z_pred, gt["dist_ab"], train_mask, val_mask,
                {**base_meta, "probe": "dist_ab"},
            )
            new_results.append(r)
            print(f"    dist_ab: val_r2={r.get('val_r2')}  alpha={r.get('ridge_alpha')}")

    results_path = os.path.join(args.out_dir, "results.json")
    all_results = merge_results(load_existing_results(results_path), new_results)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMerged {len(new_results)} ridge-refit rows into results.json "
          f"({len(all_results)} total rows) -> {results_path}")

    print("\nRendering the 2 distance-regression charts only ...")
    plot_task_relevant_line_chart(
        all_results, "dist_peg_start", "val_r2",
        "dist(peg, start_block) at t+8 (regression, RidgeCV)",
        os.path.join(task_relevant_dir, "dist_peg_start_val_r2_vs_epoch.png"),
    )
    plot_task_relevant_line_chart(
        all_results, "dist_ab", "val_r2",
        "dist(A, B) at t+8 (regression, RidgeCV)",
        os.path.join(task_relevant_dir, "dist_ab_val_r2_vs_epoch.png"),
    )
    print(f"Saved 2 charts -> {task_relevant_dir}")

    print(f"\nDone. Outputs saved under {args.out_dir}")


if __name__ == "__main__":
    main()
