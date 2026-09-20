"""
Single-epoch linear-probe snapshot for epoch 11 of
checkpoints_clip_full_fine-tuned_with_subopt_data_h16 (the HORIZON=16 variant trained by
training/dynamics_model_clip_full_finetune_combined_horizon16.py -- see that script's docstring
for how it differs from the base HORIZON=8 combined-data run also swept by
linear_probe_MLP_output.py).

Unlike linear_probe_epoch11_snapshot.py (which is pure sklearn-on-cached-arrays because a
compatible z_pred cache already exists for its HORIZON=8 checkpoint), no compatible cache
exists here: this checkpoint's action_encoder was trained on 16-action chunks and its z_pred is
a forecast of t+16, not t+8, so both the ground-truth sampling (window length, label read-time)
and the z_pred computation (ActionEncoderConv(horizon=16)) differ from every existing cache.
This script therefore does its own ground-truth sampling + z_pred computation (adapted from
linear_probe_MLP_output.py's sample_ground_truth_data / compute_zpred_for_checkpoint), then
fits the same 3 linear probes (adapted from linear_probe_epoch11_snapshot.py's
fit_classification_probe / fit_regression_probe, which additionally stash val_margins/
val_labels/val_pred/val_true for plotting) on the VALIDATION split only:
  1. ab_touch        classification (LogisticRegression) -> val_auc
     "is start_block (A) touching oracle_target_block (B) at t+16"
  2. peg_touch_start classification (LogisticRegression) -> val_auc
     "is the peg touching start_block (A) at t+16"
  3. dist_ab         regression (RidgeCV) -> pearson_r
     linear probe predicts dist(A, B) at t+16 from z_pred; pearson_r is the correlation
     between the probe's prediction and the ground-truth distance on held-out rows.
  4. dist_peg_start  regression (RidgeCV) -> pearson_r
     linear probe predicts dist(peg, A) at t+16 from z_pred; pearson_r is the correlation
     between the probe's prediction and the ground-truth peg-to-start-block distance on
     held-out rows.

The two regression probes use RidgeCV, not plain LinearRegression: with z_pred at 768 dims and
n_train ~2800-2900, plain OLS showed the classic overfitting signature (high train_r2 alongside
negative val_r2 despite a real positive pearson_r) -- confirmed when the block-identity-filtered
confound probes in linear_probe_epoch11_h16_green_cube_confound.py hit the same pattern and
RidgeCV fixed it (val_r2 flipped from negative to +0.19/+0.31). RidgeCV picks its regularization
strength per probe via leave-one-out CV on the training set only, matching that script's
methodology so the two are now a same-estimator, apples-to-apples comparison.

The fine-tuned CLIP IMAGE tower is loaded fresh from this checkpoint's own clip_state_dict and
used to encode the input frame at t into z_obs[t] (the only place any CLIP tower is used in
this script -- the ground-truth state at t+16 comes directly from privileged simulator state in
the .pkl, not from encoding a future frame). The text tower is never used.

Both caches this script produces are new and cannot collide with the existing HORIZON=8 ones:
  ground_truth_data_h16.npz                              (16-action windows, labels at t+16)
  zpred_cache/expert+subopt-model_h16_epoch011.npz        (_h16 tag suffix)

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/linear_probe_epoch11_h16_snapshot.py
"""

import os
import glob
import pickle
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HORIZON = 16                # this checkpoint dir trains on 16-action chunks (base variant is 8)
TOUCH_THRESH = 0.05          # matches cache_zhard.py / dynamics_model_clip_full_finetune_combined_horizon16.py
ACTION_NORM_CONST = 0.035    # fixed-constant rescale, unchanged from the HORIZON=8 scripts

EPOCH = 11
CKPT_TAG = "expert+subopt-model_h16"
CKPT_DIR = "/home/ashwink/Documents/swms/checkpoints_clip_full_fine-tuned_with_subopt_data_h16"
CKPT_PATH = os.path.join(CKPT_DIR, f"epoch_{EPOCH:03d}.pt")

DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHARTS_DIR = os.path.join(MAIN_PROJECT_DIR, "linear_probe_MLP_output_charts")
GT_CACHE_PATH = os.path.join(CHARTS_DIR, "ground_truth_data_h16.npz")
ZPRED_CACHE_PATH = os.path.join(CHARTS_DIR, "zpred_cache", f"{CKPT_TAG}_epoch{EPOCH:03d}.npz")
OUT_DIR = os.path.join(CHARTS_DIR, "epoch11_h16_snapshot")

# Ground-truth sampling params -- kept identical to linear_probe_epoch11_snapshot.py's upstream
# (linear_probe_MLP_output.py) defaults for apples-to-apples comparability across horizons.
N_EPISODES = 600
TIMESTEPS_PER_EPISODE = 6
VAL_FRAC = 0.2
SEED = 0
BATCH_SIZE = 64
RIDGE_ALPHAS = np.logspace(-2, 4, 13)  # RidgeCV picks the best of these via train-only LOOCV

# dataviz skill reference palette (references/palette.md) -- light chart surface.
COLOR_SURFACE = "#fcfcfb"
COLOR_SERIES_1 = "#2a78d6"   # categorical slot 1, blue
COLOR_SERIES_2 = "#eb6834"   # categorical slot 2, orange
COLOR_TEXT_PRIMARY = "#0b0b0b"
COLOR_TEXT_SECONDARY = "#52514e"
COLOR_MUTED = "#898781"
COLOR_BASELINE = "#c3c2b7"


# ---------------------------------------------------------------------------
# Inline model classes -- adapted from linear_probe_MLP_output.py, with horizon threaded
# through LatentDynamicsModel -> ActionEncoderConv (matches training/dynamics_model.py's real
# signature; the base sweep script's inline copy hardcoded horizon=8).
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
    def __init__(self, horizon: int = 8):
        super().__init__()
        self.action_encoder = ActionEncoderConv(horizon=horizon)
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


# ---------------------------------------------------------------------------
# Block names -- verbatim from linear_probe_MLP_output.py
# ---------------------------------------------------------------------------

def get_block_names(data_dir: str) -> list[str]:
    pkl_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    with open(os.path.join(data_dir, pkl_files[0]), "rb") as f:
        ep = pickle.load(f)
    names = sorted(k for k in ep["block_states"][0].keys() if k != "peg")
    assert len(names) == 8, f"expected 8 block names, got {len(names)}: {names}"
    return names


# ---------------------------------------------------------------------------
# Ground-truth sampling (HORIZON=16 windows, labels read at t+HORIZON=t+16) -- adapted from
# linear_probe_MLP_output.py's sample_ground_truth_data; body is unchanged, only the module-
# level HORIZON constant differs.
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
            actions_window = np.stack(actions[t:tf], axis=0).astype(np.float32)  # (HORIZON,2)

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
# Per-checkpoint z_pred computation -- adapted from linear_probe_MLP_output.py, with
# LatentDynamicsModel(horizon=HORIZON) to match this checkpoint's 16-action action_encoder.
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

    model = LatentDynamicsModel(horizon=HORIZON).to(DEVICE)
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
# Probe fitting -- inline-copied verbatim from linear_probe_epoch11_snapshot.py (the version
# that additionally stashes val_margins/val_labels/val_pred/val_true for plotting).
# ---------------------------------------------------------------------------

def fit_classification_probe(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                              val_mask: np.ndarray, meta: dict) -> dict:
    result = dict(meta)
    result["probe_type"] = "classification"
    result["n_train"] = int(train_mask.sum())
    result["n_val"] = int(val_mask.sum())

    if train_mask.sum() == 0 or val_mask.sum() == 0 or np.unique(y[train_mask]).size < 2:
        result["skipped"] = True
        return result

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    result["n_pos_train"] = int(y_train.sum())
    result["n_neg_train"] = int((~y_train).sum())
    result["n_pos_val"] = int(y_val.sum())
    result["n_neg_val"] = int((~y_val).sum())
    result["majority_baseline_acc"] = float(max(y_val.mean(), 1 - y_val.mean()))

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    result["train_acc"] = float(clf.score(X_train, y_train))
    result["val_acc"] = float(clf.score(X_val, y_val))

    pred_val = clf.predict(X_val)
    result["precision"] = float(precision_score(y_val, pred_val, zero_division=0))
    result["recall"] = float(recall_score(y_val, pred_val, zero_division=0))
    result["f1"] = float(f1_score(y_val, pred_val, zero_division=0))

    if np.unique(y_val).size < 2:
        result["val_auc"] = None
        result["auc_undefined"] = True
    else:
        margins = clf.decision_function(X_val)
        result["val_auc"] = float(roc_auc_score(y_val, margins))
        result["auc_undefined"] = False
        result["val_margins"] = margins
        result["val_labels"] = y_val

    result["skipped"] = False
    return result


def fit_regression_probe(X: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                          val_mask: np.ndarray, meta: dict) -> dict:
    result = dict(meta)
    result["probe_type"] = "regression"
    result["n_train"] = int(train_mask.sum())
    result["n_val"] = int(val_mask.sum())

    if train_mask.sum() < 2 or val_mask.sum() < 2:
        result["skipped"] = True
        return result

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    result["label_mean_val"] = float(y_val.mean())
    result["label_std_val"] = float(y_val.std())

    reg = RidgeCV(alphas=RIDGE_ALPHAS)
    reg.fit(X_train, y_train)
    result["ridge_alpha"] = float(reg.alpha_)

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

    result["val_pred"] = pred_val
    result["val_true"] = y_val

    result["skipped"] = False
    return result


# ---------------------------------------------------------------------------
# Plotting -- dataviz-skill styling, inline-copied from linear_probe_epoch11_snapshot.py.
# Axis labels updated from "t+8" to "t+16" since this checkpoint forecasts 16 steps ahead.
# ---------------------------------------------------------------------------

def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_BASELINE)
    ax.spines["bottom"].set_color(COLOR_BASELINE)
    ax.tick_params(colors=COLOR_MUTED)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)


def plot_score_histogram_by_label(margins: np.ndarray, labels: np.ndarray, val_auc: float,
                                   pos_name: str, neg_name: str, title: str, annotation: str,
                                   out_path: str, n_bins: int = 30):
    """Histogram of the logistic probe's validation decision-function score, split by
    true label -- blue for the negative class, orange for the positive class -- with a
    dotted vertical line at x=0, the classifier's actual decision boundary."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    neg_scores = margins[~labels]
    pos_scores = margins[labels]

    lo, hi = float(margins.min()), float(margins.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    bins = np.linspace(lo - pad, hi + pad, n_bins)

    ax.hist(neg_scores, bins=bins, color=COLOR_SERIES_1, alpha=0.75, zorder=3,
            label=f"{neg_name} (n={len(neg_scores)})")
    ax.hist(pos_scores, bins=bins, color=COLOR_SERIES_2, alpha=0.75, zorder=3,
            label=f"{pos_name} (n={len(pos_scores)})")

    ax.axvline(0.0, color=COLOR_TEXT_PRIMARY, linestyle=":", linewidth=1.75, zorder=4,
               label="decision boundary")

    ax.set_xlabel("logistic probe decision-function score (val)", color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel("count", color=COLOR_TEXT_SECONDARY)
    ax.set_title(title, color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_SECONDARY)

    _style_axes(ax)

    full_annotation = f"val_auc = {val_auc:.3f}" + (f"\n{annotation}" if annotation else "")
    ax.text(0.98, 0.98, full_annotation, transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color=COLOR_TEXT_PRIMARY,
            bbox=dict(boxstyle="round", facecolor=COLOR_SURFACE, edgecolor=COLOR_BASELINE, alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


def plot_regression_scatter(y_true: np.ndarray, y_pred: np.ndarray, pearson_r: float,
                             val_r2: float, title: str, annotation: str, out_path: str,
                             quantity_name: str = "dist(A, B)"):
    """Scatter of the linear probe's predicted dist(A,B) vs the true dist(A,B) on
    validation rows, with a dotted y=x reference line (perfect prediction)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    ax.scatter(y_true, y_pred, s=14, color=COLOR_SERIES_1, alpha=0.5,
               edgecolors="none", zorder=3)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLOR_TEXT_PRIMARY,
            linestyle=":", linewidth=1.75, zorder=4, label="y = x (perfect prediction)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(f"true {quantity_name} at t+{HORIZON} (val)", color=COLOR_TEXT_SECONDARY)
    ax.set_ylabel(f"linear probe's predicted {quantity_name} (val)", color=COLOR_TEXT_SECONDARY)
    ax.set_title(title, color=COLOR_TEXT_PRIMARY, fontsize=12, pad=12)
    ax.legend(loc="upper left", frameon=False, labelcolor=COLOR_TEXT_SECONDARY)

    _style_axes(ax)
    ax.set_aspect("equal", adjustable="box")

    full_annotation = f"pearson_r = {pearson_r:.3f}\nval_r2 = {val_r2:.3f}" + (f"\n{annotation}" if annotation else "")
    ax.text(0.98, 0.02, full_annotation, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=COLOR_TEXT_PRIMARY,
            bbox=dict(boxstyle="round", facecolor=COLOR_SURFACE, edgecolor=COLOR_BASELINE, alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=COLOR_SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ZPRED_CACHE_PATH), exist_ok=True)

    print("Building block-name list ...")
    block_names = get_block_names(DATA_DIR)
    print(f"  {block_names}\n")

    print("Sampling ground-truth data (HORIZON=16, checkpoint-independent) ...")
    gt = sample_ground_truth_data(
        DATA_DIR, block_names, N_EPISODES, TIMESTEPS_PER_EPISODE, VAL_FRAC, SEED,
        GT_CACHE_PATH,
    )
    n_gt = len(gt["is_val"])
    print(f"Rows: {n_gt}\n")

    print(f"Computing z_pred for {CKPT_TAG} epoch {EPOCH} ({CKPT_PATH}) ...")
    z_pred, _ = compute_zpred_for_checkpoint(
        CKPT_PATH, gt["frames"], gt["actions_windows"], BATCH_SIZE, ZPRED_CACHE_PATH,
    )
    assert z_pred.shape[0] == n_gt, (
        f"row mismatch: z_pred has {z_pred.shape[0]} rows, ground truth has {n_gt}"
    )
    print(f"z_pred dim={z_pred.shape[1]}\n")

    is_val = gt["is_val"].astype(bool)
    train_mask = ~is_val
    val_mask = is_val
    print(f"train={train_mask.sum()}  val={val_mask.sum()}\n")

    base_meta = {"checkpoint_dir_tag": CKPT_TAG, "epoch": EPOCH, "horizon": HORIZON}
    results = {}

    print("Fitting ab_touch (A-touching-B) classification probe ...")
    r_ab = fit_classification_probe(
        z_pred, gt["ab_touch"].astype(bool), train_mask, val_mask,
        {**base_meta, "probe": "ab_touch"},
    )
    results["ab_touch"] = r_ab
    print(f"  val_auc = {r_ab.get('val_auc')}  (n_pos_val={r_ab.get('n_pos_val')}, n_neg_val={r_ab.get('n_neg_val')})")

    print("Fitting peg_touch_start (peg-touching-A) classification probe ...")
    r_peg = fit_classification_probe(
        z_pred, gt["peg_touch_start"].astype(bool), train_mask, val_mask,
        {**base_meta, "probe": "peg_touch_start"},
    )
    results["peg_touch_start"] = r_peg
    print(f"  val_auc = {r_peg.get('val_auc')}  (n_pos_val={r_peg.get('n_pos_val')}, n_neg_val={r_peg.get('n_neg_val')})")

    print("Fitting dist_ab (dist(A,B)) regression probe ...")
    r_dist = fit_regression_probe(
        z_pred, gt["dist_ab"].astype(np.float32), train_mask, val_mask,
        {**base_meta, "probe": "dist_ab"},
    )
    results["dist_ab"] = r_dist
    print(f"  val_r2 = {r_dist.get('val_r2')}  pearson_r = {r_dist.get('pearson_r')}  "
          f"ridge_alpha = {r_dist.get('ridge_alpha')}")

    print("Fitting dist_peg_start (dist(peg, A)) regression probe ...")
    r_peg_dist = fit_regression_probe(
        z_pred, gt["dist_peg_start"].astype(np.float32), train_mask, val_mask,
        {**base_meta, "probe": "dist_peg_start"},
    )
    results["dist_peg_start"] = r_peg_dist
    print(f"  val_r2 = {r_peg_dist.get('val_r2')}  pearson_r = {r_peg_dist.get('pearson_r')}  "
          f"ridge_alpha = {r_peg_dist.get('ridge_alpha')}")

    ARRAY_KEYS = ("val_margins", "val_labels", "val_pred", "val_true")
    results_json = {
        probe_name: {k: v for k, v in r.items() if k not in ARRAY_KEYS}
        for probe_name, r in results.items()
    }
    results_path = os.path.join(OUT_DIR, "results_epoch11_h16.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nSaved results -> {results_path}")

    # -- Charts --
    if not r_ab.get("skipped") and r_ab.get("val_auc") is not None:
        plot_score_histogram_by_label(
            r_ab["val_margins"], r_ab["val_labels"], r_ab["val_auc"],
            pos_name="A touching B", neg_name="A not touching B",
            title="A-touching-B: linear separability from z_pred",
            annotation=f"epoch {EPOCH} ({CKPT_TAG}, H={HORIZON})  n_val={r_ab['n_val']}",
            out_path=os.path.join(OUT_DIR, "ab_touch_val_auc.png"),
        )
        print("Saved ab_touch_val_auc.png")
    else:
        print("Skipped ab_touch chart (probe was skipped or AUC undefined)")

    if not r_peg.get("skipped") and r_peg.get("val_auc") is not None:
        plot_score_histogram_by_label(
            r_peg["val_margins"], r_peg["val_labels"], r_peg["val_auc"],
            pos_name="peg touching A", neg_name="peg not touching A",
            title="peg-touching-A: linear separability from z_pred",
            annotation=f"epoch {EPOCH} ({CKPT_TAG}, H={HORIZON})  n_val={r_peg['n_val']}",
            out_path=os.path.join(OUT_DIR, "peg_touch_start_val_auc.png"),
        )
        print("Saved peg_touch_start_val_auc.png")
    else:
        print("Skipped peg_touch_start chart (probe was skipped or AUC undefined)")

    if not r_dist.get("skipped") and r_dist.get("pearson_r") is not None:
        plot_regression_scatter(
            r_dist["val_true"], r_dist["val_pred"], r_dist["pearson_r"], r_dist["val_r2"],
            title="dist(A,B): linear probe correlation with ground truth",
            annotation=f"epoch {EPOCH} ({CKPT_TAG}, H={HORIZON})  n_val={r_dist['n_val']}",
            out_path=os.path.join(OUT_DIR, "dist_ab_val_pearson_r.png"),
        )
        print("Saved dist_ab_val_pearson_r.png")
    else:
        print("Skipped dist_ab chart (probe was skipped or pearson_r undefined)")

    if not r_peg_dist.get("skipped") and r_peg_dist.get("pearson_r") is not None:
        plot_regression_scatter(
            r_peg_dist["val_true"], r_peg_dist["val_pred"], r_peg_dist["pearson_r"], r_peg_dist["val_r2"],
            title="dist(peg, A): linear probe correlation with ground truth",
            annotation=f"epoch {EPOCH} ({CKPT_TAG}, H={HORIZON})  n_val={r_peg_dist['n_val']}",
            out_path=os.path.join(OUT_DIR, "dist_peg_start_val_pearson_r.png"),
            quantity_name="dist(peg, A)",
        )
        print("Saved dist_peg_start_val_pearson_r.png")
    else:
        print("Skipped dist_peg_start chart (probe was skipped or pearson_r undefined)")

    print(f"\nDone. Outputs saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
