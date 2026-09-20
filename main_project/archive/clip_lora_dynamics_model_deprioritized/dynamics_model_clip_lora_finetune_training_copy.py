"""
Joint fine-tuning of the CLIP ViT-L/14 image encoder (via LoRA) with a fresh
FiLM-MLP latent dynamics model — diagnostic experiment.

Motivation: a prior fully-frozen-CLIP run of dynamics_model.py showed the InfoNCE
contrastive loss couldn't learn a generalizable touching/not-touching predicate,
suspected to be because frozen CLIP TEXT embeddings collapse semantically-opposite
statements ("A touching B" vs "A not touching B") into near-identical vectors,
weakening the contrastive margin no matter how the MLP is trained. This script
tests whether unfreezing the CLIP IMAGE encoder via LoRA (text encoder stays fully
frozen) gives enough flexibility to compensate, by producing better-separated
image embeddings.

Reuses model/loss/data conventions from dynamics_model.py unmodified; reuses only
the LoRA insertion mechanism (not the training loop) from clip_lora.py.

Usage:
    python dynamics_model_clip_lora_finetune.py --smoke_test
    python dynamics_model_clip_lora_finetune.py --train
"""

import sys
import os
import json
import math
import time
import random
import pickle
import traceback
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import open_clip
import matplotlib
matplotlib.use("Agg")  # force non-interactive backend; DISPLAY is set on this desktop
                        # session's tmux pane, so matplotlib would otherwise pick TkAgg
                        # and hang the process during figure teardown
import matplotlib.pyplot as plt

from langtable_dataset import LangTableDataset, train_val_split
from dynamics_model import (
    LatentDynamicsModel,
    infonce_logits,
    ranking_metrics,
    similarity_metrics,
    mean_pairwise_cosine,
)
from clip_lora import apply_lora_to_visual_encoder, lora_state_dict, load_lora_state_dict


# ---------------------------------------------------------------------------
# Dataset: raw frames + deterministic preprocess (no CLIP model, no grad) —
# the gradient-enabled CLIP forward pass happens in JointModel inside the loop.
#
# Raw frames are NOT held resident in memory (that OOMs at the full 4250/750-
# episode scale — ~50GB of uint8 pixels). Instead each episode's frames are
# pre-cached as a plain {stem}_frames.npy file (via cache_frames_npy.py) and
# read on demand via np.load(path, mmap_mode="r") — see load_episode_frames_mmap
# below, shared by the Dataset and the margin-probe/progress-curve diagnostics.
# ---------------------------------------------------------------------------

def load_episode_frames_mmap(frames_path: str) -> np.memmap:
    """Opens a fresh mmap handle per call (not cached across calls). Confirmed
    safe under a multi-worker fork-based DataLoader: the OS page cache shares
    physical pages across worker processes reading the same file, no races."""
    return np.load(frames_path, mmap_mode="r")


class LangTableImageDataset(LangTableDataset):
    """LangTableDataset variant for CLIP-image-encoder fine-tuning. Stores paths
    to pre-cached {stem}_frames.npy files (not raw frames, not cached frozen
    z_obs) plus a shared deterministic CLIP preprocess transform, applying ONLY
    that transform (no model, no grad) in __getitem__. Safe with num_workers > 0
    — no GPU-resident nn.Module lives on the Dataset.
    """

    def __init__(self, data_dir: str, preprocess, file_list: list[str] | None = None):
        self.data_dir = data_dir
        self.preprocess = preprocess

        self.frames_paths: list[str] = []
        self.actions: list[np.ndarray] = []
        self.z_sem: list[np.ndarray] = []
        self.touching: list[np.ndarray] = []
        self.peg_touching: list[np.ndarray] = []
        self.unique_zsem: list[torch.Tensor] = []
        self.neg_indices: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []

        pkl_files = file_list if file_list is not None else sorted(
            f for f in os.listdir(data_dir) if f.endswith(".pkl")
        )

        for pkl_file in pkl_files:
            pkl_path = os.path.join(data_dir, pkl_file)
            npz_path = pkl_path.replace(".pkl", "_zsem.npz")
            if not os.path.exists(npz_path):
                warnings.warn(f"Missing zsem file for {pkl_file}, skipping.")
                continue
            zhard_path = pkl_path.replace(".pkl", "_zhard.npz")
            if not os.path.exists(zhard_path):
                warnings.warn(f"Missing zhard file for {pkl_file}, skipping episode.")
                continue
            frames_path = pkl_path.replace(".pkl", "_frames.npy")
            if not os.path.exists(frames_path):
                warnings.warn(f"Missing frames_npy file for {pkl_file} "
                               f"(run cache_frames_npy.py first), skipping episode.")
                continue

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)
            z_sem_npz = np.load(npz_path)
            z_sem_data = z_sem_npz["z_sem"]
            T = len(z_sem_data)
            zhard = np.load(zhard_path)

            ep_idx = len(self.actions)
            self.frames_paths.append(frames_path)
            self.actions.append(ep["actions"])
            self.z_sem.append(z_sem_data)
            self.touching.append(z_sem_npz["touching"])
            self.peg_touching.append(z_sem_npz["peg_touching"])
            self.unique_zsem.append(torch.tensor(zhard["unique_zsem"], dtype=torch.float32))
            self.neg_indices.append(zhard["neg_indices"])

            for t in range(T - self.HORIZON):
                self.indices.append((ep_idx, t))

        all_actions = np.concatenate(self.actions, axis=0)
        self.action_stats = {
            "min": torch.tensor(all_actions.min(axis=0), dtype=torch.float32),
            "max": torch.tensor(all_actions.max(axis=0), dtype=torch.float32),
        }

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.indices[idx]
        frame = np.asarray(load_episode_frames_mmap(self.frames_paths[ep_idx])[t])
        pixel_values = self.preprocess(Image.fromarray(frame))

        actions_norm = self._normalize_actions(self.actions[ep_idx][t : t + self.HORIZON])
        z_sem_target = torch.tensor(self.z_sem[ep_idx][t + self.HORIZON], dtype=torch.float32)
        neg_idx = self.neg_indices[ep_idx][t + self.HORIZON].astype(np.int64)
        z_hard_neg = self.unique_zsem[ep_idx][neg_idx]

        return {
            "pixel_values": pixel_values,
            "actions": actions_norm,
            "z_sem_target": z_sem_target,
            "z_hard_neg": z_hard_neg,
        }


class JointModel(nn.Module):
    """Wraps the LoRA-injected CLIP model + LatentDynamicsModel MLP so a single
    nn.DataParallel(...) replicates+scatters both together in one forward call."""

    def __init__(self, clip_model: nn.Module, mlp: LatentDynamicsModel):
        super().__init__()
        self.clip_model = clip_model
        self.mlp = mlp

    def forward(self, pixel_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z_t = self.clip_model.encode_image(pixel_values, normalize=True)  # (B, 768), grad-enabled
        return self.mlp(z_t, actions)  # (B, 768) L2-normalized


# ---------------------------------------------------------------------------
# Margin probe + progress curve diagnostics
# ---------------------------------------------------------------------------

def build_probe_episode_records(data_dir: str, stems: list[str]) -> dict[str, dict]:
    """For each episode stem: cached touching label + ground-truth task-relevant
    block distance dist_ab(t) (same formula as cache_zhard.py), plus the path to
    the pre-cached {stem}_frames.npy for lazy mmap access — never retains the
    unpickled ep["frames"] list itself (that's the OOM this was fixed from)."""
    records = {}
    for stem in stems:
        pkl_path = os.path.join(data_dir, stem + ".pkl")
        zsem_path = os.path.join(data_dir, stem + "_zsem.npz")
        frames_path = pkl_path.replace(".pkl", "_frames.npy")
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)
        zsem = np.load(zsem_path)
        touching = zsem["touching"]

        block_states = ep["block_states"]
        a = ep["metadata"]["start_block"]
        b = ep["metadata"]["oracle_target_block"]
        T = len(block_states)
        dist_ab = np.array(
            [np.linalg.norm(block_states[t][a] - block_states[t][b]) for t in range(T)],
            dtype=np.float32,
        )

        records[stem] = {
            "frames_path": frames_path,
            "touching": touching[:T].astype(bool),
            "dist_ab": dist_ab,
        }
    return records


def encode_probe_episodes(
    clip_model: nn.Module, records: dict[str, dict], preprocess, device: str, batch_size: int = 64,
) -> dict[str, dict]:
    """No-grad CLIP image encode of every probe episode's frames (read-only diagnostic,
    not a training step)."""
    was_training = clip_model.training
    clip_model.eval()
    encoded = {}
    with torch.no_grad():
        for stem, rec in records.items():
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
    if was_training:
        clip_model.train()
    return encoded


def _mean_excl_self(sim: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray, same_set: bool):
    sub = sim[np.ix_(idx_i, idx_j)]
    if same_set:
        n = len(idx_i)
        if n < 2:
            return None
        total = sub.sum() - np.trace(sub)
        count = n * n - n
        return total / count if count > 0 else None
    return sub.mean()


def compute_margin_probe(encoded: dict[str, dict]) -> tuple[dict, dict]:
    """Within-episode pairwise contrastive margin (never compares across episodes,
    since each episode only has 2 of the 8 blocks as task-relevant — pooling across
    episodes would confound scene/block identity with the touching signal).

    For each qualifying episode: margin_i = (within_pos + within_neg)/2 - between.
    Returns (aggregate {mean,std,min,max,n_qualifying,n_total}, per_episode
    {stem: {touching_centroid, not_touching_centroid, margin}})."""
    margins = []
    per_episode = {}
    for stem, d in encoded.items():
        Z = d["Z"]
        touching = d["touching"]
        pos_idx = np.where(touching)[0]
        neg_idx = np.where(~touching)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        sim = Z @ Z.T
        within_pos = _mean_excl_self(sim, pos_idx, pos_idx, True)
        within_neg = _mean_excl_self(sim, neg_idx, neg_idx, True)
        between = _mean_excl_self(sim, pos_idx, neg_idx, False)
        if within_pos is None or within_neg is None or between is None:
            continue

        margin_i = (within_pos + within_neg) / 2.0 - between
        margins.append(margin_i)

        touching_centroid = Z[pos_idx].mean(axis=0)
        not_touching_centroid = Z[neg_idx].mean(axis=0)
        touching_centroid /= np.linalg.norm(touching_centroid) + 1e-8
        not_touching_centroid /= np.linalg.norm(not_touching_centroid) + 1e-8

        per_episode[stem] = {
            "touching_centroid": touching_centroid,
            "not_touching_centroid": not_touching_centroid,
            "margin": float(margin_i),
        }

    n_total = len(encoded)
    if not margins:
        aggregate = {"mean": float("nan"), "std": float("nan"), "min": float("nan"),
                     "max": float("nan"), "n_qualifying": 0, "n_total": n_total}
    else:
        arr = np.array(margins, dtype=np.float64)
        aggregate = {
            "mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()),
            "n_qualifying": len(arr), "n_total": n_total,
        }
    return aggregate, per_episode


def compute_progress_curve_correlation(encoded: dict[str, dict], per_episode_centroids: dict[str, dict]) -> float:
    """Pools signed_progress(t) = cos(z_t, touching_centroid) - cos(z_t, not_touching_centroid)
    vs. -dist_ab(t) across all timesteps of all qualifying probe episodes; Pearson correlation."""
    progress_all, neg_dist_all = [], []
    for stem, centroids in per_episode_centroids.items():
        Z = encoded[stem]["Z"]
        dist_ab = encoded[stem]["dist_ab"]
        tc, ntc = centroids["touching_centroid"], centroids["not_touching_centroid"]
        signed_progress = Z @ tc - Z @ ntc
        progress_all.append(signed_progress)
        neg_dist_all.append(-dist_ab)

    if not progress_all:
        return float("nan")
    progress_all = np.concatenate(progress_all)
    neg_dist_all = np.concatenate(neg_dist_all)
    if progress_all.std() == 0 or neg_dist_all.std() == 0:
        return float("nan")
    return float(np.corrcoef(progress_all, neg_dist_all)[0, 1])


def plot_progress_curves(
    encoded: dict[str, dict], per_episode_centroids: dict[str, dict],
    episode_stems_to_plot: list[str], out_dir: str, checkpoint_tag: str,
) -> dict[str, str]:
    """Returns {stem: png_path} for every episode actually plotted (skips episodes
    that didn't qualify for this checkpoint), so callers can wandb.Image-log exactly
    the files that were written."""
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for stem in episode_stems_to_plot:
        if stem not in per_episode_centroids:
            continue  # episode didn't qualify (no touching/not-touching split) this checkpoint
        Z = encoded[stem]["Z"]
        dist_ab = encoded[stem]["dist_ab"]
        centroids = per_episode_centroids[stem]
        tc, ntc = centroids["touching_centroid"], centroids["not_touching_centroid"]
        signed_progress = Z @ tc - Z @ ntc
        t_axis = np.arange(len(signed_progress))

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(t_axis, signed_progress, color="tab:blue", marker="o", markersize=3)
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("signed_progress(t) = cos(z_t, touch) - cos(z_t, not_touch)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(t_axis, dist_ab, color="tab:red", marker="x", markersize=3)
        ax2.set_ylabel("block distance (ground truth)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        ax1.set_title(f"{stem} — {checkpoint_tag}")
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"{checkpoint_tag}_{stem}.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        written[stem] = png_path
    return written


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str = "/home/ashwink/full_data_5000/langtable/demos/",
    device: str = "cuda",
    micro_batch_size: int = 256,
    effective_batch_size: int = 256,
    num_epochs: int = 15,
    lr_lora: float = 1e-4,
    lr_mlp: float = 3e-4,
    wd_lora: float = 0.0,
    temperature: float = 0.1,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.2,
    wandb_project: str = "langtable-dynamics",
    checkpoint_dir: str = "checkpoints_clip_image_finetuned",
    resume_from: str | None = None,
    n_val_episodes: int = 750,
    val_seed: int = 0,
    num_workers: int = 8,
    probe_n_episodes: int = 75,
    probe_seed: int = 123,
    full_probe_every: int = 5,
    margin_probe_divergence_threshold: float = 0.03,
    diagnostic_plot_episodes: int = 5,
    diagnostics_dir: str = "clip_finetune_diagnostics",
    smoke_test: bool = False,
    smoke_test_steps: int = 100,
):
    grad_accum_steps = math.ceil(effective_batch_size / micro_batch_size)
    grad_accum_triggered = grad_accum_steps > 1

    # ---- data split (reused verbatim: n_val=750, seed=0 -> 4250 train / 750 val) ----
    train_files, val_files = train_val_split(data_dir, n_val=n_val_episodes, seed=val_seed)

    # ---- single CLIP construction (shared preprocess transform + the model that gets LoRA) ----
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(device)
    apply_lora_to_visual_encoder(clip_model, r=r, alpha=alpha, dropout=dropout)

    mlp = LatentDynamicsModel().to(device)  # fresh random init — no checkpoint loaded
    joint_model = JointModel(clip_model, mlp).to(device)

    lora_params = [p for n, p in joint_model.clip_model.named_parameters()
                   if p.requires_grad and "lora_" in n]
    mlp_params = list(joint_model.mlp.parameters())

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr_lora, "weight_decay": wd_lora},
        {"params": mlp_params, "lr": lr_mlp},  # weight_decay unset -> AdamW default 0.01
    ])

    # ---- optimizer param-group sanity assertions ----
    n_opt_params = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    n_trainable = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
    assert n_opt_params == n_trainable, (
        f"optimizer param count {n_opt_params} != joint_model trainable count {n_trainable}"
    )
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    for p in joint_model.parameters():
        if p.requires_grad:
            assert id(p) in opt_ids, "a trainable parameter is missing from a param group"
        else:
            assert id(p) not in opt_ids, "a frozen parameter leaked into the optimizer"

    ddp_model = nn.DataParallel(joint_model, device_ids=[0, 1]) if torch.cuda.device_count() > 1 else joint_model

    train_dataset = LangTableImageDataset(data_dir, preprocess, file_list=train_files)
    val_dataset = LangTableImageDataset(data_dir, preprocess, file_list=val_files)
    val_dataset.action_stats = train_dataset.action_stats

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False, pin_memory=True)
    print(f"Train dataset: {len(train_dataset)} samples, {len(train_loader)} micro-batches/epoch")
    print(f"Val dataset:   {len(val_dataset)} samples, {len(val_loader)} micro-batches/epoch")

    n_steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = n_steps_per_epoch * num_epochs

    # ---- fixed probe subsample (sampled once, seeded, from the val split) ----
    rng = random.Random(probe_seed)
    val_stems_all = [f[:-4] for f in val_files]  # strip ".pkl"
    probe_stems = rng.sample(val_stems_all, min(probe_n_episodes, len(val_stems_all)))
    plot_stems = rng.sample(probe_stems, min(diagnostic_plot_episodes, len(probe_stems)))
    probe_records = build_probe_episode_records(data_dir, probe_stems)
    print(f"Probe subsample: {len(probe_records)} episodes (seed={probe_seed}); "
          f"{len(plot_stems)} fixed for progress-curve plots")

    # ===========================================================================
    # Smoke test
    # ===========================================================================
    if smoke_test:
        print(f"\n[smoke_test] micro_batch_size={micro_batch_size}, "
              f"grad_accum_steps={grad_accum_steps} (triggered={grad_accum_triggered})")
        joint_model.train()
        step_times = []
        step_losses = []
        step = 0
        accum_counter = 0
        step_start = time.perf_counter()
        train_iter = iter(train_loader)

        oom = False
        try:
            while step < smoke_test_steps:
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)

                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                z_sem_target = batch["z_sem_target"].to(device, non_blocking=True)
                z_hard_neg = batch["z_hard_neg"].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_pred = ddp_model(pixel_values, actions)
                    logits = infonce_logits(z_pred.float(), z_sem_target, z_hard_neg, temperature)
                    targets = torch.zeros(z_pred.shape[0], dtype=torch.long, device=device)
                    loss = F.cross_entropy(logits, targets)
                (loss / grad_accum_steps).backward()
                accum_counter += 1

                if accum_counter == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(lora_params + mlp_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_counter = 0
                    torch.cuda.synchronize() if device == "cuda" else None
                    step_time = time.perf_counter() - step_start
                    step_times.append(step_time)
                    step_losses.append(loss.item())
                    step += 1
                    step_start = time.perf_counter()
        except torch.cuda.OutOfMemoryError:
            oom = True

        if oom:
            print(f"\n[smoke_test] CUDA OutOfMemoryError at step {step} with "
                  f"micro_batch_size={micro_batch_size}. Retry with a smaller --micro_batch_size.")
            return

        # exclude first step from the mean as a flagged CUDA-warmup outlier
        timed = step_times[1:] if len(step_times) > 1 else step_times
        avg_step_time = sum(timed) / len(timed)
        extrapolated_total = avg_step_time * total_steps

        print(f"\n[smoke_test] completed {step} optimizer steps")
        print(f"(a) avg wall-clock/optimizer-step: {avg_step_time:.3f}s "
              f"(first step {step_times[0]:.3f}s excluded as warmup)")
        print(f"(b) total_steps for {num_epochs} epochs @ effective batch {effective_batch_size}: "
              f"{total_steps} -> extrapolated total time: {extrapolated_total/60:.1f} min "
              f"({extrapolated_total/3600:.2f} hr)")
        print(f"(c) grad_accum_steps={grad_accum_steps}, triggered={grad_accum_triggered}, "
              f"micro_batch_size={micro_batch_size}")

        losses = np.array(step_losses)
        if np.isnan(losses).any() or np.isinf(losses).any():
            print("(d) WARNING: loss trajectory contains NaN/Inf — LR likely too high.")
        else:
            first10, last10 = losses[:10].mean(), losses[-10:].mean()
            print(f"(d) loss[:10] mean={first10:.4f}, loss[-10:] mean={last10:.4f}")
            if abs(first10 - last10) < 0.02 * max(abs(first10), 1e-6):
                print("(d) WARNING: loss looks flat over 100 steps — LR may be too low "
                      "(or too high and stuck) for this batch size.")
        return

    # ===========================================================================
    # Full training run
    # ===========================================================================
    import wandb

    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "train_val_split.json"), "w") as f:
        json.dump({"seed": val_seed, "n_val": n_val_episodes,
                    "train_files": train_files, "val_files": val_files}, f, indent=2)

    wandb.init(
        project=wandb_project,
        name=f"clip_lora_finetune_r{r}_a{alpha}",
        config=dict(
            lora_r=r, lora_alpha=alpha, lora_dropout=dropout,
            lora_scaling=alpha / math.sqrt(r), lora_projections=["q", "k", "v", "out"],
            lora_layers="all_24", lora_scope="image_encoder_only",
            changed_weight=1.0, negative_count=448, temperature=temperature,
            n_train_episodes=len(train_files), n_val_episodes=len(val_files), val_seed=val_seed,
            effective_batch_size=effective_batch_size, micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps, grad_accum_triggered=grad_accum_triggered,
            mixed_precision="bf16", data_parallel_device_ids=[0, 1],
            lr_lora=lr_lora, lr_mlp=lr_mlp, wd_lora=wd_lora, wd_mlp=0.01,
            num_epochs=num_epochs, total_steps=total_steps,
            probe_n_episodes=probe_n_episodes, probe_seed=probe_seed,
            full_probe_every=full_probe_every,
        ),
    )

    try:
        start_epoch = 0
        step = 0
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=device)
            load_lora_state_dict(joint_model.clip_model, ckpt["lora_state_dict"])
            joint_model.mlp.load_state_dict(ckpt["mlp_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            step = ckpt.get("step", 0)
            print(f"Resumed from {resume_from} at epoch {ckpt['epoch']}")

        for epoch in range(start_epoch, num_epochs):
            joint_model.train()
            epoch_loss_sum = 0.0
            epoch_metrics_sum = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
            n_optimizer_steps_this_epoch = 0

            accum_counter = 0
            accum_loss = 0.0
            accum_metrics = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
            accum_extra = {"mean_pairwise_cosine": 0.0, "pos_sim_mean": 0.0, "hard_sim_mean": 0.0}
            step_start = time.perf_counter()

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                z_sem_target = batch["z_sem_target"].to(device, non_blocking=True)
                z_hard_neg = batch["z_hard_neg"].to(device, non_blocking=True)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_pred = ddp_model(pixel_values, actions)
                    logits = infonce_logits(z_pred.float(), z_sem_target, z_hard_neg, temperature)
                    targets = torch.zeros(z_pred.shape[0], dtype=torch.long, device=device)
                    loss = F.cross_entropy(logits, targets)

                if not torch.isfinite(loss):
                    print(f"[step {step}] WARNING: non-finite loss ({loss.item()}) — "
                          f"skipping backward for this micro-batch.")
                    continue

                (loss / grad_accum_steps).backward()
                accum_counter += 1

                with torch.no_grad():
                    accum_loss += loss.item()
                    accum_metrics_b = ranking_metrics(logits)
                    accum_sims_b = similarity_metrics(logits, temperature)
                    mean_pairwise = mean_pairwise_cosine(z_pred.float())
                    for k in accum_metrics:
                        accum_metrics[k] += accum_metrics_b[k]
                    accum_extra["mean_pairwise_cosine"] += mean_pairwise.item()
                    accum_extra["pos_sim_mean"] += accum_sims_b["pos_sim_mean"]
                    accum_extra["hard_sim_mean"] += accum_sims_b["hard_sim_mean"]

                if accum_counter == grad_accum_steps:
                    grad_norm = torch.nn.utils.clip_grad_norm_(lora_params + mlp_params, 1.0)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        print(f"[step {step}] WARNING: non-finite grad_norm ({grad_norm.item()}) — "
                              f"skipping optimizer.step() for this window.")
                    optimizer.zero_grad()

                    step_time = time.perf_counter() - step_start
                    n = grad_accum_steps
                    step_loss = accum_loss / n
                    wandb.log({
                        "train/loss": step_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/mean_pairwise_cosine": accum_extra["mean_pairwise_cosine"] / n,
                        "train/pos_sim_mean": accum_extra["pos_sim_mean"] / n,
                        "train/hard_sim_mean": accum_extra["hard_sim_mean"] / n,
                        "train/step_time_sec": step_time,
                        "step": step,
                    })
                    epoch_loss_sum += step_loss
                    for k in epoch_metrics_sum:
                        epoch_metrics_sum[k] += accum_metrics[k] / n
                    n_optimizer_steps_this_epoch += 1

                    step += 1
                    accum_counter = 0
                    accum_loss = 0.0
                    accum_metrics = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
                    accum_extra = {"mean_pairwise_cosine": 0.0, "pos_sim_mean": 0.0, "hard_sim_mean": 0.0}
                    step_start = time.perf_counter()

            # flush any leftover partial accumulation at epoch end
            if accum_counter > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(lora_params + mlp_params, 1.0)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    print(f"[step {step}] WARNING: non-finite grad_norm ({grad_norm.item()}) — "
                          f"skipping optimizer.step() for this window.")
                optimizer.zero_grad()
                n = accum_counter
                step_loss = accum_loss / n
                wandb.log({
                    "train/loss": step_loss, "train/grad_norm": grad_norm.item(),
                    "train/mean_pairwise_cosine": accum_extra["mean_pairwise_cosine"] / n,
                    "train/pos_sim_mean": accum_extra["pos_sim_mean"] / n,
                    "train/hard_sim_mean": accum_extra["hard_sim_mean"] / n,
                    "step": step,
                })
                epoch_loss_sum += step_loss
                for k in epoch_metrics_sum:
                    epoch_metrics_sum[k] += accum_metrics[k] / n
                n_optimizer_steps_this_epoch += 1
                step += 1

            avg_train_loss = epoch_loss_sum / n_optimizer_steps_this_epoch
            avg_train_metrics = {k: v / n_optimizer_steps_this_epoch for k, v in epoch_metrics_sum.items()}

            # ---- validation pass ----
            joint_model.eval()
            val_loss_sum = 0.0
            val_metrics_sum = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
            val_extra_sum = {"mean_pairwise_cosine": 0.0, "pos_sim_mean": 0.0, "hard_sim_mean": 0.0}
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    actions = batch["actions"].to(device, non_blocking=True)
                    z_sem_target = batch["z_sem_target"].to(device, non_blocking=True)
                    z_hard_neg = batch["z_hard_neg"].to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        z_pred = ddp_model(pixel_values, actions)
                    logits = infonce_logits(z_pred.float(), z_sem_target, z_hard_neg, temperature)
                    targets = torch.zeros(z_pred.shape[0], dtype=torch.long, device=device)
                    val_loss_sum += F.cross_entropy(logits, targets).item()
                    batch_metrics = ranking_metrics(logits)
                    for k in val_metrics_sum:
                        val_metrics_sum[k] += batch_metrics[k]
                    batch_sims = similarity_metrics(logits, temperature)
                    val_extra_sum["mean_pairwise_cosine"] += mean_pairwise_cosine(z_pred.float()).item()
                    val_extra_sum["pos_sim_mean"] += batch_sims["pos_sim_mean"]
                    val_extra_sum["hard_sim_mean"] += batch_sims["hard_sim_mean"]

            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_extra = {k: v / len(val_loader) for k, v in val_extra_sum.items()}
            avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

            # ---- checkpoint (saved before diagnostics, so a diagnostics crash/hang never
            # loses this epoch's already-trained weights) ----
            ckpt = {
                "epoch": epoch,
                "step": step,
                "lora_state_dict": lora_state_dict(joint_model.clip_model),
                "mlp_state_dict": joint_model.mlp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, "latest.pt"))
            torch.save(ckpt, os.path.join(checkpoint_dir, f"epoch_{epoch:03d}.pt"))

            log_dict = {
                "epoch": epoch,
                "train/epoch_loss": avg_train_loss,
                "train/epoch_top1": avg_train_metrics["top1"],
                "train/epoch_top5": avg_train_metrics["top5"],
                "train/epoch_mean_rank": avg_train_metrics["mean_rank"],
                "val/epoch_loss": avg_val_loss,
                "val/epoch_top1": avg_val_metrics["top1"],
                "val/epoch_top5": avg_val_metrics["top5"],
                "val/epoch_mean_rank": avg_val_metrics["mean_rank"],
                "val/mean_pairwise_cosine": avg_val_extra["mean_pairwise_cosine"],
                "val/pos_sim_mean": avg_val_extra["pos_sim_mean"],
                "val/hard_sim_mean": avg_val_extra["hard_sim_mean"],
            }

            # ---- margin probe + progress curve (subsampled, every epoch) ----
            # Wrapped defensively: this block does matplotlib plotting and extra CLIP
            # forward passes that are not needed for training itself. A failure here
            # (or a repeat of the matplotlib/Tk hang that killed a previous run) should
            # not take down the whole training run or lose the checkpoint saved above.
            margin_agg = None
            progress_corr = None
            try:
                encoded = encode_probe_episodes(joint_model.clip_model, probe_records, preprocess, device)
                margin_agg, per_episode_centroids = compute_margin_probe(encoded)
                progress_corr = compute_progress_curve_correlation(encoded, per_episode_centroids)
                checkpoint_tag = f"epoch_{epoch:03d}"
                plotted = plot_progress_curves(encoded, per_episode_centroids, plot_stems,
                                                os.path.join(os.path.dirname(os.path.abspath(__file__)), diagnostics_dir),
                                                checkpoint_tag)

                log_dict.update({
                    f"progress_curve_plots/{stem}": wandb.Image(png_path)
                    for stem, png_path in plotted.items()
                })
                log_dict.update({
                    "margin_probe_subsample75/mean": margin_agg["mean"],
                    "margin_probe_subsample75/std": margin_agg["std"],
                    "margin_probe_subsample75/min": margin_agg["min"],
                    "margin_probe_subsample75/max": margin_agg["max"],
                    "margin_probe_subsample75/n_qualifying": margin_agg["n_qualifying"],
                    "margin_probe_subsample75/n_total": margin_agg["n_total"],
                    "progress_curve/pearson_corr": progress_corr,
                })

                # ---- periodic full-750 probe ----
                if (epoch + 1) % full_probe_every == 0 or epoch == num_epochs - 1:
                    full_records = build_probe_episode_records(data_dir, val_stems_all)
                    full_encoded = encode_probe_episodes(joint_model.clip_model, full_records, preprocess, device)
                    full_margin_agg, _ = compute_margin_probe(full_encoded)
                    log_dict.update({
                        "margin_probe_full750/mean": full_margin_agg["mean"],
                        "margin_probe_full750/std": full_margin_agg["std"],
                        "margin_probe_full750/min": full_margin_agg["min"],
                        "margin_probe_full750/max": full_margin_agg["max"],
                        "margin_probe_full750/n_qualifying": full_margin_agg["n_qualifying"],
                        "margin_probe_full750/n_total": full_margin_agg["n_total"],
                    })
                    divergence = abs(full_margin_agg["mean"] - margin_agg["mean"])
                    diverged = divergence > margin_probe_divergence_threshold
                    log_dict["margin_probe/divergence_warning"] = int(diverged)
                    if diverged:
                        print(f"[epoch {epoch}] WARNING: subsample75 margin ({margin_agg['mean']:.4f}) "
                              f"and full750 margin ({full_margin_agg['mean']:.4f}) diverge by "
                              f"{divergence:.4f} (> {margin_probe_divergence_threshold}) — "
                              f"subsample may not be representative.")
                    del full_records, full_encoded
            except Exception:
                print(f"[epoch {epoch}] WARNING: margin probe / progress-curve diagnostics failed, "
                      f"skipping diagnostics for this epoch (checkpoint already saved):")
                traceback.print_exc()

            wandb.log(log_dict)

            margin_str = f"{margin_agg['mean']:.4f}" if margin_agg is not None else "N/A"
            progress_str = f"{progress_corr:.4f}" if progress_corr is not None else "N/A"
            print(f"[epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                  f"margin_subsample75={margin_str} progress_corr={progress_str}")
    except Exception as e:
        wandb.alert(
            title="Training crashed",
            text=f"Run '{wandb.run.name}' crashed with {type(e).__name__}: {e}",
            level=wandb.AlertLevel.ERROR,
        )
        raise
    else:
        wandb.alert(
            title="Training completed",
            text=f"Run '{wandb.run.name}' completed all {num_epochs} epochs successfully.",
            level=wandb.AlertLevel.INFO,
        )

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run full 15-epoch training with wandb")
    parser.add_argument("--smoke_test", action="store_true", help="run 100 optimizer steps, report timing/OOM/LR sanity")
    parser.add_argument("--micro_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume_from", type=str, default=None,
                         help="path to a checkpoint (e.g. checkpoints_clip_image_finetuned/latest.pt) "
                              "to resume training from")
    parser.add_argument("--slack_smoke_test", action="store_true",
                         help="send a single test wandb.alert() and exit, without running training")
    args = parser.parse_args()

    if args.slack_smoke_test:
        import wandb
        wandb.init(project="langtable-dynamics", name="slack_alert_smoke_test", job_type="smoke_test")
        wandb.alert(
            title="Slack smoke test",
            text="Test alert from dynamics_model_clip_lora_finetune.py --slack_smoke_test. "
                 "If you see this in Slack, wandb.alert() is wired up correctly.",
            level=wandb.AlertLevel.INFO,
        )
        wandb.finish()
    elif args.smoke_test:
        train(smoke_test=True, micro_batch_size=args.micro_batch_size, num_workers=args.num_workers,
              resume_from=args.resume_from)
    elif args.train:
        train(micro_batch_size=args.micro_batch_size, num_workers=args.num_workers,
              resume_from=args.resume_from)
    else:
        parser.print_help()
