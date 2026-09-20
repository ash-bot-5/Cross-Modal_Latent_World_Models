"""
Full fine-tuning of BOTH the CLIP ViT-L/14 image encoder AND text encoder, jointly with
a fresh FiLM-MLP latent dynamics model — diagnostic experiment.

Motivation: dynamics_model_clip_lora_finetune.py LoRA-fine-tuned only the image tower,
motivated by the suspicion that frozen CLIP TEXT embeddings collapse "A touching B" and
"A not touching B" into near-identical vectors, capping how well InfoNCE can separate the
touching/not-touching predicate. This script tests the natural next step -- unfreeze BOTH
towers -- but with FULL fine-tuning rather than LoRA (CLIP ViT-L/14 is only ~428M params;
full AdamW optimizer state fits comfortably in a single 3090 Ti's 24GB, so LoRA's memory
savings aren't needed here), at a conservative LR to avoid catastrophically destroying
CLIP's pretrained representations on this comparatively small/narrow dataset.

Key design points (see the approved plan for full rationale):
  - No LoRA at all. clip_lora.py is not used by this script.
  - Global statement table: every compound statement (positive or negative) is built from
    only the 8 fixed block identities, so the entire universe of possible statements is
    combinatorially bounded (56 ordered pairs x 2 states x 8 peg-blocks x 2 states = 1792),
    independent of episode count. This table is tokenized once and re-encoded fresh every
    training step (fixed cost, batch-composition-independent); every sample's positive/
    negative is a pure integer index into it -- no runtime string handling, no per-batch
    dict/set dedup.
  - Both A->B and B->A phrasings of the positive statement are trained every timestep
    (touching is symmetric) by calling the shared, UNMODIFIED infonce_logits/cross_entropy
    twice per step and averaging -- no dataset duplication, no randomization.
  - Held-out pair: forward-direction blue_moon->red_pentagon episodes are moved from train
    to val (one-directional, not a re-split) so the text-side margin probe can distinguish
    genuine generalization from memorization on a truly unseen pair.

Usage:
    python dynamics_model_clip_full_finetune.py --smoke_test
    python dynamics_model_clip_full_finetune.py --train
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
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import open_clip
import matplotlib
matplotlib.use("Agg")  # force non-interactive backend, same reason as dynamics_model_clip_lora_finetune.py
import matplotlib.pyplot as plt

from langtable_dataset import LangTableDataset, train_val_split
from dynamics_model import (
    LatentDynamicsModel,
    infonce_logits,
    ranking_metrics,
    similarity_metrics,
    mean_pairwise_cosine,
)
# Rebalances the training-loop-only forward pass across both GPUs (image tower + MLP get
# an uneven split, GPU 1's smaller shard is paired with the full text tower) instead of
# DataParallel's even split + sequential single-GPU encode_text -- see that module's
# docstring for the full rationale and the precision fix it applies. Validation,
# smoke_test, and diagnostics below are untouched and keep using ddp_model/encode_text
# directly, same as before.
from gpu_split_joint_step import ImageTextJointStep, calibrate_split, run_split_step
# Reused verbatim for the IMAGE-side margin probe / progress-curve diagnostics. This module
# has no side-effecting top-level code (everything executable lives under
# `if __name__ == "__main__":`), so importing it is safe.
from dynamics_model_clip_lora_finetune import (
    load_episode_frames_mmap,
    build_probe_episode_records,
    encode_probe_episodes,
    compute_margin_probe,
    compute_progress_curve_correlation,
    plot_progress_curves,
)

TOUCH_THRESH = 0.05  # matches cache_zhard.py exactly
HORIZON = 8          # matches LangTableDataset.HORIZON


# ---------------------------------------------------------------------------
# Global statement table (see plan §3) — every compound statement (positive or
# negative) is built from only the 8 fixed block identities, so the entire universe
# of possible statements is combinatorially bounded: 56 ordered pairs x 2 touching_ab
# states x 8 peg-blocks x 2 touching_peg states = 1792, independent of episode count.
# Built once and re-encoded fresh every training step; every sample's positive/negative
# is a pure integer index into it.
# ---------------------------------------------------------------------------

def get_block_names(data_dir: str) -> list[str]:
    """Derive the 8 fixed block names from real data, same convention as
    cache_zhard.py's `sorted(k for k in block_states[0].keys() if k != 'peg')` —
    block_states dicts also carry a 'peg' key (confirmed against real data), so it
    must be filtered out explicitly rather than assumed absent."""
    pkl_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    with open(os.path.join(data_dir, pkl_files[0]), "rb") as f:
        ep = pickle.load(f)
    names = sorted(k for k in ep["block_states"][0].keys() if k != "peg")
    assert len(names) == 8, f"expected 8 block names, got {len(names)}: {names}"
    return names


class GlobalStatementTable:
    """Canonical, deterministic enumeration of all 1792 possible compound statements.
    No episode data needed beyond the fixed 8 block names. Both the training
    negative/positive generation and the text margin probe's grounding scan share this
    same index arithmetic."""

    def __init__(self, block_names: list[str]):
        self.block_names = block_names
        self.n_blocks = len(block_names)
        self.pairs = list(permutations(block_names, 2))  # 56 ordered pairs
        self.n_pairs = len(self.pairs)
        assert self.n_pairs == 56 and self.n_blocks == 8

        self.pair_to_idx = {pair: i for i, pair in enumerate(self.pairs)}
        self.block_to_idx = {b: i for i, b in enumerate(block_names)}
        # (i, j) index-pairs mirroring self.pairs, for vectorized per-timestep lookups
        self.pair_block_idx = list(permutations(range(self.n_blocks), 2))

        self.size = self.n_pairs * 2 * self.n_blocks * 2  # 1792
        self.strings = self._build_strings()
        # ab_state[i] = 1 if entry i's A-B clause asserts "touching", else 0 -- derived
        # directly from the index arithmetic in index(): entries are laid out in blocks
        # of (n_blocks*2) sharing the same (pair_idx, ab_state), so this is a pure
        # arithmetic inverse of index(), no re-parsing of the string needed.
        self.ab_state = (np.arange(self.size) // (self.n_blocks * 2)) % 2
        # pair_idx_per_entry[i] = which of the 56 ordered (a,b) pairs entry i belongs to --
        # each pair spans a block of (2 ab_states * n_blocks peg_blocks * 2 peg_states) =
        # 4*n_blocks consecutive entries, so this is integer division by that block size.
        self.pair_idx_per_entry = np.arange(self.size) // (4 * self.n_blocks)

    def _build_strings(self) -> list[str]:
        strings: list[str | None] = [None] * self.size
        for pair_idx, (a, b) in enumerate(self.pairs):
            a_disp, b_disp = a.replace("_", " "), b.replace("_", " ")
            for ab_state in (0, 1):  # 0 = not touching, 1 = touching
                touching_ab = "touching" if ab_state else "not touching"
                for peg_idx, c in enumerate(self.block_names):
                    c_disp = c.replace("_", " ")
                    for peg_state in (0, 1):
                        touching_peg = "touching" if peg_state else "not touching"
                        idx = self.index(pair_idx, ab_state, peg_idx, peg_state)
                        strings[idx] = (
                            f"The {a_disp} is {touching_ab} the {b_disp}. "
                            f"The peg is {touching_peg} the {c_disp}."
                        )
        assert all(s is not None for s in strings)
        return strings  # type: ignore[return-value]

    def index(self, pair_idx: int, ab_state: int, peg_idx: int, peg_state: int) -> int:
        return ((pair_idx * 2 + ab_state) * self.n_blocks + peg_idx) * 2 + peg_state

    def index_for(self, a: str, b: str, ab_touching: bool, c: str, peg_touching: bool) -> int:
        pair_idx = self.pair_to_idx[(a, b)]
        peg_idx = self.block_to_idx[c]
        return self.index(pair_idx, int(ab_touching), peg_idx, int(peg_touching))


def compute_pos_neg_indices_for_episode(
    table: GlobalStatementTable,
    block_states: list[dict],
    peg_positions: np.ndarray,
    start_block: str,
    target_block: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized per-timestep computation of (pos_idx_ab, pos_idx_ba, neg_idx),
    mirroring cache_zhard.py's build_hard_neg_data logic exactly for negatives
    (56 pairs x 8 peg-blocks, inverted truth), but resolving directly to indices
    into the shared global table instead of a per-episode local dedup table. The
    positive indices use the TRUE (non-inverted) state for the episode's
    task-relevant pair, in BOTH orderings (touching is symmetric)."""
    T = len(block_states)
    block_names = table.block_names
    n_blocks = table.n_blocks

    n_negatives = table.n_pairs * n_blocks  # 56 * 8 = 448
    pos_idx_ab = np.zeros(T, dtype=np.int64)
    pos_idx_ba = np.zeros(T, dtype=np.int64)
    neg_idx = np.zeros((T, n_negatives), dtype=np.int64)

    pair_ab = table.pair_to_idx[(start_block, target_block)]
    pair_ba = table.pair_to_idx[(target_block, start_block)]
    start_i = table.block_to_idx[start_block]

    pair_block_idx = table.pair_block_idx  # 56 (i, j) tuples, fixed order

    for t in range(T):
        bpos = block_states[t]
        peg = peg_positions[t]

        block_pos_arr = np.stack([bpos[name] for name in block_names])  # (8, 2)
        diff = block_pos_arr[:, None, :] - block_pos_arr[None, :, :]     # (8, 8, 2)
        dist_matrix = np.linalg.norm(diff, axis=-1)                     # (8, 8)
        touch_matrix = dist_matrix < TOUCH_THRESH                       # (8, 8) bool, symmetric

        peg_dist = np.linalg.norm(block_pos_arr - peg[None, :], axis=-1)  # (8,)
        peg_touch = peg_dist < TOUCH_THRESH                               # (8,)

        # --- positive: TRUE (non-inverted) state for the task-relevant pair, both orderings ---
        ab_touch_true = bool(touch_matrix[table.block_to_idx[start_block], table.block_to_idx[target_block]])
        peg_touch_start = bool(peg_touch[start_i])
        pos_idx_ab[t] = table.index(pair_ab, int(ab_touch_true), start_i, int(peg_touch_start))
        pos_idx_ba[t] = table.index(pair_ba, int(ab_touch_true), start_i, int(peg_touch_start))

        # --- negatives: INVERTED truth, all 56 pairs x 8 peg-blocks ---
        ab_inv_per_pair = np.array([not touch_matrix[i, j] for (i, j) in pair_block_idx], dtype=np.int64)  # (56,)
        peg_inv_per_block = (~peg_touch).astype(np.int64)  # (8,)

        pair_idx_arr = np.arange(table.n_pairs)
        idx_grid = (
            (pair_idx_arr[:, None] * 2 + ab_inv_per_pair[:, None]) * n_blocks + np.arange(n_blocks)[None, :]
        ) * 2 + peg_inv_per_block[None, :]  # (56, 8)
        neg_idx[t] = idx_grid.reshape(-1)

    return pos_idx_ab, pos_idx_ba, neg_idx


# ---------------------------------------------------------------------------
# Held-out pair split adjustment (plan §0)
# ---------------------------------------------------------------------------

def apply_held_out_pair(
    train_files: list[str],
    val_files: list[str],
    data_dir: str,
    start_block: str,
    oracle_target_block: str,
    log_path: str,
) -> tuple[list[str], list[str], list[str]]:
    """Moves every train episode matching (metadata.start_block == start_block,
    metadata.oracle_target_block == oracle_target_block) into val. Does NOT touch
    val_files otherwise -- pre-existing matches there are counted/logged but left in
    place. train_val_split() itself is never modified (shared by other scripts).
    Returns (new_train_files, new_val_files, held_out_pair_stems) where the third
    list is ALL matching stems (moved + already-in-val), for the held-out-pair-
    specific margin probe diagnostic."""
    moved = []
    already_in_val = []

    remaining_train = []
    for fn in train_files:
        with open(os.path.join(data_dir, fn), "rb") as f:
            ep = pickle.load(f)
        meta = ep["metadata"]
        del ep
        if meta["start_block"] == start_block and meta["oracle_target_block"] == oracle_target_block:
            moved.append(fn)
        else:
            remaining_train.append(fn)

    for fn in val_files:
        with open(os.path.join(data_dir, fn), "rb") as f:
            ep = pickle.load(f)
        meta = ep["metadata"]
        del ep
        if meta["start_block"] == start_block and meta["oracle_target_block"] == oracle_target_block:
            already_in_val.append(fn)

    new_train_files = sorted(remaining_train)
    new_val_files = sorted(val_files + moved)
    held_out_pair_stems = sorted(fn[:-4] for fn in (moved + already_in_val))  # strip ".pkl"

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"Held-out pair: {start_block} -> {oracle_target_block} (forward direction only)\n\n")
        f.write(f"Episodes moved train->val: {len(moved)}\n")
        for fn in moved:
            f.write(f"  {fn}\n")
        f.write(f"\nOld train count: {len(train_files)}\n")
        f.write(f"Old val count:   {len(val_files)}\n")
        f.write(f"New train count: {len(new_train_files)}\n")
        f.write(f"New val count:   {len(new_val_files)}\n")
        f.write(f"\nAlready in val before move (not moved, just logged): {len(already_in_val)}\n")
        for fn in already_in_val:
            f.write(f"  {fn}\n")

    print(f"[held_out_pair] moved {len(moved)} episodes train->val "
          f"(train {len(train_files)}->{len(new_train_files)}, val {len(val_files)}->{len(new_val_files)}); "
          f"{len(already_in_val)} already in val. Log: {log_path}")

    return new_train_files, new_val_files, held_out_pair_stems


# ---------------------------------------------------------------------------
# Dataset: raw frames (mmap, unchanged approach) + integer indices into the global
# statement table (no runtime string handling at all).
# ---------------------------------------------------------------------------

class LangTableImageTextDataset(LangTableDataset):
    """LangTableDataset variant for joint image+text CLIP fine-tuning. Stores paths to
    pre-cached {stem}_frames.npy files plus precomputed integer indices into the shared
    GlobalStatementTable (no cached _zsem.npz/_zhard.npz needed for training data anymore
    -- the text tower is trainable now, so cached frozen-CLIP embeddings would be stale)."""

    def __init__(self, data_dir: str, preprocess, table: GlobalStatementTable, file_list: list[str] | None = None):
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.table = table

        self.frames_paths: list[str] = []
        self.actions: list[np.ndarray] = []
        self.pos_idx_ab: list[np.ndarray] = []
        self.pos_idx_ba: list[np.ndarray] = []
        self.neg_idx: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []

        pkl_files = file_list if file_list is not None else sorted(
            f for f in os.listdir(data_dir) if f.endswith(".pkl")
        )

        for pkl_file in pkl_files:
            pkl_path = os.path.join(data_dir, pkl_file)
            frames_path = pkl_path.replace(".pkl", "_frames.npy")
            if not os.path.exists(frames_path):
                warnings.warn(f"Missing frames_npy file for {pkl_file} "
                               f"(run cache_frames_npy.py first), skipping episode.")
                continue

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)
            actions = ep["actions"]
            block_states = ep["block_states"]
            peg_positions = ep["actual_peg_positions"]
            start_block = ep["metadata"]["start_block"]
            target_block = ep["metadata"]["oracle_target_block"]
            del ep  # let the bulky frames array fall out of scope

            T = len(block_states)
            pos_ab, pos_ba, neg = compute_pos_neg_indices_for_episode(
                self.table, block_states, peg_positions, start_block, target_block
            )

            ep_idx = len(self.actions)
            self.frames_paths.append(frames_path)
            self.actions.append(np.asarray(actions, dtype=np.float32))
            self.pos_idx_ab.append(pos_ab)
            self.pos_idx_ba.append(pos_ba)
            self.neg_idx.append(neg)

            for t in range(T - HORIZON):
                self.indices.append((ep_idx, t))

    def _normalize_actions(self, actions: np.ndarray) -> torch.Tensor:
        # Fixed-constant rescale: 0.035 is the max magnitude of the raw delta-x/y actions
        # in this environment, so this is a dataset-independent rescale rather than a
        # per-dataset min/max statistic fit at load time.
        a = torch.from_numpy(np.array(actions, dtype=np.float32))
        return a / 0.035

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, t = self.indices[idx]
        frame = np.asarray(load_episode_frames_mmap(self.frames_paths[ep_idx])[t])
        pixel_values = self.preprocess(Image.fromarray(frame))

        actions_norm = self._normalize_actions(self.actions[ep_idx][t : t + HORIZON])
        pos_ab = int(self.pos_idx_ab[ep_idx][t + HORIZON])
        pos_ba = int(self.pos_idx_ba[ep_idx][t + HORIZON])
        neg = torch.tensor(self.neg_idx[ep_idx][t + HORIZON], dtype=torch.int64)

        return {
            "pixel_values": pixel_values,
            "actions": actions_norm,
            "pos_global_idx_ab": pos_ab,
            "pos_global_idx_ba": pos_ba,
            "neg_global_idx": neg,
        }


class JointModel(nn.Module):
    """Wraps the (fully unfrozen) CLIP model's IMAGE tower + LatentDynamicsModel MLP so a
    single nn.DataParallel(...) replicates+scatters both together in one forward call.
    Text encoding is deliberately NOT part of this forward -- see the training loop for why
    (DataParallel's automatic batch-dim scatter/gather doesn't compose with indexing that
    assumes one global, unsharded statement table)."""

    def __init__(self, clip_model: nn.Module, mlp: LatentDynamicsModel):
        super().__init__()
        self.clip_model = clip_model
        self.mlp = mlp

    def forward(self, pixel_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z_t = self.clip_model.encode_image(pixel_values, normalize=True)  # (B, 768), grad-enabled
        return self.mlp(z_t, actions)  # (B, 768) L2-normalized


# ---------------------------------------------------------------------------
# Unfreezing both towers + optimizer param groups (plan §1, §2)
# ---------------------------------------------------------------------------

def unfreeze_both_towers(clip_model: nn.Module) -> None:
    """Full fine-tuning: every image-tower and text-tower parameter trainable.
    logit_scale/logit_bias frozen (unused by our fixed-temperature InfoNCE loss)."""
    for name, p in clip_model.named_parameters():
        if name in ("logit_scale", "logit_bias"):
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)


def split_decay(named_params: list[tuple[str, torch.nn.Parameter]]) -> tuple[list, list]:
    """Standard transformer-fine-tuning convention: exclude bias, LayerNorm, and
    embedding-like params (positional/token/class embeddings) from weight decay."""
    decay, no_decay = [], []
    for n, p in named_params:
        if n.endswith(".bias") or "ln_" in n or "LayerNorm" in n or "embedding" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(clip_model: nn.Module, mlp: nn.Module, lr_image: float, lr_text: float, lr_mlp: float):
    image_named = [(n, p) for n, p in clip_model.named_parameters()
                   if n.startswith("visual.") and p.requires_grad]
    text_named = [(n, p) for n, p in clip_model.named_parameters()
                  if not n.startswith("visual.") and n not in ("logit_scale", "logit_bias") and p.requires_grad]

    image_decay, image_no_decay = split_decay(image_named)
    text_decay, text_no_decay = split_decay(text_named)
    mlp_params = list(mlp.parameters())

    optimizer = torch.optim.AdamW([
        {"params": image_decay, "lr": lr_image, "weight_decay": 0.01},
        {"params": image_no_decay, "lr": lr_image, "weight_decay": 0.0},
        {"params": text_decay, "lr": lr_text, "weight_decay": 0.01},
        {"params": text_no_decay, "lr": lr_text, "weight_decay": 0.0},
        {"params": mlp_params, "lr": lr_mlp},  # weight_decay unset -> AdamW default 0.01
    ])

    all_trainable = [p for _, p in image_named] + [p for _, p in text_named] + mlp_params
    return optimizer, all_trainable


def lr_warmup_multiplier(step: int, warmup_steps: int) -> float:
    """Linear warmup: ramps from 1/warmup_steps up to 1.0 over the first warmup_steps
    optimizer steps (never a literal 0 multiplier, so no step is fully wasted), then
    stays at 1.0 for the rest of training. `step` is the same 0-indexed counter already
    used for wandb logging/checkpointing/resume -- one increment per optimizer-step
    window processed (including windows later skipped for non-finite gradients,
    consistent with existing step-counting semantics)."""
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup_steps)


def apply_lr_warmup(optimizer: torch.optim.Optimizer, base_lrs: list[float], step: int, warmup_steps: int) -> float:
    """Scales every param group's LR by the current warmup multiplier, in place.
    Call this right before optimizer.step() every window (even ones later skipped for
    non-finite gradients). Pure function of `step` -- no scheduler object, no extra
    checkpoint state -- so resuming from `step = ckpt.get("step", 0)` already gets the
    warmup ramp (or correct flat post-warmup LR) right with zero new bookkeeping.
    Returns the multiplier actually applied, for logging."""
    mult = lr_warmup_multiplier(step, warmup_steps)
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = base_lr * mult
    return mult


# ---------------------------------------------------------------------------
# Text-side margin probe: grounds the global table against real simulation state
# (plan §6). Reuses the exact same combinatorial structure as training negatives,
# just without inverting truth.
# ---------------------------------------------------------------------------

def ground_global_table(table: GlobalStatementTable, data_dir: str, stems: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Scans real block_states/actual_peg_positions of the given episodes, for ALL 56
    pairs (not just each episode's task-relevant one) and all 8 peg-blocks, at every
    timestep. Returns (touching_group_idx, not_touching_group_idx): deduplicated int
    arrays of indices into the global table, split by whether the A-B clause was ever
    observed TRUE (touching) or FALSE (not touching) at some real (episode, t)."""
    touching_set: set[int] = set()
    not_touching_set: set[int] = set()

    block_names = table.block_names
    n_blocks = table.n_blocks
    pair_block_idx = table.pair_block_idx
    pair_idx_arr = np.arange(table.n_pairs)

    for stem in stems:
        pkl_path = os.path.join(data_dir, stem + ".pkl")
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)
        block_states = ep["block_states"]
        peg_positions = ep["actual_peg_positions"]
        del ep
        T = len(block_states)

        for t in range(T):
            bpos = block_states[t]
            peg = peg_positions[t]

            block_pos_arr = np.stack([bpos[name] for name in block_names])
            diff = block_pos_arr[:, None, :] - block_pos_arr[None, :, :]
            dist_matrix = np.linalg.norm(diff, axis=-1)
            touch_matrix = dist_matrix < TOUCH_THRESH

            peg_dist = np.linalg.norm(block_pos_arr - peg[None, :], axis=-1)
            peg_touch = peg_dist < TOUCH_THRESH

            ab_true_per_pair = np.array([touch_matrix[i, j] for (i, j) in pair_block_idx], dtype=np.int64)  # (56,)
            peg_true_per_block = peg_touch.astype(np.int64)  # (8,)

            idx_grid = (
                (pair_idx_arr[:, None] * 2 + ab_true_per_pair[:, None]) * n_blocks + np.arange(n_blocks)[None, :]
            ) * 2 + peg_true_per_block[None, :]  # (56, 8)

            flat_idx = idx_grid.reshape(-1)
            flat_ab_true = np.repeat(ab_true_per_pair, n_blocks)
            for idx, is_touching in zip(flat_idx.tolist(), flat_ab_true.tolist()):
                (touching_set if is_touching else not_touching_set).add(idx)

    return np.array(sorted(touching_set), dtype=np.int64), np.array(sorted(not_touching_set), dtype=np.int64)


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


def compute_film_gamma_beta_stats(mlp: LatentDynamicsModel, actions: torch.Tensor) -> dict:
    """Mean |gamma|/|beta| per FiLM layer on a real action batch -- distinguishes
    'FiLM unused' (values still near their zero-init identity) from 'FiLM used but
    insufficient' (values have moved, trunk still can't separate touching states)."""
    with torch.no_grad():
        cond = mlp.action_encoder(actions)
        gamma1, beta1 = mlp.film1.proj(cond).chunk(2, dim=-1)
        gamma2, beta2 = mlp.film2.proj(cond).chunk(2, dim=-1)
    return {
        "film/film1_gamma_abs_mean": gamma1.abs().mean().item(),
        "film/film1_beta_abs_mean":  beta1.abs().mean().item(),
        "film/film2_gamma_abs_mean": gamma2.abs().mean().item(),
        "film/film2_beta_abs_mean":  beta2.abs().mean().item(),
    }


def _append_jsonl(path: str, record: dict) -> None:
    """Appends one JSON line -- a durable, local, wandb-independent mirror of
    everything sent to wandb.log(), so the full metric history survives on disk
    even if wandb's own storage is ever unavailable. `default=str` is a safety
    net for any stray non-JSON-serializable value slipping through."""
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def compute_text_margin(unique_z: np.ndarray, touching_idx: np.ndarray, not_touching_idx: np.ndarray) -> dict:
    """Same within/between margin formula as compute_margin_probe, applied to text
    table embeddings pooled globally (table entries aren't tied to a single episode's
    scene identity, so global pooling is valid here unlike the per-episode image probe)."""
    if len(touching_idx) == 0 or len(not_touching_idx) == 0:
        return {"mean": float("nan"), "n_touching": len(touching_idx), "n_not_touching": len(not_touching_idx)}

    sim = unique_z @ unique_z.T
    within_pos = _mean_excl_self(sim, touching_idx, touching_idx, True)
    within_neg = _mean_excl_self(sim, not_touching_idx, not_touching_idx, True)
    between = _mean_excl_self(sim, touching_idx, not_touching_idx, False)
    if within_pos is None or within_neg is None or between is None:
        return {"mean": float("nan"), "n_touching": len(touching_idx), "n_not_touching": len(not_touching_idx)}

    margin = (within_pos + within_neg) / 2.0 - between
    return {
        "mean": float(margin),
        "within_touching": float(within_pos),
        "within_not_touching": float(within_neg),
        "between": float(between),
        "n_touching": len(touching_idx),
        "n_not_touching": len(not_touching_idx),
    }


def fit_umap_projection(embeddings: np.ndarray, random_state: int = 0):
    """Fits UMAP ONCE on the given (1792, 768) embeddings (the pretrained-CLIP
    baseline, before any fine-tuning). Returns the fitted reducer; callers must
    reuse it via .transform() for every later epoch -- never re-fit -- so the 2D
    coordinate system stays fixed and visual movement reflects real embedding
    drift during fine-tuning, not a re-fit artifact."""
    import umap
    reducer = umap.UMAP(n_components=2, random_state=random_state)
    reducer.fit(embeddings)
    return reducer


def plot_text_embedding_umap(
    reducer, unique_z: np.ndarray, table: GlobalStatementTable, out_dir: str, checkpoint_tag: str,
) -> str:
    """Projects the CURRENT epoch's (1792, 768) table embeddings through the fixed
    reducer via .transform() (never .fit_transform()), scatter-plots the 2D result
    colored by table.ab_state (2 colors, legend), saves as PNG, returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    coords = reducer.transform(unique_z)  # (1792, 2)

    fig, ax = plt.subplots(figsize=(8, 8))
    touching = table.ab_state.astype(bool)
    ax.scatter(coords[touching, 0], coords[touching, 1], s=6, alpha=0.6, color="tab:blue", label="touching")
    ax.scatter(coords[~touching, 0], coords[~touching, 1], s=6, alpha=0.6, color="tab:red", label="not touching")
    ax.legend()
    ax.set_title(f"Text embedding UMAP (fixed basis) — {checkpoint_tag}")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{checkpoint_tag}_text_embedding_umap.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def plot_text_embedding_umap_task_identity(
    reducer, unique_z: np.ndarray, table: GlobalStatementTable, out_dir: str, checkpoint_tag: str,
) -> str:
    """Same fixed reducer, same embeddings as plot_text_embedding_umap -- reuses
    .transform() again (still never .fit_transform()), just recolors the identical
    point cloud by which of the 56 start_block->target_block pairs each entry
    belongs to, instead of by touching state."""
    os.makedirs(out_dir, exist_ok=True)
    coords = reducer.transform(unique_z)  # (1792, 2)

    cmap = plt.cm.gist_rainbow
    n_pairs = table.n_pairs
    fig, ax = plt.subplots(figsize=(10, 6))
    for pair_idx, (a, b) in enumerate(table.pairs):
        m = table.pair_idx_per_entry == pair_idx
        color = cmap(pair_idx / max(n_pairs - 1, 1))
        ax.scatter(coords[m, 0], coords[m, 1], s=20, alpha=0.7, linewidths=0,
                   color=color, label=f"{a} → {b}")
    ax.set_title(f"Text embedding UMAP (fixed basis) — task identity — {checkpoint_tag}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=5, framealpha=0.8, ncol=2,
              title="start_block → target_block")
    png_path = os.path.join(out_dir, f"{checkpoint_tag}_text_embedding_umap_task_identity.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str = "/home/ashwink/full_data_5000/langtable/demos/",
    device: str = "cuda",
    micro_batch_size: int = 256,
    effective_batch_size: int = 256,
    num_epochs: int = 16,
    lr_image: float = 1e-6,
    lr_text: float = 1e-6,
    lr_mlp: float = 3e-4,
    temperature: float = 0.1,
    wandb_project: str = "langtable-dynamics",
    checkpoint_dir: str = "checkpoints_clip_image_text_finetuned",
    resume_from: str | None = None,
    n_val_episodes: int = 750,
    val_seed: int = 0,
    num_workers: int = 8,
    probe_n_episodes: int = 75,
    probe_seed: int = 123,
    full_probe_every: int = 5,
    diagnostic_plot_episodes: int = 5,
    diagnostics_dir: str = "clip_full_finetune_diagnostics",
    held_out_start_block: str = "blue_moon",
    held_out_target_block: str = "red_pentagon",
    smoke_test: bool = False,
    smoke_test_steps: int = 100,
    legacy_even_split: bool = False,
):
    grad_accum_steps = math.ceil(effective_batch_size / micro_batch_size)
    grad_accum_triggered = grad_accum_steps > 1

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- data split + held-out pair adjustment (plan §0) ----
    train_files, val_files = train_val_split(data_dir, n_val=n_val_episodes, seed=val_seed)
    train_files, val_files, held_out_pair_stems = apply_held_out_pair(
        train_files, val_files, data_dir,
        start_block=held_out_start_block, oracle_target_block=held_out_target_block,
        log_path=os.path.join(checkpoint_dir, "held_out_pair_move_log.txt"),
    )

    # ---- global statement table (plan §3) ----
    block_names = get_block_names(data_dir)
    table = GlobalStatementTable(block_names)
    print(f"Global statement table: {table.size} entries ({table.n_pairs} pairs x 2 x "
          f"{table.n_blocks} peg-blocks x 2)")

    # ---- CLIP construction, both towers unfrozen (plan §1) ----
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(device)
    unfreeze_both_towers(clip_model)
    # Gradient checkpointing (both towers): without this, the text tower alone -- forwarding
    # all 1792 global-table statements WITH gradients every step -- retains all 12 layers'
    # activations simultaneously (nothing can be freed until backward() runs), which alone
    # needs 20+ GB and OOMs regardless of image micro_batch_size. Verified empirically: with
    # this enabled, the same 1792-statement forward+backward drops to ~11GB peak. Trades some
    # compute for memory (recomputes activations during backward instead of storing them).
    clip_model.set_grad_checkpointing(True)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    global_tokens = tokenizer(table.strings).to(device)  # tokenized once, reused every step

    mlp = LatentDynamicsModel().to(device)  # fresh random init
    joint_model = JointModel(clip_model, mlp).to(device)

    optimizer, all_trainable_params = build_optimizer(clip_model, mlp, lr_image, lr_text, lr_mlp)
    base_lrs = [g["lr"] for g in optimizer.param_groups]  # captured before any warmup scaling touches them

    ddp_model = nn.DataParallel(joint_model, device_ids=[0, 1]) if torch.cuda.device_count() > 1 else joint_model

    # ---- rebalanced GPU split (train loop + smoke_test only -- see gpu_split_joint_step.py).
    # Calibrated fresh for this run's actual micro_batch_size/hardware rather than a
    # hardcoded ratio, done once here so both smoke_test and the real training loop below
    # share the same calibration. Falls back to the untouched ddp_model + sequential
    # encode_text path (identical to today's behavior) if only 1 GPU is visible,
    # --legacy_even_split was passed, or calibration itself fails. Validation and
    # diagnostics remain on the untouched path regardless. ----
    use_rebalanced_split = False
    step_module = None
    global_tokens_gpu1 = None
    split_n0 = split_n1 = None
    gpu_split_calib = None
    gpu_split_device_ids = [0, 1]
    if not legacy_even_split and torch.cuda.device_count() > 1:
        global_tokens_gpu1 = global_tokens.to(gpu_split_device_ids[1])
        step_module = ImageTextJointStep(clip_model, mlp)
        gpu_split_calib = calibrate_split(clip_model, mlp, gpu_split_device_ids, micro_batch_size, HORIZON, global_tokens)
        if gpu_split_calib is not None:
            use_rebalanced_split = True
            split_n0, split_n1 = gpu_split_calib["n0"], gpu_split_calib["n1"]

    train_dataset = LangTableImageTextDataset(data_dir, preprocess, table, file_list=train_files)
    val_dataset = LangTableImageTextDataset(data_dir, preprocess, table, file_list=val_files)

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False, pin_memory=True)
    print(f"Train dataset: {len(train_dataset)} samples, {len(train_loader)} micro-batches/epoch")
    print(f"Val dataset:   {len(val_dataset)} samples, {len(val_loader)} micro-batches/epoch")

    n_steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = n_steps_per_epoch * num_epochs

    # ---- LR warmup: 5% of total_steps, floor 50 -- shared by smoke_test and full training ----
    warmup_steps = max(50, int(0.05 * total_steps))
    print(f"LR warmup: {warmup_steps} steps (5% of {total_steps} total steps, floor 50)")

    # ---- fixed probe subsample (sampled once, seeded, from the val split) ----
    rng = random.Random(probe_seed)
    val_stems_all = [f[:-4] for f in val_files]
    probe_stems = rng.sample(val_stems_all, min(probe_n_episodes, len(val_stems_all)))
    plot_stems = rng.sample(probe_stems, min(diagnostic_plot_episodes, len(probe_stems)))
    probe_records = build_probe_episode_records(data_dir, probe_stems)
    print(f"Probe subsample: {len(probe_records)} episodes (seed={probe_seed}); "
          f"{len(plot_stems)} fixed for progress-curve plots; "
          f"{len(held_out_pair_stems)} held-out-pair episodes for the generalization diagnostic")

    # ---- text-side margin probe grounding (deterministic, built once) ----
    touching_idx_sub, not_touching_idx_sub = ground_global_table(table, data_dir, probe_stems)
    touching_idx_full, not_touching_idx_full = ground_global_table(table, data_dir, val_stems_all)
    touching_idx_held, not_touching_idx_held = ground_global_table(table, data_dir, held_out_pair_stems)

    # ===========================================================================
    # Smoke test
    # ===========================================================================
    if smoke_test:
        print(f"\n[smoke_test] micro_batch_size={micro_batch_size}, "
              f"grad_accum_steps={grad_accum_steps} (triggered={grad_accum_triggered})")
        if use_rebalanced_split:
            print(f"[smoke_test] rebalanced GPU split active: n0={split_n0} (GPU "
                  f"{gpu_split_device_ids[0]}, image-only) / n1={split_n1} (GPU "
                  f"{gpu_split_device_ids[1]}, image+text) -- pass --legacy_even_split to compare "
                  f"against today's even-split DataParallel + sequential encode_text instead.")
        else:
            print(f"[smoke_test] using even-split DataParallel + sequential encode_text "
                  f"({'--legacy_even_split passed' if legacy_even_split else 'rebalanced split unavailable, see warning above if any'}).")
        joint_model.train()
        step_times = []
        step_losses = []
        coverage_ratios = []
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
                pos_ab = batch["pos_global_idx_ab"].to(device, non_blocking=True)
                pos_ba = batch["pos_global_idx_ba"].to(device, non_blocking=True)
                neg_idx = batch["neg_global_idx"].to(device, non_blocking=True)

                referenced = torch.unique(torch.cat([pos_ab, pos_ba, neg_idx.flatten()]))
                coverage_ratios.append(referenced.numel() / table.size)

                if use_rebalanced_split:
                    z_pred, unique_z = run_split_step(
                        step_module, pixel_values, actions, global_tokens_gpu1,
                        gpu_split_device_ids, split_n0, split_n1,
                    )
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        z_pred = ddp_model(pixel_values, actions)
                        unique_z = F.normalize(joint_model.clip_model.encode_text(global_tokens), dim=-1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_sem_target_ab = unique_z[pos_ab]
                    z_sem_target_ba = unique_z[pos_ba]
                    z_hard_neg = unique_z[neg_idx]

                    logits_ab = infonce_logits(z_pred.float(), z_sem_target_ab.float(), z_hard_neg.float(), temperature)
                    logits_ba = infonce_logits(z_pred.float(), z_sem_target_ba.float(), z_hard_neg.float(), temperature)
                    targets = torch.zeros(z_pred.shape[0], dtype=torch.long, device=device)
                    loss = 0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets))

                (loss / grad_accum_steps).backward()
                accum_counter += 1

                if accum_counter == grad_accum_steps:
                    torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                    apply_lr_warmup(optimizer, base_lrs, step, warmup_steps)
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
        print(f"(e) global table coverage per step: mean={np.mean(coverage_ratios):.3f} "
              f"(fraction of {table.size} entries referenced)")

        last_completed_step = max(step - 1, 0)
        end_mult = lr_warmup_multiplier(last_completed_step, warmup_steps)
        still_warming_up = last_completed_step < warmup_steps - 1
        print(f"(f) LR warmup: warmup_steps={warmup_steps}, multiplier at step {last_completed_step} = "
              f"{end_mult:.4f} ({'still ramping' if still_warming_up else 'fully warmed up'})")

        losses = np.array(step_losses)
        if np.isnan(losses).any() or np.isinf(losses).any():
            print("(d) WARNING: loss trajectory contains NaN/Inf — LR likely too high.")
        else:
            first10, last10 = losses[:10].mean(), losses[-10:].mean()
            print(f"(d) loss[:10] mean={first10:.4f}, loss[-10:] mean={last10:.4f}")
            if abs(first10 - last10) < 0.02 * max(abs(first10), 1e-6):
                flat_msg = ("(d) WARNING: loss looks flat over 100 steps — at LR=1e-6 for both "
                            "towers this may be expected this early, but check longer runs.")
                if still_warming_up:
                    flat_msg += (f" Also note: LR warmup is still ramping (multiplier={end_mult:.4f} "
                                 f"of target at this step) — flatness may just reflect a still-small "
                                 f"effective LR, not necessarily a real problem.")
                print(flat_msg)
        return

    # ===========================================================================
    # Full training run
    # ===========================================================================
    import wandb

    with open(os.path.join(checkpoint_dir, "train_val_split.json"), "w") as f:
        json.dump({"seed": val_seed, "n_val": len(val_files),
                    "train_files": train_files, "val_files": val_files,
                    "held_out_pair_stems": held_out_pair_stems}, f, indent=2)

    wandb.init(
        project=wandb_project,
        name="clip_full_finetune_image_text",
        config=dict(
            finetune_scope="full_image_text",
            image_lr=lr_image, text_lr=lr_text, mlp_lr=lr_mlp,
            weight_decay_policy="exclude bias/LayerNorm/embedding from decay on both towers",
            changed_weight=1.0, negative_count=448, temperature=temperature,
            n_train_episodes=len(train_files), n_val_episodes=len(val_files), val_seed=val_seed,
            held_out_pair=f"{held_out_start_block}->{held_out_target_block} (forward only)",
            n_held_out_pair_episodes=len(held_out_pair_stems),
            global_table_size=table.size,
            effective_batch_size=effective_batch_size, micro_batch_size=micro_batch_size,
            grad_accum_steps=grad_accum_steps, grad_accum_triggered=grad_accum_triggered,
            mixed_precision="bf16", data_parallel_device_ids=[0, 1],
            num_epochs=num_epochs, total_steps=total_steps,
            warmup_steps=warmup_steps, warmup_frac=0.05, warmup_floor=50,
            probe_n_episodes=probe_n_episodes, probe_seed=probe_seed,
            full_probe_every=full_probe_every,
        ),
    )

    # Local, wandb-independent mirror of everything sent to wandb.log() -- plain JSONL
    # files on disk, one record per line, plus registered with wandb.save(policy="live")
    # so they also show up as real downloadable files in the run (not just chart data
    # trapped inside wandb's own storage).
    step_log_path = os.path.join(checkpoint_dir, "step_metrics.jsonl")
    epoch_log_path = os.path.join(checkpoint_dir, "epoch_metrics.jsonl")
    file_mode = "a" if resume_from is not None else "w"  # fresh run wipes stale logs; resume appends
    open(step_log_path, file_mode).close()
    open(epoch_log_path, file_mode).close()
    wandb.save(step_log_path, policy="live")
    wandb.save(epoch_log_path, policy="live")

    # ---- fixed-basis UMAP: fit ONCE on the pretrained-CLIP text embeddings (before
    # any fine-tuning gradient step touches the text tower), then reuse via .transform()
    # for every later epoch so visual movement reflects real embedding drift, not a
    # re-fit artifact. Logged as a "pretrain" baseline frame before the epoch loop starts.
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pretrained_unique_z = F.normalize(clip_model.encode_text(global_tokens), dim=-1)
        pretrained_unique_z = pretrained_unique_z.float().cpu().numpy()

    umap_reducer = fit_umap_projection(pretrained_unique_z)
    umap_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), diagnostics_dir)
    pretrain_umap_path = plot_text_embedding_umap(umap_reducer, pretrained_unique_z, table, umap_out_dir, "epoch_pretrain")
    wandb.log({"text_embedding_umap/pretrain": wandb.Image(pretrain_umap_path)})
    _append_jsonl(epoch_log_path, {"text_embedding_umap/pretrain": pretrain_umap_path})

    pretrain_umap_task_path = plot_text_embedding_umap_task_identity(umap_reducer, pretrained_unique_z, table, umap_out_dir, "epoch_pretrain")
    wandb.log({"text_embedding_umap_task_identity/pretrain": wandb.Image(pretrain_umap_task_path)})
    _append_jsonl(epoch_log_path, {"text_embedding_umap_task_identity/pretrain": pretrain_umap_task_path})

    try:
        start_epoch = 0
        step = 0
        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=device)
            clip_model.load_state_dict(ckpt["clip_state_dict"])
            mlp.load_state_dict(ckpt["mlp_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            step = ckpt.get("step", 0)
            print(f"Resumed from {resume_from} at epoch {ckpt['epoch']}")

        # rebalanced-split calibration already ran above (shared with smoke_test); just
        # record the result into this run's wandb.config now that wandb.init() has happened
        if gpu_split_calib is not None:
            wandb.config.update({
                "gpu_split/n0": split_n0, "gpu_split/n1": split_n1,
                "gpu_split/t_img_full_sec": gpu_split_calib["t_img_full"],
                "gpu_split/t_text_sec": gpu_split_calib["t_text"],
            })

        for epoch in range(start_epoch, num_epochs):
            joint_model.train()
            epoch_loss_sum = 0.0
            epoch_metrics_sum = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
            n_optimizer_steps_this_epoch = 0
            epoch_referenced = set()

            accum_counter = 0
            accum_loss = 0.0
            accum_metrics = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
            accum_extra = {"mean_pairwise_cosine": 0.0, "pos_sim_mean": 0.0, "hard_sim_mean": 0.0}
            step_start = time.perf_counter()

            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                actions = batch["actions"].to(device, non_blocking=True)
                pos_ab = batch["pos_global_idx_ab"].to(device, non_blocking=True)
                pos_ba = batch["pos_global_idx_ba"].to(device, non_blocking=True)
                neg_idx = batch["neg_global_idx"].to(device, non_blocking=True)

                with torch.no_grad():
                    referenced = torch.unique(torch.cat([pos_ab, pos_ba, neg_idx.flatten()]))
                    epoch_referenced.update(referenced.tolist())

                if use_rebalanced_split:
                    z_pred, unique_z = run_split_step(
                        step_module, pixel_values, actions, global_tokens_gpu1,
                        gpu_split_device_ids, split_n0, split_n1,
                    )
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        z_pred = ddp_model(pixel_values, actions)
                        unique_z = F.normalize(joint_model.clip_model.encode_text(global_tokens), dim=-1)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_sem_target_ab = unique_z[pos_ab]
                    z_sem_target_ba = unique_z[pos_ba]
                    z_hard_neg = unique_z[neg_idx]

                    logits_ab = infonce_logits(z_pred.float(), z_sem_target_ab.float(), z_hard_neg.float(), temperature)
                    logits_ba = infonce_logits(z_pred.float(), z_sem_target_ba.float(), z_hard_neg.float(), temperature)
                    targets = torch.zeros(z_pred.shape[0], dtype=torch.long, device=device)
                    loss = 0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets))

                if not torch.isfinite(loss):
                    print(f"[step {step}] WARNING: non-finite loss ({loss.item()}) — "
                          f"skipping backward for this micro-batch.")
                    continue

                (loss / grad_accum_steps).backward()
                accum_counter += 1

                with torch.no_grad():
                    accum_loss += loss.item()
                    metrics_ab = ranking_metrics(logits_ab)
                    metrics_ba = ranking_metrics(logits_ba)
                    sims_ab = similarity_metrics(logits_ab, temperature)
                    sims_ba = similarity_metrics(logits_ba, temperature)
                    mean_pairwise = mean_pairwise_cosine(z_pred.float())
                    for k in accum_metrics:
                        accum_metrics[k] += 0.5 * (metrics_ab[k] + metrics_ba[k])
                    accum_extra["mean_pairwise_cosine"] += mean_pairwise.item()
                    accum_extra["pos_sim_mean"] += 0.5 * (sims_ab["pos_sim_mean"] + sims_ba["pos_sim_mean"])
                    accum_extra["hard_sim_mean"] += 0.5 * (sims_ab["hard_sim_mean"] + sims_ba["hard_sim_mean"])

                if accum_counter == grad_accum_steps:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                    apply_lr_warmup(optimizer, base_lrs, step, warmup_steps)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        print(f"[step {step}] WARNING: non-finite grad_norm ({grad_norm.item()}) — "
                              f"skipping optimizer.step() for this window.")
                    optimizer.zero_grad()

                    step_time = time.perf_counter() - step_start
                    n = grad_accum_steps
                    step_loss = accum_loss / n
                    step_record = {
                        "train/loss": step_loss,
                        "train/grad_norm": grad_norm.item(),
                        "train/mean_pairwise_cosine": accum_extra["mean_pairwise_cosine"] / n,
                        "train/pos_sim_mean": accum_extra["pos_sim_mean"] / n,
                        "train/hard_sim_mean": accum_extra["hard_sim_mean"] / n,
                        "train/step_time_sec": step_time,
                        "train/lr_image": optimizer.param_groups[0]["lr"],
                        "train/lr_text": optimizer.param_groups[2]["lr"],
                        "train/lr_mlp": optimizer.param_groups[4]["lr"],
                        "step": step,
                    }
                    wandb.log(step_record)
                    _append_jsonl(step_log_path, step_record)
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

            if accum_counter > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, 1.0)
                apply_lr_warmup(optimizer, base_lrs, step, warmup_steps)
                if torch.isfinite(grad_norm):
                    optimizer.step()
                else:
                    print(f"[step {step}] WARNING: non-finite grad_norm ({grad_norm.item()}) — "
                          f"skipping optimizer.step() for this window.")
                optimizer.zero_grad()
                n = accum_counter
                step_loss = accum_loss / n
                step_record = {
                    "train/loss": step_loss, "train/grad_norm": grad_norm.item(),
                    "train/mean_pairwise_cosine": accum_extra["mean_pairwise_cosine"] / n,
                    "train/pos_sim_mean": accum_extra["pos_sim_mean"] / n,
                    "train/hard_sim_mean": accum_extra["hard_sim_mean"] / n,
                    "train/lr_image": optimizer.param_groups[0]["lr"],
                    "train/lr_text": optimizer.param_groups[2]["lr"],
                    "train/lr_mlp": optimizer.param_groups[4]["lr"],
                    "step": step,
                }
                wandb.log(step_record)
                _append_jsonl(step_log_path, step_record)
                epoch_loss_sum += step_loss
                for k in epoch_metrics_sum:
                    epoch_metrics_sum[k] += accum_metrics[k] / n
                n_optimizer_steps_this_epoch += 1
                step += 1

            avg_train_loss = epoch_loss_sum / n_optimizer_steps_this_epoch
            avg_train_metrics = {k: v / n_optimizer_steps_this_epoch for k, v in epoch_metrics_sum.items()}
            table_coverage = len(epoch_referenced) / table.size

            # ---- validation pass ----
            joint_model.eval()
            val_loss_sum = 0.0
            val_metrics_sum = {"top1": 0.0, "top5": 0.0, "mean_rank": 0.0}
            val_extra_sum = {"mean_pairwise_cosine": 0.0, "pos_sim_mean": 0.0, "hard_sim_mean": 0.0}
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                    actions = batch["actions"].to(device, non_blocking=True)
                    pos_ab = batch["pos_global_idx_ab"].to(device, non_blocking=True)
                    pos_ba = batch["pos_global_idx_ba"].to(device, non_blocking=True)
                    neg_idx = batch["neg_global_idx"].to(device, non_blocking=True)

                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        z_pred = ddp_model(pixel_values, actions)
                        unique_z = F.normalize(joint_model.clip_model.encode_text(global_tokens), dim=-1)
                    z_sem_target_ab = unique_z[pos_ab]
                    z_sem_target_ba = unique_z[pos_ba]
                    z_hard_neg = unique_z[neg_idx]

                    logits_ab = infonce_logits(z_pred.float(), z_sem_target_ab.float(), z_hard_neg.float(), temperature)
                    logits_ba = infonce_logits(z_pred.float(), z_sem_target_ba.float(), z_hard_neg.float(), temperature)
                    targets = torch.zeros(z_pred.shape[0], dtype=torch.long, device=device)
                    val_loss_sum += (0.5 * (F.cross_entropy(logits_ab, targets) + F.cross_entropy(logits_ba, targets))).item()

                    metrics_ab = ranking_metrics(logits_ab)
                    metrics_ba = ranking_metrics(logits_ba)
                    for k in val_metrics_sum:
                        val_metrics_sum[k] += 0.5 * (metrics_ab[k] + metrics_ba[k])
                    sims_ab = similarity_metrics(logits_ab, temperature)
                    sims_ba = similarity_metrics(logits_ba, temperature)
                    val_extra_sum["mean_pairwise_cosine"] += mean_pairwise_cosine(z_pred.float()).item()
                    val_extra_sum["pos_sim_mean"] += 0.5 * (sims_ab["pos_sim_mean"] + sims_ba["pos_sim_mean"])
                    val_extra_sum["hard_sim_mean"] += 0.5 * (sims_ab["hard_sim_mean"] + sims_ba["hard_sim_mean"])

            avg_val_loss = val_loss_sum / len(val_loader)
            avg_val_extra = {k: v / len(val_loader) for k, v in val_extra_sum.items()}
            avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics_sum.items()}

            # ---- FiLM gamma/beta magnitude diagnostic (last val batch's actions; call `mlp`
            # directly rather than `ddp_model` since DataParallel replica attributes don't
            # propagate back to the source module -- `mlp` always holds the trained params) ----
            film_stats = compute_film_gamma_beta_stats(mlp, actions)

            # ---- checkpoint (saved before diagnostics) ----
            ckpt = {
                "epoch": epoch,
                "step": step,
                "clip_state_dict": clip_model.state_dict(),
                "mlp_state_dict": mlp.state_dict(),
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
                "train/global_table_coverage": table_coverage,
                "val/epoch_loss": avg_val_loss,
                "val/epoch_top1": avg_val_metrics["top1"],
                "val/epoch_top5": avg_val_metrics["top5"],
                "val/epoch_mean_rank": avg_val_metrics["mean_rank"],
                "val/mean_pairwise_cosine": avg_val_extra["mean_pairwise_cosine"],
                "val/pos_sim_mean": avg_val_extra["pos_sim_mean"],
                "val/hard_sim_mean": avg_val_extra["hard_sim_mean"],
                **film_stats,
            }

            # ---- image-side margin probe + progress curve (subsampled, every epoch) ----
            margin_agg = None
            progress_corr = None
            plotted = {}
            umap_path = None
            umap_task_path = None
            checkpoint_tag = f"epoch_{epoch:03d}"  # hoisted out of the try -- pure function of
                                                    # epoch, needed by the local-log swap below
                                                    # even if diagnostics raise before reaching it
            try:
                encoded = encode_probe_episodes(clip_model, probe_records, preprocess, device)
                margin_agg, per_episode_centroids = compute_margin_probe(encoded)
                progress_corr = compute_progress_curve_correlation(encoded, per_episode_centroids)
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
                    "margin_probe_subsample75/n_qualifying": margin_agg["n_qualifying"],
                    "margin_probe_subsample75/n_total": margin_agg["n_total"],
                    "progress_curve/pearson_corr": progress_corr,
                })

                # ---- text-side margin probe (subsample every epoch) ----
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        unique_z_probe = F.normalize(clip_model.encode_text(global_tokens), dim=-1)
                    unique_z_probe = unique_z_probe.float().cpu().numpy()
                text_margin_sub = compute_text_margin(unique_z_probe, touching_idx_sub, not_touching_idx_sub)
                log_dict.update({
                    "text/margin_probe_subsample/mean": text_margin_sub["mean"],
                    "text/margin_probe_subsample/n_touching": text_margin_sub["n_touching"],
                    "text/margin_probe_subsample/n_not_touching": text_margin_sub["n_not_touching"],
                })

                # ---- fixed-basis UMAP plot, every epoch (decoupled from full_probe_every --
                # unique_z_probe above is already computed every epoch regardless, so this
                # needs no extra text-encoder forward pass) ----
                umap_path = plot_text_embedding_umap(umap_reducer, unique_z_probe, table, umap_out_dir, checkpoint_tag)
                log_dict[f"text_embedding_umap/{checkpoint_tag}"] = wandb.Image(umap_path)

                umap_task_path = plot_text_embedding_umap_task_identity(umap_reducer, unique_z_probe, table, umap_out_dir, checkpoint_tag)
                log_dict[f"text_embedding_umap_task_identity/{checkpoint_tag}"] = wandb.Image(umap_task_path)

                # ---- periodic full-val probes (image + text + held-out-pair) ----
                if epoch == 0 or (epoch + 1) % full_probe_every == 0 or epoch == num_epochs - 1:
                    full_records = build_probe_episode_records(data_dir, val_stems_all)
                    full_encoded = encode_probe_episodes(clip_model, full_records, preprocess, device)
                    full_margin_agg, _ = compute_margin_probe(full_encoded)
                    log_dict.update({
                        "margin_probe_full_val/mean": full_margin_agg["mean"],
                        "margin_probe_full_val/std": full_margin_agg["std"],
                        "margin_probe_full_val/n_qualifying": full_margin_agg["n_qualifying"],
                        "margin_probe_full_val/n_total": full_margin_agg["n_total"],
                    })
                    del full_records, full_encoded

                    text_margin_full = compute_text_margin(unique_z_probe, touching_idx_full, not_touching_idx_full)
                    text_margin_held = compute_text_margin(unique_z_probe, touching_idx_held, not_touching_idx_held)
                    log_dict.update({
                        "text/margin_probe_full_val/mean": text_margin_full["mean"],
                        "text/margin_probe_full_val/n_touching": text_margin_full["n_touching"],
                        "text/margin_probe_full_val/n_not_touching": text_margin_full["n_not_touching"],
                        "text/margin_probe_held_out_pair/mean": text_margin_held["mean"],
                        "text/margin_probe_held_out_pair/n_touching": text_margin_held["n_touching"],
                        "text/margin_probe_held_out_pair/n_not_touching": text_margin_held["n_not_touching"],
                    })
            except Exception:
                print(f"[epoch {epoch}] WARNING: margin probe / progress-curve diagnostics failed, "
                      f"skipping diagnostics for this epoch (checkpoint already saved):")
                traceback.print_exc()

            wandb.log(log_dict)

            # local mirror: swap wandb.Image objects for their plain file-path strings
            # (wandb.Image isn't JSON-serializable, and the path is the meaningful record)
            local_log_dict = dict(log_dict)
            for stem, png_path in plotted.items():
                local_log_dict[f"progress_curve_plots/{stem}"] = png_path
            if umap_path is not None and f"text_embedding_umap/{checkpoint_tag}" in local_log_dict:
                local_log_dict[f"text_embedding_umap/{checkpoint_tag}"] = umap_path
            if umap_task_path is not None and f"text_embedding_umap_task_identity/{checkpoint_tag}" in local_log_dict:
                local_log_dict[f"text_embedding_umap_task_identity/{checkpoint_tag}"] = umap_task_path
            _append_jsonl(epoch_log_path, local_log_dict)

            margin_str = f"{margin_agg['mean']:.4f}" if margin_agg is not None else "N/A"
            progress_str = f"{progress_corr:.4f}" if progress_corr is not None else "N/A"
            print(f"[epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} "
                  f"margin_subsample75={margin_str} progress_corr={progress_str} "
                  f"table_coverage={table_coverage:.3f}")
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
        # explicit final sync -- backstop in case the "live" policy's background
        # watcher hasn't caught the very last append before the run ends
        wandb.save(step_log_path, policy="now")
        wandb.save(epoch_log_path, policy="now")

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run full 16-epoch training with wandb")
    parser.add_argument("--smoke_test", action="store_true", help="run 100 optimizer steps, report timing/OOM/LR sanity")
    parser.add_argument("--micro_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume_from", type=str, default=None,
                         help="path to a checkpoint (e.g. checkpoints_clip_image_text_finetuned/latest.pt) "
                              "to resume training from")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_clip_image_text_finetuned",
                         help="directory to write checkpoints/logs/diagnostics to")
    parser.add_argument("--slack_smoke_test", action="store_true",
                         help="send a single test wandb.alert() and exit, without running training")
    parser.add_argument("--legacy_even_split", action="store_true",
                         help="disable the rebalanced GPU split for the train loop (see gpu_split_joint_step.py) "
                              "and use today's plain even-split DataParallel + sequential encode_text instead -- "
                              "for A/B wall-clock comparison or as a safety net")
    args = parser.parse_args()

    if args.slack_smoke_test:
        import wandb
        wandb.init(project="langtable-dynamics", name="slack_alert_smoke_test", job_type="smoke_test")
        wandb.alert(
            title="Slack smoke test",
            text="Test alert from dynamics_model_clip_full_finetune.py --slack_smoke_test. "
                 "If you see this in Slack, wandb.alert() is wired up correctly.",
            level=wandb.AlertLevel.INFO,
        )
        wandb.finish()
    elif args.smoke_test:
        train(smoke_test=True, micro_batch_size=args.micro_batch_size, num_workers=args.num_workers,
              resume_from=args.resume_from, checkpoint_dir=args.checkpoint_dir,
              legacy_even_split=args.legacy_even_split)
    elif args.train:
        train(micro_batch_size=args.micro_batch_size, num_workers=args.num_workers,
              resume_from=args.resume_from, checkpoint_dir=args.checkpoint_dir,
              legacy_even_split=args.legacy_even_split)
    else:
        parser.print_help()
