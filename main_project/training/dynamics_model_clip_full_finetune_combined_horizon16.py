"""
Full fine-tuning of BOTH the CLIP ViT-L/14 image encoder AND text encoder, jointly with
a fresh FiLM-MLP latent dynamics model -- COMBINED expert + suboptimal data variant,
HORIZON-16 variant.

Identical to dynamics_model_clip_full_finetune_combined.py (see that file for the full
changelog vs. dynamics_model_clip_full_finetune.py) except the action chunk horizon is
16 steps instead of 8: HORIZON = 16 below, and the dynamics model is constructed as
LatentDynamicsModel(horizon=16) (dynamics_model.py's LatentDynamicsModel now takes an
optional horizon param, default 8, so every other caller is unaffected). Checkpoint/
diagnostics/wandb run names are also distinct so this run's artifacts never collide
with the horizon-8 run's.

Same as dynamics_model_clip_full_finetune.py, except:
  - Training data now combines the 5000-episode expert set (langtable/demos/) with a new
    1675-episode "suboptimal" set (langtable_switch/demos/, mostly failure trajectories).
    Both are split independently (plain random, no held-out-pair adjustment) via the same,
    unmodified train_val_split(): a random 750/5000 expert split, and a random 15%/1675
    suboptimal split. The two splits are logged separately (train_val_split.json) so the
    expert/suboptimal breakdown of train vs. val stays auditable.
  - No held-out-pair mechanism at all (apply_held_out_pair, held_out_pair_stems, and the
    corresponding "text/margin_probe_held_out_pair/*" wandb block are all removed --
    confirmed via exploration to be the only diagnostic tied to that mechanism).
  - Positive/negative statement generation for suboptimal episodes uses a DIFFERENT rule
    than expert episodes (expert-episode logic is completely unchanged): since suboptimal
    (failure) episodes don't have one coherent task-relevant pair to trust, both the
    positive pair AND the peg-clause target are recomputed fresh every timestep --
    A = whichever block is currently closest to the peg, B = whichever other block is
    currently closest to A. Positive statement = the TRUE state of "A is (not) touching B.
    The peg is (not) touching A." (both A->B and B->A orderings trained, same as expert).
    Negatives are identical in form to the expert path: all 56 ordered pairs x 8 peg-blocks
    (448 total), BOTH clauses inverted from the real state -- guaranteed false by
    construction, via one shared helper used by both data sources so this guarantee holds
    identically for both rather than by two independently-written copies.
  - Both data sources produce statements from the exact same shared 1792-entry
    GlobalStatementTable / string grammar -- nothing about the new data source changes or
    shrinks that combinatorial space, so the table itself is unchanged and still necessary
    (it's what makes "re-encode once per step, not per string" possible for both sources).
  - Diagnostics (margin probe, progress-curve plots, per-episode UMAP/t-SNE coloring) run
    on EXPERT VAL EPISODES ONLY -- those probes assume one fixed task-relevant pair per
    episode, which doesn't cleanly apply to suboptimal episodes' dynamic per-timestep pair.
    Suboptimal val episodes still fully participate in the actual validation loss/InfoNCE
    computation (train_loader/val_loader), just not in these descriptive probes.
  - No _zsem.npz dependency anywhere in this script. The one place the original script read
    a cached zsem file (build_probe_episode_records's "touching" label) is replaced with a
    direct recompute from block_states/actual_peg_positions (same TOUCH_THRESH formula used
    everywhere else in this file) -- see build_probe_episode_records_no_zsem.
  - Text-embedding-space visualization switched from UMAP (fit once on the pretrained-CLIP
    baseline, reused via .transform() every epoch) to t-SNE, refit fresh every epoch (t-SNE
    has no out-of-sample .transform() anyway, so this is also the only way to use it). Both
    same-epoch plots (touching-state coloring, task-identity coloring) share one fit rather
    than redundantly re-running t-SNE twice on identical input.

Usage:
    python dynamics_model_clip_full_finetune_combined_horizon16.py --smoke_test
    python dynamics_model_clip_full_finetune_combined_horizon16.py --train
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
# `if __name__ == "__main__":`), so importing it is safe. Note: build_probe_episode_records
# is NOT imported from here -- this script has its own zsem-free version, see below.
from dynamics_model_clip_lora_finetune import (
    load_episode_frames_mmap,
    encode_probe_episodes,
    compute_margin_probe,
    compute_progress_curve_correlation,
    plot_progress_curves,
)

TOUCH_THRESH = 0.05  # matches cache_zhard.py exactly
HORIZON = 16         # 16-step action chunk variant (base script uses LangTableDataset.HORIZON = 8)


# ---------------------------------------------------------------------------
# Global statement table (see plan §3) — every compound statement (positive or
# negative) is built from only the 8 fixed block identities, so the entire universe
# of possible statements is combinatorially bounded: 56 ordered pairs x 2 touching_ab
# states x 8 peg-blocks x 2 touching_peg states = 1792, independent of episode count.
# This table is tokenized once and re-encoded fresh every training step; every sample's
# positive/negative is a pure integer index into it. Shared identically by expert and
# suboptimal episodes -- both sources only ever pick different indices into this same
# fixed grammar, never a different statement form.
# ---------------------------------------------------------------------------

def get_block_names(data_dir: str) -> list[str]:
    """Derive the 8 fixed block names from real data, same convention as
    cache_zhard.py's `sorted(k for k in block_states[0].keys() if k != 'peg')` —
    block_states dicts also carry a 'peg' key (confirmed against real data), so it
    must be filtered out explicitly rather than assumed absent. Vocabulary confirmed
    identical between the expert and suboptimal data directories, so this only ever
    needs to be called on one of them."""
    pkl_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    with open(os.path.join(data_dir, pkl_files[0]), "rb") as f:
        ep = pickle.load(f)
    names = sorted(k for k in ep["block_states"][0].keys() if k != "peg")
    assert len(names) == 8, f"expected 8 block names, got {len(names)}: {names}"
    return names


class GlobalStatementTable:
    """Canonical, deterministic enumeration of all 1792 possible compound statements.
    No episode data needed beyond the fixed 8 block names. Training positive/negative
    generation (for BOTH expert and suboptimal episodes) and the text margin probe's
    grounding scan all share this same index arithmetic."""

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


def compute_hard_negatives_for_timestep(
    table: GlobalStatementTable, touch_matrix: np.ndarray, peg_touch: np.ndarray,
) -> np.ndarray:
    """All 56 ordered block pairs x 8 peg-blocks (448 total), BOTH clauses' wording
    inverted from the real state -- guaranteed false by construction, identical
    convention to cache_zhard.py's build_hard_neg_data. Shared by both the expert-
    episode and suboptimal-episode positive/negative computation below, so "both
    clauses false" holds identically for both data sources rather than by two
    independently-written copies that could drift apart."""
    n_blocks = table.n_blocks
    pair_block_idx = table.pair_block_idx

    ab_inv_per_pair = np.array([not touch_matrix[i, j] for (i, j) in pair_block_idx], dtype=np.int64)  # (56,)
    peg_inv_per_block = (~peg_touch).astype(np.int64)  # (8,)

    pair_idx_arr = np.arange(table.n_pairs)
    idx_grid = (
        (pair_idx_arr[:, None] * 2 + ab_inv_per_pair[:, None]) * n_blocks + np.arange(n_blocks)[None, :]
    ) * 2 + peg_inv_per_block[None, :]  # (56, 8)
    return idx_grid.reshape(-1)  # (448,)


def compute_pos_neg_indices_for_episode(
    table: GlobalStatementTable,
    block_states: list[dict],
    peg_positions: np.ndarray,
    start_block: str,
    target_block: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """EXPERT-episode path, behavior unchanged from dynamics_model_clip_full_finetune.py.
    Vectorized per-timestep computation of (pos_idx_ab, pos_idx_ba, neg_idx), mirroring
    cache_zhard.py's build_hard_neg_data logic exactly for negatives (via the shared
    compute_hard_negatives_for_timestep helper). Positive indices use the TRUE
    (non-inverted) state for the episode's fixed task-relevant pair, in BOTH orderings
    (touching is symmetric)."""
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

        # --- negatives: shared helper, all 56 pairs x 8 peg-blocks, both clauses inverted ---
        neg_idx[t] = compute_hard_negatives_for_timestep(table, touch_matrix, peg_touch)

    return pos_idx_ab, pos_idx_ba, neg_idx


def compute_pos_neg_indices_for_switch_episode(
    table: GlobalStatementTable,
    block_states: list[dict],
    peg_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SUBOPTIMAL-episode path (new). No fixed metadata task-pair is trusted -- these are
    failure trajectories, so the "intended" start_block/oracle_target_block may never
    actually become relevant. Instead, every timestep: A = whichever block is currently
    closest to the peg, B = whichever OTHER block is currently closest to A -- both
    recomputed fresh per timestep, unlike the expert path's fixed (start_block,
    target_block). Positive statement: the TRUE (non-inverted) state of "A is (not)
    touching B. The peg is (not) touching A." (peg clause is about A itself, the
    closest-to-peg block). Both orderings (A->B, B->A) trained as positives, same as the
    expert path. Negatives: identical 448-combo, both-clauses-inverted construction via
    the shared helper -- same "guaranteed false" guarantee as expert episodes."""
    T = len(block_states)
    block_names = table.block_names
    n_blocks = table.n_blocks

    n_negatives = table.n_pairs * n_blocks  # 448
    pos_idx_ab = np.zeros(T, dtype=np.int64)
    pos_idx_ba = np.zeros(T, dtype=np.int64)
    neg_idx = np.zeros((T, n_negatives), dtype=np.int64)

    for t in range(T):
        bpos = block_states[t]
        peg = peg_positions[t]

        block_pos_arr = np.stack([bpos[name] for name in block_names])  # (8, 2)
        diff = block_pos_arr[:, None, :] - block_pos_arr[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1)                     # (8, 8)
        touch_matrix = dist_matrix < TOUCH_THRESH

        peg_dist = np.linalg.norm(block_pos_arr - peg[None, :], axis=-1)  # (8,)
        peg_touch = peg_dist < TOUCH_THRESH

        # A = block closest to the peg
        a_i = int(np.argmin(peg_dist))
        # B = block closest to A, excluding A itself
        dist_from_a = dist_matrix[a_i].copy()
        dist_from_a[a_i] = np.inf
        b_i = int(np.argmin(dist_from_a))

        a_name, b_name = block_names[a_i], block_names[b_i]
        ab_touch_true = bool(touch_matrix[a_i, b_i])
        peg_touch_a = bool(peg_touch[a_i])

        pair_ab = table.pair_to_idx[(a_name, b_name)]
        pair_ba = table.pair_to_idx[(b_name, a_name)]
        pos_idx_ab[t] = table.index(pair_ab, int(ab_touch_true), a_i, int(peg_touch_a))
        pos_idx_ba[t] = table.index(pair_ba, int(ab_touch_true), a_i, int(peg_touch_a))

        # --- negatives: same shared helper as the expert path, both clauses inverted ---
        neg_idx[t] = compute_hard_negatives_for_timestep(table, touch_matrix, peg_touch)

    return pos_idx_ab, pos_idx_ba, neg_idx


# ---------------------------------------------------------------------------
# Dataset: raw frames (mmap, unchanged approach) + integer indices into the global
# statement table (no runtime string handling at all). Combines expert + suboptimal
# episodes -- each file_list entry now carries its own data_dir and an is_switch flag
# so the constructor can route to the right positive/negative computation per episode.
# ---------------------------------------------------------------------------

class LangTableImageTextDataset(LangTableDataset):
    """LangTableDataset variant for joint image+text CLIP fine-tuning across combined
    expert + suboptimal data. Stores paths to pre-cached {stem}_frames.npy files plus
    precomputed integer indices into the shared GlobalStatementTable (no cached
    _zsem.npz/_zhard.npz needed for training data -- the text tower is trainable now,
    so cached frozen-CLIP embeddings would be stale)."""

    def __init__(
        self,
        table: GlobalStatementTable,
        preprocess,
        file_list: list[tuple[str, str, bool]],
    ):
        """file_list: list of (data_dir, pkl_filename, is_switch) tuples -- is_switch
        selects compute_pos_neg_indices_for_switch_episode (no metadata needed) vs.
        compute_pos_neg_indices_for_episode (expert path, needs metadata's
        start_block/oracle_target_block)."""
        self.preprocess = preprocess
        self.table = table

        self.frames_paths: list[str] = []
        self.actions: list[np.ndarray] = []
        self.pos_idx_ab: list[np.ndarray] = []
        self.pos_idx_ba: list[np.ndarray] = []
        self.neg_idx: list[np.ndarray] = []
        self.indices: list[tuple[int, int]] = []

        for data_dir, pkl_file, is_switch in file_list:
            pkl_path = os.path.join(data_dir, pkl_file)
            frames_path = pkl_path.replace(".pkl", "_frames.npy")
            if not os.path.exists(frames_path):
                warnings.warn(f"Missing frames_npy file for {pkl_file} in {data_dir} "
                               f"(run cache_frames_npy.py / cache_frames_npy_switch.py first), "
                               f"skipping episode.")
                continue

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)
            actions = ep["actions"]
            block_states = ep["block_states"]
            peg_positions = ep["actual_peg_positions"]

            if is_switch:
                pos_ab, pos_ba, neg = compute_pos_neg_indices_for_switch_episode(
                    self.table, block_states, peg_positions
                )
            else:
                start_block = ep["metadata"]["start_block"]
                target_block = ep["metadata"]["oracle_target_block"]
                pos_ab, pos_ba, neg = compute_pos_neg_indices_for_episode(
                    self.table, block_states, peg_positions, start_block, target_block
                )
            del ep  # let the bulky frames array fall out of scope

            T = len(block_states)
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
# just without inverting truth. Only ever called against the EXPERT data directory
# and expert stems (see train()) -- diagnostics are scoped to expert val episodes only.
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


# ---------------------------------------------------------------------------
# Text-embedding-space visualization: t-SNE, refit fresh every epoch (per user request --
# no persisted reducer, no out-of-sample transform; t-SNE doesn't support the latter
# anyway). Both same-epoch plots share one fit computed once at the call site in train().
# ---------------------------------------------------------------------------

def fit_tsne(embeddings: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Fresh t-SNE fit every call -- no persisted reducer, no out-of-sample transform.
    Called once per epoch; the resulting (N,2) coords are shared by both
    plot_text_embedding_tsne* calls for that epoch so t-SNE isn't redundantly re-run
    twice on identical input within the same epoch."""
    from sklearn.manifold import TSNE
    return TSNE(n_components=2, random_state=random_state, init="pca").fit_transform(embeddings)


def plot_text_embedding_tsne(
    coords: np.ndarray, table: GlobalStatementTable, out_dir: str, checkpoint_tag: str,
) -> str:
    """coords: (1792, 2), already fit fresh this epoch via fit_tsne() -- see train()
    call site. Scatter-plots colored by table.ab_state (2 colors, legend), saves as
    PNG, returns the path."""
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    touching = table.ab_state.astype(bool)
    ax.scatter(coords[touching, 0], coords[touching, 1], s=6, alpha=0.6, color="tab:blue", label="touching")
    ax.scatter(coords[~touching, 0], coords[~touching, 1], s=6, alpha=0.6, color="tab:red", label="not touching")
    ax.legend()
    ax.set_title(f"Text embedding t-SNE (refit every epoch) — {checkpoint_tag}")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"{checkpoint_tag}_text_embedding_tsne.png")
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def plot_text_embedding_tsne_task_identity(
    coords: np.ndarray, table: GlobalStatementTable, out_dir: str, checkpoint_tag: str,
) -> str:
    """Same coords as plot_text_embedding_tsne (one shared fit per epoch), recolors the
    identical point cloud by which of the 56 start_block->target_block pairs each entry
    belongs to, instead of by touching state."""
    os.makedirs(out_dir, exist_ok=True)

    cmap = plt.cm.gist_rainbow
    n_pairs = table.n_pairs
    fig, ax = plt.subplots(figsize=(10, 6))
    for pair_idx, (a, b) in enumerate(table.pairs):
        m = table.pair_idx_per_entry == pair_idx
        color = cmap(pair_idx / max(n_pairs - 1, 1))
        ax.scatter(coords[m, 0], coords[m, 1], s=20, alpha=0.7, linewidths=0,
                   color=color, label=f"{a} → {b}")
    ax.set_title(f"Text embedding t-SNE (refit every epoch) — task identity — {checkpoint_tag}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=5, framealpha=0.8, ncol=2,
              title="start_block → target_block")
    png_path = os.path.join(out_dir, f"{checkpoint_tag}_text_embedding_tsne_task_identity.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# Zsem-free probe episode records -- same as dynamics_model_clip_lora_finetune.py's
# build_probe_episode_records, except the "touching" label is recomputed directly from
# block_states/actual_peg_positions instead of read from a cached {stem}_zsem.npz. Only
# ever called against the EXPERT data directory (diagnostics are scoped to expert val
# episodes only -- see train()).
# ---------------------------------------------------------------------------

def build_probe_episode_records_no_zsem(data_dir: str, stems: list[str]) -> dict[str, dict]:
    """For each episode stem: touching label + ground-truth task-relevant block distance
    dist_ab(t), both recomputed directly from block_states (same TOUCH_THRESH formula
    used everywhere else in this file -- no _zsem.npz read at all), plus the path to the
    pre-cached {stem}_frames.npy for lazy mmap access -- never retains the unpickled
    ep["frames"] list itself (that's the OOM this pattern was fixed from originally)."""
    records = {}
    for stem in stems:
        pkl_path = os.path.join(data_dir, stem + ".pkl")
        frames_path = pkl_path.replace(".pkl", "_frames.npy")
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        block_states = ep["block_states"]
        a = ep["metadata"]["start_block"]
        b = ep["metadata"]["oracle_target_block"]
        T = len(block_states)
        dist_ab = np.array(
            [np.linalg.norm(block_states[t][a] - block_states[t][b]) for t in range(T)],
            dtype=np.float32,
        )
        touching = dist_ab < TOUCH_THRESH

        records[stem] = {
            "frames_path": frames_path,
            "touching": touching.astype(bool),
            "dist_ab": dist_ab,
        }
    return records


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    expert_data_dir: str = "/home/ashwink/full_data_5000/langtable/demos/",
    switch_data_dir: str = "/home/ashwink/full_data_5000/langtable_switch/demos/",
    switch_val_fraction: float = 0.15,
    device: str = "cuda",
    micro_batch_size: int = 256,
    effective_batch_size: int = 256,
    num_epochs: int = 25,
    lr_image: float = 1e-6,
    lr_text: float = 1e-6,
    lr_mlp: float = 3e-4,
    temperature: float = 0.1,
    wandb_project: str = "langtable-dynamics",
    checkpoint_dir: str = "checkpoints_clip_full_fine-tuned_with_subopt_data_h16",
    resume_from: str | None = None,
    n_val_episodes: int = 750,
    val_seed: int = 0,
    num_workers: int = 8,
    probe_n_episodes: int = 75,
    probe_seed: int = 123,
    full_probe_every: int = 5,
    diagnostic_plot_episodes: int = 5,
    diagnostics_dir: str = "clip_full_finetune_combined_h16_diagnostics",
    smoke_test: bool = False,
    smoke_test_steps: int = 100,
    legacy_even_split: bool = False,
):
    grad_accum_steps = math.ceil(effective_batch_size / micro_batch_size)
    grad_accum_triggered = grad_accum_steps > 1

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- data split: expert (plain random n_val_episodes) + suboptimal (random
    # switch_val_fraction), each split independently via the same, unmodified
    # train_val_split() -- no held-out-pair adjustment for either source. ----
    expert_train, expert_val = train_val_split(expert_data_dir, n_val=n_val_episodes, seed=val_seed)
    switch_pkl_files = sorted(f for f in os.listdir(switch_data_dir) if f.endswith(".pkl"))
    switch_n_val = round(switch_val_fraction * len(switch_pkl_files))
    switch_train, switch_val = train_val_split(switch_data_dir, n_val=switch_n_val, seed=val_seed)

    train_files = (
        [(expert_data_dir, f, False) for f in expert_train]
        + [(switch_data_dir, f, True) for f in switch_train]
    )
    val_files = (
        [(expert_data_dir, f, False) for f in expert_val]
        + [(switch_data_dir, f, True) for f in switch_val]
    )

    with open(os.path.join(checkpoint_dir, "train_val_split.json"), "w") as f:
        json.dump({
            "val_seed": val_seed,
            "switch_val_fraction": switch_val_fraction,
            "expert_n_train": len(expert_train), "expert_n_val": len(expert_val),
            "switch_n_train": len(switch_train), "switch_n_val": len(switch_val),
            "expert_train_files": expert_train, "expert_val_files": expert_val,
            "switch_train_files": switch_train, "switch_val_files": switch_val,
        }, f, indent=2)

    # ---- global statement table (plan §3) ----
    block_names = get_block_names(expert_data_dir)
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

    mlp = LatentDynamicsModel(horizon=HORIZON).to(device)  # fresh random init
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

    train_dataset = LangTableImageTextDataset(table, preprocess, file_list=train_files)
    val_dataset = LangTableImageTextDataset(table, preprocess, file_list=val_files)

    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True,
                               num_workers=num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False, pin_memory=True)
    print(f"Train dataset: {len(train_dataset)} samples, {len(train_loader)} micro-batches/epoch "
          f"({len(expert_train)} expert + {len(switch_train)} suboptimal episodes)")
    print(f"Val dataset:   {len(val_dataset)} samples, {len(val_loader)} micro-batches/epoch "
          f"({len(expert_val)} expert + {len(switch_val)} suboptimal episodes)")

    n_steps_per_epoch = len(train_loader) // grad_accum_steps
    total_steps = n_steps_per_epoch * num_epochs

    # ---- LR warmup: 5% of total_steps, floor 50 -- shared by smoke_test and full training ----
    warmup_steps = max(50, int(0.05 * total_steps))
    print(f"LR warmup: {warmup_steps} steps (5% of {total_steps} total steps, floor 50)")

    # ---- fixed probe subsample: EXPERT VAL EPISODES ONLY (seeded, from the expert val
    # split). Diagnostics assume one fixed task-relevant pair per episode, which doesn't
    # cleanly apply to suboptimal episodes' dynamic per-timestep pair -- suboptimal val
    # episodes still fully participate in the actual val_loader loss/InfoNCE computation
    # above, just not in these descriptive probes. ----
    rng = random.Random(probe_seed)
    expert_val_stems = [f[:-4] for f in expert_val]
    probe_stems = rng.sample(expert_val_stems, min(probe_n_episodes, len(expert_val_stems)))
    plot_stems = rng.sample(probe_stems, min(diagnostic_plot_episodes, len(probe_stems)))
    probe_records = build_probe_episode_records_no_zsem(expert_data_dir, probe_stems)
    print(f"Probe subsample: {len(probe_records)} episodes (seed={probe_seed}), expert-only; "
          f"{len(plot_stems)} fixed for progress-curve plots")

    # ---- text-side margin probe grounding (deterministic, built once, expert-only) ----
    touching_idx_sub, not_touching_idx_sub = ground_global_table(table, expert_data_dir, probe_stems)
    touching_idx_full, not_touching_idx_full = ground_global_table(table, expert_data_dir, expert_val_stems)

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

    wandb.init(
        project=wandb_project,
        name="clip_full_finetune_combined_h16",
        config=dict(
            finetune_scope="full_image_text_combined_expert_switch",
            image_lr=lr_image, text_lr=lr_text, mlp_lr=lr_mlp,
            weight_decay_policy="exclude bias/LayerNorm/embedding from decay on both towers",
            changed_weight=1.0, negative_count=448, temperature=temperature,
            n_train_episodes=len(train_files), n_val_episodes=len(val_files), val_seed=val_seed,
            n_expert_train=len(expert_train), n_expert_val=len(expert_val),
            n_switch_train=len(switch_train), n_switch_val=len(switch_val),
            switch_val_fraction=switch_val_fraction,
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

    # ---- t-SNE baseline frame (pretrained CLIP, before any fine-tuning gradient step
    # touches the text tower) -- fit fresh, logged as a "pretrain" frame before the
    # epoch loop starts. No reducer object persists past this call. ----
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pretrained_unique_z = F.normalize(clip_model.encode_text(global_tokens), dim=-1)
        pretrained_unique_z = pretrained_unique_z.float().cpu().numpy()

    tsne_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), diagnostics_dir)
    pretrain_coords = fit_tsne(pretrained_unique_z)
    pretrain_tsne_path = plot_text_embedding_tsne(pretrain_coords, table, tsne_out_dir, "epoch_pretrain")
    wandb.log({"text_embedding_tsne/pretrain": wandb.Image(pretrain_tsne_path)})
    _append_jsonl(epoch_log_path, {"text_embedding_tsne/pretrain": pretrain_tsne_path})

    pretrain_tsne_task_path = plot_text_embedding_tsne_task_identity(pretrain_coords, table, tsne_out_dir, "epoch_pretrain")
    wandb.log({"text_embedding_tsne_task_identity/pretrain": wandb.Image(pretrain_tsne_task_path)})
    _append_jsonl(epoch_log_path, {"text_embedding_tsne_task_identity/pretrain": pretrain_tsne_task_path})

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

            # ---- image-side margin probe + progress curve (subsampled, every epoch,
            # expert-only) ----
            margin_agg = None
            progress_corr = None
            plotted = {}
            tsne_path = None
            tsne_task_path = None
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

                # ---- t-SNE plot, fresh fit every epoch (decoupled from full_probe_every --
                # unique_z_probe above is already computed every epoch regardless, so this
                # needs no extra text-encoder forward pass) ----
                epoch_coords = fit_tsne(unique_z_probe)
                tsne_path = plot_text_embedding_tsne(epoch_coords, table, tsne_out_dir, checkpoint_tag)
                log_dict[f"text_embedding_tsne/{checkpoint_tag}"] = wandb.Image(tsne_path)

                tsne_task_path = plot_text_embedding_tsne_task_identity(epoch_coords, table, tsne_out_dir, checkpoint_tag)
                log_dict[f"text_embedding_tsne_task_identity/{checkpoint_tag}"] = wandb.Image(tsne_task_path)

                # ---- periodic full-val probes (image + text, expert-only) ----
                if epoch == 0 or (epoch + 1) % full_probe_every == 0 or epoch == num_epochs - 1:
                    full_records = build_probe_episode_records_no_zsem(expert_data_dir, expert_val_stems)
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
                    log_dict.update({
                        "text/margin_probe_full_val/mean": text_margin_full["mean"],
                        "text/margin_probe_full_val/n_touching": text_margin_full["n_touching"],
                        "text/margin_probe_full_val/n_not_touching": text_margin_full["n_not_touching"],
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
            if tsne_path is not None and f"text_embedding_tsne/{checkpoint_tag}" in local_log_dict:
                local_log_dict[f"text_embedding_tsne/{checkpoint_tag}"] = tsne_path
            if tsne_task_path is not None and f"text_embedding_tsne_task_identity/{checkpoint_tag}" in local_log_dict:
                local_log_dict[f"text_embedding_tsne_task_identity/{checkpoint_tag}"] = tsne_task_path
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
    parser.add_argument("--effective_batch_size", type=int, default=None,
                         help="defaults to --micro_batch_size if not set (grad_accum_steps=1). Set explicitly "
                              "if you want gradient accumulation (effective_batch_size > micro_batch_size).")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--resume_from", type=str, default=None,
                         help="path to a checkpoint (e.g. checkpoints_clip_full_fine-tuned_with_subopt_data_h16/latest.pt) "
                              "to resume training from")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_clip_full_fine-tuned_with_subopt_data_h16",
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
            text="Test alert from dynamics_model_clip_full_finetune_combined.py --slack_smoke_test. "
                 "If you see this in Slack, wandb.alert() is wired up correctly.",
            level=wandb.AlertLevel.INFO,
        )
        wandb.finish()
    elif args.smoke_test:
        train(smoke_test=True, micro_batch_size=args.micro_batch_size,
              effective_batch_size=args.effective_batch_size or args.micro_batch_size,
              num_workers=args.num_workers,
              resume_from=args.resume_from, checkpoint_dir=args.checkpoint_dir,
              legacy_even_split=args.legacy_even_split)
    elif args.train:
        train(micro_batch_size=args.micro_batch_size,
              effective_batch_size=args.effective_batch_size or args.micro_batch_size,
              num_workers=args.num_workers,
              resume_from=args.resume_from, checkpoint_dir=args.checkpoint_dir,
              legacy_even_split=args.legacy_even_split)
    else:
        parser.print_help()
