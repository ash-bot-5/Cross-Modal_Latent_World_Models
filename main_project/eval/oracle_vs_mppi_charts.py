"""
Per-timestep chart: oracle-policy proposals vs. fresh MPPI-probe proposals vs. what the peg
actually did, all from the EXACT real pybullet state at that timestep of a closed-loop rollout.

Motivation: mppi_global_or_local_min.py answers "does the planner converge to one basin or
scatter, from ONE fixed hand-built probe state?" This script asks the same question but walked
across every real timestep of an ACTUAL rollout, so it can also be compared against what a
scripted oracle policy would have done and what really happened -- not just against a synthetic
near-perfect reference chunk.

Inputs (both produced under a rollouts_5_per/ epoch directory):
  {stem}_pybullet_states.pkl  -- written by mppi_rollout_5_per_checkpoint.py: a list of
                                 inner.get_pybullet_state() dicts, one per real env step,
                                 index-aligned with that rollout's saved mp4 frames.
  {stem}_oracle_chunks.pkl    -- written in a SEPARATE repo (the expert-oracle-policy repo),
                                 using the pybullet_states.pkl above (its meta.states_pkl field
                                 points straight back at it -- confirmed by dist_at_t in this
                                 file matching this rollout's own logged distances exactly). For
                                 every real timestep t (stride=1) it records what N_ORACLE_SEEDS
                                 different oracle-policy seeds would do over the next `horizon`
                                 steps, if replayed from the EXACT real state at t. Actions are
                                 RAW physical units (8, 2) float32, not normalized to [-1, 1].
  where stem = f"epoch{rollout_epoch}_rollout{rollout_idx}" and rollout_epoch is parsed from
  --rollout_dir's basename (e.g. ".../epoch11" -> 11).

For every timestep t in [--t_start, --t_end] (inclusive, default 0..50) this restores the exact
pybullet state at t (inner.set_pybullet_state -- NOT the mp4 frame, which is lossy H.264; the
saved state gives a lossless re-render), and plots THREE kinds of 8-step action-chunk paths:

  1. Oracle proposals   -- the N_ORACLE_SEEDS chunks already in oracle_chunks.pkl, scored by
                           this checkpoint's dynamics model for direct comparison.
  2. MPPI-probe proposals -- --n_mppi_seeds FRESH single-command() MPPI planning passes from
                           this exact z_t (mirrors mppi_global_or_local_min.py: a brand-new
                           MPPI instance per seed, so prev_best_action is always None and every
                           seed draws from the same Uniform(-1,1) initial-population shape --
                           warm start never engages). Default hyperparameters (K=512,
                           mppi_temperature=0.1, evals=1000 i.e. n_planning_itrs=999) match
                           THIS rollout's own log header exactly, not the cheaper budget
                           mppi_temperature_sweep.py/mppi_global_or_local_min.py use elsewhere --
                           confirmed by grep "MPPI:" on this rollout's own .log.
  3. Actual outcome (ground truth) -- NOT a proposal: the peg's ACTUALLY MEASURED trajectory
                           from t to t+horizon, reconstructed by restoring pybullet_states[t]
                           through pybullet_states[t+horizon] in sequence and reading
                           inner._robot.forward_kinematics().translation[:2] at each (the
                           achieved pose -- NOT block_states_at_t['peg']/get_block_states()
                           ['peg'], which is the effector TARGET, not what was actually
                           achieved). Turned into an "action chunk" as consecutive position
                           deltas purely so it can be scored by the model on the same footing as
                           the other two. Because the real rollout only replans every
                           n_execute steps, for t not a multiple of n_execute this 8-step
                           window can straddle two different real planning decisions -- it is
                           simply "whatever really happened next", not tied to one plan.

Deliberately NOT included, after review: any reference/correct-direction arrow, and no
distance or other text annotation on the per-timestep charts -- each chart shows only the
three kinds of paths above and their cos_sim in the legend.

Dynamics-model checkpoint (both the fine-tuned CLIP tower AND the MLP, matching this project's
established convention of never reusing a pretrained-CLIP cache once a checkpoint has its own
fine-tuned text/image tower) is loaded from --ckpt_dir / --epoch, both explicitly CLI flags so
this can be pointed at any checkpoint directory, not just the one that produced this rollout.

Model classes (FiLMLayer/ActionEncoderConv/LatentDynamicsModel) are inline-copied verbatim from
mppi_global_or_local_min.py, per this project's eval/ convention of not importing
dynamics_model.py (whose module level triggers LangTableDataset/open_clip side effects).

Cost: --n_mppi_seeds (default 3) x (--t_end - --t_start + 1) (default 51) x ~6s per command()
call at evals=1000 -- roughly 15-16 minutes of GPU time for the defaults. Oracle scoring and the
ground-truth reconstruction are a handful of batch-1 forward passes per timestep (negligible);
restoring a pybullet state is a set of pose resets, not a physics step (negligible).

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/oracle_vs_mppi_charts.py \\
        --rollout_dir main_project/rollouts_5_per/expert+subopt_iters1000_exec8_with_pkls/epoch11 \\
        --rollout_idx 0
"""

import os
import re
import sys
import glob
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.mppi_core import MPPI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K = 512
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1
HORIZON = 8

DEFAULT_CKPT_DIR = os.path.join(
    MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data"
)

# Colors: MPPI-probe seeds get tab10 (blue/orange/green/...), oracle seeds get a fixed,
# visually distinct 3-color set reused from mppi_global_or_local_min.py's REF_ROT_COLORS
# convention (so it's never confused with a tab10 MPPI color), ground truth is a single
# unmistakable black.
MPPI_COLORS = [matplotlib.colormaps["tab10"](i) for i in range(10)]
ORACLE_COLORS = ["teal", "saddlebrown", "dimgray"]
GT_COLOR = "black"

# Reused identically at every timestep for a given seed index, so "MPPI-probe seed k" means the
# same initial-population draw pattern at every t -- isolates what changes to the real state
# z_t, not to a different RNG stream per timestep (same rationale as mppi_global_or_local_min.py
# reusing one fixed SAMPLE_SEEDS list across every epoch).
MPPI_SEED_BASE = 5000


# ---------------------------------------------------------------------------
# Inline model classes -- verbatim from mppi_global_or_local_min.py.
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
    """Conv1d action encoder: treats the 2 action dims as channels and the 8 timesteps as
    the sequence axis, so neighboring timesteps can interact via the kernel before the
    temporal axis is collapsed by flattening (unlike a flatten-first Linear, which discards
    step order entirely)."""

    def __init__(self, action_dim: int = 2, horizon: int = 8,
                 conv_channels: int = 64, kernel_size: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(action_dim, conv_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.act = nn.Mish()
        self.out = nn.Linear(conv_channels * horizon, out_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, horizon, action_dim) -> (B, action_dim, horizon) for Conv1d
        h = self.act(self.conv(actions.transpose(1, 2)))  # (B, conv_channels, horizon)
        return self.out(h.flatten(1))                      # (B, out_dim)


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
        # actions: (B, 8, 2)
        global_cond = self.action_encoder(actions)
        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_chunk(chunk_raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Raw action units -> [-1, 1], the same way LangTableDataset (and every mppi_* script) does."""
    return np.clip(2.0 * (chunk_raw - lo) / (hi - lo) - 1.0, -1.0, 1.0).astype(np.float32)


def cumulative_path(chunk_raw: np.ndarray) -> np.ndarray:
    """(8, 2) per-step displacements -> (9, 2) path starting at the origin."""
    return np.concatenate([[[0.0, 0.0]], np.cumsum(chunk_raw, axis=0)], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="epoch directory holding {stem}_pybullet_states.pkl and "
                             "{stem}_oracle_chunks.pkl, e.g. "
                             ".../rollouts_5_per/expert+subopt_iters1000_exec8_with_pkls/epoch11")
    parser.add_argument("--rollout_idx", type=int, default=0,
                        help="which rollout in --rollout_dir (builds epoch{N}_rollout{idx}_* names)")
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR,
                        help="checkpoint directory for the dynamics model + fine-tuned CLIP "
                             f"tower (default {DEFAULT_CKPT_DIR}, the dir that produced this "
                             "rollout -- override to score these same real states with a "
                             "different checkpoint's model)")
    parser.add_argument("--epoch", type=int, default=None,
                        help="which epoch_NNN.pt to load from --ckpt_dir. Default: parsed from "
                             "--rollout_dir's basename (e.g. 'epoch11' -> 11). This is "
                             "independent of --rollout_dir's own epoch number -- set it "
                             "explicitly to score this rollout's real trajectory against a "
                             "DIFFERENT checkpoint's dynamics model.")
    parser.add_argument("--t_start", type=int, default=0)
    parser.add_argument("--t_end", type=int, default=50)
    parser.add_argument("--n_mppi_seeds", type=int, default=3)
    parser.add_argument("--mppi_evals", type=int, default=1000,
                        help="total dynamics evaluations per MPPI command() call; "
                             "n_planning_itrs = evals - 1. Default 1000 matches this rollout's "
                             "own log header (K=512, mppi_temperature=0.1, evals=1000).")
    parser.add_argument("--mppi_temperature", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default=None,
                         help="default: MPPI_charts/oracle_vs_mppi/<parent>_<epoch>_rollout<idx>/")
    args = parser.parse_args()

    rollout_dir = os.path.abspath(args.rollout_dir.rstrip("/"))
    rollout_idx = args.rollout_idx

    epoch_dirname = os.path.basename(rollout_dir)
    m = re.search(r"epoch(\d+)$", epoch_dirname)
    if not m:
        raise ValueError(
            f"Could not parse an epoch number from --rollout_dir's basename {epoch_dirname!r} "
            f"-- expected something like '.../epoch11'."
        )
    rollout_epoch = int(m.group(1))
    ckpt_epoch = args.epoch if args.epoch is not None else rollout_epoch

    stem = f"epoch{rollout_epoch}_rollout{rollout_idx}"
    oracle_path = os.path.join(rollout_dir, f"{stem}_oracle_chunks.pkl")
    states_path = os.path.join(rollout_dir, f"{stem}_pybullet_states.pkl")
    for p in (oracle_path, states_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found")

    print(f"Loading oracle chunks from {oracle_path} ...")
    with open(oracle_path, "rb") as f:
        oracle_data = pickle.load(f)
    meta = oracle_data["meta"]
    probes = oracle_data["probes"]
    n_oracle_seeds = meta["num_seeds"]
    horizon = meta["horizon"]
    assert horizon == HORIZON, (
        f"oracle_chunks.pkl horizon={horizon} != this script's assumed HORIZON={HORIZON}"
    )
    block_a = meta["start_block"]
    block_b = meta["target_block"]
    print(f"  {n_oracle_seeds} oracle seeds, horizon={horizon}, "
          f"start_block={block_a}, target_block={block_b}")

    print(f"Loading pybullet states from {states_path} ...")
    with open(states_path, "rb") as f:
        pybullet_states = pickle.load(f)
    n_states = len(pybullet_states)
    assert n_states == meta["num_states"], (
        f"pybullet_states.pkl has {n_states} states but oracle_chunks.pkl's meta says "
        f"num_states={meta['num_states']} -- they may not be the matching pair for this rollout."
    )

    if args.t_start < 0 or args.t_start not in probes or args.t_end not in probes:
        raise ValueError(
            f"--t_start/--t_end ({args.t_start}/{args.t_end}) must both be keys in "
            f"oracle_chunks.pkl's probes dict (available: {min(probes)}..{max(probes)})"
        )
    if args.t_end + horizon >= n_states:
        raise ValueError(
            f"--t_end={args.t_end} + horizon={horizon} = {args.t_end + horizon} exceeds the "
            f"available pybullet_states ({n_states}) -- reduce --t_end so the ground-truth path "
            f"always has a full t..t+horizon window of real states to read."
        )

    # Goal text: read verbatim from this rollout's own .log rather than re-deriving it from
    # block_a/block_b, so this generalizes correctly even if a rollout's goal phrasing differs.
    log_candidates = sorted(glob.glob(os.path.join(rollout_dir, f"{stem}_*.log")))
    if not log_candidates:
        raise FileNotFoundError(f"no {stem}_*.log found in {rollout_dir} to read the goal from")
    goal_text = None
    with open(log_candidates[0]) as f:
        for line in f:
            gm = re.match(r"^Goal \(fixed.*?\):\s*(.+?)\s*$", line)
            if gm:
                goal_text = gm.group(1)
                break
    if goal_text is None:
        raise ValueError(f"no 'Goal (fixed...)' line found in {log_candidates[0]}")
    print(f"Goal text: {goal_text!r}")

    # -----------------------------------------------------------------------
    # Action normalization stats -- same source/convention as every other eval/ script.
    # -----------------------------------------------------------------------
    print(f"Loading action stats from {TRAIN_DIR} ...")
    all_actions = []
    for pkl_file in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl")):
        with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
            ep = pickle.load(f)
        all_actions.append(np.array(ep["actions"], dtype=np.float32))
    all_actions = np.concatenate(all_actions, axis=0)
    action_lo = all_actions.min(axis=0)
    action_hi = all_actions.max(axis=0)
    print(f"  lo={action_lo.tolist()}  hi={action_hi.tolist()}")

    # -----------------------------------------------------------------------
    # Env -- one persistent physics client, states restored into it per timestep. Must match
    # block_combo/seed exactly, since set_pybullet_state restores objects by obj_id, which only
    # lines up if this process's object-loading order matches the one that produced the pickled
    # states (already externally verified: the other repo's oracle run round-tripped through
    # this same pybullet_states.pkl successfully).
    # -----------------------------------------------------------------------
    print("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env({"ood": False, "block_combo": [block_a, block_b]}, seed=42)
    inner = env.env
    print("Env ready.")

    # Restore-fidelity check: recompute dist(block_a, block_b) at t_start from the freshly
    # restored env and compare against the cached dist_at_t. If this fails, nothing downstream
    # can be trusted -- the env construction doesn't match what produced pybullet_states.pkl.
    inner.set_pybullet_state(pybullet_states[args.t_start])
    blocks_now = inner.get_block_states()
    dist_now = float(np.linalg.norm(blocks_now[block_a] - blocks_now[block_b]))
    expected = float(probes[args.t_start]["dist_at_t"])
    assert abs(dist_now - expected) < 1e-3, (
        f"restore fidelity check FAILED at t={args.t_start}: recomputed dist={dist_now:.6f} "
        f"!= recorded dist_at_t={expected:.6f}"
    )
    print(f"Restore fidelity check passed at t={args.t_start}: dist={dist_now:.4f} "
          f"(recorded {expected:.4f})")

    # -----------------------------------------------------------------------
    # Checkpoint + model
    # -----------------------------------------------------------------------
    ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{ckpt_epoch:03d}.pt")
    print(f"Loading checkpoint {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_star = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)

    def dynamics_fn(state, action):
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        return 1.0 - (terminal_state * z_star).sum(dim=-1)

    # -----------------------------------------------------------------------
    # Output dir
    # -----------------------------------------------------------------------
    parent_dirname = os.path.basename(os.path.dirname(rollout_dir))
    out_dir = args.out_dir or os.path.join(
        MAIN_PROJECT_DIR, "MPPI_charts", "oracle_vs_mppi",
        f"{parent_dirname}_{epoch_dirname}_rollout{rollout_idx}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output -> {out_dir}")

    n_mppi = args.n_mppi_seeds
    mppi_seeds = [MPPI_SEED_BASE + k for k in range(n_mppi)]
    timesteps = list(range(args.t_start, args.t_end + 1))
    print(f"\n{len(timesteps)} timesteps x ({n_oracle_seeds} oracle + {n_mppi} MPPI-probe seeds "
          f"+ 1 ground-truth path) -- MPPI budget: K={K} mppi_temperature={args.mppi_temperature} "
          f"evals={args.mppi_evals} (n_planning_itrs={args.mppi_evals - 1})\n")

    # -----------------------------------------------------------------------
    # Main computation loop
    # -----------------------------------------------------------------------
    results = {}

    for t in timesteps:
        inner.set_pybullet_state(pybullet_states[t])
        pil_frame = env.get_frame()
        with torch.no_grad():
            img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
            z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)

        probe_t = probes[t]

        # ---- Oracle proposals ----
        oracle_chunks_raw = np.stack(
            [np.asarray(s["actions"], dtype=np.float32) for s in probe_t["seeds"]], axis=0
        )  # (n_oracle_seeds, 8, 2)
        oracle_cos = np.zeros(oracle_chunks_raw.shape[0], dtype=np.float64)
        with torch.no_grad():
            for j in range(oracle_chunks_raw.shape[0]):
                chunk_norm = normalize_chunk(oracle_chunks_raw[j], action_lo, action_hi)
                chunk_t = torch.tensor(chunk_norm, device=DEVICE).unsqueeze(0)
                z_pred = model(z_t.unsqueeze(0), chunk_t).squeeze(0)
                oracle_cos[j] = float((z_pred * z_star).sum().item())

        # ---- MPPI-probe proposals: fresh instance per seed, exactly one command() call each ----
        mppi_chunks_raw = []
        mppi_cos = []
        for k in range(n_mppi):
            mppi = MPPI(
                dynamics_fn=dynamics_fn,
                cost_fn=cost_fn,
                pred_horizon=HORIZON,
                action_dim=2,
                num_samples=K,
                mppi_temperature=args.mppi_temperature,
                n_planning_itrs=args.mppi_evals - 1,
                device=DEVICE,
                max_action_value=MAX_ACTION_VALUE,
                warm_start_std=WARM_START_STD,
            )
            torch.manual_seed(mppi_seeds[k])
            with torch.no_grad():
                best_action_chunk = mppi.command(z_t)  # (8, 2) normalized

            best_idx = int(torch.argmax(mppi.last_returns))
            assert mppi.last_actions[best_idx].allclose(best_action_chunk), (
                "argmax of last_returns does not match mppi.command()'s own selection"
            )
            cos_sim = float(mppi.last_returns[best_idx].item() + 1.0)
            chunk_raw = (
                (best_action_chunk.cpu().numpy() + 1.0) / 2.0 * (action_hi - action_lo) + action_lo
            ).astype(np.float32)
            mppi_chunks_raw.append(chunk_raw)
            mppi_cos.append(cos_sim)
        mppi_chunks_raw = np.stack(mppi_chunks_raw, axis=0)  # (n_mppi, 8, 2)
        mppi_cos = np.array(mppi_cos)

        # ---- Ground-truth peg path: what actually happened, t -> t+horizon ----
        gt_states = pybullet_states[t: t + horizon + 1]
        peg_positions = []
        for s in gt_states:
            inner.set_pybullet_state(s)
            fk = inner._robot.forward_kinematics()
            peg_positions.append(np.asarray(fk.translation[:2], dtype=np.float64).copy())
        peg_positions = np.stack(peg_positions, axis=0)  # (horizon+1, 2)
        gt_path = peg_positions - peg_positions[0]        # already a path from the origin
        gt_action_raw = np.diff(peg_positions, axis=0).astype(np.float32)  # (horizon, 2)
        with torch.no_grad():
            gt_norm = normalize_chunk(gt_action_raw, action_lo, action_hi)
            gt_t = torch.tensor(gt_norm, device=DEVICE).unsqueeze(0)
            z_pred_gt = model(z_t.unsqueeze(0), gt_t).squeeze(0)
            gt_cos = float((z_pred_gt * z_star).sum().item())

        results[t] = dict(
            oracle_chunks_raw=oracle_chunks_raw, oracle_cos=oracle_cos,
            mppi_chunks_raw=mppi_chunks_raw, mppi_cos=mppi_cos,
            gt_path=gt_path, gt_cos=gt_cos,
        )
        print(f"t={t:3d}  oracle_cos={np.round(oracle_cos, 4)}  "
              f"mppi_cos={np.round(mppi_cos, 4)}  gt_cos={gt_cos:.4f}")

    # -----------------------------------------------------------------------
    # Shared global axis scale across every timestep and every series, so the per-timestep
    # charts are directly comparable at a glance (magnitude only -- there is no reference
    # direction/arrow in this script).
    # -----------------------------------------------------------------------
    all_points = [np.zeros((1, 2))]
    for t in timesteps:
        r = results[t]
        for j in range(r["oracle_chunks_raw"].shape[0]):
            all_points.append(cumulative_path(r["oracle_chunks_raw"][j]))
        for k in range(r["mppi_chunks_raw"].shape[0]):
            all_points.append(cumulative_path(r["mppi_chunks_raw"][k]))
        all_points.append(r["gt_path"])
    all_points = np.concatenate(all_points, axis=0)
    axis_half = float(np.abs(all_points).max() * 1.15)
    axis_lim = (-axis_half, axis_half)
    print(f"\nShared axis scale: axis_lim=[{-axis_half:.4f}, {axis_half:.4f}]")

    # -----------------------------------------------------------------------
    # Per-timestep charts: ONLY the paths and their cos_sim in the legend -- no reference arrow,
    # no distance/other annotation (explicitly confirmed).
    # -----------------------------------------------------------------------
    for t in timesteps:
        r = results[t]
        fig, ax = plt.subplots(figsize=(8, 8))

        for j in range(r["oracle_chunks_raw"].shape[0]):
            path = cumulative_path(r["oracle_chunks_raw"][j])
            color = ORACLE_COLORS[j % len(ORACLE_COLORS)]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, linestyle="--",
                    label=f"oracle seed {j + 1}  cos_sim={r['oracle_cos'][j]:.4f}",
                    zorder=3 + j)
            ax.annotate("", xy=tuple(path[-1]), xytext=tuple(path[-2]),
                        arrowprops=dict(facecolor=color, edgecolor=color, width=1.2, headwidth=7),
                        zorder=4 + j)

        n_o = r["oracle_chunks_raw"].shape[0]
        for k in range(r["mppi_chunks_raw"].shape[0]):
            path = cumulative_path(r["mppi_chunks_raw"][k])
            color = MPPI_COLORS[k % len(MPPI_COLORS)]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.2,
                    label=f"MPPI-probe seed {k + 1}  cos_sim={r['mppi_cos'][k]:.4f}",
                    zorder=3 + n_o + k)
            ax.annotate("", xy=tuple(path[-1]), xytext=tuple(path[-2]),
                        arrowprops=dict(facecolor=color, edgecolor=color, width=1.4, headwidth=8),
                        zorder=4 + n_o + k)

        gt_path = r["gt_path"]
        ax.plot(gt_path[:, 0], gt_path[:, 1], color=GT_COLOR, linewidth=3.0,
                label=f"actual outcome (peg)  cos_sim={r['gt_cos']:.4f}",
                zorder=10 + n_o + r["mppi_chunks_raw"].shape[0])
        ax.annotate("", xy=tuple(gt_path[-1]), xytext=tuple(gt_path[-2]),
                    arrowprops=dict(facecolor=GT_COLOR, edgecolor=GT_COLOR, width=1.8, headwidth=10),
                    zorder=11 + n_o + r["mppi_chunks_raw"].shape[0])

        ax.set_xlim(axis_lim)
        ax.set_ylim(axis_lim)
        ax.set_aspect("equal")
        ax.set_xlabel("net displacement dx")
        ax.set_ylabel("net displacement dy")
        ax.set_title(f"{stem} -- timestep {t}")
        ax.legend(loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"t{t:03d}.png"), dpi=150)
        plt.close()

    print(f"Saved {len(timesteps)} per-timestep charts -> {out_dir}")

    # -----------------------------------------------------------------------
    # Summary chart: cos_sim of every series vs. timestep, across the full range.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for j in range(n_oracle_seeds):
        ys = [results[t]["oracle_cos"][j] for t in timesteps]
        ax.plot(timesteps, ys, color=ORACLE_COLORS[j % len(ORACLE_COLORS)], linestyle="--",
                marker="o", markersize=4, linewidth=1.2, label=f"oracle seed {j + 1}")
    for k in range(n_mppi):
        ys = [results[t]["mppi_cos"][k] for t in timesteps]
        ax.plot(timesteps, ys, color=MPPI_COLORS[k % len(MPPI_COLORS)],
                marker="s", markersize=4, linewidth=1.2, label=f"MPPI-probe seed {k + 1}")
    ys = [results[t]["gt_cos"] for t in timesteps]
    ax.plot(timesteps, ys, color=GT_COLOR, linewidth=2.5, marker="^", markersize=5,
            label="actual outcome (peg)")

    ax.set_xlabel("timestep")
    ax.set_ylabel("cos_sim")
    ax.set_title(f"{stem}: cos_sim vs. timestep, all series")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    summary_path = os.path.join(out_dir, "summary_cos_sim_vs_timestep.png")
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"Saved summary chart -> {summary_path}")

    # -----------------------------------------------------------------------
    # Cache the computed data so it never needs silently recomputing.
    # -----------------------------------------------------------------------
    npz_path = os.path.join(out_dir, "data.npz")
    np.savez(
        npz_path,
        timesteps=np.array(timesteps),
        oracle_chunks_raw=np.stack([results[t]["oracle_chunks_raw"] for t in timesteps], axis=0),
        oracle_cos=np.stack([results[t]["oracle_cos"] for t in timesteps], axis=0),
        mppi_chunks_raw=np.stack([results[t]["mppi_chunks_raw"] for t in timesteps], axis=0),
        mppi_cos=np.stack([results[t]["mppi_cos"] for t in timesteps], axis=0),
        gt_path=np.stack([results[t]["gt_path"] for t in timesteps], axis=0),
        gt_cos=np.array([results[t]["gt_cos"] for t in timesteps]),
        axis_lim=np.array(axis_lim),
        mppi_temperature=args.mppi_temperature,
        mppi_evals=args.mppi_evals,
        n_mppi_seeds=n_mppi,
        mppi_seed_base=MPPI_SEED_BASE,
        rollout_epoch=rollout_epoch,
        ckpt_epoch=ckpt_epoch,
        rollout_idx=rollout_idx,
        goal_text=goal_text,
    )
    print(f"Saved data -> {npz_path}")
    print(f"\nDone. Outputs under {out_dir}")


if __name__ == "__main__":
    main()
