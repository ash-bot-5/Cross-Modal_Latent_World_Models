"""
h8 obstacle-detour probe: does the model rate a PHYSICALLY-CORRECT detour-and-push chunk
above the chunks its own MPPI loop converges to?

Duplicate of mppi_obstacle_expert_chunk_probe.py (that script's docstring covers the full
rationale -- same obstacle setup, same MPPI-vs-expert-chunk cos_sim comparison, same "isolate
planner blame from dynamics-model blame" motivation) but pointed at the h8 model
(checkpoints_clip_full_fine-tuned_with_subopt_data/epoch_011.pt, action horizon 8) instead of the
h16 one, with a hand-crafted expert chunk redesigned for an 8-action budget instead of 16.

What changed vs. the h16 script:

1. Checkpoint: h8 epoch 11 (CKPT_DIR/EPOCH below), HORIZON=8. Everything about how the
   checkpoint is loaded/detected (action_horizon from action_encoder.out.weight.shape) and how
   action_lo/action_hi normalization stats are computed (full per-timestep min/max over every
   raw ep["actions"] in TRAIN_DIR, independent of chunk length) is unchanged from the h16
   script -- both are horizon-agnostic by construction.

2. NEW_BLOCK_DIST raised from h16's 0.10 to 0.0925 (NOT lowered to 0.07 -- see the staging-bug
   note below), and EXPERT_PHASES completely redesigned for an 8-step budget (down from 16).
   Found by direct empirical iteration (physics-only, no CLIP/model/MPPI involved) in a scratch
   script, watching per-step dist(green_cube,blue_moon) and dist(peg,green_cube)/dist(peg,
   blue_moon), the same discipline the h16 script's own docstring describes using to tune its
   push phase:

     - First attempt: NEW_BLOCK_DIST=0.07 (an aggressively small gap, on the theory that an
       8-step budget could only afford a small push). This actually worked physically -- final
       dist(green_cube,blue_moon)=0.0477, dist(peg,green_cube)=0.0363, both comfortably
       TOUCHING at t=8 -- but the deliverable failed on its FIRST REAL RUN with
       `AssertionError: realized peg is 0.0373 from blue_moon, not comfortably outside touching
       range` (the `dist_peg_blue > TOUCH_THRESH` staging assertion, copied unchanged from the
       h16 script). Root cause: relocating blue_moon that close to green_cube (target 0.07)
       doesn't actually settle at 0.07 -- physics clamps it to a ~0.078 floor (blue_moon's own
       collision radius against green_cube), and with PEG_START_OFFSET=0.04 fixed, dist_peg_blue
       = 0.078-0.04 = ~0.037 < TOUCH_THRESH. This never showed up in scratch tuning because the
       scratch script (deliberately physics-only, no model loading) never exercised that
       specific assertion -- a gap in the tuning process, not a property of the physics.
     - Tried shrinking PEG_START_OFFSET instead (to keep the peg farther from blue_moon without
       changing block geometry) -- FAILED WORSE: offset=0.015 made the peg's OWN final-approach
       move disturb green_cube (moved 0.0218 during staging, tripping the
       `green_moved_peg < 1e-3` assertion), and offset=0.03 still moved it 0.004 (over the 1e-3
       tolerance). dist_peg_green was consistently ~0.0407 across every successful configuration
       tried -- that's green_cube's actual physical collision radius from the peg, and
       PEG_START_OFFSET=0.04 sits right at it. It's a floor, not a free parameter.
     - With PEG_START_OFFSET pinned at 0.04, the only remaining lever is a LARGER NEW_BLOCK_DIST
       (more room between the blocks, so dist_peg_blue clears TOUCH_THRESH even after the
       peg's fixed ~0.04 offset is subtracted) -- but a larger gap is a bigger target for the
       push phase to close in the same 3-step budget (lateral_out(2)+detour_back(2)+
       lateral_in(1)=5 steps fixed, same as before -- see below -- leaving only 3 for push).
       A uniform push (matching the original 0.07 attempt's magnitude) plateaued around
       dist=0.050-0.055 at NEW_BLOCK_DIST=0.093-0.095 regardless of exact magnitude -- too much
       push and the last step overshoots and REBOUNDS (final ends up worse than an earlier step
       in the same rollout, e.g. 0.0501->0.0574); too little and it just doesn't reach.
     - A TAPERING push -- decaying per-step magnitude, same technique the h16 script's own
       PUSH_MAGNITUDES now uses -- avoids the overshoot while still making full use of all 3
       steps (front-loaded distance-closing, tail end just barely nudges rather than slamming
       into an already-near-contact block). PUSH_MAGNITUDES=[0.055, 0.037, 0.025] at
       NEW_BLOCK_DIST=0.0925 (the minimum block separation that keeps dist_peg_blue's staging
       margin positive, ~0.0019, given PEG_START_OFFSET=0.04 is fixed) landed at
       dist(green_cube,blue_moon)=0.0495 and dist(peg,green_cube)=0.0407 at t=8 -- both
       TOUCHING, monotonically decreasing across all 3 push steps (0.0556->0.0543->0.0495, no
       rebound), reproduced identically across repeated runs from a pristine env.
     - A "merge lateral_in into the first push step" idea (one diagonal action doing
       realignment and pushing at once, to free a 4th push step) was also tried and abandoned:
       per-axis action clipping distorts a diagonal request badly enough that the realignment is
       lost entirely -- the peg missed green_cube on the following steps (dist_peg_green blew
       out to 0.09-0.12). Clipping only composes safely along a single (f or p) direction per
       step, not a combined one.

Everything else -- env setup, block-teleport + peg-staging (parking every block except
green_cube off-table before ANY peg motion, staging the peg with only green_cube on the table,
then restoring the other 6 blocks + blue_moon at its new closer position), the 5 independent
MPPI planning passes, physically executing all 6 chunks (5 MPPI winners + expert) from an
identical snapshotted starting state via set_pybullet_state, saving a frame + rollout video per
chunk, the chart, and the .npz/summary.txt output -- is copied unchanged from
mppi_obstacle_expert_chunk_probe.py; none of it depends on the model's action horizon. Each video
is VIDEO_FPS=5 fps, HORIZON+1=9 frames (starting frame + one per action) covering that one
8-step chunk.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_obstacle_expert_chunk_probe_h8.py
"""

import os
import sys
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.mppi_core import MPPI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"   # start_block -- the one to be pushed
BLOCK_B = "blue_moon"    # target_block -- the destination

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(MAIN_PROJECT_DIR)
OUT_DIR = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "obstacle_avoidance_expert_chunk_h8")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# h8 checkpoint dir -- this checkpoint trains on 8-action chunks (h16 variant also exists at
# the repo root, checkpoints_clip_full_fine-tuned_with_subopt_data_h16/).
CKPT_DIR = os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data")
EPOCH = 11  # last completed epoch of this h8 run

VIDEO_FPS = 5  # slower than mppi_rollout_5_per_checkpoint.py's fps=10 -- each of these clips is
               # only 9 frames (start + 8 actions) covering one deliberate action chunk, not a
               # 200+-step episode, so a lower fps keeps it watchable rather than a ~0.9s blur

# MPPI hyperparameters -- matches the h16 rollout-generation campaign, not the older
# mppi_obstacle_avoidance_probe.py precedent (temp=0.03).
MPPI_TEMPERATURE = 0.1
N_PLANNING_ITRS = 999   # evals=1000
NUM_SAMPLES = 2000
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1    # API completeness only -- each run is a single command() call on a
                         # fresh instance (prev_best_action=None), so this is never exercised.

N_MPPI_RUNS = 5
SAMPLE_SEED = 123
SAMPLE_SEEDS = [SAMPLE_SEED + i for i in range(N_MPPI_RUNS)]

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])  # original (pre-relocation) position
TOUCH_THRESH = 0.05
NEW_BLOCK_DIST = 0.0925      # new dist(green_cube, blue_moon) after relocating blue_moon
                              # (0.0925, not h16's 0.10 -- see module docstring. NOT smaller --
                              # a first attempt at 0.07 physically worked but failed the
                              # dist_peg_blue staging assertion below; PEG_START_OFFSET is a
                              # fixed floor, so this had to go UP, not down, to clear it)
OTHER_BLOCK_CLEARANCE = 0.06  # min clearance blue_moon's new position must keep from every
                               # OTHER block (not green_cube) to avoid a post-teleport physics pop
PEG_PARK_OFFSET = 0.10       # safe again now that ALL other blocks are parked off-table first
PEG_START_OFFSET = 0.04
OFF_TABLE_SPACING = 0.2      # per-block x-offset used to park blocks off-table without overlapping

HORIZON = 8

# Expert chunk phase design (steps, world-frame direction, total displacement magnitude OR an
# explicit per-step magnitude array -- same convention as the h16 script's tapering push phase).
# f = unit vector green_cube -> blue_moon; p = perpendicular to f. Peg starts at local
# f=+PEG_START_OFFSET (0.04), p=0 (i.e. "between" the blocks, near-touching green_cube).
# Empirically tuned (physics-only, see module docstring) for the 8-step budget:
#   phase 1: +p, 2 steps, total 0.05   -- move laterally clear of green_cube (same magnitude as
#                                          h16 -- this clearance is governed by the block's own
#                                          collision geometry, not by the action budget; 1 step
#                                          (0.035) was tried first and was NOT enough -- the peg
#                                          grazed and shoved green_cube during detour_back)
#   phase 2: -f, 2 steps, total 0.08   -- travel past green_cube to its far side (shorter than
#                                          h16's 3 steps/0.10 -- an 8-step budget can't afford
#                                          more, and 0.08 was confirmed empirically sufficient:
#                                          dist(green_cube,blue_moon) stays flat through this
#                                          whole phase, i.e. no disturbance, at every
#                                          NEW_BLOCK_DIST tried)
#   phase 3: -p, 1 step, total 0.035   -- realign with the green_cube-blue_moon axis (asymmetric
#                                          with phase 1's 2 steps -- validated empirically, not
#                                          derived analytically; 1 step was enough here). A
#                                          "merge this into the first push step" idea (one
#                                          diagonal action instead of a separate realign step, to
#                                          free a 4th push step) was tried and abandoned: per-axis
#                                          clipping distorts a diagonal (f,p) request badly enough
#                                          that the realignment is lost -- the peg missed
#                                          green_cube on later steps (dist_peg_green blew out to
#                                          0.09-0.12). Clipping only composes safely along a
#                                          single direction per step.
#   phase 4: +f, 3 steps, TAPERING magnitudes PUSH_MAGNITUDES=[0.055, 0.037, 0.025] -- close the
#             gap and push green_cube into blue_moon, decaying each step rather than a uniform
#             push (same technique as h16's own tapering push, scaled down to a 3-step budget).
#             Load-bearing that it decays: a UNIFORM 3-step push at comparable total magnitude
#             plateaued around final dist=0.050-0.055 regardless of exact magnitude, and pushing
#             harder made it WORSE, not better -- the last step overshoots an already-near-
#             contact block and REBOUNDS it back apart (e.g. one uniform attempt: 0.0501 at its
#             best intermediate step, then 0.0574 by the actual final step). The taper avoids
#             this: each step is confirmed to MONOTONICALLY decrease dist(green_cube,blue_moon)
#             with no rebound (0.0556 -> 0.0543 -> 0.0495), landing t=8 at
#             dist(green_cube,blue_moon)=0.0495 and dist(peg,green_cube)=0.0407, both TOUCHING
#             (TOUCH_THRESH=0.05), reproduced identically across repeated runs from a pristine
#             env. NEW_BLOCK_DIST=0.0925 is the smallest gap that keeps the dist_peg_blue
#             staging assertion below comfortably positive (~0.0019 margin) given
#             PEG_START_OFFSET=0.04 is fixed (see module docstring) -- pushing NEW_BLOCK_DIST
#             any smaller would reopen the original staging-assertion failure.
PUSH_MAGNITUDES = [0.055, 0.037, 0.025]
EXPERT_PHASES = [
    ("lateral_out", "p", +1, 2, 0.05),
    ("detour_back", "f", -1, 2, 0.08),
    ("lateral_in",  "p", -1, 1, 0.035),
    ("push",        "f", +1, 3, PUSH_MAGNITUDES),
]
assert sum(n for *_, n, _ in EXPERT_PHASES) == HORIZON, "expert chunk phases must sum to HORIZON"

# Validated categorical palette (dataviz skill convention, matches
# mppi_obstacle_avoidance_probe.py) for the 5 per-seed run paths.
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
REF_COLOR = "#c0399f"   # expert chunk -- distinct from RUN_COLORS and the block icon colors
INK = "#333333"
INK_MUTED = "#888888"
ICON_GREEN = "#3a7d3a"
ICON_BLUE = "#4d6f99"


# ---------------------------------------------------------------------------
# Inline model classes (matches every other eval script's convention).
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
# Helpers
# ---------------------------------------------------------------------------

def cumulative_path(chunk_raw: np.ndarray) -> np.ndarray:
    """(horizon, 2) per-step displacements -> (horizon+1, 2) path starting at the origin."""
    return np.concatenate([[[0.0, 0.0]], np.cumsum(chunk_raw, axis=0)], axis=0)


def denormalize_chunk(chunk_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return ((chunk_norm + 1.0) / 2.0 * (hi - lo) + lo)


def normalize_chunk(chunk_raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip(2.0 * (chunk_raw - lo) / (hi - lo) - 1.0, -1.0, 1.0).astype(np.float32)


def draw_block_icons(ax, green_rel: np.ndarray, blue_rel: np.ndarray):
    """Both blocks as labeled icons at their position relative to the peg's start (the plot
    origin). green_cube: filled square + dashed TOUCH_THRESH circle. blue_moon: filled star."""
    ax.scatter([green_rel[0]], [green_rel[1]], marker="s", s=260, color=ICON_GREEN,
               edgecolors=INK, linewidths=1.0, zorder=10)
    ax.add_patch(Circle(tuple(green_rel), TOUCH_THRESH, fill=False, linestyle="--",
                         edgecolor=ICON_GREEN, alpha=0.7, linewidth=1.3, zorder=9))
    ax.text(green_rel[0], green_rel[1] - TOUCH_THRESH - 0.008, "green_cube",
            color=ICON_GREEN, fontsize=9, fontweight="bold", ha="center", va="top", zorder=10)

    ax.scatter([blue_rel[0]], [blue_rel[1]], marker="*", s=420, color=ICON_BLUE,
               edgecolors=INK, linewidths=1.0, zorder=10)
    ax.text(blue_rel[0], blue_rel[1] - 0.014, "blue_moon",
            color=ICON_BLUE, fontsize=9, fontweight="bold", ha="center", va="top", zorder=10)

    ax.scatter([0.0], [0.0], marker="P", s=140, color=INK, zorder=10)
    ax.text(0.0, 0.008, "peg start", color=INK, fontsize=8, ha="center", va="bottom", zorder=10)


def build_expert_chunk_raw(f_dir: np.ndarray, p_dir: np.ndarray) -> np.ndarray:
    """Hand-crafted (HORIZON, 2) raw action chunk implementing the detour-and-push maneuver,
    per EXPERT_PHASES. f_dir/p_dir are world-frame unit vectors. Each phase's last element is
    either a scalar (uniform per-step magnitude = total_disp / n_steps) or an array of exactly
    n_steps explicit per-step magnitudes (used by the tapering push phase, see EXPERT_PHASES)."""
    basis = {"f": f_dir, "p": p_dir}
    rows = []
    for _name, axis, sign, n_steps, spec in EXPERT_PHASES:
        direction = basis[axis] * sign
        if np.isscalar(spec):
            per_step = (spec / n_steps) * direction
            rows.append(np.tile(per_step, (n_steps, 1)))
        else:
            magnitudes = np.asarray(spec, dtype=np.float32)
            assert len(magnitudes) == n_steps
            rows.append(magnitudes[:, None] * direction[None, :])
    return np.concatenate(rows, axis=0).astype(np.float32)  # (HORIZON, 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Action normalization stats.
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
    print(f"Action normalization stats: lo={action_lo.tolist()} hi={action_hi.tolist()}\n")

    # -----------------------------------------------------------------------
    # 2. Env + compute the new (closer) blue_moon target position. The actual relocation
    #    happens together with peg staging in section 3 below (both need every other block
    #    parked off-table first -- see module docstring).
    # -----------------------------------------------------------------------
    print("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env({"ood": False, "block_combo": None}, seed=42)
    print("Env ready.\n")

    from language_table.environments.utils.pose3d import Pose3d
    from scipy.spatial.transform import Rotation as R

    inner = env.env
    blocks = inner.get_block_states()
    green_xy = blocks[BLOCK_A]
    blue_xy_orig = blocks[BLOCK_B]

    assert np.allclose(green_xy, EXPECTED_GREEN_XY, atol=1e-5), (
        f"{BLOCK_A} position {green_xy} != expected {EXPECTED_GREEN_XY} -- table arrangement diverged"
    )
    assert np.allclose(blue_xy_orig, EXPECTED_BLUE_XY, atol=1e-5), (
        f"{BLOCK_B} position {blue_xy_orig} != expected {EXPECTED_BLUE_XY} -- table arrangement diverged"
    )

    toward_blue_direction = (blue_xy_orig - green_xy) / np.linalg.norm(blue_xy_orig - green_xy)
    away_from_blue_direction = -toward_blue_direction
    perp_direction = np.array([-toward_blue_direction[1], toward_blue_direction[0]])  # unit, arbitrary sign

    new_blue_xy = green_xy + toward_blue_direction * NEW_BLOCK_DIST

    other_blocks = {k: v for k, v in blocks.items() if k not in (BLOCK_A, BLOCK_B, "peg")}
    for name, xy in other_blocks.items():
        d = float(np.linalg.norm(new_blue_xy - xy))
        assert d > OTHER_BLOCK_CLEARANCE, (
            f"new {BLOCK_B} position {new_blue_xy} is only {d:.4f} from {name} "
            f"(need > {OTHER_BLOCK_CLEARANCE}) -- would risk a post-teleport physics pop"
        )
    print(f"Clearance to other 6 blocks OK (min "
          f"{min(np.linalg.norm(new_blue_xy - xy) for xy in other_blocks.values()):.4f} "
          f"> {OTHER_BLOCK_CLEARANCE})\n")

    def _teleport_block(name, xy, z=None, orn=None):
        bid = inner._block_to_pybullet_id[name]
        if z is None or orn is None:
            _pos, _orn = inner.pybullet_client.getBasePositionAndOrientation(bid)
            z = _pos[2] if z is None else z
            orn = _orn if orn is None else orn
        inner.pybullet_client.resetBasePositionAndOrientation(
            bid, [float(xy[0]), float(xy[1]), z], orn,
        )
        return z, orn

    def _move_peg_to(xy):
        inner._set_robot_target_effector_pose(Pose3d(
            rotation=R.from_rotvec([0, np.pi, 0]),
            translation=np.array([xy[0], xy[1], 0.145]),
        ))
        inner._step_simulation_to_stabilize()

    # -----------------------------------------------------------------------
    # 3. Stage the peg between green_cube and blue_moon's NEW (closer) position.
    #    Every block except green_cube is parked off-table first (see module docstring for
    #    why -- the arm's rest pose is uncontrolled and a direct move can sweep through
    #    whichever blocks happen to lie in its path), the peg is staged with nothing else in
    #    the workspace to hit, then all 7 blocks are restored -- 6 to their original spot,
    #    blue_moon to its new closer one.
    # -----------------------------------------------------------------------
    other_block_names = list(other_blocks.keys()) + [BLOCK_B]  # 6 unrelated + blue_moon
    saved_z_orn = {}
    for i, name in enumerate(other_block_names):
        saved_z_orn[name] = _teleport_block(name, [5.0 + i * OFF_TABLE_SPACING, 5.0])
    inner._step_simulation_to_stabilize()

    park_xy = green_xy + toward_blue_direction * PEG_PARK_OFFSET
    final_xy = green_xy + toward_blue_direction * PEG_START_OFFSET
    _move_peg_to(park_xy)
    _move_peg_to(final_xy)

    green_xy_after_peg = inner.get_block_states()[BLOCK_A]
    green_moved_peg = float(np.linalg.norm(green_xy_after_peg - green_xy))
    assert green_moved_peg < 1e-3, (
        f"{BLOCK_A} moved {green_moved_peg:.6f} during peg staging -- unexpected disturbance"
    )

    for name in other_block_names:
        target = new_blue_xy if name == BLOCK_B else other_blocks.get(name, blue_xy_orig)
        z, orn = saved_z_orn[name]
        _teleport_block(name, target, z=z, orn=orn)
    inner._step_simulation_to_stabilize()

    blocks_after_peg = inner.get_block_states()
    green_xy_after = blocks_after_peg[BLOCK_A]
    blue_xy_after = blocks_after_peg[BLOCK_B]
    peg_xy_after = np.asarray(inner._robot.forward_kinematics().translation[:2], dtype=np.float64)
    dist_peg_green = float(np.linalg.norm(peg_xy_after - green_xy_after))
    dist_peg_blue = float(np.linalg.norm(peg_xy_after - blue_xy_after))
    dist_green_blue = float(np.linalg.norm(green_xy_after - blue_xy_after))
    green_moved = float(np.linalg.norm(green_xy_after - green_xy))
    blue_moved = float(np.linalg.norm(blue_xy_after - new_blue_xy))

    print(f"Peg start (realized): {peg_xy_after}  dist(peg,{BLOCK_A})={dist_peg_green:.4f}  "
          f"dist(peg,{BLOCK_B})={dist_peg_blue:.4f}  dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}\n")

    assert dist_peg_green < TOUCH_THRESH, (
        f"realized peg is {dist_peg_green:.4f} from {BLOCK_A}, not < {TOUCH_THRESH}"
    )
    assert dist_peg_blue > TOUCH_THRESH, (
        f"realized peg is {dist_peg_blue:.4f} from {BLOCK_B}, not comfortably outside touching range"
    )
    assert dist_peg_blue < dist_green_blue, (
        f"realized peg is {dist_peg_blue:.4f} from {BLOCK_B}, but dist({BLOCK_A},{BLOCK_B})="
        f"{dist_green_blue:.4f} -- peg is NOT between the blocks"
    )
    assert green_moved < 1e-3, f"{BLOCK_A} moved {green_moved:.6f} -- unexpected disturbance"
    assert blue_moved < 1e-3, f"{BLOCK_B} did not land at its target position (off by {blue_moved:.6f})"
    for name in other_block_names:
        if name == BLOCK_B:
            continue
        moved = float(np.linalg.norm(blocks_after_peg[name] - other_blocks[name]))
        assert moved < 1e-3, f"{name} moved {moved:.6f} -- unexpected disturbance"
    print(f"Obstacle geometry verified: peg is between {BLOCK_A} and {BLOCK_B}, "
          f"nearly touching {BLOCK_A}, dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}, "
          f"all other blocks undisturbed.\n")

    pil_frame = env.get_frame()
    frame_path = os.path.join(OUT_DIR, "frame_00_start.png")
    pil_frame.save(frame_path)
    print(f"Saved start frame -> {frame_path}\n")

    # Full pybullet state at this exact moment (peg staged, blue_moon relocated, everything
    # verified undisturbed) -- snapshotted once so every chunk below (5 MPPI winners + the
    # expert) can be physically executed from the IDENTICAL starting point via
    # set_pybullet_state, without repeating the staged-teleport setup 6 times.
    initial_pybullet_state = inner.get_pybullet_state()

    def _execute_chunk_and_capture(chunk_raw, tag):
        """Resets to the exact staged starting state, executes chunk_raw open-loop for
        HORIZON real env steps, and returns (final_frame, min_dist, success, dist_hist,
        first_touch_frame, first_touch_step, first_touch_dist, video_frames) -- min_dist is
        the minimum dist(BLOCK_A, BLOCK_B) over the whole rollout (matching the project-wide
        success convention), not just the final step. first_touch_* capture the frame/step/
        dist at the FIRST step where dist drops to <= TOUCH_THRESH (None if it never does) --
        the moment the blocks are actually touching, not just wherever the rollout ends up.
        video_frames is a list of HORIZON+1 (H,W,3) uint8 arrays (starting frame + one per
        step), ready for imageio.mimsave."""
        inner.set_pybullet_state(initial_pybullet_state)
        cur0 = inner.get_block_states()
        dist_hist = [float(np.linalg.norm(cur0[BLOCK_A] - cur0[BLOCK_B]))]
        frame_now = env.get_frame()
        video_frames = [np.array(frame_now)]
        first_touch_frame = first_touch_step = first_touch_dist = None
        for step_idx in range(HORIZON):
            env.step(chunk_raw[step_idx])
            cur = inner.get_block_states()
            d = float(np.linalg.norm(cur[BLOCK_A] - cur[BLOCK_B]))
            dist_hist.append(d)
            frame_now = env.get_frame()
            video_frames.append(np.array(frame_now))
            print(f"    [{tag}] step {step_idx:>2}: dist({BLOCK_A},{BLOCK_B})={d:.4f}")
            if first_touch_frame is None and d <= TOUCH_THRESH:
                first_touch_frame = frame_now
                first_touch_step = step_idx
                first_touch_dist = d
        final_frame = frame_now
        min_dist = float(min(dist_hist))
        success = min_dist <= TOUCH_THRESH
        return (final_frame, min_dist, success, dist_hist,
                first_touch_frame, first_touch_step, first_touch_dist, video_frames)

    green_rel = green_xy_after - peg_xy_after
    blue_rel = blue_xy_after - peg_xy_after

    goal_text = (
        f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
        f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
    )
    print(f"Goal: {goal_text}\n")

    # -----------------------------------------------------------------------
    # 4. Hand-crafted expert detour-and-push chunk. Clipped to the dataset's raw per-axis
    #    action bounds BEFORE anything else, so the physically-executed chunk (section 7) and
    #    the model-facing normalized chunk are the exact same actions -- the phase magnitudes
    #    are specified in (f, p) space and project onto world (x, y) anisotropically, so some
    #    steps do exceed action_hi/lo on one axis before this clip.
    # -----------------------------------------------------------------------
    expert_chunk_raw = build_expert_chunk_raw(toward_blue_direction, perp_direction)
    expert_chunk_raw = np.clip(expert_chunk_raw, action_lo, action_hi).astype(np.float32)
    expert_disp = expert_chunk_raw.sum(axis=0)
    print(f"Expert chunk: net_disp=({expert_disp[0]:+.4f},{expert_disp[1]:+.4f})  "
          f"phases={[p[0] for p in EXPERT_PHASES]}\n")

    # -----------------------------------------------------------------------
    # 5. Load h8 checkpoint, encode z_t / z_sem through its OWN fine-tuned CLIP towers.
    # -----------------------------------------------------------------------
    ckpt_path = os.path.join(CKPT_DIR, f"epoch_{EPOCH:03d}.pt")
    print(f"=== h8 epoch {EPOCH}: {ckpt_path} ===")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}")

    action_horizon = ckpt["mlp_state_dict"]["action_encoder.out.weight"].shape[1] // 64
    print(f"  Detected action horizon: {action_horizon}")
    assert action_horizon == HORIZON, f"expected horizon {HORIZON}, checkpoint has {action_horizon}"

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model = LatentDynamicsModel(horizon=action_horizon).to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_sem = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)   # (768,)
        img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
        z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)     # (768,)

        expert_chunk_norm = normalize_chunk(expert_chunk_raw, action_lo, action_hi)
        expert_chunk_t = torch.tensor(expert_chunk_norm, device=DEVICE).unsqueeze(0)  # (1,8,2)
        z_pred_expert = model(z_t.unsqueeze(0), expert_chunk_t)                        # (1,768)
        expert_cos_sim = float((z_pred_expert * z_sem).sum(dim=-1).item())
    print(f"  expert chunk: cos_sim={expert_cos_sim:.4f}\n")

    def dynamics_fn(state, action):
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        return 1.0 - (terminal_state * z_sem).sum(dim=-1)

    # -----------------------------------------------------------------------
    # 6. 5 independent single-shot MPPI passes.
    # -----------------------------------------------------------------------
    print(f"=== {N_MPPI_RUNS} independent MPPI planning passes ===")
    run_chunks_raw, run_cos_sim, run_min_dist, run_success = [], [], [], []
    for run_idx in range(N_MPPI_RUNS):
        mppi = MPPI(
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            pred_horizon=action_horizon,
            action_dim=2,
            num_samples=NUM_SAMPLES,
            mppi_temperature=MPPI_TEMPERATURE,
            n_planning_itrs=N_PLANNING_ITRS,
            device=DEVICE,
            max_action_value=MAX_ACTION_VALUE,
            warm_start_std=WARM_START_STD,
        )
        torch.manual_seed(SAMPLE_SEEDS[run_idx])
        with torch.no_grad():
            best_action_chunk = mppi.command(z_t)

        final_returns = mppi.last_returns
        best_idx = int(torch.argmax(final_returns))
        assert mppi.last_actions[best_idx].allclose(best_action_chunk)
        cos_sim = float(final_returns[best_idx].item() + 1.0)

        chunk_raw = denormalize_chunk(best_action_chunk.cpu().numpy(), action_lo, action_hi)
        disp = chunk_raw.sum(axis=0)
        disp_toward_blue = float(np.dot(disp, toward_blue_direction))
        print(f"  run {run_idx + 1} (seed={SAMPLE_SEEDS[run_idx]})  cos_sim={cos_sim:.4f}  "
              f"net_disp=({disp[0]:+.4f},{disp[1]:+.4f})  disp.toward_blue={disp_toward_blue:+.4f}")

        # Physically execute this winner from the identical staged starting state (resets via
        # set_pybullet_state inside the helper) and save the resulting frame + a rollout video
        # -- this is now done for every MPPI winner too, not just the expert chunk, so
        # effectiveness can be compared visually.
        run_frame, run_dist, run_ok, _, *_, run_video_frames = _execute_chunk_and_capture(
            chunk_raw, f"mppi_run{run_idx + 1}"
        )
        outcome = "SUCCESS" if run_ok else "FAILURE"
        name_stem = (f"{run_idx + 1:02d}_mppi_run{run_idx + 1}_seed{SAMPLE_SEEDS[run_idx]}_"
                     f"cos{cos_sim:.4f}_dist{run_dist:.4f}_{outcome}")
        run_frame.save(os.path.join(OUT_DIR, f"frame_{name_stem}.png"))
        video_path = os.path.join(OUT_DIR, f"video_{name_stem}.mp4")
        imageio.mimsave(video_path, run_video_frames, fps=VIDEO_FPS)
        print(f"    physical: min_dist={run_dist:.4f} -> {outcome}  "
              f"saved -> frame_{name_stem}.png, video_{name_stem}.mp4")

        run_chunks_raw.append(chunk_raw)
        run_cos_sim.append(cos_sim)
        run_min_dist.append(run_dist)
        run_success.append(run_ok)

    run_chunks_raw = np.stack(run_chunks_raw, axis=0)  # (5, 8, 2)
    run_cos_sim = np.array(run_cos_sim)
    run_min_dist = np.array(run_min_dist)
    run_success = np.array(run_success)
    print(f"\n  expert cos_sim={expert_cos_sim:.4f}  vs.  MPPI winners "
          f"min/mean/max={run_cos_sim.min():.4f}/{run_cos_sim.mean():.4f}/{run_cos_sim.max():.4f}\n")

    del clip_model, model
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # 7. Physically execute the expert chunk from the identical staged starting state.
    # -----------------------------------------------------------------------
    print("=== Physically executing the expert chunk (env.step x 8) ===")
    (expert_frame, min_dist_achieved, expert_success, expert_dist_history,
     expert_touch_frame, expert_touch_step, expert_touch_dist,
     expert_video_frames) = _execute_chunk_and_capture(expert_chunk_raw, "expert")
    # final_dist (state at t=8, i.e. expert_frame itself) is the metric that actually matches
    # what the dynamics model predicts -- it's trained to predict the state AFTER the full
    # 8-step chunk, not at any intermediate step, so cos_sim should be compared against THIS,
    # not min_dist_achieved (the trajectory's closest approach, which can occur mid-chunk and
    # then separate again by the final step -- the tapering push phase above was tuned
    # specifically so these two now coincide).
    final_dist = expert_dist_history[-1]
    final_touching = final_dist <= TOUCH_THRESH
    expert_outcome = "SUCCESS" if final_touching else "FAILURE"
    expert_name_stem = f"{N_MPPI_RUNS + 1:02d}_expert_cos{expert_cos_sim:.4f}_finaldist{final_dist:.4f}_{expert_outcome}"
    expert_frame_name = f"frame_{expert_name_stem}.png"
    expert_frame.save(os.path.join(OUT_DIR, expert_frame_name))
    expert_video_path = os.path.join(OUT_DIR, f"video_{expert_name_stem}.mp4")
    imageio.mimsave(expert_video_path, expert_video_frames, fps=VIDEO_FPS)
    print(f"  final dist (t=8)={final_dist:.4f}  min dist (any step)={min_dist_achieved:.4f}  "
          f"-> {expert_outcome} at t=8 (touch thresh {TOUCH_THRESH})  "
          f"saved -> {expert_frame_name}, video_{expert_name_stem}.mp4\n")

    # The moment the blocks are ACTUALLY touching during the expert rollout (first step where
    # dist <= TOUCH_THRESH) -- distinct from the final frame above, since the push can overshoot
    # past that point.
    expert_touch_frame_name = None
    if expert_touch_frame is not None:
        expert_touch_frame_name = (
            f"frame_{N_MPPI_RUNS + 1:02d}b_expert_first_touch_step{expert_touch_step:02d}_"
            f"dist{expert_touch_dist:.4f}.png"
        )
        expert_touch_frame.save(os.path.join(OUT_DIR, expert_touch_frame_name))
        print(f"  first touching at step {expert_touch_step} (dist={expert_touch_dist:.4f}) "
              f"-> saved {expert_touch_frame_name}\n")
    else:
        print("  expert chunk never reached touching range -- no first-touch frame to save\n")

    # -----------------------------------------------------------------------
    # 8. Chart.
    # -----------------------------------------------------------------------
    all_axis_points = [np.zeros((1, 2)), green_rel.reshape(1, 2), blue_rel.reshape(1, 2),
                        cumulative_path(expert_chunk_raw)]
    for i in range(N_MPPI_RUNS):
        all_axis_points.append(cumulative_path(run_chunks_raw[i]))
    axis_half = float(np.abs(np.concatenate(all_axis_points, axis=0)).max() * 1.2)
    axis_lim = (-axis_half, axis_half)

    fig, ax = plt.subplots(figsize=(9, 9))

    for run_idx in range(N_MPPI_RUNS):
        path = cumulative_path(run_chunks_raw[run_idx])
        color = RUN_COLORS[run_idx]
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0,
                label=f"run {run_idx + 1}  cos_sim={run_cos_sim[run_idx]:.4f}", zorder=3 + run_idx)
        ax.annotate("", xy=tuple(path[-1]), xytext=tuple(path[-2]),
                    arrowprops=dict(facecolor=color, edgecolor=color, width=1.4, headwidth=8),
                    zorder=4 + run_idx)

    expert_path = cumulative_path(expert_chunk_raw)
    expert_label = (f"expert (detour+push)  cos_sim={expert_cos_sim:.4f}  "
                     f"[t=8 {'TOUCHING' if final_touching else 'not touching'}, "
                     f"final_dist={final_dist:.4f}]")
    ax.plot(expert_path[:, 0], expert_path[:, 1], color=REF_COLOR, linewidth=3.0, linestyle="--",
             label=expert_label, zorder=3 + N_MPPI_RUNS)
    ax.annotate("", xy=tuple(expert_path[-1]), xytext=tuple(expert_path[-2]),
                arrowprops=dict(facecolor=REF_COLOR, edgecolor=REF_COLOR, width=1.8, headwidth=10),
                zorder=4 + N_MPPI_RUNS)
    ax.text(expert_path[-1, 0], expert_path[-1, 1], "  expert", color=REF_COLOR, fontsize=11,
             fontweight="bold", va="center", zorder=4 + N_MPPI_RUNS)

    draw_block_icons(ax, green_rel, blue_rel)
    ax.set_xlim(axis_lim); ax.set_ylim(axis_lim); ax.set_aspect("equal")
    ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("net displacement dx", color=INK)
    ax.set_ylabel("net displacement dy", color=INK)
    ax.set_title(
        f"expert+subopt_h8 epoch{EPOCH} (horizon={HORIZON}): obstacle-detour expert-chunk probe\n"
        f"dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}  {N_MPPI_RUNS} MPPI passes, "
        f"temp={MPPI_TEMPERATURE}, K={NUM_SAMPLES}, {N_PLANNING_ITRS} planning iters\n"
        f"dashed = hand-crafted physically-correct detour+push reference",
        color=INK,
    )
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"epoch{EPOCH}_expert_probe.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved chart -> {plot_path}")

    # -----------------------------------------------------------------------
    # 9. Save data + summary.
    # -----------------------------------------------------------------------
    data_path = os.path.join(OUT_DIR, f"epoch{EPOCH}_expert_probe_data.npz")
    np.savez(
        data_path,
        run_chunks_raw=run_chunks_raw, run_cos_sim=run_cos_sim,
        run_min_dist=run_min_dist, run_success=run_success,
        expert_chunk_raw=expert_chunk_raw, expert_cos_sim=expert_cos_sim,
        expert_final_touching=final_touching, expert_final_dist=final_dist,
        expert_min_dist=min_dist_achieved,
        expert_dist_history=np.array(expert_dist_history),
        expert_first_touch_step=(expert_touch_step if expert_touch_step is not None else -1),
        expert_first_touch_dist=(expert_touch_dist if expert_touch_dist is not None else np.nan),
        green_xy=green_xy_after, blue_xy=blue_xy_after, peg_xy=peg_xy_after,
        dist_green_blue_start=dist_green_blue,
        action_horizon=action_horizon, epoch=EPOCH,
        mppi_temperature=MPPI_TEMPERATURE, num_samples=NUM_SAMPLES,
        n_planning_itrs=N_PLANNING_ITRS, sample_seeds=np.array(SAMPLE_SEEDS),
    )
    print(f"Saved data -> {data_path}")

    lines = [
        "Obstacle-detour expert-chunk probe (h8): peg starts BETWEEN green_cube and blue_moon "
        "(relocated closer together), nearly touching green_cube.",
        f"goal: {goal_text}",
        f"peg={peg_xy_after}  green_cube={green_xy_after}  blue_moon={blue_xy_after}  "
        f"dist(green_cube,blue_moon)={dist_green_blue:.4f} (target {NEW_BLOCK_DIST})",
        f"mppi_temperature={MPPI_TEMPERATURE}  n_planning_itrs={N_PLANNING_ITRS}  "
        f"num_samples={NUM_SAMPLES}  seeds={SAMPLE_SEEDS}",
        "cos_sim for every chunk (5 MPPI winners + expert) is the model's OWN predicted "
        "outcome. Every chunk (not just the expert) is ALSO physically executed once from the "
        "identical staged starting state, so cos_sim can be compared against real task "
        "success -- one PNG of the resulting frame is saved per chunk, named "
        "frame_<idx>_..._<cos_sim>_<min_dist>_<SUCCESS|FAILURE>.png.",
        "",
        f"expert chunk: cos_sim={expert_cos_sim:.4f}  "
        f"t=8 (final) result={'TOUCHING' if final_touching else 'NOT TOUCHING'}  "
        f"final_dist={final_dist:.4f}  min_dist_any_step={min_dist_achieved:.4f}  "
        f"(touch thresh {TOUCH_THRESH})  -- final_dist is what matters, since the model only "
        f"predicts the state after the full 8-step chunk",
        (f"  first touching at step {expert_touch_step} (dist={expert_touch_dist:.4f}) -> "
         f"{expert_touch_frame_name}" if expert_touch_frame is not None
         else "  never reached touching range"),
        "",
    ]
    for i in range(N_MPPI_RUNS):
        lines.append(
            f"  run {i + 1} (seed={SAMPLE_SEEDS[i]}): cos_sim={run_cos_sim[i]:.4f}  "
            f"physical_result={'SUCCESS' if run_success[i] else 'FAILURE'}  "
            f"min_dist={run_min_dist[i]:.4f}"
        )
    lines.append(f"  mean cos_sim={run_cos_sim.mean():.4f}  max cos_sim={run_cos_sim.max():.4f}  "
                 f"physical successes={int(run_success.sum())}/{N_MPPI_RUNS}")
    lines.append("")
    lines.append(
        f"expert cos_sim {'>' if expert_cos_sim > run_cos_sim.max() else '<='} "
        f"max MPPI winner cos_sim "
        f"({'model correctly rates the physically-correct chunk highest' if expert_cos_sim > run_cos_sim.max() else 'model does NOT rate the physically-correct chunk above its own MPPI winners'})"
    )

    summary_txt = "\n".join(lines)
    print(summary_txt)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")
    print(f"\nDone. Outputs under {OUT_DIR}")


if __name__ == "__main__":
    main()
