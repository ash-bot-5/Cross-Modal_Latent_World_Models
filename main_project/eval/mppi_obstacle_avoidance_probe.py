"""
Obstacle-avoidance probe: the peg starts BETWEEN green_cube and blue_moon, nearly touching
green_cube -- the mirror image of every other fixed-scene probe in this repo
(mppi_global_or_local_min.py, mppi_rollout_5_per_checkpoint.py, mppi_temp_samples_sweep.py),
which all place the peg on the FAR side of green_cube from blue_moon so a straight push is
always correct. Here, a straight push toward blue_moon just drives the peg past/into
green_cube without ever pushing it the right way -- the only way to actually get green_cube to
blue_moon is for the peg to first move to green_cube's FAR side (away from blue_moon) and push
from there, i.e. route AROUND the block. This tests whether the horizon-16 model's longer
lookahead (vs. horizon-8) is enough to plan that detour instead of a myopic straight push.

Peg placement: same two-stage staged-teleport technique as mppi_global_or_local_min.py
(park clear on the green_cube/blue_moon axis, then move inward), see
[[project_langtable_peg_teleport_staged]], just with the direction mirrored --
peg_xy = green_xy + toward_blue_direction * offset instead of away_from_blue_direction.
PEG_START_OFFSET = 0.04 (closer than the 0.045 "just inside touching" convention used
elsewhere, per explicit request -- with only 16 actions of budget, starting further away
burns steps on approach before any detour/push can happen. 0.04 is the closest offset the
peg-teleport memory's own measurements verify stays safe under staging -- green_cube stays
put to <1e-4 "for every offset tested down to 0.040"; going tighter risks actually pressing
the block during setup itself, contaminating the scene before any planning starts).

Runs BOTH models on the identical obstacle scene for a direct comparison:
  - checkpoints_clip_full_fine-tuned_with_subopt_data (h8), epoch 8
  - checkpoints_clip_full_fine-tuned_with_subopt_data_h16 (h16), epoch 10
5 independent seeded MPPI passes per model (fresh instance per seed, so every run starts from
an identical Uniform(-1,1) initial population -- same rationale as
mppi_global_or_local_min.py), mppi_temperature=0.03, n_planning_itrs=1000, num_samples=2000 --
the single configuration requested, not a sweep.

Like every other probe this session settled on: NO real sim execution. cos_sim is the
dynamics model's own predicted outcome (cost_fn = 1 - cos(model(state,action), z_sem)); this
script only visualizes the PLANNED chunk's shape (env.step() is never called), which is what
actually answers "does the plan curve around the block" -- no execution needed to see that.

Chart: the peg's start is the origin (0,0), matching cumulative_path's convention. Both
blocks are drawn as labeled icons at their position relative to the peg's start (not implied
by a direction arrow) -- green_cube as a filled square + dashed TOUCH_THRESH circle, blue_moon
as a filled star -- so a run's path relative to both is legible at a glance.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_obstacle_avoidance_probe.py
"""

import os
import sys
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
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
OUT_DIR_BASE = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "obstacle_avoidance")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = [
    dict(
        tag="expert+subopt",
        ckpt_dir=os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data"),
        epoch=8,
    ),
    dict(
        tag="expert+subopt_h16",
        ckpt_dir=os.path.join(REPO_ROOT, "checkpoints_clip_full_fine-tuned_with_subopt_data_h16"),
        epoch=10,
    ),
]

# The single configuration requested -- not a sweep.
MPPI_TEMPERATURE = 0.03
N_PLANNING_ITRS = 1000   # "1000 planning iterations", taken literally (this script builds
                         # MPPI directly rather than going through an --evals CLI flag, so
                         # there's no evals-vs-n_planning_itrs off-by-one to resolve here).
NUM_SAMPLES = 2000
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1    # passed for API completeness; never exercised since each run is a
                         # single command() call on a fresh instance (prev_best_action=None).

N_MPPI_RUNS = 5
SAMPLE_SEED = 123
SAMPLE_SEEDS = [SAMPLE_SEED + i for i in range(N_MPPI_RUNS)]

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])
TOUCH_THRESH = 0.05
PEG_PARK_OFFSET = 0.10
PEG_START_OFFSET = 0.04

# Validated categorical palette (dataviz skill, checked via scripts/validate_palette.js,
# ALL CHECKS PASS -- see mppi_temp_samples_sweep.py for the failing-tab10 comparison) for the
# 5 per-seed run paths.
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#333333"
INK_MUTED = "#888888"

# Block icon colors -- deliberately muted/desaturated relative to RUN_COLORS (which are the
# vivid foreground data series) so a filled block icon never reads as "another run's path".
# Chosen to still evoke the blocks' real colors (green_cube / blue_moon) since these are
# pictograms of physical objects, not a categorical data series.
ICON_GREEN = "#3a7d3a"
ICON_BLUE = "#4d6f99"


# ---------------------------------------------------------------------------
# Inline model classes -- horizon-parametrized (mppi_temp_samples_sweep.py's pattern).
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


def draw_block_icons(ax, green_rel: np.ndarray, blue_rel: np.ndarray):
    """Both blocks as labeled icons at their position relative to the peg's start (the plot
    origin) -- not implied by a direction arrow. green_cube: filled square + dashed
    TOUCH_THRESH circle (the "nearly touching" boundary this probe starts inside of).
    blue_moon: filled star at its exact position."""
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


DEFAULT_GOAL_TEXT = (
    f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
    f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--goal_text", type=str, default=None,
                        help=f"goal sentence fed to the text tower. Default (unflagged): "
                             f"{DEFAULT_GOAL_TEXT!r}")
    parser.add_argument("--tag", type=str, default=None,
                        help="output subdirectory suffix, e.g. --tag not_touching writes to "
                             "obstacle_avoidance_not_touching/ instead of obstacle_avoidance/. "
                             "REQUIRED when --goal_text differs from the default, so a "
                             "different-goal run never silently overwrites the default run's "
                             "outputs.")
    args = parser.parse_args()

    goal_text = args.goal_text if args.goal_text is not None else DEFAULT_GOAL_TEXT
    if goal_text != DEFAULT_GOAL_TEXT and not args.tag:
        parser.error(
            "--goal_text differs from the default goal text, which requires --tag -- "
            "without it this run would overwrite the default run's outputs under "
            f"{OUT_DIR_BASE}/."
        )
    out_dir = OUT_DIR_BASE if not args.tag else f"{OUT_DIR_BASE}_{args.tag}"

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Action normalization stats -- shared by both models.
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
    # 2. Fixed env + OBSTACLE peg placement (peg BETWEEN the blocks, mirrored two-stage
    #    teleport). One env for the whole script -- no rollout, no repeated resets needed;
    #    the same static frame is reused (re-encoded through each model's own CLIP tower).
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
    blue_xy = blocks[BLOCK_B]

    assert np.allclose(green_xy, EXPECTED_GREEN_XY, atol=1e-5), (
        f"{BLOCK_A} position {green_xy} != expected {EXPECTED_GREEN_XY} -- table arrangement diverged"
    )
    assert np.allclose(blue_xy, EXPECTED_BLUE_XY, atol=1e-5), (
        f"{BLOCK_B} position {blue_xy} != expected {EXPECTED_BLUE_XY} -- table arrangement diverged"
    )

    away_from_blue_direction = (green_xy - blue_xy) / np.linalg.norm(green_xy - blue_xy)
    toward_blue_direction = -away_from_blue_direction

    def _move_peg_to(xy):
        inner._set_robot_target_effector_pose(Pose3d(
            rotation=R.from_rotvec([0, np.pi, 0]),
            translation=np.array([xy[0], xy[1], 0.145]),
        ))
        inner._step_simulation_to_stabilize()

    park_xy = green_xy + toward_blue_direction * PEG_PARK_OFFSET     # stage 1: park clear,
                                                                       # between the blocks
    final_xy = green_xy + toward_blue_direction * PEG_START_OFFSET   # stage 2: move to nearly
                                                                       # touching green_cube,
                                                                       # still between them
    _move_peg_to(park_xy)
    _move_peg_to(final_xy)

    # Verify the REALIZED geometry (IK settles a fraction of a mm short of target), not just
    # the commanded pose -- same discipline as every other staged-teleport script.
    blocks_after = inner.get_block_states()
    green_xy_after = blocks_after[BLOCK_A]
    blue_xy_after = blocks_after[BLOCK_B]
    peg_xy_after = np.asarray(inner._robot.forward_kinematics().translation[:2], dtype=np.float64)
    dist_peg_green = float(np.linalg.norm(peg_xy_after - green_xy_after))
    dist_peg_blue = float(np.linalg.norm(peg_xy_after - blue_xy_after))
    dist_green_blue = float(np.linalg.norm(green_xy_after - blue_xy_after))
    green_moved = float(np.linalg.norm(green_xy_after - green_xy))

    print(f"Peg start (realized): {peg_xy_after}  dist(peg,{BLOCK_A})={dist_peg_green:.4f}  "
          f"dist(peg,{BLOCK_B})={dist_peg_blue:.4f}  dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}")
    print(f"  {BLOCK_A} moved {green_moved:.6f} during peg placement\n")

    assert dist_peg_green < TOUCH_THRESH, (
        f"realized peg is {dist_peg_green:.4f} from {BLOCK_A}, not < {TOUCH_THRESH} -- "
        f"peg is not nearly touching {BLOCK_A}"
    )
    assert dist_peg_blue < dist_green_blue, (
        f"realized peg is {dist_peg_blue:.4f} from {BLOCK_B}, but dist({BLOCK_A},{BLOCK_B})="
        f"{dist_green_blue:.4f} -- peg is NOT between the blocks (it's further from {BLOCK_B} "
        f"than {BLOCK_A} itself is, i.e. past {BLOCK_A} rather than between them)"
    )
    print(f"Obstacle geometry verified: peg is between {BLOCK_A} and {BLOCK_B}, "
          f"nearly touching {BLOCK_A}.\n")

    pil_frame = env.get_frame()
    frame_path = os.path.join(out_dir, "obstacle_frame.png")
    pil_frame.save(frame_path)
    print(f"Saved probe frame -> {frame_path}\n")

    green_rel = green_xy_after - peg_xy_after
    blue_rel = blue_xy_after - peg_xy_after

    print(f"Goal: {goal_text}\n")

    def cost_fn_factory(z_sem):
        def cost_fn(terminal_state):
            return 1.0 - (terminal_state * z_sem).sum(dim=-1)
        return cost_fn

    # -----------------------------------------------------------------------
    # 3. Per-model probe.
    # -----------------------------------------------------------------------
    all_axis_points = [np.zeros((1, 2)), green_rel.reshape(1, 2), blue_rel.reshape(1, 2)]
    model_results = {}

    for m in MODELS:
        tag, ckpt_dir, epoch = m["tag"], m["ckpt_dir"], m["epoch"]
        ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
        print(f"=== {tag} epoch {epoch}: {ckpt_path} ===")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}")

        action_horizon = ckpt["mlp_state_dict"]["action_encoder.out.weight"].shape[1] // 64
        print(f"  Detected action horizon: {action_horizon}")

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

        def dynamics_fn(state, action):
            with torch.no_grad():
                return model(state, action)

        cost_fn = cost_fn_factory(z_sem)

        run_chunks_raw, run_cos_sim, run_disp_toward_blue = [], [], []
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
                collect_diagnostics=True,
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
                  f"net_disp=({disp[0]:+.4f},{disp[1]:+.4f})  "
                  f"disp.toward_blue={disp_toward_blue:+.4f}  "
                  f"({'toward/through blue_moon -- no detour' if disp_toward_blue > 0 else 'back toward far side of green_cube -- detour starting'})")

            run_chunks_raw.append(chunk_raw)
            run_cos_sim.append(cos_sim)
            run_disp_toward_blue.append(disp_toward_blue)
            all_axis_points.append(cumulative_path(chunk_raw))

        model_results[tag] = dict(
            epoch=epoch, action_horizon=action_horizon,
            run_chunks_raw=np.stack(run_chunks_raw, axis=0),
            run_cos_sim=np.array(run_cos_sim),
            run_disp_toward_blue=np.array(run_disp_toward_blue),
        )
        del clip_model, model
        torch.cuda.empty_cache()
        print()

    # -----------------------------------------------------------------------
    # 4. Charts -- one per model, plus a combined side-by-side figure, all sharing one axis
    #    range (computed across BOTH models' paths + both block icon positions) so they're
    #    directly comparable.
    # -----------------------------------------------------------------------
    axis_half = float(np.abs(np.concatenate(all_axis_points, axis=0)).max() * 1.2)
    axis_lim = (-axis_half, axis_half)
    print(f"Shared plot scale: axis_lim=[{-axis_half:.4f}, {axis_half:.4f}]\n")

    def plot_model_paths(ax, tag: str, r: dict):
        for run_idx in range(N_MPPI_RUNS):
            path = cumulative_path(r["run_chunks_raw"][run_idx])
            color = RUN_COLORS[run_idx]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0,
                    label=f"run {run_idx + 1}  cos_sim={r['run_cos_sim'][run_idx]:.4f}",
                    zorder=3 + run_idx)
            ax.annotate("", xy=tuple(path[-1]), xytext=tuple(path[-2]),
                        arrowprops=dict(facecolor=color, edgecolor=color, width=1.4, headwidth=8),
                        zorder=4 + run_idx)
        draw_block_icons(ax, green_rel, blue_rel)
        ax.set_xlim(axis_lim); ax.set_ylim(axis_lim); ax.set_aspect("equal")
        ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("net displacement dx", color=INK)
        ax.set_ylabel("net displacement dy", color=INK)
        ax.set_title(
            f"{tag}  epoch{r['epoch']}  (horizon={r['action_horizon']})\n"
            f"{N_MPPI_RUNS} planned chunks, temp={MPPI_TEMPERATURE}, "
            f"K={NUM_SAMPLES}, {N_PLANNING_ITRS} planning iters",
            color=INK,
        )
        ax.legend(loc="upper left", fontsize=8)

    for tag, r in model_results.items():
        fig, ax = plt.subplots(figsize=(9, 9))
        plot_model_paths(ax, tag, r)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"{tag}_epoch{r['epoch']}_obstacle_probe.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        np.savez(
            os.path.join(out_dir, f"{tag}_epoch{r['epoch']}_obstacle_probe_data.npz"),
            run_chunks_raw=r["run_chunks_raw"], run_cos_sim=r["run_cos_sim"],
            run_disp_toward_blue=r["run_disp_toward_blue"],
            green_xy=green_xy_after, blue_xy=blue_xy_after, peg_xy=peg_xy_after,
            action_horizon=r["action_horizon"], epoch=r["epoch"],
            mppi_temperature=MPPI_TEMPERATURE, num_samples=NUM_SAMPLES,
            n_planning_itrs=N_PLANNING_ITRS, sample_seeds=np.array(SAMPLE_SEEDS),
        )
        print(f"Saved {tag} -> {plot_path}")

    fig, axes = plt.subplots(1, 2, figsize=(17, 9))
    for ax, (tag, r) in zip(axes, model_results.items()):
        plot_model_paths(ax, tag, r)
    plt.tight_layout()
    combined_path = os.path.join(out_dir, "combined_h8_vs_h16_obstacle_probe.png")
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"Saved combined comparison -> {combined_path}\n")

    # -----------------------------------------------------------------------
    # 5. summary.txt
    # -----------------------------------------------------------------------
    lines = [
        "Obstacle-avoidance probe: peg starts BETWEEN green_cube and blue_moon, nearly "
        "touching green_cube.",
        f"goal: {goal_text}",
        f"peg={peg_xy_after}  green_cube={green_xy_after}  blue_moon={blue_xy_after}  "
        f"dist(peg,green_cube)={dist_peg_green:.4f}  dist(green_cube,blue_moon)={dist_green_blue:.4f}",
        f"mppi_temperature={MPPI_TEMPERATURE}  n_planning_itrs={N_PLANNING_ITRS}  "
        f"num_samples={NUM_SAMPLES}  seeds={SAMPLE_SEEDS}",
        "cos_sim is the dynamics model's OWN predicted outcome -- no env.step() is ever called.",
        "disp.toward_blue > 0 means the planned chunk drives further toward/through blue_moon "
        "(no detour); < 0 means it moves back toward the far side of green_cube (a detour "
        "starting). Descriptive only, no pass/fail threshold asserted.",
        "",
    ]
    for tag, r in model_results.items():
        lines.append(f"--- {tag} epoch{r['epoch']} (horizon={r['action_horizon']}) ---")
        for i in range(N_MPPI_RUNS):
            lines.append(
                f"  run {i + 1} (seed={SAMPLE_SEEDS[i]}): cos_sim={r['run_cos_sim'][i]:.4f}  "
                f"disp.toward_blue={r['run_disp_toward_blue'][i]:+.4f}"
            )
        lines.append(f"  mean cos_sim={r['run_cos_sim'].mean():.4f}  "
                     f"mean disp.toward_blue={r['run_disp_toward_blue'].mean():+.4f}")
        lines.append("")

    summary_txt = "\n".join(lines)
    print(summary_txt)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")
    print(f"Done. Outputs under {out_dir}")


if __name__ == "__main__":
    main()
