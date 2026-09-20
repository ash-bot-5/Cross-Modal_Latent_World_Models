"""
Softmax-temperature sweep OR num_samples (K) sweep on a fixed single-frame MPPI probe, for any
checkpoint -- generalized from mppi_temperature_sweep.py (the "inspiration script") to also
target the h16 checkpoints (16-step action chunks) rather than only the hardcoded h8/epoch11
default.

Two things changed relative to the inspiration script, both per explicit instruction:

  1. --sweep {temperature,num_samples} selects ONE parameter to vary; the other is held fixed
     at a flag-overridable default. This is NOT a crossed 2D grid -- run the script twice (once
     per --sweep value) to get both sweeps, each with its own output subdirectory.
  2. The inspiration script's baseline-reproduction machinery is dropped entirely. It hinges on
     ref_chunk_raw, a hand-derived "near-perfect" (8, 2) action chunk and a cached
     baseline_run_cos_sim, BOTH specific to the h8/epoch11 checkpoint's 8-step action space --
     neither means anything for a checkpoint with a different action_horizon (e.g. 16). No
     ref_chunk_raw, no ref_cos_sim, no vs_ref column, no assert-against-cached-baseline. What IS
     still reused from the inspiration script's cached artifacts, because it's pure scene
     geometry independent of any checkpoint: first_frame.png (a rendered RGB frame) and
     green_xy/blue_xy/peg_xy/correct_direction from its epoch11_mppi_5runs_data.npz (the fixed
     block/peg positions in that one frame). In place of the dropped numeric reproduction check,
     a lightweight sanity print (detected action_horizon, z_t shape, checkpoint epoch/step) still
     surfaces a broken load immediately.

Like the inspiration script: no LangTable env, no pybullet, no env.step() anywhere -- cos_sim is
the dynamics model's OWN prediction (cost_fn = 1 - cos(model(state, action), z_sem)), graded
against its own imagined outcome, never checked against a real rollout. That is deliberate here,
matching the existing codebase convention for this kind of probe.

action_horizon is auto-detected per checkpoint from action_encoder.out.weight's shape (same
one-liner mppi_rollout_random_scenes.py uses), so this runs both h8 and h16 checkpoints
unmodified.

Default checkpoint: main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data_h16,
epoch 4 (the "h16/epoch4" model from the prior random-scenes comparison).

Run (two separate invocations -- NOT a crossed grid):
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_temp_samples_sweep.py --sweep temperature
    python main_project/eval/mppi_temp_samples_sweep.py --sweep num_samples
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
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.mppi_core import MPPI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Probe artifacts (first_frame.png + geometry) are reused read-only from the inspiration
# script's own baseline directory -- pure scene geometry, independent of checkpoint.
PROBE_DATA_DIR = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "min_determination", "expert+subopt-model")
PROBE_EPOCH_FOR_GEOMETRY = 11   # any epoch's npz has identical geometry fields; this one is what
                                # mppi_temperature_sweep.py already reads by default

DEFAULT_CKPT_DIR = os.path.join(
    MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data_h16"
)
DEFAULT_EPOCH = 4
OUT_ROOT = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "min_determination", "expert+subopt_h16")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_TEMPERATURES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
DEFAULT_NUM_SAMPLES_GRID = [512, 1000, 2000, 4000]
FIXED_TEMPERATURE_DEFAULT = 0.01   # held fixed when --sweep num_samples (the inspiration
                                   # script's own BASELINE_TEMPERATURE)
FIXED_NUM_SAMPLES_DEFAULT = 512    # held fixed when --sweep temperature (the inspiration
                                   # script's own pinned K)

EVALS_DEFAULT = 10   # total dynamics evaluations per planning call = n_planning_itrs + 1, same
                     # meaning as mppi_rollout_random_scenes.py's --evals (n_planning_itrs =
                     # evals - 1); default 10 reproduces this script's original hardcoded
                     # N_PLANNING_ITRS=9.
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1

N_MPPI_RUNS_DEFAULT = 5
SAMPLE_SEED = 123

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])
TOUCH_THRESH = 0.05

# Validated categorical palette (dataviz skill references/palette.md, light mode, slots 1-5:
# blue/orange/aqua/yellow/magenta) -- checked via scripts/validate_palette.js, ALL CHECKS PASS
# (tab10's default green/orange pair FAILs CVD separation, ΔE 0.7 protan). Contrast-vs-surface
# WARNs on 3 of the 5 are mitigated by this chart's existing direct labels + legend.
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
RUN_MARKERS = ["o", "s", "^", "D", "v"]
INK = "#333333"
INK_MUTED = "#888888"


# ---------------------------------------------------------------------------
# Inline model classes -- horizon-parametrized (mppi_rollout_random_scenes.py's pattern),
# unlike the inspiration script's hardcoded-8 copy.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR,
                        help=f"checkpoint directory (default: h16, {DEFAULT_CKPT_DIR})")
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                        help=f"checkpoint epoch to probe (default {DEFAULT_EPOCH})")
    parser.add_argument("--sweep", type=str, choices=["temperature", "num_samples"], required=True,
                        help="which parameter to sweep; the other is held fixed (see "
                             "--fixed_temperature / --fixed_num_samples). NOT a crossed grid -- "
                             "run this script twice, once per value, for both sweeps.")
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES,
                        help="mppi_temperature grid, used when --sweep temperature")
    parser.add_argument("--num_samples_grid", type=int, nargs="+", default=DEFAULT_NUM_SAMPLES_GRID,
                        help="num_samples (K) grid, used when --sweep num_samples")
    parser.add_argument("--fixed_num_samples", type=int, default=FIXED_NUM_SAMPLES_DEFAULT,
                        help=f"K held fixed when --sweep temperature (default "
                             f"{FIXED_NUM_SAMPLES_DEFAULT})")
    parser.add_argument("--fixed_temperature", type=float, default=FIXED_TEMPERATURE_DEFAULT,
                        help=f"mppi_temperature held fixed when --sweep num_samples (default "
                             f"{FIXED_TEMPERATURE_DEFAULT})")
    parser.add_argument("--n_runs", type=int, default=N_MPPI_RUNS_DEFAULT,
                        help=f"independent seeded MPPI passes per arm (default {N_MPPI_RUNS_DEFAULT})")
    parser.add_argument("--evals", type=int, default=EVALS_DEFAULT,
                        help=f"total dynamics evaluations per planning call; the planner gets "
                             f"n_planning_itrs = evals - 1 (default {EVALS_DEFAULT}). Same "
                             f"meaning as mppi_rollout_random_scenes.py's --evals. Dominates "
                             f"wall-clock -- scales linearly with this value.")
    parser.add_argument("--tag", type=str, default=None,
                        help="output subdirectory suffix, e.g. --tag fine writes to "
                             "temp_sweep_fine/ instead of temp_sweep/.")
    args = parser.parse_args()

    if args.n_runs > len(RUN_COLORS):
        parser.error(
            f"--n_runs {args.n_runs} exceeds the validated palette's {len(RUN_COLORS)} "
            f"categorical slots -- add another checked slot to RUN_COLORS rather than "
            f"reusing a color (which would break run/seed color identity)."
        )
    if args.evals < 1:
        parser.error(f"--evals must be >= 1 (1 = initial evaluation only), got {args.evals}")
    n_planning_itrs = args.evals - 1

    epoch = args.epoch
    sample_seeds = [SAMPLE_SEED + i for i in range(args.n_runs)]

    sweep_name = "temp_sweep" if args.sweep == "temperature" else "num_samples_sweep"
    out_dir = os.path.join(OUT_ROOT, sweep_name if not args.tag else f"{sweep_name}_{args.tag}")

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("  (CUDA unavailable -- running on CPU; fine for this workload)")
    print(f"Sweep: {args.sweep}")
    print(f"Output dir: {out_dir}")

    # -----------------------------------------------------------------------
    # 1. Fixed probe scene: reuse first_frame.png + geometry, read-only, from the inspiration
    #    script's cached baseline directory. Pure scene geometry, independent of checkpoint --
    #    NOT a numeric-reproduction check (that machinery is h8/epoch11-specific and dropped).
    # -----------------------------------------------------------------------
    frame_path = os.path.join(PROBE_DATA_DIR, "first_frame.png")
    npz_path = os.path.join(PROBE_DATA_DIR, f"epoch{PROBE_EPOCH_FOR_GEOMETRY}_mppi_5runs_data.npz")
    for path in (frame_path, npz_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. It is produced by mppi_global_or_local_min.py -- re-run\n"
                f"  python main_project/eval/mppi_global_or_local_min.py\n"
                f"to regenerate the probe artifacts, then re-run this sweep."
            )

    geom = np.load(npz_path)
    npz_green = geom["green_xy"]
    npz_blue = geom["blue_xy"]
    npz_peg = geom["peg_xy"]
    correct_direction = geom["correct_direction"]

    assert np.allclose(npz_green, EXPECTED_GREEN_XY, atol=1e-5), (
        f"cached {BLOCK_A} {npz_green} != expected {EXPECTED_GREEN_XY} -- probe geometry diverged"
    )
    assert np.allclose(npz_blue, EXPECTED_BLUE_XY, atol=1e-5), (
        f"cached {BLOCK_B} {npz_blue} != expected {EXPECTED_BLUE_XY} -- probe geometry diverged"
    )
    dist_peg_green = float(np.linalg.norm(npz_peg - npz_green))
    assert dist_peg_green < TOUCH_THRESH, (
        f"cached peg is {dist_peg_green:.4f} from {BLOCK_A}, not < {TOUCH_THRESH}"
    )
    print(f"Probe geometry verified from {os.path.basename(npz_path)}: "
          f"{BLOCK_A}={npz_green}, {BLOCK_B}={npz_blue}, "
          f"dist(peg,{BLOCK_A})={dist_peg_green:.4f} (touching)")

    # -----------------------------------------------------------------------
    # 2. Action normalization stats.
    # -----------------------------------------------------------------------
    print(f"\nLoading action stats from {TRAIN_DIR} ...")
    all_actions = []
    for pkl_file in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl")):
        with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
            ep = pickle.load(f)
        all_actions.append(np.array(ep["actions"], dtype=np.float32))
    all_actions = np.concatenate(all_actions, axis=0)
    action_lo = all_actions.min(axis=0)
    action_hi = all_actions.max(axis=0)
    print(f"Action normalization stats: lo={action_lo.tolist()} hi={action_hi.tolist()}")

    # -----------------------------------------------------------------------
    # 3. Checkpoint + CLIP + model, loaded ONCE and shared by every arm.
    # -----------------------------------------------------------------------
    ckpt_path = os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pt")
    print(f"\nLoading checkpoint {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}")

    # Action horizon is a property of the checkpoint's own architecture -- detect it from the
    # saved weights (mppi_rollout_random_scenes.py's pattern) instead of assuming 8.
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

    goal_text = (
        f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
        f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
    )
    print(f"Goal: {goal_text}")

    pil_frame = Image.open(frame_path).convert("RGB")
    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_sem = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)   # (768,)

        img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
        z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)     # (768,)
    print(f"z_t shape: {tuple(z_t.shape)}  (sanity check -- no numeric baseline to reproduce "
          f"for this checkpoint, see module docstring)\n")

    # -----------------------------------------------------------------------
    # 4. The sweep. dynamics_fn / cost_fn identical to the inspiration script's -- model-
    #    predicted cos_sim only, no env, no execution.
    # -----------------------------------------------------------------------
    def dynamics_fn(state, action):
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        return 1.0 - (terminal_state * z_sem).sum(dim=-1)

    if args.sweep == "temperature":
        grid = list(args.temperatures)
        fixed_num_samples = args.fixed_num_samples
        print(f"Grid: temperatures={[f'{t:g}' for t in grid]}  (K fixed at {fixed_num_samples})\n")
    else:
        grid = list(args.num_samples_grid)
        fixed_temperature = args.fixed_temperature
        print(f"Grid: num_samples={grid}  (mppi_temperature fixed at {fixed_temperature:g})\n")

    arms = {}
    for value in grid:
        temp = value if args.sweep == "temperature" else args.fixed_temperature
        K = args.fixed_num_samples if args.sweep == "temperature" else value

        print(f"=== {args.sweep}={value:g}: {args.n_runs} independent MPPI passes "
              f"(temp={temp:g}, K={K}) ===")
        run_chunks_raw, run_net_disp, run_cos_sim = [], [], []
        run_final_ess, run_mean_ess, run_final_std = [], [], []

        for run_idx in range(args.n_runs):
            mppi = MPPI(
                dynamics_fn=dynamics_fn,
                cost_fn=cost_fn,
                pred_horizon=action_horizon,
                action_dim=2,
                num_samples=K,
                mppi_temperature=temp,
                n_planning_itrs=n_planning_itrs,
                device=DEVICE,
                max_action_value=MAX_ACTION_VALUE,
                warm_start_std=WARM_START_STD,
                collect_diagnostics=True,
            )

            torch.manual_seed(sample_seeds[run_idx])
            with torch.no_grad():
                best_action_chunk = mppi.command(z_t)

            final_returns = mppi.last_returns
            best_idx = int(torch.argmax(final_returns))
            assert mppi.last_actions[best_idx].allclose(best_action_chunk), (
                "argmax of last_returns does not match mppi.command()'s own selection"
            )
            cos_sim = float(final_returns[best_idx].item() + 1.0)

            chunk_raw = denormalize_chunk(
                best_action_chunk.cpu().numpy(), action_lo, action_hi
            )                                                       # (action_horizon, 2)
            disp = chunk_raw.sum(axis=0)
            mag = float(np.linalg.norm(disp))
            dir_align = float(np.dot(disp / (mag + 1e-8), correct_direction))

            diags = mppi.last_diagnostics
            final_ess = diags[-1]["ess"]
            mean_ess = float(np.mean([d["ess"] for d in diags]))
            final_std = diags[-1]["std_mean"]

            print(f"  run {run_idx + 1} (seed={sample_seeds[run_idx]})  cos_sim={cos_sim:.4f}  "
                  f"net_disp=({disp[0]:+.4f},{disp[1]:+.4f})  |disp|={mag:.4f}  "
                  f"dir_align={dir_align:+.4f}  ESS_final={final_ess:6.2f}  "
                  f"ESS_mean={mean_ess:6.2f}  std_final={final_std:.2e}")

            run_chunks_raw.append(chunk_raw)
            run_net_disp.append(disp)
            run_cos_sim.append(cos_sim)
            run_final_ess.append(final_ess)
            run_mean_ess.append(mean_ess)
            run_final_std.append(final_std)

        run_chunks_raw = np.stack(run_chunks_raw, axis=0)
        run_net_disp = np.stack(run_net_disp, axis=0)
        run_cos_sim = np.array(run_cos_sim)

        pdist = [float(np.linalg.norm(run_net_disp[i] - run_net_disp[j]))
                 for i in range(args.n_runs) for j in range(i + 1, args.n_runs)]
        stats = dict(
            mean=float(run_cos_sim.mean()),
            std=float(run_cos_sim.std(ddof=0)),
            var=float(run_cos_sim.var(ddof=0)),
            best=float(run_cos_sim.max()),
            worst=float(run_cos_sim.min()),
            rng=float(run_cos_sim.max() - run_cos_sim.min()),
            disp_spread=float(np.mean(pdist)) if pdist else 0.0,
            disp_spread_max=float(np.max(pdist)) if pdist else 0.0,
            ess_final=float(np.mean(run_final_ess)),
            ess_mean=float(np.mean(run_mean_ess)),
            std_final=float(np.mean(run_final_std)),
        )
        print(f"  cross-seed: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"var={stats['var']:.6f}  range={stats['rng']:.4f}  "
              f"net-disp spread={stats['disp_spread']:.4f}")

        arms[value] = dict(
            run_chunks_raw=run_chunks_raw,
            run_net_disp=run_net_disp,
            run_cos_sim=run_cos_sim,
            run_final_ess=np.array(run_final_ess),
            run_mean_ess=np.array(run_mean_ess),
            temp=temp, K=K,
            **stats,
        )

    # -----------------------------------------------------------------------
    # 5. Per-arm charts, sharing one axis range and one correct-direction arrow.
    # -----------------------------------------------------------------------
    all_mags = np.array([np.linalg.norm(d) for a in arms.values() for d in a["run_net_disp"]])
    ref_len = float(np.median(all_mags)) if len(all_mags) else 1.0
    arrow_end = correct_direction * ref_len

    all_points = [np.zeros((1, 2)), arrow_end.reshape(1, 2)]
    for a in arms.values():
        for chunk in a["run_chunks_raw"]:
            all_points.append(cumulative_path(chunk))
    axis_half = float(np.abs(np.concatenate(all_points, axis=0)).max() * 1.15)
    axis_lim = (-axis_half, axis_half)
    print(f"\nShared plot scale: ref_len={ref_len:.4f}  axis_lim=[{-axis_half:.4f}, {axis_half:.4f}]")

    os.makedirs(out_dir, exist_ok=True)
    for value, a in arms.items():
        arm_dir = os.path.join(out_dir, f"{args.sweep}_{value:g}")
        os.makedirs(arm_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.annotate("", xy=tuple(arrow_end), xytext=(0, 0),
                    arrowprops=dict(facecolor=INK, edgecolor=INK, width=1.5, headwidth=8), zorder=2)
        ax.text(arrow_end[0], arrow_end[1], "  correct dir", color=INK, fontsize=9,
                va="center", zorder=2)

        for run_idx in range(args.n_runs):
            path = cumulative_path(a["run_chunks_raw"][run_idx])
            color = RUN_COLORS[run_idx]
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0,
                    label=f"run {run_idx + 1}  cos_sim={a['run_cos_sim'][run_idx]:.4f}",
                    zorder=3 + run_idx)
            ax.annotate("", xy=tuple(path[-1]), xytext=tuple(path[-2]),
                        arrowprops=dict(facecolor=color, edgecolor=color, width=1.4, headwidth=8),
                        zorder=4 + run_idx)
            ax.text(path[-1, 0], path[-1, 1], f"  {run_idx + 1}", color=INK, fontsize=10,
                    va="center", zorder=4 + run_idx)

        ax.set_xlim(axis_lim); ax.set_ylim(axis_lim); ax.set_aspect("equal")
        ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("net displacement dx", color=INK)
        ax.set_ylabel("net displacement dy", color=INK)
        ax.set_title(
            f"epoch{epoch} {args.sweep} sweep -- {args.sweep}={value:g}  "
            f"(temp={a['temp']:g}, K={a['K']})\n"
            f"{args.n_runs} independent MPPI passes (seeds {sample_seeds[0]}-{sample_seeds[-1]}), "
            f"{n_planning_itrs} refinement iters\n"
            f"cross-seed std={a['std']:.4f}  range={a['rng']:.4f}  "
            f"mean ESS={a['ess_mean']:.1f}/{a['K']}",
            color=INK,
        )
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(arm_dir, f"epoch{epoch}_{args.n_runs}runs.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        np.savez(
            os.path.join(arm_dir, f"epoch{epoch}_{args.n_runs}runs_data.npz"),
            run_chunks_raw=a["run_chunks_raw"],
            run_net_disp=a["run_net_disp"],
            run_cos_sim=a["run_cos_sim"],
            run_final_ess=a["run_final_ess"],
            run_mean_ess=a["run_mean_ess"],
            correct_direction=correct_direction,
            green_xy=npz_green, blue_xy=npz_blue, peg_xy=npz_peg,
            mppi_temperature=a["temp"], K=a["K"],
            n_planning_itrs=n_planning_itrs, n_mppi_runs=args.n_runs,
            sample_seeds=np.array(sample_seeds), epoch=epoch, action_horizon=action_horizon,
            **{k: a[k] for k in ("mean", "std", "var", "best", "worst", "rng",
                                 "disp_spread", "disp_spread_max", "ess_final", "ess_mean",
                                 "std_final")},
        )
        print(f"Saved arm {args.sweep}={value:g} -> {arm_dir}")

    # -----------------------------------------------------------------------
    # 6. Summary chart: per-seed cos_sim vs the swept parameter.
    # -----------------------------------------------------------------------
    values = np.array(sorted(arms.keys()))
    means = np.array([arms[v]["mean"] for v in values])
    stds = np.array([arms[v]["std"] for v in values])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(values, means - stds, means + stds, color=INK_MUTED, alpha=0.18,
                    label="mean +/- 1 std across seeds", zorder=2)
    ax.plot(values, means, color=INK, linewidth=2.0, marker="o", markersize=5,
            label="mean across seeds", zorder=4)
    for run_idx in range(args.n_runs):
        ys = [arms[v]["run_cos_sim"][run_idx] for v in values]
        ax.plot(values, ys, color=RUN_COLORS[run_idx], marker=RUN_MARKERS[run_idx % len(RUN_MARKERS)],
                markersize=7, linewidth=1.0, alpha=0.85,
                label=f"seed {sample_seeds[run_idx]}", zorder=5 + run_idx)

    fixed_desc = (f"K={args.fixed_num_samples} fixed" if args.sweep == "temperature"
                  else f"mppi_temperature={args.fixed_temperature:g} fixed")
    if args.sweep == "temperature":
        ax.set_xscale("log")
    ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"{'mppi_temperature (log scale)' if args.sweep == 'temperature' else 'num_samples (K)'}",
                  color=INK)
    ax.set_ylabel("cos_sim of selected action chunk (model-predicted)", color=INK)
    ax.set_title(
        f"epoch{epoch} ({os.path.basename(args.ckpt_dir)}): MPPI {args.sweep} sweep vs. "
        f"selected-chunk cos_sim\n"
        f"{args.n_runs} seeds per {args.sweep} value, {fixed_desc}, "
        f"{n_planning_itrs} iters -- higher is better, tighter is more reproducible",
        color=INK,
    )
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    summary_plot = os.path.join(out_dir, f"summary_cos_sim_vs_{args.sweep}.png")
    plt.savefig(summary_plot, dpi=150)
    plt.close()
    print(f"Saved summary chart -> {summary_plot}")

    # -----------------------------------------------------------------------
    # 7. Summary table (stdout + summary.txt + summary.npz)
    # -----------------------------------------------------------------------
    header = (f"{args.sweep:>12}  {'per-seed cos_sim':<40}  {'mean':>7}  {'std':>7}  {'var':>9}  "
              f"{'best':>7}  {'ESS':>7}  {'disp_spr':>9}")
    lines = [
        f"epoch {epoch} ({args.ckpt_dir}) MPPI {args.sweep} sweep",
        f"action_horizon={action_horizon}  n_planning_itrs={n_planning_itrs}  "
        f"max_action_value={MAX_ACTION_VALUE}  warm_start_std={WARM_START_STD}  "
        f"seeds={sample_seeds}  {fixed_desc}",
        f"goal: {goal_text}",
        "",
        header,
        "-" * len(header),
    ]
    for v in values:
        a = arms[v]
        per_seed = " ".join(f"{c:.4f}" for c in a["run_cos_sim"])
        lines.append(
            f"{v:>12g}  {per_seed:<38}  {a['mean']:>7.4f}  {a['std']:>7.4f}  "
            f"{a['var']:>9.6f}  {a['best']:>7.4f}  {a['ess_mean']:>7.1f}  {a['disp_spread']:>9.4f}"
        )
    lines += ["", "ESS = mean effective sample size 1/sum(w^2) across refit iterations, out of K;",
              "      ~K means the softmax averages over a near-tied population, ~1 means it "
              "collapsed onto", "      a single sample and could not explore further.",
              "NOTE: cos_sim here is the dynamics model's OWN predicted outcome (cost_fn = "
              "1 - cos(model(...), z_sem)).", "      No env.step() is ever called -- nothing "
              "here is checked against a real rollout."]

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    np.savez(
        os.path.join(out_dir, "summary.npz"),
        values=values,
        run_cos_sim=np.stack([arms[v]["run_cos_sim"] for v in values], axis=0),
        mean=means, std=stds,
        var=np.array([arms[v]["var"] for v in values]),
        best=np.array([arms[v]["best"] for v in values]),
        ess_mean=np.array([arms[v]["ess_mean"] for v in values]),
        ess_final=np.array([arms[v]["ess_final"] for v in values]),
        std_final=np.array([arms[v]["std_final"] for v in values]),
        disp_spread=np.array([arms[v]["disp_spread"] for v in values]),
        sample_seeds=np.array(sample_seeds),
        epoch=epoch, action_horizon=action_horizon, sweep=args.sweep,
    )
    print(f"\nDone. Outputs under {out_dir}")
    print(f"Probe artifacts in {PROBE_DATA_DIR} were only read, never modified.")


if __name__ == "__main__":
    main()
