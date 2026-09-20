"""
Softmax-temperature sweep on the epoch-11 5-seed MPPI probe.

mppi_global_or_local_min.py showed high run-to-run seed volatility at the single fixed
mppi_temperature=0.01: epoch-11 cos_sim [0.5630, 0.6311, 0.6286, 0.5079, 0.6020] against a
near-perfect reference chunk at 0.6388 -- a ~0.12 spread with no clear convergence. This script
asks whether that volatility is a function of the CEM softmax temperature, by re-running the
IDENTICAL 5-seed probe across a grid of temperatures and reporting per-seed cos_sim plus
cross-seed variance for each.

Why temperature and not elite truncation: mppi_core.py is a Gaussian CEM whose softmax over
K=512 already behaves like an elite selector at temperature 0.01. For a realistic
final-population return spread (~0.02 in cos_sim) the effective sample size 1/sum(w^2) is only
~5 of 512, i.e. the refit is already dominated by a handful of samples, so explicitly
truncating to the top ~1% would be close to a no-op. Lowering temperature further drives ESS
toward 1, at which point mean collapses onto a single sample and std -> 1e-9, freezing the
population for the remaining iterations -- the returned argmax then just reflects the best
member of the initial uniform draw, which should make seed volatility WORSE. So the grid
deliberately brackets 0.01 in both directions and the informative direction is expected to be
upward (more exploration). ESS is measured per iteration rather than assumed, via
mppi_core.py's opt-in collect_diagnostics flag.

Note that command() returns actions[best_idx] -- the argmax of the final population, never the
refit mean. Temperature changes WHERE the population searches, not the selection rule.

Relationship to mppi_global_or_local_min.py (which this script does NOT modify or import --
it is top-level script code with no __main__ guard, so importing it would reload all 5000
training pkls, rebuild the pybullet env and rerun every MPPI pass as an import side effect):
this script reproduces that script's probe exactly by reading the two artifacts it already
wrote, rather than re-deriving them --

  first_frame.png             the exact probe frame (peg touching green_cube, collinear with
                              blue_moon). Saved lossless, so clip_preprocess() on it yields a
                              bit-identical z_t -- no env, no pybullet, no peg-placement logic
                              needed here at all.
  epoch{N}_mppi_5runs_data.npz  ref_chunk_raw, correct_direction, green_xy/blue_xy/peg_xy,
                              sample_seeds, and the baseline run_cos_sim / ref_cos_sim.

Two asserts make that reuse trustworthy rather than merely convenient:
  1. the npz's block/peg geometry must match the EXPECTED_* constants and still be touching;
  2. the temp=0.01 arm must reproduce the npz's cached run_cos_sim to 1e-4. That is a full
     end-to-end proof that this script's z_t, action-normalization stats, hyperparameters and
     per-seed RNG all replicate mppi_global_or_local_min.py. If it fails, every other arm is
     suspect too, so it is checked and reported explicitly.

Outputs land under MPPI_charts/min_determination/expert+subopt-model/temp_sweep/ -- a per-arm
subdirectory plus a cross-arm summary. Nothing is written into the parent directory, so the
existing baseline epoch*_mppi_5runs* charts and first_frame.png are only ever read.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_temperature_sweep.py
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
# Config -- every MPPI hyperparameter except mppi_temperature is pinned to the value
# mppi_global_or_local_min.py used, so temperature is the only thing that varies.
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"
HORIZON = 8

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "min_determination", "expert+subopt-model")
OUT_DIR = os.path.join(DATA_DIR, "temp_sweep")
CKPT_DIR = os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_EPOCH = 11
# Log-spaced grid bracketing the current 0.01 in both directions. See module docstring for why
# the interesting direction is expected to be upward.
DEFAULT_TEMPERATURES = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
BASELINE_TEMPERATURE = 0.01  # the value mppi_global_or_local_min.py used -> reproduction check

K = 512
N_PLANNING_ITRS = 9
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1

N_MPPI_RUNS = 5
SAMPLE_SEED = 123
SAMPLE_SEEDS = [SAMPLE_SEED + i for i in range(N_MPPI_RUNS)]

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])
TOUCH_THRESH = 0.05

BASELINE_RTOL = 1e-4  # tolerance for the temp=0.01 reproduction assert

# Chart style: identical to mppi_global_or_local_min.py's, deliberately -- these charts are meant
# to be compared side by side with the existing epoch11_mppi_5runs.png, so seed i keeps the same
# color in every chart (color follows the seed, never its rank). Identity is never carried by
# color alone: every path is both direct-labeled and in the legend.
RUN_COLORS = [matplotlib.colormaps["tab10"](i) for i in range(10)]
RUN_MARKERS = ["o", "s", "^", "D", "v"]  # secondary encoding for the summary chart
REF_COLOR = "magenta"
INK = "#333333"        # text/axis ink -- never a series color
INK_MUTED = "#888888"


# ---------------------------------------------------------------------------
# Inline model classes -- verbatim from mppi_global_or_local_min.py (lines 142-189), which is
# itself the eval/ convention: inline-copy rather than import dynamics_model.py, whose module
# level pulls in LangTableDataset/open_clip side effects.
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
    temporal axis is collapsed by flattening."""

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_chunk(chunk_raw: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Raw action units -> [-1, 1], the same way LangTableDataset (and every mppi_* script) does."""
    return np.clip(2.0 * (chunk_raw - lo) / (hi - lo) - 1.0, -1.0, 1.0).astype(np.float32)


def denormalize_chunk(chunk_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[-1, 1] -> raw action units (inverse of normalize_chunk)."""
    return ((chunk_norm + 1.0) / 2.0 * (hi - lo) + lo)


def cumulative_path(chunk_raw: np.ndarray) -> np.ndarray:
    """(8, 2) per-step displacements -> (9, 2) path starting at the origin."""
    return np.concatenate([[[0.0, 0.0]], np.cumsum(chunk_raw, axis=0)], axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                        help=f"checkpoint epoch to probe (default {DEFAULT_EPOCH})")
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES,
                        help="mppi_temperature values to sweep")
    parser.add_argument("--tag", type=str, default=None,
                        help="output subdirectory suffix -- e.g. --tag fine writes to "
                             "temp_sweep_fine/ instead of temp_sweep/. Use it for any grid other "
                             "than the default one, so independent sweeps do not overwrite each "
                             "other's per-arm directories or summary files.")
    args = parser.parse_args()
    epoch = args.epoch
    temperatures = list(args.temperatures)

    # Untagged runs keep writing to OUT_DIR, so the original coarse sweep stays reproducible.
    out_dir = OUT_DIR if not args.tag else f"{OUT_DIR}_{args.tag}"

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("  (CUDA unavailable -- running on CPU; fine for this workload)")
    print(f"Grid: {[f'{t:g}' for t in temperatures]}")
    print(f"Output dir: {out_dir}")

    # -----------------------------------------------------------------------
    # 1. Baseline artifacts from mppi_global_or_local_min.py (read-only).
    # -----------------------------------------------------------------------
    frame_path = os.path.join(DATA_DIR, "first_frame.png")
    npz_path = os.path.join(DATA_DIR, f"epoch{epoch}_mppi_5runs_data.npz")
    for path in (frame_path, npz_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. It is produced by mppi_global_or_local_min.py -- re-run\n"
                f"  python main_project/eval/mppi_global_or_local_min.py\n"
                f"to regenerate the probe artifacts, then re-run this sweep."
            )

    baseline = np.load(npz_path)
    npz_green = baseline["green_xy"]
    npz_blue = baseline["blue_xy"]
    npz_peg = baseline["peg_xy"]
    correct_direction = baseline["correct_direction"]
    ref_chunk_raw = baseline["ref_chunk_raw"].astype(np.float32)   # (8, 2)
    baseline_run_cos_sim = baseline["run_cos_sim"].astype(float)   # (5,)
    baseline_ref_cos_sim = float(baseline["ref_cos_sim"])
    npz_seeds = baseline["sample_seeds"]

    # The sweep is only comparable to the baseline if the cached probe geometry is the one this
    # script assumes. Fail loudly rather than silently sweeping a different starting state.
    assert np.allclose(npz_green, EXPECTED_GREEN_XY, atol=1e-5), (
        f"cached {BLOCK_A} {npz_green} != expected {EXPECTED_GREEN_XY} -- probe state diverged"
    )
    assert np.allclose(npz_blue, EXPECTED_BLUE_XY, atol=1e-5), (
        f"cached {BLOCK_B} {npz_blue} != expected {EXPECTED_BLUE_XY} -- probe state diverged"
    )
    dist_peg_green = float(np.linalg.norm(npz_peg - npz_green))
    assert dist_peg_green < TOUCH_THRESH, (
        f"cached peg is {dist_peg_green:.4f} from {BLOCK_A}, not < {TOUCH_THRESH} -- the cached "
        f"probe does not have the peg touching {BLOCK_A}, so this sweep would not match it"
    )
    assert list(npz_seeds) == SAMPLE_SEEDS, (
        f"cached sample_seeds {list(npz_seeds)} != this script's {SAMPLE_SEEDS} -- per-seed "
        f"comparison against the baseline would not be like-for-like"
    )
    print(f"Probe geometry verified from {os.path.basename(npz_path)}: "
          f"{BLOCK_A}={npz_green}, {BLOCK_B}={npz_blue}, "
          f"dist(peg,{BLOCK_A})={dist_peg_green:.4f} (touching)")
    print(f"Baseline (temp={BASELINE_TEMPERATURE}) cached cos_sim: "
          f"{np.round(baseline_run_cos_sim, 4)}  ref={baseline_ref_cos_sim:.4f}")

    # -----------------------------------------------------------------------
    # 2. Action normalization stats -- recomputed from the same source as the baseline so
    #    ACTION_LO/HI are guaranteed identical (they are not stored in the npz).
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
    # 3. Checkpoint + CLIP + model, loaded ONCE and shared by every arm, so all arms see an
    #    identical z_t / z_sem and differ only by temperature.
    # -----------------------------------------------------------------------
    ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt")
    print(f"\nLoading checkpoint {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}")

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

        ref_norm_t = torch.tensor(
            normalize_chunk(ref_chunk_raw, action_lo, action_hi), device=DEVICE
        ).unsqueeze(0)                                                          # (1, 8, 2)
        ref_cos_sim = float((model(z_t.unsqueeze(0), ref_norm_t) * z_sem).sum(dim=-1).item())

    # Early, cheap check that z_t / action stats / model all match the baseline: the reference
    # chunk's cos_sim is temperature-independent, so it must equal the cached value.
    ref_delta = abs(ref_cos_sim - baseline_ref_cos_sim)
    print(f"Reference near-perfect chunk: cos_sim={ref_cos_sim:.6f} "
          f"(cached {baseline_ref_cos_sim:.6f}, delta={ref_delta:.2e})")
    assert ref_delta < BASELINE_RTOL, (
        f"reference chunk cos_sim {ref_cos_sim:.6f} != cached {baseline_ref_cos_sim:.6f}. z_t, "
        f"the action-normalization stats or the checkpoint differ from the baseline run, so no "
        f"arm of this sweep would be comparable to it."
    )

    # -----------------------------------------------------------------------
    # 4. The sweep. dynamics_fn / cost_fn are identical to mppi_global_or_local_min.py's.
    # -----------------------------------------------------------------------
    def dynamics_fn(state, action):
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        return 1.0 - (terminal_state * z_sem).sum(dim=-1)

    arms = {}  # temp -> dict of results
    for temp in temperatures:
        print(f"\n=== temperature {temp:g}: {N_MPPI_RUNS} independent MPPI passes ===")
        run_chunks_raw, run_net_disp, run_cos_sim = [], [], []
        run_final_ess, run_mean_ess, run_final_std = [], [], []

        for run_idx in range(N_MPPI_RUNS):
            # Fresh instance per run: prev_best_action is None, so the initial population comes
            # from Uniform(-1,1). Combined with the per-run reseed below, run i draws the
            # IDENTICAL initial population in every arm -- arms diverge only through the refit
            # weighting, which is the whole point of the sweep.
            mppi = MPPI(
                dynamics_fn=dynamics_fn,
                cost_fn=cost_fn,
                pred_horizon=HORIZON,
                action_dim=2,
                num_samples=K,
                mppi_temperature=temp,
                n_planning_itrs=N_PLANNING_ITRS,
                device=DEVICE,
                max_action_value=MAX_ACTION_VALUE,
                warm_start_std=WARM_START_STD,
                collect_diagnostics=True,
            )

            torch.manual_seed(SAMPLE_SEEDS[run_idx])
            with torch.no_grad():
                best_action_chunk = mppi.command(z_t)

            # cost_fn is exactly 1 - cos_sim, so last_returns == cos_sim - 1 for the population.
            final_returns = mppi.last_returns
            best_idx = int(torch.argmax(final_returns))
            assert mppi.last_actions[best_idx].allclose(best_action_chunk), (
                "argmax of last_returns does not match mppi.command()'s own selection"
            )
            cos_sim = float(final_returns[best_idx].item() + 1.0)

            chunk_raw = denormalize_chunk(
                best_action_chunk.cpu().numpy(), action_lo, action_hi
            )                                                       # (8, 2)
            disp = chunk_raw.sum(axis=0)
            mag = float(np.linalg.norm(disp))
            dir_align = float(np.dot(disp / (mag + 1e-8), correct_direction))

            diags = mppi.last_diagnostics
            final_ess = diags[-1]["ess"]
            mean_ess = float(np.mean([d["ess"] for d in diags]))
            final_std = diags[-1]["std_mean"]

            print(f"  run {run_idx + 1} (seed={SAMPLE_SEEDS[run_idx]})  cos_sim={cos_sim:.4f}  "
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

        # Cross-seed volatility -- the quantity the sweep exists to measure.
        pdist = [float(np.linalg.norm(run_net_disp[i] - run_net_disp[j]))
                 for i in range(N_MPPI_RUNS) for j in range(i + 1, N_MPPI_RUNS)]
        stats = dict(
            mean=float(run_cos_sim.mean()),
            std=float(run_cos_sim.std(ddof=0)),
            var=float(run_cos_sim.var(ddof=0)),
            best=float(run_cos_sim.max()),
            worst=float(run_cos_sim.min()),
            rng=float(run_cos_sim.max() - run_cos_sim.min()),
            vs_ref=float(run_cos_sim.max() - ref_cos_sim),
            disp_spread=float(np.mean(pdist)),
            disp_spread_max=float(np.max(pdist)),
            ess_final=float(np.mean(run_final_ess)),
            ess_mean=float(np.mean(run_mean_ess)),
            std_final=float(np.mean(run_final_std)),
        )
        print(f"  cross-seed: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"var={stats['var']:.6f}  range={stats['rng']:.4f}  "
              f"best-ref={stats['vs_ref']:+.4f}  net-disp spread={stats['disp_spread']:.4f}")

        arms[temp] = dict(
            run_chunks_raw=run_chunks_raw,
            run_net_disp=run_net_disp,
            run_cos_sim=run_cos_sim,
            run_final_ess=np.array(run_final_ess),
            run_mean_ess=np.array(run_mean_ess),
            **stats,
        )

    # -----------------------------------------------------------------------
    # 5. Baseline reproduction check -- the assert that makes the whole sweep trustworthy.
    # -----------------------------------------------------------------------
    print("\n=== baseline reproduction check ===")
    if BASELINE_TEMPERATURE in arms:
        got = arms[BASELINE_TEMPERATURE]["run_cos_sim"]
        delta = np.abs(got - baseline_run_cos_sim)
        print(f"  temp={BASELINE_TEMPERATURE} this run: {np.round(got, 4)}")
        print(f"  cached baseline:        {np.round(baseline_run_cos_sim, 4)}")
        print(f"  max abs delta: {delta.max():.2e}  (tolerance {BASELINE_RTOL:.0e})")
        assert delta.max() < BASELINE_RTOL, (
            f"temp={BASELINE_TEMPERATURE} arm does not reproduce the cached epoch-{epoch} "
            f"baseline (max delta {delta.max():.2e}). This script's probe state, z_t, action "
            f"stats or per-seed RNG differ from mppi_global_or_local_min.py, so no arm of this "
            f"sweep is comparable to it. Investigate before trusting any result above."
        )
        print("  PASS -- probe replicates mppi_global_or_local_min.py exactly")
    else:
        print(f"  SKIPPED -- {BASELINE_TEMPERATURE} not in the swept grid {temperatures}")

    # -----------------------------------------------------------------------
    # 6. Per-arm charts, sharing one axis range and one correct-direction arrow so the arms are
    #    directly comparable with each other and with the existing epoch11_mppi_5runs.png.
    # -----------------------------------------------------------------------
    all_mags = np.array([np.linalg.norm(d) for a in arms.values() for d in a["run_net_disp"]])
    ref_len = float(np.median(all_mags))
    arrow_end = correct_direction * ref_len

    all_points = [np.zeros((1, 2)), arrow_end.reshape(1, 2), cumulative_path(ref_chunk_raw)]
    for a in arms.values():
        for chunk in a["run_chunks_raw"]:
            all_points.append(cumulative_path(chunk))
    axis_half = float(np.abs(np.concatenate(all_points, axis=0)).max() * 1.15)
    axis_lim = (-axis_half, axis_half)
    print(f"\nShared plot scale: ref_len={ref_len:.4f}  axis_lim=[{-axis_half:.4f}, {axis_half:.4f}]")

    ref_path = cumulative_path(ref_chunk_raw)
    for temp, a in arms.items():
        arm_dir = os.path.join(out_dir, f"temp_{temp:g}")
        os.makedirs(arm_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.annotate("", xy=tuple(arrow_end), xytext=(0, 0),
                    arrowprops=dict(facecolor=INK, edgecolor=INK, width=1.5, headwidth=8), zorder=2)
        ax.text(arrow_end[0], arrow_end[1], "  correct dir", color=INK, fontsize=9,
                va="center", zorder=2)

        for run_idx in range(N_MPPI_RUNS):
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

        ax.plot(ref_path[:, 0], ref_path[:, 1], color=REF_COLOR, linewidth=2.6, linestyle="--",
                label=f"ref near-perfect  cos_sim={ref_cos_sim:.4f}", zorder=3 + N_MPPI_RUNS)
        ax.text(ref_path[-1, 0], ref_path[-1, 1], "  ref", color=REF_COLOR, fontsize=11,
                fontweight="bold", va="center", zorder=4 + N_MPPI_RUNS)

        ax.set_xlim(axis_lim); ax.set_ylim(axis_lim); ax.set_aspect("equal")
        ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("net displacement dx", color=INK)
        ax.set_ylabel("net displacement dy", color=INK)
        ax.set_title(
            f"epoch{epoch} temperature sweep -- mppi_temperature={temp:g}"
            f"{'  (baseline)' if temp == BASELINE_TEMPERATURE else ''}\n"
            f"{N_MPPI_RUNS} independent MPPI passes (seeds {SAMPLE_SEEDS[0]}-{SAMPLE_SEEDS[-1]}), "
            f"K={K}, {N_PLANNING_ITRS} refinement iters\n"
            f"cross-seed std={a['std']:.4f}  range={a['rng']:.4f}  "
            f"mean ESS={a['ess_mean']:.1f}/{K}",
            color=INK,
        )
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plot_path = os.path.join(arm_dir, f"epoch{epoch}_5runs.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        np.savez(
            os.path.join(arm_dir, f"epoch{epoch}_5runs_data.npz"),
            run_chunks_raw=a["run_chunks_raw"],
            run_net_disp=a["run_net_disp"],
            run_cos_sim=a["run_cos_sim"],
            run_final_ess=a["run_final_ess"],
            run_mean_ess=a["run_mean_ess"],
            ref_chunk_raw=ref_chunk_raw,
            ref_cos_sim=ref_cos_sim,
            correct_direction=correct_direction,
            green_xy=npz_green, blue_xy=npz_blue, peg_xy=npz_peg,
            mppi_temperature=temp,
            K=K, n_planning_itrs=N_PLANNING_ITRS, n_mppi_runs=N_MPPI_RUNS,
            sample_seeds=np.array(SAMPLE_SEEDS), epoch=epoch,
            **{k: a[k] for k in ("mean", "std", "var", "best", "worst", "rng", "vs_ref",
                                 "disp_spread", "disp_spread_max", "ess_final", "ess_mean",
                                 "std_final")},
        )
        print(f"Saved arm temp={temp:g} -> {arm_dir}")

    # -----------------------------------------------------------------------
    # 7. Summary chart: per-seed cos_sim vs temperature. Single y-axis (never a dual axis);
    #    seed identity carried by color AND marker shape AND the legend.
    # -----------------------------------------------------------------------
    temps = np.array(sorted(arms.keys()))
    means = np.array([arms[t]["mean"] for t in temps])
    stds = np.array([arms[t]["std"] for t in temps])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(temps, means - stds, means + stds, color=INK_MUTED, alpha=0.18,
                    label="mean +/- 1 std across seeds", zorder=2)
    ax.plot(temps, means, color=INK, linewidth=2.0, marker="o", markersize=5,
            label="mean across seeds", zorder=4)
    for run_idx in range(N_MPPI_RUNS):
        ys = [arms[t]["run_cos_sim"][run_idx] for t in temps]
        ax.plot(temps, ys, color=RUN_COLORS[run_idx], marker=RUN_MARKERS[run_idx],
                markersize=7, linewidth=1.0, alpha=0.85,
                label=f"seed {SAMPLE_SEEDS[run_idx]}", zorder=5 + run_idx)

    ax.axhline(ref_cos_sim, color=REF_COLOR, linestyle="--", linewidth=2.0,
               label=f"ref near-perfect chunk ({ref_cos_sim:.4f})", zorder=3)
    ax.axvline(BASELINE_TEMPERATURE, color=INK_MUTED, linestyle=":", linewidth=1.5, zorder=1)
    ax.text(BASELINE_TEMPERATURE, ax.get_ylim()[0], " baseline", color=INK_MUTED,
            fontsize=9, va="bottom", ha="left")

    ax.set_xscale("log")
    ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("mppi_temperature (log scale)", color=INK)
    ax.set_ylabel("cos_sim of selected action chunk", color=INK)
    ax.set_title(
        f"epoch{epoch}: MPPI softmax temperature vs. selected-chunk cos_sim\n"
        f"{N_MPPI_RUNS} seeds per temperature, all other hyperparameters fixed "
        f"(K={K}, {N_PLANNING_ITRS} iters) -- higher is better, tighter is more reproducible",
        color=INK,
    )
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    summary_plot = os.path.join(out_dir, "summary_cos_sim_vs_temperature.png")
    plt.savefig(summary_plot, dpi=150)
    plt.close()
    print(f"Saved summary chart -> {summary_plot}")

    # -----------------------------------------------------------------------
    # 8. Summary table (stdout + summary.txt + summary.npz)
    # -----------------------------------------------------------------------
    header = (f"{'temp':>8}  {'per-seed cos_sim':<40}  {'mean':>7}  {'std':>7}  {'var':>9}  "
              f"{'best':>7}  {'vs_ref':>8}  {'ESS':>7}  {'disp_spr':>9}")
    lines = [
        f"epoch {epoch} MPPI softmax-temperature sweep",
        f"K={K}  n_planning_itrs={N_PLANNING_ITRS}  max_action_value={MAX_ACTION_VALUE}  "
        f"warm_start_std={WARM_START_STD}  seeds={SAMPLE_SEEDS}",
        f"ref near-perfect chunk cos_sim = {ref_cos_sim:.4f}  (the target to converge toward)",
        "",
        header,
        "-" * len(header),
    ]
    for t in temps:
        a = arms[t]
        per_seed = " ".join(f"{c:.4f}" for c in a["run_cos_sim"])
        marker = " *" if t == BASELINE_TEMPERATURE else "  "
        lines.append(
            f"{t:>8g}{marker}{per_seed:<38}  {a['mean']:>7.4f}  {a['std']:>7.4f}  "
            f"{a['var']:>9.6f}  {a['best']:>7.4f}  {a['vs_ref']:>+8.4f}  "
            f"{a['ess_mean']:>7.1f}  {a['disp_spread']:>9.4f}"
        )
    lines += ["", "* = baseline (the value mppi_global_or_local_min.py used)",
              "ESS = mean effective sample size 1/sum(w^2) across refit iterations, out of "
              f"K={K};", "      ~K means the softmax averages over a near-tied population, "
              "~1 means it collapsed onto", "      a single sample and could not explore further.",
              "vs_ref = best-seed cos_sim minus the reference chunk's cos_sim (0 = matched it)."]

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    np.savez(
        os.path.join(out_dir, "summary.npz"),
        temperatures=temps,
        run_cos_sim=np.stack([arms[t]["run_cos_sim"] for t in temps], axis=0),  # (n_temp, 5)
        mean=means, std=stds,
        var=np.array([arms[t]["var"] for t in temps]),
        best=np.array([arms[t]["best"] for t in temps]),
        vs_ref=np.array([arms[t]["vs_ref"] for t in temps]),
        ess_mean=np.array([arms[t]["ess_mean"] for t in temps]),
        ess_final=np.array([arms[t]["ess_final"] for t in temps]),
        std_final=np.array([arms[t]["std_final"] for t in temps]),
        disp_spread=np.array([arms[t]["disp_spread"] for t in temps]),
        ref_cos_sim=ref_cos_sim,
        baseline_run_cos_sim=baseline_run_cos_sim,
        sample_seeds=np.array(SAMPLE_SEEDS),
        epoch=epoch,
    )
    print(f"\nDone. Outputs under {out_dir}")
    print(f"Baseline artifacts in {DATA_DIR} were only read, never modified.")


if __name__ == "__main__":
    main()
