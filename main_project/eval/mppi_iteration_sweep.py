"""
Planner-iteration-count sweep on the epoch-11 5-seed MPPI probe.

mppi_core.MPPI runs n_planning_itrs refits plus 1 initial evaluation, so the shipped
n_planning_itrs=9 means 10 dynamics evaluations per command() call. Every result in this
project so far used that value. This script asks what happens with far more: does the
planner keep improving, converge, or collapse long before iteration 10 and waste
everything after?

The grid is expressed in TOTAL EVALUATIONS (10, 20, 50, 100, 500, 1000), which is
n_planning_itrs + 1. Output directories and chart titles use total evaluations; the
MPPI constructor receives n_planning_itrs = total - 1.

Why this is a separate script from mppi_temperature_sweep.py: that script sweeps
mppi_temperature with n_planning_itrs pinned, and its baseline-reproduction assert is
temperature-specific. This one pins temperature and sweeps iterations. Both are
deliberately left alone; so is mppi_global_or_local_min.py, whose chart style this
reproduces (including the 90/180/270deg wrong-direction controls, which
mppi_temperature_sweep.py's charts omit).

Relationship to the artifacts it reads (all READ-ONLY, nothing is written into the
parent directory):

  first_frame.png               the exact probe frame (peg touching green_cube, collinear
                                with blue_moon), saved lossless by mppi_global_or_local_min.py,
                                so clip_preprocess() on it yields a bit-identical z_t. No env,
                                no pybullet, no peg-placement logic needed here at all.
  epoch{N}_mppi_5runs_data.npz  ref_chunk_raw, correct_direction, green_xy/blue_xy/peg_xy,
                                sample_seeds.
  temp_sweep/temp_0.1/
    epoch{N}_5runs_data.npz     THE REPRODUCTION ANCHOR. mppi_temperature_sweep.py already ran
                                exactly this configuration -- temp=0.1, K=512, n_planning_itrs=9,
                                seeds 123-127 -- so this script's 10-evaluation arm must
                                reproduce its run_cos_sim to 1e-4. That single assert is what
                                proves this script's z_t, action-normalization stats, checkpoint
                                and per-seed RNG all match the existing tooling. If it fails,
                                every other arm is suspect too, so it is checked and reported
                                explicitly.

Two structural facts worth knowing when reading the output:

  1. The arms are NESTED. Same seed gives the same initial population, and n_planning_itrs
     only sets the loop count, so the 1000-evaluation run's first 9 refits are bit-identical
     to the 10-evaluation run's. This script asserts that prefix property across arms; a
     failure means determinism is broken somewhere and no arm can be trusted. (It also means
     the whole sweep could in principle be read off one long run, but six arms cost well under
     a minute of GPU, so it is not worth restructuring MPPI to expose intermediate states.)
  2. command() returns actions[best_idx] -- the argmax of the FINAL population, never the refit
     mean. Iteration count changes where the population searches, not the selection rule.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_iteration_sweep.py                      # all six arms
    python main_project/eval/mppi_iteration_sweep.py --evals 10 --tag smoke   # single-arm smoke test
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
# Config -- every MPPI hyperparameter except n_planning_itrs is pinned to the value
# mppi_global_or_local_min.py / mppi_temperature_sweep.py used, so the iteration count
# is the only thing that varies.
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"
HORIZON = 8

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "min_determination", "expert+subopt-model")
OUT_DIR = os.path.join(DATA_DIR, "iteration_sweep")
CKPT_DIR = os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_EPOCH = 11
# Total dynamics evaluations per command() call = n_planning_itrs + 1.
DEFAULT_EVALS = [10, 20, 50, 100, 500, 1000]
DEFAULT_TEMPERATURE = 0.1     # the best-performing arm of the temperature sweep

# The anchor arm: mppi_temperature_sweep.py's temp=0.1 run used n_planning_itrs=9, i.e. 10 evals.
ANCHOR_EVALS = 10
ANCHOR_NPZ = os.path.join(DATA_DIR, "temp_sweep", "temp_0.1", "epoch{epoch}_5runs_data.npz")

K = 512
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1

N_MPPI_RUNS = 5
SAMPLE_SEED = 123
SAMPLE_SEEDS = [SAMPLE_SEED + i for i in range(N_MPPI_RUNS)]

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])
TOUCH_THRESH = 0.05

ANCHOR_RTOL = 1e-4  # tolerance for the reproduction assert

# Wrong-direction controls: the same near-perfect chunk rotated 90/180/270 deg. A well-behaved
# model should rate all three BELOW the aligned reference (they push the wrong way), so their
# predicted cos_sim is a sanity check that the model penalizes wrong-direction actions.
# Ported verbatim from mppi_global_or_local_min.py:127-131.
REF_ROTATIONS_DEG = [90, 180, 270]
REF_ROT_COLORS = ["teal", "saddlebrown", "dimgray"]  # one per rotation; distinct from runs + ref

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
# Inline model classes -- verbatim from mppi_temperature_sweep.py, which is itself the
# eval/ convention: inline-copy rather than import dynamics_model.py, whose module level
# pulls in LangTableDataset/open_clip side effects.
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


def rotate_chunk(chunk_raw: np.ndarray, deg: float) -> np.ndarray:
    """Rotate every per-step (dx, dy) of a chunk by deg. Ported from
    mppi_global_or_local_min.py:353-357."""
    th = np.radians(deg)
    rm = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th),  np.cos(th)]], dtype=np.float32)
    return (chunk_raw @ rm.T).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--epoch", type=int, default=DEFAULT_EPOCH,
                        help=f"checkpoint epoch to probe (default {DEFAULT_EPOCH})")
    parser.add_argument("--evals", type=int, nargs="+", default=DEFAULT_EVALS,
                        help="total dynamics evaluations per command() call to sweep. The MPPI "
                             "constructor gets n_planning_itrs = evals - 1. Default "
                             f"{DEFAULT_EVALS}.")
    parser.add_argument("--mppi_temperature", type=float, default=DEFAULT_TEMPERATURE,
                        help=f"softmax temperature, held fixed across the sweep (default "
                             f"{DEFAULT_TEMPERATURE}, the best arm of the temperature sweep).")
    parser.add_argument("--tag", type=str, default=None,
                        help="output subdirectory suffix -- e.g. --tag smoke writes to "
                             "iteration_sweep_smoke/ instead of iteration_sweep/. Use it for any "
                             "grid other than the default one, so independent sweeps do not "
                             "overwrite each other's per-arm directories or summary files.")
    args = parser.parse_args()
    epoch = args.epoch
    evals_grid = sorted(set(args.evals))
    temperature = args.mppi_temperature

    if any(e < 1 for e in evals_grid):
        parser.error(f"--evals must all be >= 1 (1 = initial evaluation only), got {evals_grid}")

    out_dir = OUT_DIR if not args.tag else f"{OUT_DIR}_{args.tag}"

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("  (CUDA unavailable -- running on CPU; the 1000-eval arm will be slow)")
    print(f"Grid (total evaluations): {evals_grid}")
    print(f"  -> n_planning_itrs:     {[e - 1 for e in evals_grid]}")
    print(f"Temperature (fixed): {temperature:g}")
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
    npz_seeds = baseline["sample_seeds"]

    # The sweep is only comparable to the existing charts if the cached probe geometry is the one
    # this script assumes. Fail loudly rather than silently sweeping a different starting state.
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
        f"comparison against the existing charts would not be like-for-like"
    )
    print(f"Probe geometry verified from {os.path.basename(npz_path)}: "
          f"{BLOCK_A}={npz_green}, {BLOCK_B}={npz_blue}, "
          f"dist(peg,{BLOCK_A})={dist_peg_green:.4f} (touching)")

    # -----------------------------------------------------------------------
    # 1b. The reproduction anchor: mppi_temperature_sweep.py's temp=0.1 arm.
    # -----------------------------------------------------------------------
    anchor_path = ANCHOR_NPZ.format(epoch=epoch)
    anchor = None
    if ANCHOR_EVALS in evals_grid and abs(temperature - DEFAULT_TEMPERATURE) < 1e-12:
        if not os.path.exists(anchor_path):
            print(f"WARNING: anchor {anchor_path} not found -- the {ANCHOR_EVALS}-eval arm cannot "
                  f"be checked against mppi_temperature_sweep.py. Results are unverified.")
        else:
            anchor = np.load(anchor_path)
            assert int(anchor["n_planning_itrs"]) == ANCHOR_EVALS - 1, (
                f"anchor npz has n_planning_itrs={int(anchor['n_planning_itrs'])}, expected "
                f"{ANCHOR_EVALS - 1} -- it is not the {ANCHOR_EVALS}-evaluation configuration"
            )
            assert list(anchor["sample_seeds"]) == SAMPLE_SEEDS, (
                f"anchor seeds {list(anchor['sample_seeds'])} != {SAMPLE_SEEDS}"
            )
            assert abs(float(anchor["mppi_temperature"]) - temperature) < 1e-12
            print(f"Anchor loaded from {os.path.relpath(anchor_path, DATA_DIR)}: "
                  f"run_cos_sim={np.round(anchor['run_cos_sim'].astype(float), 4)}")
    else:
        print(f"No anchor check: it requires {ANCHOR_EVALS} in the grid and temperature "
              f"{DEFAULT_TEMPERATURE:g} (got temperature {temperature:g}).")

    # -----------------------------------------------------------------------
    # 2. Action normalization stats -- recomputed from the same source as every other
    #    mppi_* script so ACTION_LO/HI are guaranteed identical (not stored in the npz).
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
    #    identical z_t / z_sem and differ only by iteration count.
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

        # --- Wrong-direction rotations of the reference chunk (90/180/270 deg) ---
        # Ported from mppi_global_or_local_min.py:347-363 and :420-426. Controls for "does
        # the model rate wrong-direction actions as bad". Iteration-independent, so computed
        # once and drawn identically on every arm's chart.
        ref_rot_chunks_raw, ref_rot_cos_sim = [], []
        for deg in REF_ROTATIONS_DEG:
            rot_raw = rotate_chunk(ref_chunk_raw, deg)
            rot_norm_t = torch.tensor(
                normalize_chunk(rot_raw, action_lo, action_hi), device=DEVICE
            ).unsqueeze(0)
            ref_rot_chunks_raw.append(rot_raw)
            ref_rot_cos_sim.append(
                float((model(z_t.unsqueeze(0), rot_norm_t) * z_sem).sum(dim=-1).item())
            )

    print(f"Reference near-perfect chunk: cos_sim={ref_cos_sim:.6f}")
    for deg, c in zip(REF_ROTATIONS_DEG, ref_rot_cos_sim):
        print(f"    rot {deg:>3}deg (wrong dir): cos_sim={c:.4f}")

    # Cheap early check that z_t / action stats / model match the anchor run: the reference
    # chunk's cos_sim involves no MPPI at all, so it must equal the cached value.
    if anchor is not None:
        ref_delta = abs(ref_cos_sim - float(anchor["ref_cos_sim"]))
        print(f"  vs anchor ref_cos_sim {float(anchor['ref_cos_sim']):.6f}  delta={ref_delta:.2e}")
        assert ref_delta < ANCHOR_RTOL, (
            f"reference chunk cos_sim {ref_cos_sim:.6f} != anchor {float(anchor['ref_cos_sim']):.6f}. "
            f"z_t, the action-normalization stats or the checkpoint differ from the anchor run, so "
            f"no arm of this sweep would be comparable to it."
        )

    # -----------------------------------------------------------------------
    # 4. The sweep. dynamics_fn / cost_fn are identical to mppi_temperature_sweep.py's.
    # -----------------------------------------------------------------------
    def dynamics_fn(state, action):
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        return 1.0 - (terminal_state * z_sem).sum(dim=-1)

    arms = {}  # total_evals -> dict of results
    for n_evals in evals_grid:
        n_itrs = n_evals - 1
        print(f"\n=== {n_evals} evaluations (n_planning_itrs={n_itrs}): "
              f"{N_MPPI_RUNS} independent MPPI passes ===")
        run_chunks_raw, run_net_disp, run_cos_sim = [], [], []
        run_diagnostics = []

        for run_idx in range(N_MPPI_RUNS):
            # Fresh instance per run: prev_best_action is None, so the initial population comes
            # from Uniform(-1,1). Combined with the per-run reseed below, run i draws the
            # IDENTICAL initial population in every arm -- arms diverge only by how many refits
            # they then perform, which is the whole point of the sweep.
            mppi = MPPI(
                dynamics_fn=dynamics_fn,
                cost_fn=cost_fn,
                pred_horizon=HORIZON,
                action_dim=2,
                num_samples=K,
                mppi_temperature=temperature,
                n_planning_itrs=n_itrs,
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

            diags = mppi.last_diagnostics or []
            final_std = diags[-1]["std_mean"] if diags else float("nan")
            final_ess = diags[-1]["ess"] if diags else float("nan")

            print(f"  run {run_idx + 1} (seed={SAMPLE_SEEDS[run_idx]})  cos_sim={cos_sim:.4f}  "
                  f"net_disp=({disp[0]:+.4f},{disp[1]:+.4f})  |disp|={mag:.4f}  "
                  f"dir_align={dir_align:+.4f}  std_final={final_std:.2e}  "
                  f"ESS_final={final_ess:6.2f}")

            run_chunks_raw.append(chunk_raw)
            run_net_disp.append(disp)
            run_cos_sim.append(cos_sim)
            run_diagnostics.append(diags)

        run_chunks_raw = np.stack(run_chunks_raw, axis=0)
        run_net_disp = np.stack(run_net_disp, axis=0)
        run_cos_sim = np.array(run_cos_sim)

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
            std_final=float(np.mean([d[-1]["std_mean"] for d in run_diagnostics if d])),
            ess_final=float(np.mean([d[-1]["ess"] for d in run_diagnostics if d])),
        )
        print(f"  cross-seed: mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"range={stats['rng']:.4f}  best-ref={stats['vs_ref']:+.4f}  "
              f"net-disp spread={stats['disp_spread']:.4f}")

        arms[n_evals] = dict(
            n_planning_itrs=n_itrs,
            run_chunks_raw=run_chunks_raw,
            run_net_disp=run_net_disp,
            run_cos_sim=run_cos_sim,
            run_diagnostics=run_diagnostics,
            **stats,
        )

    # -----------------------------------------------------------------------
    # 5a. Anchor reproduction check -- the assert that makes the whole sweep trustworthy.
    # -----------------------------------------------------------------------
    print("\n=== anchor reproduction check ===")
    if anchor is not None:
        got = arms[ANCHOR_EVALS]["run_cos_sim"]
        want = anchor["run_cos_sim"].astype(float)
        delta = np.abs(got - want)
        print(f"  {ANCHOR_EVALS}-eval arm this run: {np.round(got, 4)}")
        print(f"  mppi_temperature_sweep:  {np.round(want, 4)}")
        print(f"  max abs delta: {delta.max():.2e}  (tolerance {ANCHOR_RTOL:.0e})")
        assert delta.max() < ANCHOR_RTOL, (
            f"the {ANCHOR_EVALS}-evaluation arm does not reproduce mppi_temperature_sweep.py's "
            f"temp=0.1 result (max delta {delta.max():.2e}). This script's probe state, z_t, "
            f"action stats or per-seed RNG differ from it, so no arm of this sweep is "
            f"trustworthy. Investigate before reading any result above."
        )
        print("  PASS -- replicates mppi_temperature_sweep.py's temp=0.1 arm exactly")
    else:
        print("  SKIPPED -- no anchor available for this grid/temperature")

    # -----------------------------------------------------------------------
    # 5b. Nesting check -- arms are prefixes of one another (see the module docstring).
    # -----------------------------------------------------------------------
    print("\n=== nesting check (shorter arms must be exact prefixes of longer ones) ===")
    ordered = sorted(arms.keys())
    n_checked = 0
    for short, long in zip(ordered, ordered[1:]):
        for run_idx in range(N_MPPI_RUNS):
            ds = arms[short]["run_diagnostics"][run_idx]
            dl = arms[long]["run_diagnostics"][run_idx]
            for i, d in enumerate(ds):
                for key in ("ess", "std_mean", "return_spread"):
                    assert abs(d[key] - dl[i][key]) < 1e-9, (
                        f"nesting violated at evals {short} vs {long}, seed "
                        f"{SAMPLE_SEEDS[run_idx]}, iter {i}, field {key}: "
                        f"{d[key]} != {dl[i][key]}. The arms are not sharing an RNG stream, so "
                        f"they differ by more than iteration count and are not comparable."
                    )
            n_checked += 1
    print(f"  PASS -- {n_checked} arm-pair x seed prefixes verified"
          if n_checked else "  SKIPPED -- only one arm in the grid")

    # -----------------------------------------------------------------------
    # 6. Per-arm charts, sharing one axis range and one correct-direction arrow so the arms are
    #    directly comparable with each other and with the existing epoch11_mppi_5runs.png.
    # -----------------------------------------------------------------------
    all_mags = np.array([np.linalg.norm(d) for a in arms.values() for d in a["run_net_disp"]])
    ref_len = float(np.median(all_mags))
    arrow_end = correct_direction * ref_len

    all_points = [np.zeros((1, 2)), arrow_end.reshape(1, 2), cumulative_path(ref_chunk_raw)]
    all_points += [cumulative_path(c) for c in ref_rot_chunks_raw]
    for a in arms.values():
        for chunk in a["run_chunks_raw"]:
            all_points.append(cumulative_path(chunk))
    axis_half = float(np.abs(np.concatenate(all_points, axis=0)).max() * 1.15)
    axis_lim = (-axis_half, axis_half)
    print(f"\nShared plot scale: ref_len={ref_len:.4f}  axis_lim=[{-axis_half:.4f}, {axis_half:.4f}]")

    ref_path = cumulative_path(ref_chunk_raw)
    rot_paths = [cumulative_path(c) for c in ref_rot_chunks_raw]

    for n_evals, a in arms.items():
        arm_dir = os.path.join(out_dir, f"iters_{n_evals}")
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

        # Wrong-direction controls, identical on every arm (they involve no MPPI).
        for path, deg, color, c in zip(rot_paths, REF_ROTATIONS_DEG, REF_ROT_COLORS,
                                        ref_rot_cos_sim):
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.8, linestyle=":",
                    label=f"ref rot {deg}deg (wrong dir)  cos_sim={c:.4f}", zorder=2)
            ax.annotate("", xy=tuple(path[-1]), xytext=tuple(path[-2]),
                        arrowprops=dict(facecolor=color, edgecolor=color, width=1.2, headwidth=7),
                        zorder=2)
            ax.text(path[-1, 0], path[-1, 1], f"  {deg}°", color=color, fontsize=9,
                    va="center", zorder=2)

        ax.set_xlim(axis_lim); ax.set_ylim(axis_lim); ax.set_aspect("equal")
        ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("net displacement dx", color=INK)
        ax.set_ylabel("net displacement dy", color=INK)
        ax.set_title(
            f"epoch{epoch} iteration sweep -- {n_evals} evaluations "
            f"(n_planning_itrs={a['n_planning_itrs']})\n"
            f"{N_MPPI_RUNS} independent MPPI passes (seeds {SAMPLE_SEEDS[0]}-{SAMPLE_SEEDS[-1]}), "
            f"K={K}, temperature={temperature:g}\n"
            f"cross-seed std={a['std']:.4f}  range={a['rng']:.4f}  "
            f"final refit std={a['std_final']:.2e}",
            color=INK,
        )
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(arm_dir, f"epoch{epoch}_5runs.png"), dpi=150)
        plt.close()

        np.savez(
            os.path.join(arm_dir, f"epoch{epoch}_5runs_data.npz"),
            run_chunks_raw=a["run_chunks_raw"],
            run_net_disp=a["run_net_disp"],
            run_cos_sim=a["run_cos_sim"],
            ref_chunk_raw=ref_chunk_raw,
            ref_cos_sim=ref_cos_sim,
            ref_rot_chunks_raw=np.stack(ref_rot_chunks_raw, axis=0),
            ref_rot_cos_sim=np.array(ref_rot_cos_sim),
            ref_rotations_deg=np.array(REF_ROTATIONS_DEG),
            correct_direction=correct_direction,
            green_xy=npz_green, blue_xy=npz_blue, peg_xy=npz_peg,
            mppi_temperature=temperature,
            n_evals=n_evals, n_planning_itrs=a["n_planning_itrs"],
            K=K, n_mppi_runs=N_MPPI_RUNS,
            sample_seeds=np.array(SAMPLE_SEEDS), epoch=epoch,
            **{k: a[k] for k in ("mean", "std", "var", "best", "worst", "rng", "vs_ref",
                                 "disp_spread", "disp_spread_max", "std_final", "ess_final")},
        )
        print(f"Saved arm {n_evals} evals -> {arm_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 7. Summary chart: per-seed cos_sim vs total evaluations. Single y-axis (never a dual
    #    axis); seed identity carried by color AND marker shape AND the legend.
    # -----------------------------------------------------------------------
    xs = np.array(ordered, dtype=float)
    means = np.array([arms[e]["mean"] for e in ordered])
    stds = np.array([arms[e]["std"] for e in ordered])

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.fill_between(xs, means - stds, means + stds, color=INK_MUTED, alpha=0.18,
                    label="mean +/- 1 std across seeds", zorder=2)
    ax.plot(xs, means, color=INK, linewidth=2.0, marker="o", markersize=5,
            label="mean across seeds", zorder=4)
    for run_idx in range(N_MPPI_RUNS):
        ys = [arms[e]["run_cos_sim"][run_idx] for e in ordered]
        ax.plot(xs, ys, color=RUN_COLORS[run_idx], marker=RUN_MARKERS[run_idx],
                markersize=7, linewidth=1.0, alpha=0.85,
                label=f"seed {SAMPLE_SEEDS[run_idx]}", zorder=5 + run_idx)

    ax.axhline(ref_cos_sim, color=REF_COLOR, linestyle="--", linewidth=2.0,
               label=f"ref near-perfect chunk ({ref_cos_sim:.4f})", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("total dynamics evaluations per command() call (log scale)", color=INK)
    ax.set_ylabel("cos_sim of selected action chunk", color=INK)
    ax.set_title(
        f"epoch{epoch}: planner iteration count vs. selected-chunk cos_sim\n"
        f"{N_MPPI_RUNS} seeds per arm, all other hyperparameters fixed "
        f"(K={K}, temperature={temperature:g}) -- higher is better, tighter is more reproducible",
        color=INK,
    )
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_cos_sim_vs_iterations.png"), dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 8. Collapse trace: std_mean and ESS per iteration WITHIN a single command() call.
    #    This is the chart that answers the mechanism question -- whether the sampling
    #    distribution bottoms out early, making the long arms arithmetically identical to
    #    the short ones. Averaged across seeds; one line per arm.
    # -----------------------------------------------------------------------
    fig, (ax_std, ax_ess) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    arm_colors = [matplotlib.colormaps["viridis"](i / max(len(ordered) - 1, 1))
                  for i in range(len(ordered))]
    for (n_evals, color) in zip(ordered, arm_colors):
        diags = arms[n_evals]["run_diagnostics"]
        n_itr = len(diags[0])
        if n_itr == 0:
            continue
        itrs = np.arange(1, n_itr + 1)
        std_mat = np.array([[d[i]["std_mean"] for i in range(n_itr)] for d in diags])
        ess_mat = np.array([[d[i]["ess"] for i in range(n_itr)] for d in diags])
        ax_std.plot(itrs, std_mat.mean(axis=0), color=color, linewidth=1.8,
                    label=f"{n_evals} evals")
        ax_ess.plot(itrs, ess_mat.mean(axis=0), color=color, linewidth=1.8,
                    label=f"{n_evals} evals")

    ax_std.axhline(0.577, color=INK_MUTED, linestyle=":", linewidth=1.5)
    ax_std.text(1.05, 0.577, " initial Uniform(-1,1) std = 0.577", color=INK_MUTED,
                fontsize=8, va="bottom")
    ax_std.set_yscale("log")
    ax_std.set_ylabel("mean refit std (log)", color=INK)
    ax_std.set_title(
        f"epoch{epoch}: within-call collapse trace, mean over {N_MPPI_RUNS} seeds "
        f"(K={K}, temperature={temperature:g})\n"
        f"arms are nested -- a shorter arm is an exact prefix of a longer one, so these "
        f"curves overlay by construction",
        color=INK,
    )
    ax_std.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
    ax_std.set_axisbelow(True)
    ax_std.legend(loc="best", fontsize=8)

    ax_ess.axhline(K, color=INK_MUTED, linestyle=":", linewidth=1.5)
    ax_ess.text(1.05, K, f" K = {K}", color=INK_MUTED, fontsize=8, va="top")
    ax_ess.set_xscale("log")
    ax_ess.set_yscale("log")
    ax_ess.set_xlabel("refit iteration within one command() call (log scale)", color=INK)
    ax_ess.set_ylabel("effective sample size 1/sum(w^2) (log)", color=INK)
    ax_ess.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
    ax_ess.set_axisbelow(True)
    ax_ess.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "collapse_trace.png"), dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 9. Summary table (stdout + summary.txt + summary.npz)
    # -----------------------------------------------------------------------
    header = (f"{'evals':>7}  {'itrs':>5}  {'per-seed cos_sim':<40}  {'mean':>7}  {'std':>7}  "
              f"{'best':>7}  {'vs_ref':>8}  {'std_final':>10}  {'disp_spr':>9}")
    lines = [
        f"epoch {epoch} MPPI planner-iteration sweep",
        f"K={K}  mppi_temperature={temperature:g}  max_action_value={MAX_ACTION_VALUE}  "
        f"warm_start_std={WARM_START_STD}  seeds={SAMPLE_SEEDS}",
        f"ref near-perfect chunk cos_sim = {ref_cos_sim:.4f}",
        "wrong-dir controls: " + "  ".join(
            f"{d}deg={c:.4f}" for d, c in zip(REF_ROTATIONS_DEG, ref_rot_cos_sim)),
        "",
        header,
        "-" * len(header),
    ]
    for e in ordered:
        a = arms[e]
        per_seed = " ".join(f"{c:.4f}" for c in a["run_cos_sim"])
        mark = " *" if e == ANCHOR_EVALS else "  "
        lines.append(
            f"{e:>7}{mark}{a['n_planning_itrs']:>4}  {per_seed:<40}  {a['mean']:>7.4f}  "
            f"{a['std']:>7.4f}  {a['best']:>7.4f}  {a['vs_ref']:>+8.4f}  "
            f"{a['std_final']:>10.2e}  {a['disp_spread']:>9.4f}"
        )
    lines += [
        "",
        f"* = the {ANCHOR_EVALS}-evaluation arm, asserted to reproduce "
        f"mppi_temperature_sweep.py's temp=0.1 result",
        "evals = total dynamics evaluations per command() call = n_planning_itrs + 1",
        "std_final = mean refit std at the last iteration; the initial Uniform(-1,1) draw has",
        "            std 0.577, so this is the direct measure of population collapse.",
        "vs_ref = best-seed cos_sim minus the reference chunk's cos_sim (0 = matched it).",
    ]

    summary_txt = "\n".join(lines)
    print("\n" + summary_txt)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_txt + "\n")

    np.savez(
        os.path.join(out_dir, "summary.npz"),
        n_evals=np.array(ordered),
        n_planning_itrs=np.array([arms[e]["n_planning_itrs"] for e in ordered]),
        run_cos_sim=np.stack([arms[e]["run_cos_sim"] for e in ordered], axis=0),  # (n_arms, 5)
        mean=means, std=stds,
        var=np.array([arms[e]["var"] for e in ordered]),
        best=np.array([arms[e]["best"] for e in ordered]),
        vs_ref=np.array([arms[e]["vs_ref"] for e in ordered]),
        std_final=np.array([arms[e]["std_final"] for e in ordered]),
        ess_final=np.array([arms[e]["ess_final"] for e in ordered]),
        disp_spread=np.array([arms[e]["disp_spread"] for e in ordered]),
        ref_cos_sim=ref_cos_sim,
        ref_rot_cos_sim=np.array(ref_rot_cos_sim),
        ref_rotations_deg=np.array(REF_ROTATIONS_DEG),
        mppi_temperature=temperature,
        sample_seeds=np.array(SAMPLE_SEEDS),
        epoch=epoch,
    )
    print(f"\nDone. Outputs under {out_dir}")
    print(f"Artifacts in {DATA_DIR} were only read, never modified.")


if __name__ == "__main__":
    main()
