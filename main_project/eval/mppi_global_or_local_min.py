"""
Global-vs-local-minima probe for the MPPI planner, built off
cost_landscape_mppi_selected.py.

cost_landscape_mppi_selected.py runs ONE full MPPI planning pass per epoch and plots the
top-5 candidates from that single run's final CEM population -- i.e. "how close were the
runners-up to the winner in one search." That does not tell us whether the planner finds
the SAME optimum every time it searches, or lands somewhere different depending on sampling
luck.

This script asks that question directly. For each epoch checkpoint it runs FIVE independent
full MPPI planning passes from the SAME fixed probe state, SAME goal text, and SAME
hyperparameters, and plots each pass's single mppi.command() winner together on one chart.
The interpretation:
  - 5 winners converging to (approximately) the same net displacement
        => one dominant basin: a global minimum the planner reliably finds.
  - 5 winners scattering to different net displacements
        => many local minima in the cos(z_pred, z_goal) landscape that the planner falls
           into unpredictably depending on the random draw.

Seeding (this is the crux of the test): MPPI is fully deterministic given dynamics_fn,
cost_fn, and the torch RNG state, so reseeding all 5 runs identically would reproduce
bit-identical output every time and trivially "prove" convergence. Instead each of the 5
runs gets a DISTINCT seed (SAMPLE_SEED + 0..4). The "same initial action distribution"
requirement is met structurally: a FRESH MPPI instance is built for each run, and
mppi_core.py's _sample_initial_actions() always draws the first population from
Uniform(-1,1) when prev_best_action is None (true for every fresh instance) -- so the
distribution is identical across runs, only the RNG draws differ. The same 5 seeds are
reused across every epoch, so run i always starts from the same seed in every epoch,
isolating what changes across epochs to the checkpoint rather than to sampling luck (same
rationale as cost_landscape_mppi_selected.py's per-epoch reseed).

Uses the same fixed probe starting state (peg -> green_cube -> blue_moon, pre-aligned
collinear) and the same single-stage "touching" goal text as cost_landscape_mppi_selected.py,
for direct comparability. One deliberate difference: the peg's standoff from green_cube is
0.045 rather than 0.06, so at t=0 the peg is already TOUCHING green_cube (dist < the
project-wide TOUCH_THRESH of 0.05) instead of sitting just outside touching range. The
direction logic is untouched -- the peg is still collinear with green_cube and blue_moon on
the far side of green_cube, so correct_direction (green_cube -> blue_moon) is unchanged.
Reaching that pose requires a two-stage peg placement (park clear of the block on the axis,
then move inward); see the PEG_PARK_OFFSET comment for why a single teleport does not work.

Because there's no real rollout, and only one env/pybullet client is ever constructed, this
runs all epochs sequentially in a single process.

Outputs land under main_project/MPPI_charts/min_determination/expert+subopt-model/: one shared
first_frame.png, plus epoch{N}_mppi_5runs.png + epoch{N}_mppi_5runs_data.npz per epoch. All
charts share one fixed axis range and one fixed-length correct-direction arrow (computed once
from all epochs' results after every epoch's MPPI passes have run), so the charts are
directly comparable at a glance instead of each auto-scaling independently.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_global_or_local_min.py
"""

import os
import re
import sys
import glob
import pickle

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
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"
HORIZON = 8

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "min_determination", "expert+subopt-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data")


def _discover_epochs(ckpt_dir: str) -> list[int]:
    """Every epoch_NNN.pt checkpoint in ckpt_dir, sorted ascending -- the user wants ALL
    epochs in the folder, not a hardcoded subset."""
    epochs = []
    for path in glob.glob(os.path.join(ckpt_dir, "epoch_*.pt")):
        m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(path))
        if m:
            epochs.append(int(m.group(1)))
    return sorted(epochs)


EPOCHS = _discover_epochs(CKPT_DIR)

# MPPI hyperparameters -- identical to cost_landscape_mppi_selected.py /
# mppi_rollout_clip_full_finetune.py
K = 512
MPPI_TEMPERATURE = 0.01
N_PLANNING_ITRS = 9
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1

N_MPPI_RUNS = 5  # independent full MPPI planning passes per epoch
SAMPLE_SEED = 123
SAMPLE_SEEDS = [SAMPLE_SEED + i for i in range(N_MPPI_RUNS)]  # distinct seed per run, reused across epochs

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])

RUN_COLORS = [matplotlib.colormaps["tab10"](i) for i in range(10)]  # one distinct color per run
REF_COLOR = "magenta"  # 6th path: expert near-perfect (max-aligned) reference chunk -- deliberately
                        # NOT a tab10 color, so it never collides with the 5 MPPI-run colors

# Reference "near-perfect" chunk: 8 steps pointing along correct_direction, each sized to the
# median expert per-step magnitude, with small fixed-seed jitter so the steps vary naturally
# (rather than being the identical action 8 times). Fixed => identical on every chart.
REF_SEED = 7
REF_JITTER_FRAC = 0.4  # per-step Gaussian jitter std, as a fraction of the median expert step size

# Wrong-direction controls: the same near-perfect chunk rotated 90/180/270 deg. A well-behaved
# model should rate all three BELOW the aligned reference (they push the wrong way), so their
# predicted cos_sim is a sanity check that the model penalizes wrong-direction actions.
REF_ROTATIONS_DEG = [90, 180, 270]
REF_ROT_COLORS = ["teal", "saddlebrown", "dimgray"]  # one per rotation; distinct from runs + ref


# ---------------------------------------------------------------------------
# Inline model classes (matches every other eval script's convention -- avoids
# triggering LangTableDataset/open_clip module-level side effects via dynamics_model.py).
# Widened hidden dim (1024) + Conv1d action encoder, matching the current
# dynamics_model.py architecture that checkpoints_clip_full_finetuned_wider_hl/ was
# trained with.
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
# 1. Action normalization stats (needed to denormalize MPPI's [-1,1] output back to
#    raw physical units for plotting) -- same source/convention as
#    cost_landscape_mppi_selected.py.
# ---------------------------------------------------------------------------

print(f"Loading action stats from {TRAIN_DIR} ...")
all_actions = []
for pkl_file in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl")):
    with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
        ep = pickle.load(f)
    all_actions.append(np.array(ep["actions"], dtype=np.float32))
all_actions = np.concatenate(all_actions, axis=0)
ACTION_LO = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)
ACTION_HI = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)
print(f"Action normalization stats: lo={ACTION_LO.tolist()} hi={ACTION_HI.tolist()}\n")


# ---------------------------------------------------------------------------
# 2. Fixed probe state: env + peg-teleport logic, verbatim from
#    cost_landscape_mppi_selected.py / cost_landscape_realistic_chunks.py.
# ---------------------------------------------------------------------------

print("Initializing LangTable env ...")
from swm.utils.envs import get_lang_table_env
env = get_lang_table_env(
    {"ood": False, "block_combo": [BLOCK_A, BLOCK_B]},
    seed=42,
)
print("Env ready.")

from language_table.environments.utils.pose3d import Pose3d
from scipy.spatial.transform import Rotation as R

inner = env.env
blocks = inner.get_block_states()
green_xy = blocks[BLOCK_A]
blue_xy = blocks[BLOCK_B]

assert np.allclose(green_xy, EXPECTED_GREEN_XY, atol=1e-5), (
    f"{BLOCK_A} position {green_xy} does not match expected {EXPECTED_GREEN_XY} "
    f"-- table arrangement has diverged."
)
assert np.allclose(blue_xy, EXPECTED_BLUE_XY, atol=1e-5), (
    f"{BLOCK_B} position {blue_xy} does not match expected {EXPECTED_BLUE_XY} "
    f"-- table arrangement has diverged."
)
print(f"Block arrangement verified identical to existing rollouts: "
      f"{BLOCK_A}={green_xy}, {BLOCK_B}={blue_xy}")

# Direction logic unchanged: the peg sits on the far side of BLOCK_A from BLOCK_B, collinear
# with both, so pushing along correct_direction (A->B) is the correct plan. Only the standoff
# distance changed -- PEG_START_OFFSET < TOUCH_THRESH so the peg starts the episode already
# TOUCHING BLOCK_A (the previous 0.06 put it just outside the touching threshold).
TOUCH_THRESH = 0.05         # peg/block touching threshold -- matches TOUCHING_DISTANCE_THRESHOLD
                            # in language_table/environments/lang_table_for_demo.py and the
                            # project-wide convention in caching/cache_zhard.py et al.
PEG_START_OFFSET = 0.045    # final peg standoff from BLOCK_A's center along away_from_blue_direction

# Two-stage approach, and this is load-bearing -- do NOT collapse it into a single teleport.
# _set_robot_target_effector_pose only sets a TARGET; the arm then drives toward it during
# _step_simulation_to_stabilize(). Commanding the final touching-range pose directly from the
# arm's rest pose makes that path sweep straight through BLOCK_A and launch it (measured: a
# single teleport to 0.045 displaces BLOCK_A by ~0.06 and inflates dist(A,B) from 0.199 to
# ~0.24; at 0.04 it displaces it by 0.138). Parking first at PEG_PARK_OFFSET -- on the same
# axis but clear of the block -- means the final inward move approaches along the axis and
# disturbs nothing: BLOCK_A stays put to <1e-4 and dist(A,B) is preserved exactly.
PEG_PARK_OFFSET = 0.10

away_from_blue_direction = (green_xy - blue_xy) / np.linalg.norm(green_xy - blue_xy)
peg_xy = green_xy + away_from_blue_direction * PEG_START_OFFSET


def _move_peg_to(xy):
    inner._set_robot_target_effector_pose(Pose3d(
        rotation=R.from_rotvec([0, np.pi, 0]),
        translation=np.array([xy[0], xy[1], 0.145]),
    ))
    inner._step_simulation_to_stabilize()


_move_peg_to(green_xy + away_from_blue_direction * PEG_PARK_OFFSET)  # stage 1: park clear
_move_peg_to(peg_xy)                                                # stage 2: move into contact range

dist_peg_green = np.linalg.norm(peg_xy - green_xy)
dist_peg_blue = np.linalg.norm(peg_xy - blue_xy)
dist_green_blue = np.linalg.norm(green_xy - blue_xy)
print(f"Peg start (commanded): {peg_xy}  dist(peg,{BLOCK_A})={dist_peg_green:.4f}  "
      f"dist(peg,{BLOCK_B})={dist_peg_blue:.4f}  dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}")
assert dist_peg_blue > dist_green_blue, (
    f"peg ended up between {BLOCK_A} and {BLOCK_B}, not on the outside past {BLOCK_A} -- geometry bug"
)
assert dist_peg_green < TOUCH_THRESH, (
    f"commanded peg start is {dist_peg_green:.4f} from {BLOCK_A}, not < {TOUCH_THRESH} -- "
    f"the peg would NOT count as touching {BLOCK_A} at t=0"
)

# The peg now starts inside touching range, so verify the REALIZED geometry rather than trusting
# the commanded pose: IK settles a fraction of a mm short of the target, and a regression in the
# staged approach above would show up here as a displaced BLOCK_A. Everything downstream
# (correct_direction, the EXPECTED_* asserts, the saved green_xy/blue_xy) assumes the blocks
# never moved, so that is asserted explicitly.
blocks_after = inner.get_block_states()
green_xy_after = blocks_after[BLOCK_A]
blue_xy_after = blocks_after[BLOCK_B]
peg_xy_after = np.asarray(inner._robot.forward_kinematics().translation[:2], dtype=np.float64)
dist_peg_green_after = np.linalg.norm(peg_xy_after - green_xy_after)
dist_peg_blue_after = np.linalg.norm(peg_xy_after - blue_xy_after)
dist_green_blue_after = np.linalg.norm(green_xy_after - blue_xy_after)
green_moved = np.linalg.norm(green_xy_after - green_xy)
print(f"Peg start (realized):  {peg_xy_after}  dist(peg,{BLOCK_A})={dist_peg_green_after:.4f} "
      f"(touching: {dist_peg_green_after < TOUCH_THRESH})  "
      f"dist(peg,{BLOCK_B})={dist_peg_blue_after:.4f}\n"
      f"  {BLOCK_A} moved {green_moved:.6f} during peg placement  "
      f"dist({BLOCK_A},{BLOCK_B})={dist_green_blue_after:.4f} (was {dist_green_blue:.4f})")
assert dist_peg_green_after < TOUCH_THRESH, (
    f"realized peg start is {dist_peg_green_after:.4f} from {BLOCK_A}, not < {TOUCH_THRESH} -- "
    f"peg is not touching {BLOCK_A} at t=0"
)
assert green_moved < 1e-3, (
    f"{BLOCK_A} moved {green_moved:.6f} while placing the peg -- the staged approach above is "
    f"supposed to leave it untouched, so the probe state (and correct_direction) is invalid"
)
assert dist_peg_blue_after > dist_green_blue_after, (
    f"realized peg position sits between {BLOCK_A} and {BLOCK_B} -- geometry bug"
)

correct_direction = (blue_xy - green_xy) / np.linalg.norm(blue_xy - green_xy)  # (2,) unit vector
print(f"Correct push direction (A->B): {correct_direction}\n")

# --- Reference "near-perfect" action chunk -------------------------------------------------
# A realistic, well-aligned chunk to contrast against the 5 MPPI winners: each of the 8 steps
# points along correct_direction at the median expert per-step magnitude, plus small
# fixed-seed Gaussian jitter so the steps vary naturally instead of being one action repeated
# 8 times. Fixed across every epoch/chart. Its net direction is essentially correct_direction
# (jitter averages out), so it represents an almost-perfectly-aligned push -- the test is
# whether the model's predicted cos_sim for it beats the MPPI winners' cos_sims.
median_step = float(np.median(np.linalg.norm(all_actions, axis=1)))
_ref_rng = np.random.default_rng(REF_SEED)
REF_CHUNK_RAW = (
    correct_direction[None, :] * median_step
    + _ref_rng.normal(0.0, REF_JITTER_FRAC * median_step, size=(HORIZON, 2))
).astype(np.float32)  # (8, 2) raw action units
# Normalize to [-1, 1] the same way LangTableDataset does, then clamp, for the model forward.
REF_CHUNK_NORM = np.clip(
    2.0 * (REF_CHUNK_RAW - ACTION_LO.numpy()) / (ACTION_HI.numpy() - ACTION_LO.numpy()) - 1.0,
    -1.0, 1.0,
).astype(np.float32)
_ref_disp = REF_CHUNK_RAW.sum(axis=0)
_ref_align = float(np.dot(_ref_disp / (np.linalg.norm(_ref_disp) + 1e-8), correct_direction))
print(f"Reference chunk: median expert step={median_step:.4f}  net_disp=({_ref_disp[0]:+.4f},"
      f"{_ref_disp[1]:+.4f})  |disp|={np.linalg.norm(_ref_disp):.4f}  "
      f"dir_align_with_correct={_ref_align:.4f}\n")
REF_CHUNK_NORM_T = torch.tensor(REF_CHUNK_NORM, device=DEVICE).unsqueeze(0)  # (1, 8, 2)

# --- Wrong-direction rotations of the reference chunk (90/180/270 deg) ----------------------
# Rotate every per-step (dx, dy) of the reference chunk by each angle, then renormalize/clamp
# for the model forward. Same jitter shape, just pointed the wrong way -- controls for "does
# the model rate wrong-direction actions as bad".
REF_ROT_CHUNKS_RAW = []      # list of (8, 2) raw
REF_ROT_CHUNKS_NORM_T = []   # list of (1, 8, 2) normalized tensors on DEVICE
for _deg in REF_ROTATIONS_DEG:
    _th = np.radians(_deg)
    _Rm = np.array([[np.cos(_th), -np.sin(_th)],
                    [np.sin(_th),  np.cos(_th)]], dtype=np.float32)
    _rot_raw = (REF_CHUNK_RAW @ _Rm.T).astype(np.float32)  # rotate each row vector
    _rot_norm = np.clip(
        2.0 * (_rot_raw - ACTION_LO.numpy()) / (ACTION_HI.numpy() - ACTION_LO.numpy()) - 1.0,
        -1.0, 1.0,
    ).astype(np.float32)
    REF_ROT_CHUNKS_RAW.append(_rot_raw)
    REF_ROT_CHUNKS_NORM_T.append(torch.tensor(_rot_norm, device=DEVICE).unsqueeze(0))

os.makedirs(OUT_DIR, exist_ok=True)
pil_frame = env.get_frame()
frame_path = os.path.join(OUT_DIR, "first_frame.png")
pil_frame.save(frame_path)
print(f"Saved first frame -> {frame_path}")

goal_text = (
    f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
    f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
)
print(f"Goal: {goal_text}")

lo_t = ACTION_LO.to(DEVICE)
hi_t = ACTION_HI.to(DEVICE)


# ---------------------------------------------------------------------------
# 3-6. Per-epoch: load checkpoint, run N_MPPI_RUNS independent MPPI planning passes (each
# a fresh MPPI instance with a distinct seed), collect each pass's winner, save data.
# Plotting is deferred to a second pass (section 7) so every epoch's chart can share one
# global axis range and one fixed-length correct-direction arrow.
# ---------------------------------------------------------------------------

epoch_results = {}  # epoch -> dict(run_chunks_raw, run_net_disp, run_cos_sim)

for epoch in EPOCHS:
    ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pt")
    print(f"\n=== Epoch {epoch}: loading checkpoint {ckpt_path} ===")
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

    # z_sem (goal) and z_t (probe frame) do not change across the 5 runs -- compute once.
    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_sem = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)

        img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
        z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)

        # Reference near-perfect chunk's predicted cos_sim under THIS epoch's model.
        z_pred_ref = model(z_t.unsqueeze(0), REF_CHUNK_NORM_T)  # (1, 768)
        ref_cos_sim = float((z_pred_ref * z_sem).sum(dim=-1).item())

        # Wrong-direction rotations' predicted cos_sim under THIS epoch's model.
        ref_rot_cos_sim = []
        for _rot_norm_t in REF_ROT_CHUNKS_NORM_T:
            _z_pred_rot = model(z_t.unsqueeze(0), _rot_norm_t)  # (1, 768)
            ref_rot_cos_sim.append(float((_z_pred_rot * z_sem).sum(dim=-1).item()))
    print(f"  reference near-perfect chunk: cos_sim={ref_cos_sim:.4f}")
    for _deg, _c in zip(REF_ROTATIONS_DEG, ref_rot_cos_sim):
        print(f"    rot {_deg:>3}deg (wrong dir): cos_sim={_c:.4f}")

    # --- MPPI dynamics_fn / cost_fn, mirroring mppi_rollout_clip_full_finetune.py ---
    def dynamics_fn(state, action):
        # state:  (K, 768)   -- batch of latent obs
        # action: (K, 8, 2)  -- normalized action chunks
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        # terminal_state: (K, 768), z_sem: (768,)
        return 1.0 - (terminal_state * z_sem).sum(dim=-1)  # (K,)

    print(f"=== Epoch {epoch}: {N_MPPI_RUNS} independent MPPI planning passes ===")
    run_chunks_raw = []
    run_net_disp = []
    run_cos_sim = []
    for run_idx in range(N_MPPI_RUNS):
        # Fresh MPPI instance per run: prev_best_action starts None, so the initial
        # population is drawn from Uniform(-1,1) -- identical distribution across all runs.
        mppi = MPPI(
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            pred_horizon=HORIZON,
            action_dim=2,
            num_samples=K,
            mppi_temperature=MPPI_TEMPERATURE,
            n_planning_itrs=N_PLANNING_ITRS,
            device=DEVICE,
            max_action_value=MAX_ACTION_VALUE,
            warm_start_std=WARM_START_STD,
        )

        # Distinct seed per run -- the whole point: different random draws, so convergence
        # of the winners reflects a real basin, not seed reuse.
        torch.manual_seed(SAMPLE_SEEDS[run_idx])
        with torch.no_grad():
            best_action_chunk = mppi.command(z_t)  # (8, 2) normalized, also sets mppi.last_*

        # cost_fn is exactly 1 - cos_sim with no other terms, so last_returns == cos_sim - 1
        # for the whole final population -- recover the winner's cos_sim directly.
        final_actions = mppi.last_actions            # (K, 8, 2) normalized
        final_returns = mppi.last_returns            # (K,)
        best_idx = int(torch.argmax(final_returns))
        assert final_actions[best_idx].allclose(best_action_chunk), (
            "argmax of last_returns does not match mppi.command()'s own selection"
        )
        cos_sim = float(final_returns[best_idx].item() + 1.0)

        chunk_norm = best_action_chunk.cpu()                                    # (8, 2)
        chunk_raw = ((chunk_norm + 1.0) / 2.0 * (ACTION_HI - ACTION_LO) + ACTION_LO).numpy()  # (8, 2)
        disp = chunk_raw.sum(axis=0)
        mag = np.linalg.norm(disp)
        angle_deg = np.degrees(np.arctan2(disp[1], disp[0]))
        disp_unit = disp / (mag + 1e-8)
        dir_align = float(np.dot(disp_unit, correct_direction))
        dir_angle_off = np.degrees(np.arccos(np.clip(dir_align, -1.0, 1.0)))
        print(f"  run {run_idx + 1} (seed={SAMPLE_SEEDS[run_idx]})  cos_sim={cos_sim:.4f}  "
              f"net_disp=({disp[0]:+.4f},{disp[1]:+.4f})  magnitude={mag:.4f}  "
              f"angle={angle_deg:+.1f}deg  dir_align_with_correct={dir_align:.4f} "
              f"({dir_angle_off:.1f}deg off)")
        run_chunks_raw.append(chunk_raw)
        run_net_disp.append(disp)
        run_cos_sim.append(cos_sim)

    # Spread diagnostic: pairwise net-displacement distances across the 5 winners. Small =>
    # convergence to one basin; large => scattered local minima.
    disp_arr = np.stack(run_net_disp, axis=0)  # (N_MPPI_RUNS, 2)
    pdist = [np.linalg.norm(disp_arr[i] - disp_arr[j])
             for i in range(N_MPPI_RUNS) for j in range(i + 1, N_MPPI_RUNS)]
    print(f"  winner net-disp spread: mean pairwise dist={np.mean(pdist):.4f}  "
          f"max={np.max(pdist):.4f}\n")

    run_chunks_raw = np.stack(run_chunks_raw, axis=0)  # (N_MPPI_RUNS, 8, 2)
    run_net_disp = np.stack(run_net_disp, axis=0)      # (N_MPPI_RUNS, 2)
    run_cos_sim = np.array(run_cos_sim)                # (N_MPPI_RUNS,)

    epoch_results[epoch] = dict(
        run_chunks_raw=run_chunks_raw,
        run_net_disp=run_net_disp,
        run_cos_sim=run_cos_sim,
        ref_cos_sim=ref_cos_sim,
        ref_rot_cos_sim=ref_rot_cos_sim,
    )

    # --- Save data ---
    data_path = os.path.join(OUT_DIR, f"epoch{epoch}_mppi_5runs_data.npz")
    np.savez(
        data_path,
        run_chunks_raw=run_chunks_raw,
        run_net_disp=run_net_disp,
        run_cos_sim=run_cos_sim,
        ref_chunk_raw=REF_CHUNK_RAW,
        ref_cos_sim=ref_cos_sim,
        ref_rot_chunks_raw=np.stack(REF_ROT_CHUNKS_RAW, axis=0),
        ref_rot_cos_sim=np.array(ref_rot_cos_sim),
        ref_rotations_deg=np.array(REF_ROTATIONS_DEG),
        correct_direction=correct_direction,
        green_xy=green_xy,
        blue_xy=blue_xy,
        peg_xy=peg_xy,
        K=K,
        n_planning_itrs=N_PLANNING_ITRS,
        mppi_temperature=MPPI_TEMPERATURE,
        n_mppi_runs=N_MPPI_RUNS,
        sample_seeds=np.array(SAMPLE_SEEDS),
    )
    print(f"Saved data -> {data_path}")


# ---------------------------------------------------------------------------
# 7. Global plot scale: one fixed-length correct-direction arrow (median winner net
# displacement magnitude across ALL epochs' ALL runs -- so the arrow doesn't shrink to
# invisibility in early/undertrained epochs) and one shared symmetric axis range (covering
# every epoch's paths AND the arrow), applied identically to every chart below so the PNGs
# are directly comparable at a glance.
# ---------------------------------------------------------------------------

all_mags = np.array([
    np.linalg.norm(disp)
    for res in epoch_results.values()
    for disp in res["run_net_disp"]
])
ref_len = np.median(all_mags)
arrow_end = correct_direction * ref_len

all_points = [np.zeros((1, 2)), arrow_end.reshape(1, 2)]
for res in epoch_results.values():
    for chunk in res["run_chunks_raw"]:
        path = np.concatenate([[[0.0, 0.0]], np.cumsum(chunk, axis=0)], axis=0)  # (9,2)
        all_points.append(path)
# The fixed reference chunk + its rotations (same every epoch) must also fit inside the shared axes.
ref_path = np.concatenate([[[0.0, 0.0]], np.cumsum(REF_CHUNK_RAW, axis=0)], axis=0)  # (9,2)
all_points.append(ref_path)
for _rot_raw in REF_ROT_CHUNKS_RAW:
    all_points.append(np.concatenate([[[0.0, 0.0]], np.cumsum(_rot_raw, axis=0)], axis=0))
all_points = np.concatenate(all_points, axis=0)  # (*, 2)

axis_half_range = np.abs(all_points).max() * 1.15  # 15% padding
AXIS_LIM = (-axis_half_range, axis_half_range)
print(f"Global plot scale: ref_len={ref_len:.4f}  axis_lim=[{AXIS_LIM[0]:.4f}, {AXIS_LIM[1]:.4f}]\n")


# ---------------------------------------------------------------------------
# 8. Plot each epoch's N_MPPI_RUNS winners together (all equally weighted -- there's no
# single "chosen" run here, each is an independent legitimate plan), using the shared axis
# range and correct-direction arrow computed above.
# ---------------------------------------------------------------------------

for epoch in EPOCHS:
    res = epoch_results[epoch]
    run_chunks_raw = res["run_chunks_raw"]
    run_cos_sim = res["run_cos_sim"]
    ref_cos_sim = res["ref_cos_sim"]
    ref_rot_cos_sim = res["ref_rot_cos_sim"]

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.annotate(
        "", xy=(arrow_end[0], arrow_end[1]), xytext=(0, 0),
        arrowprops=dict(facecolor="black", edgecolor="black", width=1.5, headwidth=8),
        zorder=2,
    )
    ax.text(arrow_end[0], arrow_end[1], "  correct dir", color="black", fontsize=9, va="center", zorder=2)

    for run_idx in range(N_MPPI_RUNS):
        chunk = run_chunks_raw[run_idx]
        cos_sim = run_cos_sim[run_idx]
        path = np.concatenate([[[0.0, 0.0]], np.cumsum(chunk, axis=0)], axis=0)  # (9,2)
        color = RUN_COLORS[run_idx % len(RUN_COLORS)]

        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.2,
                 label=f"run {run_idx + 1}  cos_sim={cos_sim:.4f}", zorder=3 + run_idx)
        ax.annotate(
            "", xy=(path[-1, 0], path[-1, 1]), xytext=(path[-2, 0], path[-2, 1]),
            arrowprops=dict(facecolor=color, edgecolor=color, width=1.4, headwidth=8),
            zorder=4 + run_idx,
        )
        ax.text(path[-1, 0], path[-1, 1], f"  {run_idx + 1}", color=color, fontsize=10,
                 va="center", zorder=4 + run_idx)

    # 6th path: the fixed expert near-perfect reference chunk (same on every chart), drawn on
    # top in a distinct color/style. Its legend cos_sim is the whole point of the test --
    # theoretically it should exceed all 5 MPPI winners' cos_sims.
    ref_path = np.concatenate([[[0.0, 0.0]], np.cumsum(REF_CHUNK_RAW, axis=0)], axis=0)  # (9,2)
    ax.plot(ref_path[:, 0], ref_path[:, 1], color=REF_COLOR, linewidth=3.0, linestyle="--",
             label=f"ref near-perfect  cos_sim={ref_cos_sim:.4f}", zorder=3 + N_MPPI_RUNS)
    ax.annotate(
        "", xy=(ref_path[-1, 0], ref_path[-1, 1]), xytext=(ref_path[-2, 0], ref_path[-2, 1]),
        arrowprops=dict(facecolor=REF_COLOR, edgecolor=REF_COLOR, width=1.8, headwidth=10),
        zorder=4 + N_MPPI_RUNS,
    )
    ax.text(ref_path[-1, 0], ref_path[-1, 1], "  ref", color=REF_COLOR, fontsize=11,
             fontweight="bold", va="center", zorder=4 + N_MPPI_RUNS)

    # Wrong-direction controls: the reference chunk rotated 90/180/270 deg. Dotted so they read
    # as "deliberately wrong". Their legend cos_sim should sit below the aligned ref's.
    for rot_idx, deg in enumerate(REF_ROTATIONS_DEG):
        rot_chunk = REF_ROT_CHUNKS_RAW[rot_idx]
        rot_cos = ref_rot_cos_sim[rot_idx]
        rot_color = REF_ROT_COLORS[rot_idx % len(REF_ROT_COLORS)]
        rot_path = np.concatenate([[[0.0, 0.0]], np.cumsum(rot_chunk, axis=0)], axis=0)  # (9,2)
        ax.plot(rot_path[:, 0], rot_path[:, 1], color=rot_color, linewidth=2.0, linestyle=":",
                 label=f"ref rot {deg}deg (wrong dir)  cos_sim={rot_cos:.4f}",
                 zorder=3 + N_MPPI_RUNS)
        ax.annotate(
            "", xy=(rot_path[-1, 0], rot_path[-1, 1]), xytext=(rot_path[-2, 0], rot_path[-2, 1]),
            arrowprops=dict(facecolor=rot_color, edgecolor=rot_color, width=1.2, headwidth=7),
            zorder=4 + N_MPPI_RUNS,
        )
        ax.text(rot_path[-1, 0], rot_path[-1, 1], f"  {deg}°", color=rot_color, fontsize=9,
                 va="center", zorder=4 + N_MPPI_RUNS)

    ax.set_xlim(AXIS_LIM)
    ax.set_ylim(AXIS_LIM)
    ax.set_aspect("equal")
    ax.set_xlabel("net displacement dx")
    ax.set_ylabel("net displacement dy")
    ax.set_title(
        f"epoch{epoch} full-finetune (expert+subopt data): global-vs-local-minima probe\n"
        f"{N_MPPI_RUNS} independent MPPI passes (distinct seeds), K={K}, "
        f"{N_PLANNING_ITRS} refinement iters -- each solid path = one pass's mppi.command() winner\n"
        f"dashed magenta = expert near-perfect ref (should score highest); "
        f"dotted = 90/180/270deg wrong-dir controls (should score low)"
    )
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"epoch{epoch}_mppi_5runs.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved diagram -> {plot_path}")

print(f"\nDone. Outputs for epochs {EPOCHS} saved under {OUT_DIR}")
