"""
MPPI-planning companion to cost_landscape_realistic_chunks.py, evaluated across
checkpoints at epochs 5-11 in checkpoints_clip_full_fine-tuned_with_subopt_data/.

cost_landscape_realistic_chunks.py scores a static pool of ~5000 REAL training action
chunks against a fixed probe state and plots them, colored by cos(z_pred, z_sem_goal), to
see what kind of action shape the dynamics model rates highest. This script asks a related
but different question: what does the MPPI planner (planning/mppi_core.py, driven the same
way eval/mppi_rollout_clip_full_finetune.py drives it) actually produce when run from that
SAME fixed probe state? Rather than a static pool, it runs one full MPPI planning pass
(pred_horizon=8, K=512 candidates, 9 refinement iterations) per epoch and plots the top
MPPI_TOP_K candidates from the final CEM population -- rank 1 is the exact chunk
mppi.command() would return, ranks 2..MPPI_TOP_K are near-best runners-up from that same
final population, for context on how close the alternatives were.

Uses the same fixed probe starting state (peg -> green_cube -> blue_moon, pre-aligned
collinear) and the same single-stage "touching" goal text as cost_landscape_realistic_chunks.py,
for direct comparability -- NOT mppi_rollout_clip_full_finetune.py's two-stage goal-switching,
since there's no real multi-step rollout here to stage across (MPPI plans once per epoch
against one fixed latent, it never steps the real env).

Because there's no real rollout, and only one env/pybullet client is ever constructed, this
runs epochs 0-15 sequentially in a single process -- no subprocess-per-checkpoint dance like
mppi_rollout_clip_full_finetune.py needs.

To isolate what changes across epochs to the model checkpoint (not sampling luck), the torch
RNG is reseeded with SAMPLE_SEED immediately before every epoch's mppi.command() call, and a
fresh MPPI instance is constructed per epoch.

Outputs land under main_project/MPPI_charts/expert+subopt-model/: one shared first_frame.png, plus
epoch{N}_mppi_selected_chunk.png + epoch{N}_mppi_selected_chunk_data.npz per epoch. All 7
charts share one fixed axis range and one fixed-length correct-direction arrow (computed
once from all epochs' results after every epoch's MPPI pass has run), so the charts are
directly comparable at a glance instead of each auto-scaling independently.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/cost_landscape_mppi_selected.py
"""

import os
import sys
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
OUT_DIR = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "expert+subopt-model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data")
EPOCHS = list(range(5, 12))  # 5..11, newly available checkpoints in CKPT_DIR

# MPPI hyperparameters -- identical to mppi_rollout_clip_full_finetune.py
K = 512
MPPI_TEMPERATURE = 0.01
N_PLANNING_ITRS = 9
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1

MPPI_TOP_K = 5  # rank 1 = mppi.command()'s actual output, 2..5 = near-best runners-up
SAMPLE_SEED = 123

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])

RANK_COLORS = [matplotlib.colormaps["tab10"](i) for i in range(10)]  # rank 2..K, one distinct color each
RANK1_COLOR = "gold"


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
#    cost_landscape_realistic_chunks.py.
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
#    cost_landscape_realistic_chunks.py / cost_landscape_heatmap.py.
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

away_from_blue_direction = (green_xy - blue_xy) / np.linalg.norm(green_xy - blue_xy)
peg_xy = green_xy + away_from_blue_direction * 0.06

pose = Pose3d(
    rotation=R.from_rotvec([0, np.pi, 0]),
    translation=np.array([peg_xy[0], peg_xy[1], 0.145]),
)
inner._set_robot_target_effector_pose(pose)
inner._step_simulation_to_stabilize()

dist_peg_green = np.linalg.norm(peg_xy - green_xy)
dist_peg_blue = np.linalg.norm(peg_xy - blue_xy)
dist_green_blue = np.linalg.norm(green_xy - blue_xy)
print(f"Peg start: {peg_xy}  dist(peg,{BLOCK_A})={dist_peg_green:.4f}  "
      f"dist(peg,{BLOCK_B})={dist_peg_blue:.4f}  dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}")
assert dist_peg_blue > dist_green_blue, (
    f"peg ended up between {BLOCK_A} and {BLOCK_B}, not on the outside past {BLOCK_A} -- geometry bug"
)

correct_direction = (blue_xy - green_xy) / np.linalg.norm(blue_xy - green_xy)  # (2,) unit vector
print(f"Correct push direction (A->B): {correct_direction}\n")

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
# 3-6. Per-epoch: load checkpoint, run one MPPI planning pass, rank the final CEM
# population's top MPPI_TOP_K candidates, save data. Plotting is deferred to a second
# pass (section 7) so every epoch's chart can share one global axis range and one
# fixed-length correct-direction arrow, rather than each auto-scaling independently.
# ---------------------------------------------------------------------------

epoch_results = {}  # epoch -> dict(ranked_chunks_raw, ranked_net_disp, ranked_cos_sim)

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

    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_sem = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)

        img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
        z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)

    # --- MPPI dynamics_fn / cost_fn, mirroring mppi_rollout_clip_full_finetune.py ---
    def dynamics_fn(state, action):
        # state:  (K, 768)   -- batch of latent obs
        # action: (K, 8, 2)  -- normalized action chunks
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        # terminal_state: (K, 768), z_sem: (768,)
        return 1.0 - (terminal_state * z_sem).sum(dim=-1)  # (K,)

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

    # Reseed immediately before command() so every epoch's CEM search starts from the
    # identical random draw -- isolates what changes across epochs to the checkpoint,
    # not sampling luck.
    torch.manual_seed(SAMPLE_SEED)
    with torch.no_grad():
        best_action_chunk = mppi.command(z_t)  # (8, 2) normalized, also sets mppi.last_*

    # cost_fn is exactly 1 - cos_sim with no other terms, so last_returns == cos_sim - 1
    # for the whole final population -- recover cos_sim directly, no extra forward passes.
    final_actions = mppi.last_actions        # (K, 8, 2) normalized
    final_cos_sim = (mppi.last_returns + 1.0).cpu().numpy()  # (K,)

    top_k = min(MPPI_TOP_K, K)
    top_idx = np.argsort(final_cos_sim)[::-1][:top_k]
    assert final_actions[top_idx[0]].allclose(best_action_chunk), (
        "rank-1 candidate by last_returns does not match mppi.command()'s own selection"
    )

    print(f"=== Epoch {epoch}: top {top_k} MPPI candidates from final CEM population ===")
    ranked_chunks_raw = []
    ranked_net_disp = []
    ranked_cos_sim = []
    for rank, idx in enumerate(top_idx, start=1):
        chunk_norm = final_actions[idx].cpu()                                   # (8, 2)
        chunk_raw = ((chunk_norm + 1.0) / 2.0 * (ACTION_HI - ACTION_LO) + ACTION_LO).numpy()  # (8, 2)
        disp = chunk_raw.sum(axis=0)
        mag = np.linalg.norm(disp)
        angle_deg = np.degrees(np.arctan2(disp[1], disp[0]))
        disp_unit = disp / (mag + 1e-8)
        dir_align = float(np.dot(disp_unit, correct_direction))
        dir_angle_off = np.degrees(np.arccos(np.clip(dir_align, -1.0, 1.0)))
        cos_sim = float(final_cos_sim[idx])
        print(f"  #{rank}  cos_sim={cos_sim:.4f}  net_disp=({disp[0]:+.4f},{disp[1]:+.4f})  "
              f"magnitude={mag:.4f}  angle={angle_deg:+.1f}deg  "
              f"dir_align_with_correct={dir_align:.4f} ({dir_angle_off:.1f}deg off)")
        ranked_chunks_raw.append(chunk_raw)
        ranked_net_disp.append(disp)
        ranked_cos_sim.append(cos_sim)
    print()

    ranked_chunks_raw = np.stack(ranked_chunks_raw, axis=0)  # (top_k, 8, 2)
    ranked_net_disp = np.stack(ranked_net_disp, axis=0)      # (top_k, 2)
    ranked_cos_sim = np.array(ranked_cos_sim)                # (top_k,)

    epoch_results[epoch] = dict(
        ranked_chunks_raw=ranked_chunks_raw,
        ranked_net_disp=ranked_net_disp,
        ranked_cos_sim=ranked_cos_sim,
        best_action_chunk_normalized=best_action_chunk.cpu().numpy(),
        top_k=top_k,
    )

    # --- Save data ---
    data_path = os.path.join(OUT_DIR, f"epoch{epoch}_mppi_selected_chunk_data.npz")
    np.savez(
        data_path,
        ranked_chunks_raw=ranked_chunks_raw,
        ranked_net_disp=ranked_net_disp,
        ranked_cos_sim=ranked_cos_sim,
        best_action_chunk_normalized=best_action_chunk.cpu().numpy(),
        correct_direction=correct_direction,
        green_xy=green_xy,
        blue_xy=blue_xy,
        peg_xy=peg_xy,
        K=K,
        n_planning_itrs=N_PLANNING_ITRS,
        mppi_temperature=MPPI_TEMPERATURE,
        sample_seed=SAMPLE_SEED,
    )
    print(f"Saved data -> {data_path}")


# ---------------------------------------------------------------------------
# 7. Global plot scale: one fixed-length correct-direction arrow (median rank-1 net
# displacement magnitude across ALL epochs, not just this epoch's own top_k -- so the
# arrow doesn't shrink to invisibility in early/undertrained epochs) and one shared
# symmetric axis range (covering every epoch's paths AND the arrow), applied identically
# to every chart below so the 16 PNGs are directly comparable at a glance.
# ---------------------------------------------------------------------------

rank1_mags = np.array([
    np.linalg.norm(res["ranked_net_disp"][0]) for res in epoch_results.values()
])
ref_len = np.median(rank1_mags)
arrow_end = correct_direction * ref_len

all_points = [np.zeros((1, 2)), arrow_end.reshape(1, 2)]
for res in epoch_results.values():
    for chunk in res["ranked_chunks_raw"]:
        path = np.concatenate([[[0.0, 0.0]], np.cumsum(chunk, axis=0)], axis=0)  # (9,2)
        all_points.append(path)
all_points = np.concatenate(all_points, axis=0)  # (*, 2)

axis_half_range = np.abs(all_points).max() * 1.15  # 15% padding
AXIS_LIM = (-axis_half_range, axis_half_range)
print(f"Global plot scale: ref_len={ref_len:.4f}  axis_lim=[{AXIS_LIM[0]:.4f}, {AXIS_LIM[1]:.4f}]\n")


# ---------------------------------------------------------------------------
# 8. Plot each epoch's top MPPI_TOP_K candidates, rank 1 highlighted distinctly, using
# the shared axis range and correct-direction arrow computed above.
# ---------------------------------------------------------------------------

for epoch in EPOCHS:
    res = epoch_results[epoch]
    ranked_chunks_raw = res["ranked_chunks_raw"]
    ranked_cos_sim = res["ranked_cos_sim"]
    top_k = res["top_k"]

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.annotate(
        "", xy=(arrow_end[0], arrow_end[1]), xytext=(0, 0),
        arrowprops=dict(facecolor="black", edgecolor="black", width=1.5, headwidth=8),
        zorder=2,
    )
    ax.text(arrow_end[0], arrow_end[1], "  correct dir", color="black", fontsize=9, va="center", zorder=2)

    for rank in range(top_k, 0, -1):  # draw runners-up first, rank 1 last/on top
        idx = rank - 1
        chunk = ranked_chunks_raw[idx]
        cos_sim = ranked_cos_sim[idx]
        path = np.concatenate([[[0.0, 0.0]], np.cumsum(chunk, axis=0)], axis=0)  # (9,2)

        if rank == 1:
            color = RANK1_COLOR
            linewidth = 3.5
            fontweight = "bold"
            fontsize = 12
        else:
            color = RANK_COLORS[(rank - 2) % len(RANK_COLORS)]
            linewidth = 1.8
            fontweight = "normal"
            fontsize = 9

        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=linewidth,
                 label=f"#{rank}  cos_sim={cos_sim:.4f}", zorder=3 + (top_k - rank))
        ax.annotate(
            "", xy=(path[-1, 0], path[-1, 1]), xytext=(path[-2, 0], path[-2, 1]),
            arrowprops=dict(facecolor=color, edgecolor=color, width=2 if rank == 1 else 1.2,
                             headwidth=10 if rank == 1 else 7),
            zorder=4 + (top_k - rank),
        )
        ax.text(path[-1, 0], path[-1, 1], f"  #{rank}", color=color, fontsize=fontsize,
                 fontweight=fontweight, va="center", zorder=4 + (top_k - rank))

    ax.set_xlim(AXIS_LIM)
    ax.set_ylim(AXIS_LIM)
    ax.set_aspect("equal")
    ax.set_xlabel("net displacement dx")
    ax.set_ylabel("net displacement dy")
    ax.set_title(
        f"epoch{epoch} full-finetune (expert+subopt data): MPPI planning output\n"
        f"top {top_k} of {K} final-population candidates "
        f"({N_PLANNING_ITRS} refinement iters) -- #1 is mppi.command()'s actual pick"
    )
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"epoch{epoch}_mppi_selected_chunk.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved diagram -> {plot_path}")

print(f"\nDone. Outputs for epochs {EPOCHS} saved under {OUT_DIR}")
