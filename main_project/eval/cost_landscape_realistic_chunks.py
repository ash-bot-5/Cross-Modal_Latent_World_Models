"""
Realistic-chunk cost-landscape diagram for the wider_hl full-finetune dynamics model,
evaluated across checkpoints at epochs 6/7/8/9/11/12/13/14.

Follow-up to cost_landscape_heatmap.py, which probed the model with a SYNTHETIC action: one
fixed (dx, dy) repeated identically for all 8 steps of the chunk. That analysis found the
model's highest-scoring action was always small-magnitude, even with the grid widened to
+/-0.03 (matching the real single-step max). A follow-up data analysis showed real 8-step
training chunks are rarely a coherent constant push (median path-coherence 0.63 -- directions
wobble within a chunk), so the effective constant-per-step magnitude implied by a real chunk's
net displacement (median 0.0057) is roughly half the raw single-step median (0.0101) -- the
synthetic "repeat one big action 8x" probe is a shape the model essentially never saw training.

This script tests that directly: feed the model thousands of REAL, unmodified 8-step action
chunk shapes (sampled from every episode in full_data_5000, magnitude-decile-stratified so
small AND large net displacement are both well represented), score each by
cos(z_pred, z_sem_goal), and plot the result as individual curved path arrows in (dx, dy)
displacement space (cumulative sum of the 8 raw steps from the origin) -- NOT a binned grid
average -- color-coded by cos_sim, with the top 10 highest-scoring real chunks highlighted in
distinct colors so it's unambiguous which real action shapes the model rated best.

Uses the same fixed probe starting state (peg -> green_cube -> blue_moon, pre-aligned
collinear) as cost_landscape_heatmap.py, for direct comparability to those heatmaps.

The candidate chunk pool, env setup, and first-frame image are built once and reused
across every epoch's checkpoint; only the CLIP + LatentDynamicsModel weights, scoring,
and plot are redone per epoch. Outputs land under cost_landscape_realistic_wider_hl/,
one epoch{N}_realistic_chunks.png + epoch{N}_realistic_chunks_data.npz pair per epoch,
plus a single shared first_frame.png.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/cost_landscape_realistic_chunks.py
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"
HORIZON = 8

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(MAIN_PROJECT_DIR, "eval", "cost_landscape_realistic_wider_hl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_finetuned_wider_hl")
EPOCHS = [6, 7, 8, 9, 11, 12, 13, 14]

N_DECILES = 10
N_PER_DECILE = 500          # -> ~5000 candidate chunks total
N_DISPLAY = 200              # random subsample of the scored pool actually drawn
TOP_K = 10
SAMPLE_SEED = 123

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])

RANK_COLORS = [matplotlib.colormaps["tab10"](i) for i in range(10)]  # rank 1..10, one distinct color each


# ---------------------------------------------------------------------------
# Inline model classes (matches every other eval script's convention -- avoids
# triggering LangTableDataset/open_clip module-level side effects via dynamics_model.py)
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
# 1. Build the real chunk pool (every sliding 8-step window, every episode) and
#    a magnitude-decile-stratified sample of ~5000 candidates.
# ---------------------------------------------------------------------------

print(f"Building real chunk pool from {TRAIN_DIR} ...")
pkl_files = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl"))

pool_chunks = []      # list of (8,2) raw action arrays
pool_net_disp = []    # list of (2,) net displacement
all_actions_flat = []  # for global lo/hi normalization stats

for i, fn in enumerate(pkl_files):
    with open(os.path.join(TRAIN_DIR, fn), "rb") as f:
        ep = pickle.load(f)
    actions = np.array(ep["actions"], dtype=np.float32)  # (T-1, 2)
    all_actions_flat.append(actions)
    n_steps = len(actions)
    for t in range(n_steps - HORIZON + 1):
        chunk = actions[t : t + HORIZON]
        pool_chunks.append(chunk)
        pool_net_disp.append(chunk.sum(axis=0))
    if (i + 1) % 1000 == 0:
        print(f"  processed {i + 1}/{len(pkl_files)} episodes, {len(pool_chunks)} chunks so far")

pool_chunks = np.stack(pool_chunks, axis=0)          # (M, 8, 2)
pool_net_disp = np.stack(pool_net_disp, axis=0)      # (M, 2)
pool_magnitude = np.linalg.norm(pool_net_disp, axis=1)  # (M,)
print(f"Total real chunk pool: {len(pool_chunks)} chunks\n")

all_actions_flat = np.concatenate(all_actions_flat, axis=0)
ACTION_LO = torch.tensor(all_actions_flat.min(axis=0), dtype=torch.float32)
ACTION_HI = torch.tensor(all_actions_flat.max(axis=0), dtype=torch.float32)
print(f"Action normalization stats: lo={ACTION_LO.tolist()} hi={ACTION_HI.tolist()}\n")

rng = np.random.RandomState(SAMPLE_SEED)
decile_edges = np.percentile(pool_magnitude, np.linspace(0, 100, N_DECILES + 1))
sample_idx = []
for d in range(N_DECILES):
    lo_edge, hi_edge = decile_edges[d], decile_edges[d + 1]
    if d == N_DECILES - 1:
        in_bin = np.where((pool_magnitude >= lo_edge) & (pool_magnitude <= hi_edge))[0]
    else:
        in_bin = np.where((pool_magnitude >= lo_edge) & (pool_magnitude < hi_edge))[0]
    take = min(N_PER_DECILE, len(in_bin))
    sample_idx.append(rng.choice(in_bin, size=take, replace=False))
sample_idx = np.concatenate(sample_idx)

candidate_chunks = pool_chunks[sample_idx]       # (N, 8, 2)
candidate_net_disp = pool_net_disp[sample_idx]   # (N, 2)
candidate_magnitude = pool_magnitude[sample_idx]  # (N,)
N = len(candidate_chunks)
print(f"Stratified sample: {N} candidate chunks across {N_DECILES} magnitude deciles "
      f"(edges: {np.round(decile_edges, 4)})\n")


# ---------------------------------------------------------------------------
# 2. Fixed probe state: env + peg-teleport logic, verbatim from cost_landscape_heatmap.py
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
frame_path = os.path.join(OUT_DIR, "epoch8_first_frame.png")
pil_frame.save(frame_path)
print(f"Saved first frame -> {frame_path}")Practical setup: refit UMAP fresh each epoch, but hold n_neighbors and min_dist constant across all epoch fits. If you let those hyperparameters drift between epochs too, you won't be able to tell whether visual differences come from the model or from the projection settings.

One more thing worth adding to the same diagnostic pass, since you're already doing per-epoch fits: log the raw metric your margin probe uses (cosine sim or whatever) alongside the UMAP chart, so you have a quantitative check that agrees or disagrees with what the plot shows. UMAP plots are good for spotting qualitative structure but easy to overread — pairing it with the margin probe number keeps you honest about whether "looks separated" corresponds to "is separated" in the metric you actually care about.


# ---------------------------------------------------------------------------
# 3. Load epoch-8 checkpoint: CLIP (full finetune, both towers) + LatentDynamicsModel
# ---------------------------------------------------------------------------

print(f"\nLoading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
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
Practical setup: refit UMAP fresh each epoch, but hold n_neighbors and min_dist constant across all epoch fits. If you let those hyperparameters drift between epochs too, you won't be able to tell whether visual differences come from the model or from the projection settings.

One more thing worth adding to the same diagnostic pass, since you're already doing per-epoch fits: log the raw metric your margin probe uses (cosine sim or whatever) alongside the UMAP chart, so you have a quantitative check that agrees or disagrees with what the plot shows. UMAP plots are good for spotting qualitative structure but easy to overread — pairing it with the margin probe number keeps you honest about whether "looks separated" corresponds to "is separated" in the metric you actually care about.
goal_text = (
    f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
    f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
)
print(f"Goal: {goal_text}")

with torch.no_grad():
    tokens = clip_tokenizer([goal_text]).to(DEVICE)
    z_sem = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)

    img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
    z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)


# ---------------------------------------------------------------------------
# 4. Batched forward pass: score all N candidate REAL chunks
# ---------------------------------------------------------------------------

lo_t = ACTION_LO.to(DEVICE)
hi_t = ACTION_HI.to(DEVICE)
candidates_raw_t = torch.tensor(candidate_chunks, dtype=torch.float32).to(DEVICE)  # (N, 8, 2)
candidates_norm = (candidates_raw_t - lo_t) / (hi_t - lo_t) * 2.0 - 1.0            # (N, 8, 2)

z_t_batch = z_t.unsqueeze(0).repeat(N, 1)  # (N, 768)

with torch.no_grad():
    z_pred_batch = model(z_t_batch, candidates_norm)               # (N, 768), L2-normalized
    cos_sim = (z_pred_batch * z_sem.unsqueeze(0)).sum(dim=-1)      # (N,)

cos_sim = cos_sim.cpu().numpy()
print(f"\nScored {N} real candidate chunks. cos_sim range: "
      f"[{cos_sim.min():.4f}, {cos_sim.max():.4f}], mean={cos_sim.mean():.4f}\n")


# ---------------------------------------------------------------------------
# 5. Top K + terminal report
# ---------------------------------------------------------------------------

top_idx = np.argsort(cos_sim)[::-1][:TOP_K]

print(f"=== Top {TOP_K} real action chunks by predicted cos_sim ===")
for rank, idx in enumerate(top_idx, start=1):
    disp = candidate_net_disp[idx]
    mag = candidate_magnitude[idx]
    angle_deg = np.degrees(np.arctan2(disp[1], disp[0]))
    disp_unit = disp / (mag + 1e-8)
    dir_align = float(np.dot(disp_unit, correct_direction))
    dir_angle_off = np.degrees(np.arccos(np.clip(dir_align, -1.0, 1.0)))
    print(f"  #{rank}  cos_sim={cos_sim[idx]:.4f}  net_disp=({disp[0]:+.4f},{disp[1]:+.4f})  "
          f"magnitude={mag:.4f}  angle={angle_deg:+.1f}deg  "
          f"dir_align_with_correct={dir_align:.4f} ({dir_angle_off:.1f}deg off)")
print()


# ---------------------------------------------------------------------------
# 6. Display subset (avoid clutter) -- random subsample of the scored pool,
#    already magnitude-decile-balanced by construction; top K always included.
# ---------------------------------------------------------------------------

display_rng = np.random.RandomState(SAMPLE_SEED + 1)
background_pool = np.setdiff1d(np.arange(N), top_idx)
n_bg = min(N_DISPLAY - TOP_K, len(background_pool))
background_idx = display_rng.choice(background_pool, size=n_bg, replace=False)


# ---------------------------------------------------------------------------
# 7. Plot: individual curved path arrows (cumulative 8-step polylines from origin)
# ---------------------------------------------------------------------------

norm = mcolors.Normalize(vmin=cos_sim.min(), vmax=cos_sim.max())
cmap = matplotlib.colormaps["viridis"]

fig, ax = plt.subplots(figsize=(9, 9))

for idx in background_idx:
    path = np.concatenate([[[0.0, 0.0]], np.cumsum(candidate_chunks[idx], axis=0)], axis=0)  # (9,2)
    ax.plot(path[:, 0], path[:, 1], color=cmap(norm(cos_sim[idx])), alpha=0.3, linewidth=1, zorder=1)

# correct-direction reference arrow, scaled to a representative display length
ref_len = np.median(candidate_magnitude)
arrow_end = correct_direction * ref_len
ax.annotate(
    "", xy=(arrow_end[0], arrow_end[1]), xytext=(0, 0),
    arrowprops=dict(facecolor="black", edgecolor="black", width=1.5, headwidth=8),
    zorder=2,
)
ax.text(arrow_end[0], arrow_end[1], "  correct dir", color="black", fontsize=9, va="center", zorder=2)

for rank, idx in enumerate(top_idx, start=1):
    color = RANK_COLORS[rank - 1]
    path = np.concatenate([[[0.0, 0.0]], np.cumsum(candidate_chunks[idx], axis=0)], axis=0)  # (9,2)
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5,
             label=f"#{rank}  cos_sim={cos_sim[idx]:.4f}", zorder=3)
    ax.annotate(
        "", xy=(path[-1, 0], path[-1, 1]), xytext=(path[-2, 0], path[-2, 1]),
        arrowprops=dict(facecolor=color, edgecolor=color, width=2, headwidth=10),
        zorder=4,
    )
    ax.text(path[-1, 0], path[-1, 1], f"  #{rank}", color=color, fontsize=11,
             fontweight="bold", va="center", zorder=4)

ax.set_aspect("equal")
ax.set_xlabel("net displacement dx")
ax.set_ylabel("net displacement dy")
ax.set_title(
    f"epoch8 full-finetune: real 8-step action chunks scored by cos(z_pred, z_sem)\n"
    f"{len(background_idx)} background paths + top {TOP_K} highlighted "
    f"(of {N} scored, {len(pool_chunks)} in full pool)"
)
ax.legend(loc="upper left", fontsize=8)

sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label="cosine similarity")

plt.tight_layout()
heatmap_path = os.path.join(OUT_DIR, "epoch8_realistic_chunks.png")
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Saved diagram -> {heatmap_path}")


# ---------------------------------------------------------------------------
# 8. Save data
# ---------------------------------------------------------------------------

data_path = os.path.join(OUT_DIR, "epoch8_realistic_chunks_data.npz")
np.savez(
    data_path,
    candidate_chunks=candidate_chunks,
    candidate_net_disp=candidate_net_disp,
    cos_sim=cos_sim,
    top_idx=top_idx,
    correct_direction=correct_direction,
    green_xy=green_xy,
    blue_xy=blue_xy,
    peg_xy=peg_xy,
)
print(f"Saved data -> {data_path}")
print(f"\nDone. Outputs saved under {OUT_DIR}")
