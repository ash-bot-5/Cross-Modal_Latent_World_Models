"""
Cost-landscape heatmap: cosine similarity to the goal's z_sem across a 6x6 grid of
candidate delta-xy actions, for the full-finetune checkpoints (epoch 0, 1, 8).

From a single, fixed starting state (peg->blockA->blockB pre-aligned collinear --
reusing the exact sim-state-generation logic from mppi_rollout_clip_full_finetune.py),
this evaluates every candidate action ONCE, rolling the frozen dynamics model forward
H=8 steps (repeating that single action, matching training's HORIZON) to get z_pred,
then plots cos(z_pred, z_sem_goal) as a 2D heatmap over (dx, dy). This visualizes what
MPPI/CEM actually optimizes over at inference time -- a planning-relevant complement to
the linear/MLP probe findings on raw embedding separability.

No live env.step() control loop here -- the env is used only once, to produce the single
starting frame each checkpoint's heatmap is computed from.

Run (produces 3 individual heatmaps + first-frame PNGs + .npz data, plus 1 combined
side-by-side comparison PNG, all under main_project/eval/cost_landscape_heatmaps/):
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/cost_landscape_heatmap.py

Each checkpoint's grid evaluation runs in its own subprocess (this script re-invokes
itself with --ckpt_path/--label), same pattern as mppi_rollout_clip_full_finetune.py --
pybullet only supports one physics client per process.
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"
H = 8            # matches training HORIZON -- repeat one action H times per rollout
GRID_RANGE = 0.01  # +/- per-step delta-xy magnitude, confirmed against real data + scale_factor
GRID_POINTS = 6    # 6x6 = 36 candidate actions

TRAIN_DIR    = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR      = os.path.join(MAIN_PROJECT_DIR, "eval", "cost_landscape_narrower_heatmaps")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = [
    ("full_finetune_epoch0", os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_image_text_finetuned", "epoch_000.pt")),
    ("full_finetune_epoch1", os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_image_text_finetuned", "epoch_001.pt")),
    ("full_finetune_epoch8", os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_image_text_finetuned", "epoch_008.pt")),
]

# Known-good block positions from the already-generated rollout logs
# (mppi_rollout_full_finetune_epoch{0,1,8}_green_cube_blue_moon_steps300_*.log), all
# three byte-identical -- asserted below so any divergence in the table arrangement
# (different seed, config, env version drift) fails loudly instead of silently.
EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])

# ---------------------------------------------------------------------------
# Inline model classes (importing dynamics_model.py triggers LangTableDataset
# + open_clip at module level — copy classes here to stay lightweight)
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


class LatentDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(16, 256), nn.Mish(), nn.Linear(256, 256),
        )
        self.fc1 = nn.Linear(768, 512);  self.ln1 = nn.LayerNorm(512)
        self.film1 = FiLMLayer(256, 512); self.act1 = nn.Mish()
        self.fc2 = nn.Linear(512, 512);  self.ln2 = nn.LayerNorm(512)
        self.film2 = FiLMLayer(256, 512); self.act2 = nn.Mish()
        self.fc_out = nn.Linear(512, 768); self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        global_cond = self.action_encoder(actions.flatten(1))
        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)


def run_worker(label: str, ckpt_path: str) -> None:
    """Runs the 36-action grid evaluation for one checkpoint and saves the heatmap
    data + PNG + first-frame PNG. Everything below (CLIP, env, model) is process-local
    -- safe to call exactly once per process."""

    print(f"=== {label} ===")
    print(f"Checkpoint: {ckpt_path}")

    # -----------------------------------------------------------------------
    # 1. Goal text + z_sem, using THIS checkpoint's own fine-tuned text tower
    # -----------------------------------------------------------------------
    goal_text = (
        f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
        f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
    )
    print(f"Goal: {goal_text}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_sem = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)

    # -----------------------------------------------------------------------
    # 2. Load LatentDynamicsModel
    # -----------------------------------------------------------------------
    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    # -----------------------------------------------------------------------
    # 3. Action stats from training pkls (for normalizing the candidate grid)
    # -----------------------------------------------------------------------
    print(f"Loading action stats from {TRAIN_DIR} ...")
    all_actions = []
    for pkl_file in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl")):
        with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
            ep = pickle.load(f)
        all_actions.append(np.array(ep["actions"], dtype=np.float32))
    all_actions = np.concatenate(all_actions, axis=0)
    lo = all_actions.min(axis=0)  # (2,)
    hi = all_actions.max(axis=0)  # (2,)
    print(f"  lo={lo.tolist()}  hi={hi.tolist()}")

    # -----------------------------------------------------------------------
    # 4. Init LangTable env (same call as mppi_rollout_clip_full_finetune.py)
    # -----------------------------------------------------------------------
    print("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env(
        {"ood": False, "block_combo": [BLOCK_A, BLOCK_B]},
        seed=42,
    )
    print("Env ready.")

    # -----------------------------------------------------------------------
    # 5. Peg-teleport logic, reused verbatim from mppi_rollout_clip_full_finetune.py
    # (forces peg collinear with, and just past, BLOCK_A, on the side opposite BLOCK_B)
    # -----------------------------------------------------------------------
    from language_table.environments.utils.pose3d import Pose3d
    from scipy.spatial.transform import Rotation as R

    inner = env.env
    blocks = inner.get_block_states()
    green_xy = blocks[BLOCK_A]
    blue_xy = blocks[BLOCK_B]

    # Hard-assert this matches the already-generated rollout videos' table arrangement.
    assert np.allclose(green_xy, EXPECTED_GREEN_XY, atol=1e-5), (
        f"{BLOCK_A} position {green_xy} does not match expected {EXPECTED_GREEN_XY} "
        f"from the existing rollout logs -- table arrangement has diverged."
    )
    assert np.allclose(blue_xy, EXPECTED_BLUE_XY, atol=1e-5), (
        f"{BLOCK_B} position {blue_xy} does not match expected {EXPECTED_BLUE_XY} "
        f"from the existing rollout logs -- table arrangement has diverged."
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
        f"peg ended up between {BLOCK_A} and {BLOCK_B}, not on the outside past {BLOCK_A} — geometry bug"
    )

    # -----------------------------------------------------------------------
    # 6. First frame -- the heatmap is generated from this single frame only,
    # no env.step() calls beyond the reset + peg-teleport above.
    # -----------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    pil_frame = env.get_frame()
    frame_path = os.path.join(OUT_DIR, f"{label}_first_frame.png")
    pil_frame.save(frame_path)
    print(f"Saved first frame -> {frame_path}")

    img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)

    # -----------------------------------------------------------------------
    # 7. Build the 36-candidate delta-xy grid, normalize, repeat H times, batch
    # -----------------------------------------------------------------------
    dx_vals = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_POINTS)
    dy_vals = np.linspace(-GRID_RANGE, GRID_RANGE, GRID_POINTS)
    dx_grid, dy_grid = np.meshgrid(dx_vals, dy_vals, indexing="xy")  # (6, 6) each
    candidates_raw = np.stack([dx_grid.ravel(), dy_grid.ravel()], axis=-1)  # (36, 2)

    lo_t = torch.tensor(lo, dtype=torch.float32)
    hi_t = torch.tensor(hi, dtype=torch.float32)
    candidates_raw_t = torch.tensor(candidates_raw, dtype=torch.float32)
    candidates_norm = (candidates_raw_t - lo_t) / (hi_t - lo_t) * 2.0 - 1.0  # (36, 2)

    actions_batch = candidates_norm.unsqueeze(1).repeat(1, H, 1).to(DEVICE)  # (36, 8, 2)
    z_t_batch = z_t.unsqueeze(0).repeat(actions_batch.shape[0], 1)          # (36, 768)

    with torch.no_grad():
        z_pred_batch = model(z_t_batch, actions_batch)                     # (36, 768), L2-normalized
        cos_sim = (z_pred_batch * z_sem.unsqueeze(0)).sum(dim=-1)          # (36,)

    cos_sim_grid = cos_sim.cpu().numpy().reshape(GRID_POINTS, GRID_POINTS)

    # -----------------------------------------------------------------------
    # 8. Geometrically correct direction: peg -> A -> B, i.e. the direction
    # from A toward B (pushing the peg this way pushes A toward B).
    # Assumes the env's raw 2D action maps directly to world-frame (dx, dy),
    # consistent with how the rollout script applies env.step(a2_raw) as-is.
    # -----------------------------------------------------------------------
    correct_direction = (blue_xy - green_xy) / np.linalg.norm(blue_xy - green_xy)  # (2,) unit vector
    print(f"Correct push direction (A->B): {correct_direction}")

    # -----------------------------------------------------------------------
    # 9. Save data + individual heatmap
    # -----------------------------------------------------------------------
    data_path = os.path.join(OUT_DIR, f"{label}_heatmap_data.npz")
    np.savez(
        data_path,
        dx_grid=dx_grid,
        dy_grid=dy_grid,
        cos_sim_grid=cos_sim_grid,
        correct_direction=correct_direction,
        green_xy=green_xy,
        blue_xy=blue_xy,
        peg_xy=peg_xy,
    )
    print(f"Saved heatmap data -> {data_path}")

    fig, ax = plt.subplots(figsize=(6, 6))
    mesh = ax.pcolormesh(dx_grid, dy_grid, cos_sim_grid, shading="nearest", cmap="viridis")
    ax.set_aspect("equal")
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    ax.set_title(f"{label}: cos(z_pred, z_sem) over action grid")
    fig.colorbar(mesh, ax=ax, label="cosine similarity")
    arrow_end = correct_direction * GRID_RANGE
    ax.annotate(
        "", xy=(arrow_end[0], arrow_end[1]), xytext=(0, 0),
        arrowprops=dict(facecolor="red", edgecolor="red", width=1.5, headwidth=8),
    )
    ax.text(arrow_end[0], arrow_end[1], "  correct dir", color="red", fontsize=9, va="center")
    plt.tight_layout()
    heatmap_path = os.path.join(OUT_DIR, f"{label}_heatmap.png")
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved heatmap -> {heatmap_path}")


def combine_heatmaps() -> None:
    """Loads the 3 per-checkpoint .npz files and renders one side-by-side comparison
    figure with a shared color scale."""
    loaded = []
    for label, _ in CHECKPOINTS:
        d = np.load(os.path.join(OUT_DIR, f"{label}_heatmap_data.npz"))
        loaded.append((label, d))

    all_grids = [d["cos_sim_grid"] for _, d in loaded]
    vmin = min(g.min() for g in all_grids)
    vmax = max(g.max() for g in all_grids)

    fig, axes = plt.subplots(1, len(loaded), figsize=(6 * len(loaded), 6))
    mesh = None
    for ax, (label, d) in zip(axes, loaded):
        mesh = ax.pcolormesh(
            d["dx_grid"], d["dy_grid"], d["cos_sim_grid"],
            shading="nearest", cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("dx")
        ax.set_ylabel("dy")
        ax.set_title(label)
        arrow_end = d["correct_direction"] * GRID_RANGE
        ax.annotate(
            "", xy=(arrow_end[0], arrow_end[1]), xytext=(0, 0),
            arrowprops=dict(facecolor="red", edgecolor="red", width=1.5, headwidth=8),
        )

    fig.colorbar(mesh, ax=axes, label="cosine similarity", shrink=0.8)
    fig.suptitle("Cost landscape comparison: cos(z_pred, z_sem) across full-finetune checkpoints")
    combined_path = os.path.join(OUT_DIR, "cost_landscape_comparison_epoch0_1_8.png")
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"Saved combined comparison -> {combined_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="internal: run a single checkpoint's grid evaluation (used by subprocess workers)",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="internal: descriptive label for output filenames (used with --ckpt_path)",
    )
    args = parser.parse_args()

    if args.ckpt_path is not None:
        # Worker mode: run exactly one checkpoint's grid evaluation in this process.
        run_worker(args.label, args.ckpt_path)
        return

    # Orchestrator mode: launch one subprocess per checkpoint, sequentially --
    # separate processes because pybullet only supports one physics client
    # per process, so a second in-process env after the first is unsafe.
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, ckpt_path in CHECKPOINTS:
        print(f"=== Running heatmap worker for {label} ({ckpt_path}) ===")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--ckpt_path", ckpt_path, "--label", label],
            check=True,
        )
    combine_heatmaps()
    print(f"Done. Outputs saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
