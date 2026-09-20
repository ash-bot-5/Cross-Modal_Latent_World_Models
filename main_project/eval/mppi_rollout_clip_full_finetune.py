"""
Live MPPI planning loop using the fully fine-tuned (both CLIP towers, no LoRA)
jointly fine-tuned dynamics model in LangTable -- EPOCH-8-ONLY, EXTENDED DIAGNOSTICS variant.

Same as the original full epochs-6-15 sweep, except:
  - the checkpoint's CLIP weights are loaded as a full state dict onto both towers
    (clip_state_dict), not a LoRA adapter onto the visual tower only -- these
    checkpoints come from dynamics_model_clip_full_finetune.py, which fine-tunes
    both the image AND text towers jointly with the FiLM-MLP dynamics model.
  - because both towers are fine-tuned, the goal embedding z_star is computed with
    this SAME loaded checkpoint's own fine-tuned text tower (clip_model.encode_text),
    not a separately-loaded frozen CLIP -- a stale/frozen text tower would not match
    the image tower's fine-tuned embedding space.
  - the peg-start-position logic (peg forced collinear with, and just past, BLOCK_A,
    on the side opposite BLOCK_B) is UNCHANGED from mppi_rollout_clip_lora.py.
  - CHECKPOINTS is restricted to epoch 8 only (same checkpoint directory), and the
    output label/filename is changed so this run doesn't overwrite the existing
    episode_0_full_finetune_wider_hl_epoch8_steps300.mp4 already in rollouts_wider_hl/.
  - adds a new per-step diagnostic, cos(zpred0,z*): the model's prediction when fed
    an all-zero action chunk instead of the actually-planned one (same "zero actions"
    probe as eval_3_expert_trajs_0_actions.py), isolating how much of the model's
    goal-similarity signal comes from the image embedding alone vs. the chosen action.
  - also logs the norm of the actually-executed raw action, ||a_raw||, so step size
    can be correlated against the cosine diagnostics.

At each env timestep:
  1. Encode current frame with the fine-tuned CLIP image tower -> z_t (768,)
  2. Call mppi.command(z_t) -> action_chunk (16,) normalized in [-1, 1]
  3. Denormalize first 2 dims to raw action space and step the env
  4. Log ground-truth closeness-to-goal, prediction-vs-reality accuracy,
     action-sensitivity/collapse diagnostics, the zero-action baseline prediction,
     and the executed action's magnitude.
  5. Repeat for MAX_STEPS steps, then save an .mp4

Run (produces one rollout video for the epoch-8 checkpoint in CHECKPOINTS below,
saved under main_project/rollouts/rollouts_wider_hl/):
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_rollout_clip_full_finetune.py

Each rollout runs in its own subprocess (this script re-invokes itself with
--ckpt_path/--label) rather than looping in-process, because the LangTable
env wraps a pybullet physics client and pybullet only supports one such
client per process — constructing a second env in the same process after
the first would be unsafe/undefined.
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import imageio

from planning.mppi_core import MPPI

# ---------------------------------------------------------------------------
# Config — edit BLOCK_A / BLOCK_B before running
# ---------------------------------------------------------------------------
BLOCK_A   = "green_cube"
BLOCK_B   = "blue_moon"
MAX_STEPS = 300
K         = 512   # num_samples

TRAIN_DIR    = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR      = os.path.join(MAIN_PROJECT_DIR, "rollouts", "rollouts_wider_hl")
NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CKPT_DIR = os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_finetuned_wider_hl")

# The full-finetune checkpoint(s) this script runs — (descriptive label,
# checkpoint path). Labels are used directly in the output .mp4 / log filenames.
# Restricted to epoch 8 only, with a new label so the output mp4 doesn't overwrite
# the existing episode_0_full_finetune_wider_hl_epoch8_steps300.mp4.
CHECKPOINTS = [
    ("full_finetune_wider_hl_epoch8_extended_diag", os.path.join(CKPT_DIR, "epoch_008.pt")),
]

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


def run_rollout(label: str, ckpt_path: str) -> None:
    """Runs one full MPPI rollout with the given checkpoint and saves an .mp4.
    Everything below is process-local (CLIP, env, model, MPPI) — safe to call
    exactly once per process."""

    # -----------------------------------------------------------------------
    # Logging — mirror to stdout and a persistent, timestamped log file so the
    # full per-step record survives regardless of how the script is launched.
    # -----------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        NOTEBOOKS_DIR, f"mppi_rollout_{label}_{BLOCK_A}_{BLOCK_B}_steps{MAX_STEPS}_{timestamp}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,  # reset handlers in case this module is re-entered
    )
    log = logging.getLogger(f"mppi_rollout.{label}")
    log.info(f"Label: {label}")
    log.info(f"Logging to {log_path}")

    # -----------------------------------------------------------------------
    # 1. Fixed goal text, held constant for the entire rollout (no stage-switching).
    # -----------------------------------------------------------------------
    goal_text = (
        f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
        f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
    )
    log.info(f"Goal (fixed, steps 0-{MAX_STEPS - 1}): {goal_text}\n")

    # -----------------------------------------------------------------------
    # 2. Checkpoint + CLIP (image + text encoder), both towers fully fine-tuned --
    # loaded as a plain full state dict, no LoRA adapter involved.
    # -----------------------------------------------------------------------
    log.info(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    log.info(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}\n")

    log.info("Loading CLIP ViT-L/14 (datacomp_xl_s13b_b90k) + fine-tuned clip_state_dict ...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # z_star uses THIS checkpoint's own fine-tuned text tower (both towers were
    # fine-tuned jointly here, so the goal embedding must come from the same model
    # instance -- a separately-loaded frozen CLIP text tower would not match this
    # checkpoint's fine-tuned embedding space).
    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_star = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)
    log.info(f"z_star shape: {z_star.shape}\n")

    # -----------------------------------------------------------------------
    # 3. Load LatentDynamicsModel
    # -----------------------------------------------------------------------
    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    # -----------------------------------------------------------------------
    # 4. Action stats from training pkls (for denormalization before env.step)
    # -----------------------------------------------------------------------
    log.info(f"Loading action stats from {TRAIN_DIR} ...")
    all_actions = []
    for pkl_file in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl")):
        with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
            ep = pickle.load(f)
        all_actions.append(np.array(ep["actions"], dtype=np.float32))
    all_actions = np.concatenate(all_actions, axis=0)
    lo = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)  # (2,)
    hi = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)  # (2,)
    log.info(f"  lo={lo.tolist()}  hi={hi.tolist()}\n")

    # -----------------------------------------------------------------------
    # 5. Init LangTable env
    # -----------------------------------------------------------------------
    log.info("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env(
        {"ood": False, "block_combo": [BLOCK_A, BLOCK_B]},
        seed=42,
    )
    log.info("Env ready.\n")

    # -----------------------------------------------------------------------
    # 5b. Force the peg to start collinear with, and just past, BLOCK_A —
    # i.e. peg -> green_cube -> blue_moon in a straight line, so a single push
    # accomplishes the task instead of needing to navigate first.
    # (Unchanged from mppi_rollout_clip_lora.py, per instruction.)
    # -----------------------------------------------------------------------
    from language_table.environments.utils.pose3d import Pose3d
    from scipy.spatial.transform import Rotation as R

    inner = env.env  # GymWrapper.__getattr__ proxies unknown attrs straight through to the
                     # inner LanguageTableDataGeneration, so no third .env is needed/exists
                     # (verified directly: env.env.env raises AttributeError, env.env works)
    blocks = inner.get_block_states()
    green_xy = blocks[BLOCK_A]
    blue_xy = blocks[BLOCK_B]

    away_from_blue_direction = (green_xy - blue_xy) / np.linalg.norm(green_xy - blue_xy)
    peg_xy = green_xy + away_from_blue_direction * 0.06  # continue PAST green_cube, away from
                                                          # blue_moon — not between them

    pose = Pose3d(
        rotation=R.from_rotvec([0, np.pi, 0]),  # same fixed downward-facing rotation the
                                                 # internal random reset uses for the effector
        translation=np.array([peg_xy[0], peg_xy[1], 0.145]),  # 0.145 = EFFECTOR_HEIGHT
    )
    inner._set_robot_target_effector_pose(pose)
    inner._step_simulation_to_stabilize()

    dist_peg_green = np.linalg.norm(peg_xy - green_xy)
    dist_peg_blue = np.linalg.norm(peg_xy - blue_xy)
    dist_green_blue = np.linalg.norm(green_xy - blue_xy)
    log.info(f"Forced peg start: {peg_xy}  ({BLOCK_A}={green_xy}, {BLOCK_B}={blue_xy})")
    log.info(f"  dist(peg,{BLOCK_A})={dist_peg_green:.4f}  dist(peg,{BLOCK_B})={dist_peg_blue:.4f}  "
             f"dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}\n")
    assert dist_peg_blue > dist_green_blue, (
        f"peg ended up between {BLOCK_A} and {BLOCK_B}, not on the outside past {BLOCK_A} — geometry bug"
    )

    # -----------------------------------------------------------------------
    # 6. MPPI dynamics_fn and cost_fn
    # -----------------------------------------------------------------------

    # dynamics_fn is called once per CEM iteration; stash its output from the
    # last call so we can inspect the model's predicted z_pred batch right
    # after mppi.command() returns, instead of discarding it.
    _last_z_pred_batch = {"value": None}

    def dynamics_fn(state, action):
        # state:  (K, 768)   — batch of latent obs
        # action: (K, 8, 2)  — normalized action chunks
        z_pred_batch = model(state, action)   # (K, 768)
        _last_z_pred_batch["value"] = z_pred_batch.detach()
        return z_pred_batch

    def cost_fn(terminal_state):
        # terminal_state: (K, 768), z_star: (768,)
        return 1.0 - (terminal_state * z_star).sum(dim=-1)           # (K,)

    # -----------------------------------------------------------------------
    # 7. Init MPPI (ported from swm/planning_algos.py's plan_model_mppi --
    # Gaussian CEM with softmax reward-weighting, plus basic cross-step
    # warm-starting layered on top: each call after the first seeds its
    # initial population from a shifted version of the prior best chunk)
    # -----------------------------------------------------------------------
    mppi = MPPI(
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        pred_horizon=8,
        action_dim=2,
        num_samples=K,
        mppi_temperature=0.01,
        n_planning_itrs=9,
        device=DEVICE,
        max_action_value=1.0,
        warm_start_std=0.1,
    )

    # -----------------------------------------------------------------------
    # 8. Rollout loop — dynamics-model debug diagnostics every step
    # -----------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []

    log.info(
        f"{'step':>4}  {'cos(zt,z*)':>10}  {'cos(zt,zt-1)':>12}  |  "
        f"{'cos(predK,z*) min/mean/max':>27}  {'cos(pred_sel,z*)':>16}  {'cos(zpred0,z*)':>15}  {'||a_raw||':>9}  |  "
        f"{'dist(peg,A)':>11}  {'dist(A,B)':>10}"
    )
    log.info("-" * 175)

    with torch.no_grad():
        z_prev = None

        for step in range(MAX_STEPS):
            # Encode current observation
            pil_frame = env.get_frame()                                           # PIL Image
            rgb_np    = np.array(pil_frame)                                       # (H, W, 3)
            frames.append(rgb_np)

            img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
            z_t   = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)

            # ---- Real / ground truth ----
            cos_t_star = (z_t * z_star).sum().item()
            cos_t_prev = (z_t * z_prev).sum().item() if z_prev is not None else float("nan")

            # ---- Ground-truth world-frame distances (not CLIP-based): peg-to-BLOCK_A and
            # BLOCK_A-to-BLOCK_B, straight from the simulator's own state -- same calls used
            # in the peg-forcing setup above (inner.get_block_states(), inner._robot.forward_
            # kinematics()), just re-queried live every step instead of once at reset. ----
            blocks_now = inner.get_block_states()
            green_xy_now = blocks_now[BLOCK_A]
            blue_xy_now = blocks_now[BLOCK_B]
            peg_xy_now = np.array(inner._robot.forward_kinematics().translation[:2], dtype=np.float32)
            dist_peg_a_now = np.linalg.norm(peg_xy_now - green_xy_now)
            dist_a_b_now = np.linalg.norm(green_xy_now - blue_xy_now)

            # Plan — returns (8, 2) normalized action chunk
            action_chunk = mppi.command(z_t)

            # ---- Model-internal stats from the K candidate rollout (last MPPI iter) ----
            z_pred_K = _last_z_pred_batch["value"]                       # (K, 768)
            cos_predK_star = (z_pred_K * z_star.unsqueeze(0)).sum(dim=-1)  # (K,)

            # Model's prediction for the specific action chunk actually chosen
            z_pred_selected = model(z_t.unsqueeze(0), action_chunk.unsqueeze(0)).squeeze(0)
            cos_pred_selected_star = (z_pred_selected * z_star).sum().item()

            # ---- Zero-action baseline: what does the model predict if fed an all-zero
            # action chunk instead of the planned one? Isolates how much of the model's
            # goal-similarity signal comes from the image embedding z_t alone vs. the
            # chosen action -- same probe as eval_3_expert_trajs_0_actions.py. ----
            actions_zero = torch.zeros(1, 8, 2, device=DEVICE)
            z_pred_zero = model(z_t.unsqueeze(0), actions_zero).squeeze(0)
            cos_zero_star = (z_pred_zero * z_star).sum().item()

            # Denormalize first real sub-step back to raw action space before stepping env
            a2_norm = action_chunk[0].cpu().float()
            a2_raw  = (a2_norm + 1.0) / 2.0 * (hi - lo) + lo                    # (2,)
            a_raw_norm = a2_raw.norm().item()

            log.info(
                f"{step:>4}  {cos_t_star:>10.4f}  {cos_t_prev:>12.4f}  |  "
                f"{cos_predK_star.min().item():>8.4f}/{cos_predK_star.mean().item():>8.4f}/{cos_predK_star.max().item():>8.4f}  "
                f"{cos_pred_selected_star:>16.4f}  {cos_zero_star:>15.4f}  {a_raw_norm:>9.4f}  |  "
                f"{dist_peg_a_now:>11.4f}  {dist_a_b_now:>10.4f}"
            )

            env.step(a2_raw.numpy())
            z_prev = z_t

    # -----------------------------------------------------------------------
    # 9. Save video
    # -----------------------------------------------------------------------
    out_path = os.path.join(OUT_DIR, f"episode_0_{label}_steps{MAX_STEPS}.mp4")
    imageio.mimsave(out_path, frames, fps=10)
    log.info(f"\nSaved {len(frames)} frames -> {out_path}")
    log.info(f"Full log written to {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="internal: run a single rollout for this checkpoint (used by subprocess workers)",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="internal: descriptive label for output filenames (used with --ckpt_path)",
    )
    args = parser.parse_args()

    if args.ckpt_path is not None:
        # Worker mode: run exactly one rollout in this process.
        run_rollout(args.label, args.ckpt_path)
        return

    # Orchestrator mode: launch one subprocess per checkpoint, sequentially —
    # separate processes because pybullet only supports one physics client
    # per process, so a second in-process env after the first is unsafe.
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, ckpt_path in CHECKPOINTS:
        print(f"=== Running rollout for {label} ({ckpt_path}) ===")
        subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--ckpt_path", ckpt_path, "--label", label],
            check=True,
        )
    print(f"Done. Videos saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
