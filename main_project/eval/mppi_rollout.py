"""
Live MPPI planning loop using the trained latent dynamics model in LangTable.

This is an eval to debug the FiLM LatentDynamicsModel's prediction quality
itself (not the MPPI optimizer). At each env timestep:
  1. Encode current frame with CLIP -> z_t (768,)
  2. Call mppi.command(z_t) -> action_chunk (16,) normalized in [-1, 1]
  3. Denormalize first 2 dims to raw action space and step the env
  4. Log ground-truth closeness-to-goal, prediction-vs-reality accuracy, and
     action-sensitivity/collapse diagnostics for the dynamics model.
  5. Repeat for MAX_STEPS steps, then save an .mp4

Run (produces TWO rollout videos under main_project/rollouts/, one per checkpoint
in CHECKPOINTS below):
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_rollout.py

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
OUT_DIR      = os.path.join(MAIN_PROJECT_DIR, "rollouts")
NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# The two checkpoints this script compares — (descriptive label, checkpoint path).
# Labels are used directly in the output .mp4 / log filenames.
CHECKPOINTS = [
    ("upweighted_best", os.path.join(MAIN_PROJECT_DIR, "checkpoints_upweighted", "best.pt")),
    ("non-upweighted_best", os.path.join(MAIN_PROJECT_DIR, "checkpoints_non-upweighted", "best.pt")),
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
    # 1. Goal text + z_star
    # -----------------------------------------------------------------------
    goal_text = (
        f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
        f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
    )
    log.info(f"Goal: {goal_text}\n")

    # -----------------------------------------------------------------------
    # 2. Load CLIP (image + text encoder)
    # -----------------------------------------------------------------------
    log.info("Loading CLIP ViT-L/14 ...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-S13B-B90K"
    )
    clip_model = clip_model.to(DEVICE).eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_star = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)
    log.info(f"z_star shape: {z_star.shape}\n")

    # -----------------------------------------------------------------------
    # 3. Checkpoint
    # -----------------------------------------------------------------------
    log.info(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    log.info(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}\n")

    # -----------------------------------------------------------------------
    # 4. Load LatentDynamicsModel
    # -----------------------------------------------------------------------
    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # -----------------------------------------------------------------------
    # 5. Action stats from training pkls (for denormalization before env.step)
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
    # 6. Init LangTable env
    # -----------------------------------------------------------------------
    log.info("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env(
        {"ood": False, "block_combo": [BLOCK_A, BLOCK_B]},
        seed=42,
    )
    log.info("Env ready.\n")

    # -----------------------------------------------------------------------
    # 7. MPPI dynamics_fn and cost_fn
    # -----------------------------------------------------------------------

    # dynamics_fn is called once per MPPI iteration (horizon=1); stash its output
    # from the last call so we can inspect the model's predicted z_pred batch
    # right after mppi.command() returns, instead of discarding it.
    _last_z_pred_batch = {"value": None}

    def dynamics_fn(state, action):
        # state:  (K, 768) — batch of latent obs
        # action: (K, 16)  — normalized action chunks
        z_pred_batch = model(state, action.reshape(action.shape[0], 8, 2))   # (K, 768)
        _last_z_pred_batch["value"] = z_pred_batch.detach()
        return z_pred_batch

    def cost_fn(terminal_state):
        # terminal_state: (K, 768), z_star: (768,)
        return 1.0 - (terminal_state * z_star).sum(dim=-1)           # (K,)

    # -----------------------------------------------------------------------
    # 8. Init MPPI
    # -----------------------------------------------------------------------
    mppi = MPPI(
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        action_dim=16,
        horizon=1,
        num_samples=K,
        noise_sigma=torch.full((16,), 0.1, dtype=torch.float32, device=DEVICE),
        lambda_=0.01,
        device=DEVICE,
        u_min=-1.0,
        u_max=1.0,
        num_iters=10,
    )

    # -----------------------------------------------------------------------
    # 9. Rollout loop — dynamics-model debug diagnostics every step
    # -----------------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []

    log.info(
        f"{'step':>4}  {'cos(zt,z*)':>10}  {'cos(zt,zt-1)':>12}  {'pred_acc':>9}  |  "
        f"{'cos(predK,z*) min/mean/max':>27}  {'cos(predK,zt) min/mean/max':>27}  "
        f"{'pairwise_cos':>12}  {'cos(pred_sel,z*)':>16}"
    )
    log.info("-" * 150)

    with torch.no_grad():
        z_prev = None
        z_pred_selected_prev = None

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

            # ---- Prediction accuracy: last step's prediction vs. this step's reality ----
            pred_accuracy = (
                (z_pred_selected_prev * z_t).sum().item() if z_pred_selected_prev is not None else float("nan")
            )

            # Plan — returns (16,) normalized action chunk
            action_chunk = mppi.command(z_t)

            # ---- Model-internal stats from the K candidate rollout (last MPPI iter) ----
            z_pred_K = _last_z_pred_batch["value"]                       # (K, 768)
            cos_predK_star = (z_pred_K * z_star.unsqueeze(0)).sum(dim=-1)  # (K,)
            cos_predK_zt   = (z_pred_K * z_t.unsqueeze(0)).sum(dim=-1)     # (K,)

            pairwise_cos_matrix = z_pred_K @ z_pred_K.T
            mean_pairwise_cos = (pairwise_cos_matrix.sum() - K) / (K * (K - 1))

            # Model's prediction for the specific action chunk actually chosen
            z_pred_selected = model(z_t.unsqueeze(0), action_chunk.reshape(1, 8, 2)).squeeze(0)
            cos_pred_selected_star = (z_pred_selected * z_star).sum().item()

            log.info(
                f"{step:>4}  {cos_t_star:>10.4f}  {cos_t_prev:>12.4f}  {pred_accuracy:>9.4f}  |  "
                f"{cos_predK_star.min().item():>8.4f}/{cos_predK_star.mean().item():>8.4f}/{cos_predK_star.max().item():>8.4f}  "
                f"{cos_predK_zt.min().item():>8.4f}/{cos_predK_zt.mean().item():>8.4f}/{cos_predK_zt.max().item():>8.4f}  "
                f"{mean_pairwise_cos.item():>12.4f}  {cos_pred_selected_star:>16.4f}"
            )

            # Denormalize first 2 dims back to raw action space before stepping env
            a2_norm = action_chunk[:2].cpu().float()
            a2_raw  = (a2_norm + 1.0) / 2.0 * (hi - lo) + lo                    # (2,)
            env.step(a2_raw.numpy())
            z_prev = z_t
            z_pred_selected_prev = z_pred_selected

    # -----------------------------------------------------------------------
    # 10. Save video
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
