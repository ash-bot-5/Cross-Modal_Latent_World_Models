"""
Zero-action ablation of the CLIP-LoRA jointly fine-tuned dynamics model (epoch 8) on the 8
validation-set episodes whose episode number is divisible by 125 (750, 2000, 2875, 3125, 3625,
4000, 4250, 4500 — from checkpoints_upweighted/train_val_split.json, identical val split in all
runs).

Same as eval_3_expert_trajs_clip_lora.py, except the real 8-step action chunk is replaced with an
all-zero tensor at every timestep:

  z_pred    = model(z_t, zeros(1, HORIZON, 2))
  cos_sim   = cos(z_pred, z_sem[-1])   # fixed goal: z_sem at the episode's final timestep

This isolates whether the MLP is mostly reading touching-state off the (fine-tuned) image
embedding alone, independent of the action-conditioning signal. z_t is recomputed by running each
frame through the LoRA-modified CLIP visual encoder (loaded from the checkpoint), not read from
the pre-cached frozen-CLIP _zobs.npz.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_3_expert_trajs_0_actions_clip_lora.py
"""

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import open_clip
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.clip_lora import apply_lora_to_visual_encoder, load_lora_state_dict

# ---------------------------------------------------------------------------
# Model class (copied inline to avoid importing dynamics_model.py, which
# triggers open_clip + LangTableDataset at module level)
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
            nn.Linear(16, 256),
            nn.Mish(),
            nn.Linear(256, 256),
        )
        self.fc1 = nn.Linear(768, 512)
        self.ln1 = nn.LayerNorm(512)
        self.film1 = FiLMLayer(256, 512)
        self.act1 = nn.Mish()
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.film2 = FiLMLayer(256, 512)
        self.act2 = nn.Mish()
        self.fc_out = nn.Linear(512, 768)
        self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        global_cond = self.action_encoder(actions.flatten(1))
        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = "/home/ashwink/full_data_5000/langtable/demos/"
EP_NUMS   = [750, 2000, 2875, 3125, 3625, 4000, 4250, 4500]
HORIZON   = 8
DEVICE    = "cpu"
EPOCH     = 8

CKPT_PATH = os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", f"epoch_{EPOCH:03d}.pt")
OUT_DIR   = os.path.join(REPO_ROOT, "eval_results_clip_lora_epoch8")
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"No checkpoint at {CKPT_PATH}")

cache_dir = os.path.join(OUT_DIR, "cached_data_0_actions")
os.makedirs(cache_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load CLIP + LoRA visual encoder + MLP from the checkpoint
# ---------------------------------------------------------------------------
print(f"Checkpoint: {CKPT_PATH}")
print("Loading OpenCLIP ViT-L/14 (datacomp_xl_s13b_b90k) + LoRA adapter ...")
clip_model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
)
apply_lora_to_visual_encoder(clip_model, r=16, alpha=32, dropout=0.2)
clip_model = clip_model.to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
load_lora_state_dict(clip_model, ckpt["lora_state_dict"])
clip_model.eval()

model = LatentDynamicsModel().to(DEVICE)
model.load_state_dict(ckpt["mlp_state_dict"])
model.eval()
print(f"Model loaded (epoch {ckpt['epoch']}, step {ckpt.get('step', '?')})\n")

# ---------------------------------------------------------------------------
# Per-episode evaluation (zero actions at every timestep)
# ---------------------------------------------------------------------------
for ep_num in EP_NUMS:
    stem = f"blocktoblock_{ep_num}_success"
    with open(os.path.join(DATA_DIR, stem + ".pkl"), "rb") as f:
        ep = pickle.load(f)
    instruction = ep["metadata"]["instruction_str"]

    frames = np.load(os.path.join(DATA_DIR, stem + "_frames.npy"), mmap_mode="r")  # (T,180,320,3) uint8
    z_sem = np.load(os.path.join(DATA_DIR, stem + "_zsem.npz"))["z_sem"]  # (T, 768)

    T = frames.shape[0]
    with torch.no_grad():
        pixel_values = torch.stack([preprocess(Image.fromarray(np.asarray(frames[t]))) for t in range(T)])
        z_obs = clip_model.encode_image(pixel_values, normalize=True).numpy()  # (T, 768)

    goal = torch.tensor(z_sem[-1], dtype=torch.float32)  # fixed goal embedding

    # Peg touching Block A (start_block), computed locally from raw positions rather than the
    # cached _zsem.npz's peg_touching (which used a looser 0.05 threshold and strict "<").
    PEG_TOUCH_THRESH = 0.04
    start_block = ep["metadata"]["start_block"]
    peg_dist = np.array([
        np.linalg.norm(ep["actual_peg_positions"][t] - ep["block_states"][t][start_block])
        for t in range(T)
    ])
    peg_touching = peg_dist <= PEG_TOUCH_THRESH

    timesteps, mlp_outputs, cosine_sims = [], [], []

    actions_zero = torch.zeros(1, HORIZON, 2, dtype=torch.float32)

    with torch.no_grad():
        for t in range(T - HORIZON):
            z_t = torch.tensor(z_obs[t], dtype=torch.float32).unsqueeze(0)

            z_pred = model(z_t, actions_zero).squeeze(0)  # (768,)
            cos_sim = F.cosine_similarity(z_pred.unsqueeze(0), goal.unsqueeze(0)).item()

            timesteps.append(t)
            mlp_outputs.append(z_pred.numpy())
            cosine_sims.append(cos_sim)

    timesteps = np.array(timesteps, dtype=np.int64)
    mlp_outputs = np.array(mlp_outputs, dtype=np.float32)
    cosine_sims = np.array(cosine_sims, dtype=np.float32)

    # ---- cache ----
    cache_path = os.path.join(cache_dir, f"{ep_num}_mlp_eval.npz")
    np.savez(cache_path, timesteps=timesteps, mlp_output=mlp_outputs, cosine_sim=cosine_sims)

    # ---- plot ----
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )

    for t, is_touching in zip(timesteps, peg_touching[timesteps]):
        color = "green" if is_touching else "red"
        ax.axvspan(t - 0.5, t + 0.5, color=color, alpha=0.15, zorder=0)
        ax2.axvspan(t - 0.5, t + 0.5, color=color, alpha=0.15, zorder=0)

    ax.plot(timesteps, cosine_sims, marker="o", markersize=3, color="tab:blue", zorder=2)
    ax.set_xlim(timesteps[0] - 0.5, timesteps[-1] + 0.5)
    ax.set_ylabel("Cosine similarity to goal z_sem")
    ax.set_title(f"{stem} (zero actions, clip_lora_epoch8)\n\"{instruction}\"")

    legend_handles = [
        mpatches.Patch(color="green", alpha=0.3, label="peg touching Block A"),
        mpatches.Patch(color="red", alpha=0.3, label="peg not touching Block A"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    ax2.plot(timesteps, peg_dist[timesteps], marker="o", markersize=3, color="tab:purple", zorder=2)
    ax2.axhline(PEG_TOUCH_THRESH, linestyle="--", color="gray", linewidth=1,
                label=f"threshold ({PEG_TOUCH_THRESH})")
    ax2.set_ylabel("||peg - Block A||")
    ax2.set_xlabel("Timestep")
    ax2.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(cache_dir, f"{ep_num}_cosine_sim.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"{stem}  (T={T}, N={len(timesteps)}, instruction=\"{instruction}\")")
    print(f"  cos_sim: first={cosine_sims[0]:.4f}  last={cosine_sims[-1]:.4f}  "
          f"min={cosine_sims.min():.4f}  max={cosine_sims.max():.4f}")
    print(f"  cached: {cache_path}")
    print(f"  plot:   {plot_path}\n")
