"""
Evaluate the latent dynamics MLP on the 8 validation-set episodes whose episode
number is divisible by 125 (750, 2000, 2875, 3125, 3625, 4000, 4250, 4500 —
from checkpoints_upweighted/train_val_split.json, identical val split in both runs).

For each timestep t (up to T - HORIZON):
  z_pred    = model(z_obs[t], actions[t:t+HORIZON])
  cos_sim   = cos(z_pred, z_sem[-1])   # fixed goal: z_sem at the episode's final timestep

Caches per-timestep z_pred and cos_sim to a .npz per episode, and plots cos_sim over time.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_3_expert_trajs.py
"""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Model classes (copied inline to avoid importing dynamics_model.py, which
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
TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
DATA_DIR  = "/home/ashwink/full_data_5000/langtable/demos/"
EP_NUMS   = [750, 2000, 2875, 3125, 3625, 4000, 4250, 4500]
HORIZON   = 8
DEVICE    = "cpu"

# (label, checkpoint path, top-level output dir) — run once per checkpoint
RUNS = [
    ("upweighted", os.path.join(REPO_ROOT, "checkpoints_upweighted", "best.pt"),
     os.path.join(REPO_ROOT, "eval_results_upweighted")),
    ("non-upweighted", os.path.join(REPO_ROOT, "checkpoints_non-upweighted", "best.pt"),
     os.path.join(REPO_ROOT, "eval_results_non-upweighted")),
]
for _, ckpt_path, _ in RUNS:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

# ---------------------------------------------------------------------------
# 1. Action stats from training pkls (actions only — no frames loaded)
#    Checkpoint-independent — computed once, reused for both runs.
# ---------------------------------------------------------------------------
print(f"Computing action stats from {TRAIN_DIR} ...")
all_actions = []
pkl_files = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl"))
for pkl_file in pkl_files:
    with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
        ep = pickle.load(f)
    all_actions.append(np.array(ep["actions"], dtype=np.float32))
all_actions = np.concatenate(all_actions, axis=0)
lo = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)
hi = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)
print(f"  {len(pkl_files)} episodes  |  action shape: {all_actions.shape}  |  lo={lo.tolist()}  hi={hi.tolist()}\n")

# ---------------------------------------------------------------------------
# 2. Per-checkpoint, per-episode evaluation
# ---------------------------------------------------------------------------
for label, ckpt_path, out_dir in RUNS:
    print(f"=== {label} ===")
    print(f"Checkpoint: {ckpt_path}")

    cache_dir = os.path.join(out_dir, "cached_MLP_output_data")
    os.makedirs(cache_dir, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']}, step {ckpt.get('step', '?')})\n")

    for ep_num in EP_NUMS:
        stem = f"blocktoblock_{ep_num}_success"
        with open(os.path.join(DATA_DIR, stem + ".pkl"), "rb") as f:
            ep = pickle.load(f)
        actions_raw = ep["actions"]  # list of T-1 arrays, each (2,)
        instruction = ep["metadata"]["instruction_str"]

        z_obs = np.load(os.path.join(DATA_DIR, stem + "_zobs.npz"))["z_obs"]  # (T, 768)
        z_sem = np.load(os.path.join(DATA_DIR, stem + "_zsem.npz"))["z_sem"]  # (T, 768)

        T = z_obs.shape[0]
        goal = torch.tensor(z_sem[-1], dtype=torch.float32)  # fixed goal embedding

        timesteps, mlp_outputs, cosine_sims = [], [], []

        with torch.no_grad():
            for t in range(T - HORIZON):
                z_t = torch.tensor(z_obs[t], dtype=torch.float32).unsqueeze(0)

                chunk = torch.tensor(
                    np.array(actions_raw[t : t + HORIZON], dtype=np.float32)
                )  # (HORIZON, 2)
                actions_norm = ((chunk - lo) / (hi - lo) * 2.0 - 1.0).unsqueeze(0)  # (1, HORIZON, 2)

                z_pred = model(z_t, actions_norm).squeeze(0)  # (768,)
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
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(timesteps, cosine_sims, marker="o", markersize=3)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Cosine similarity to goal z_sem")
        ax.set_title(f"{stem} ({label})\n\"{instruction}\"")
        plt.tight_layout()
        plot_path = os.path.join(cache_dir, f"{ep_num}_cosine_sim.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"{stem}  (T={T}, N={len(timesteps)}, instruction=\"{instruction}\")")
        print(f"  cos_sim: first={cosine_sims[0]:.4f}  last={cosine_sims[-1]:.4f}  "
              f"min={cosine_sims.min():.4f}  max={cosine_sims.max():.4f}")
        print(f"  cached: {cache_path}")
        print(f"  plot:   {plot_path}\n")
