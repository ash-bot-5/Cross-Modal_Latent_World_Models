"""
Cosine-sim diagnostic plots for the CLIP fine-tuning experiments' checkpoints.

Checkpoints evaluated (4 total):
  - checkpoints_clip_image_text_finetuned/epoch_000.pt   (full finetune: image + text towers)
  - checkpoints_clip_image_text_finetuned/epoch_001.pt
  - checkpoints_clip_image_text_finetuned/epoch_008.pt
  - checkpoints_clip_image_finetuned/epoch_000.pt         (LoRA: image tower only, text frozen)

For each checkpoint x each of 3 shared validation episodes:
  z_obs(t)  = clip_model.encode_image(frame_t)             # recomputed via THIS checkpoint's
                                                             # own (possibly fine-tuned) image encoder
  z_pred(t) = model(z_obs(t), zeros(1, HORIZON, 2))         # zero-action rollout
  cos_sim(t) = cos(z_pred(t), goal)

goal is the semantic embedding of "Block A is touching Block B. The peg is touching Block A."
at the episode's final timestep:
  - LoRA checkpoint (text tower frozen/pretrained): goal = cached z_sem[-1] from the .npz.
  - Full-finetune checkpoints (text tower fine-tuned, drifted from pretrained): goal is
    RE-ENCODED through that checkpoint's own clip_model.encode_text(...) -- the cached z_sem
    is not a valid target for these, since the text embedding space has moved.

Episode selection: 3 random episodes, seeded, drawn from the LoRA checkpoint's 750-episode
val split (a confirmed strict subset of the full-finetune checkpoint's 820-episode val split,
so every episode is valid held-out data for both checkpoint families), filtered to genuine
touching at the final timestep for BOTH dist(A,B) and dist(A,peg) (ground truth from
block_states/actual_peg_positions, not the "_success" filename).

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_finetuned_checkpoints_cosine_sim.py
"""

import json
import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import open_clip
from PIL import Image

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training"))
from clip_lora import apply_lora_to_visual_encoder, load_lora_state_dict


# ---------------------------------------------------------------------------
# Model class (copied inline, per the established convention in
# eval_3_expert_trajs_clip_lora.py / eval_3_expert_trajs_0_actions.py, to avoid
# triggering open_clip/LangTableDataset module-level side effects via dynamics_model.py)
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
DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
HORIZON = 8
DEVICE = "cpu"
TOUCH_THRESH = 0.05  # matches cache_zhard.py / compute_pos_neg_indices_for_episode convention
EPISODE_SEED = 42
N_EPISODES = 3

LORA_VAL_SPLIT_JSON = os.path.join(
    REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "train_val_split.json"
)

CHECKPOINTS = [
    ("clip_image_text_epoch000", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_text_finetuned", "epoch_000.pt"), "full"),
    ("clip_image_text_epoch001", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_text_finetuned", "epoch_001.pt"), "full"),
    ("clip_image_text_epoch008", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_text_finetuned", "epoch_008.pt"), "full"),
    ("clip_image_epoch000", os.path.join(REPO_ROOT, "training", "checkpoints_clip_image_finetuned", "epoch_000.pt"), "lora"),
]

OUT_DIR = os.path.join(REPO_ROOT, "eval", "finetuned_checkpoints_cosine_sim_plots")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Goal statement text (matches GlobalStatementTable's phrasing in
# dynamics_model_clip_full_finetune.py:135-151 exactly)
# ---------------------------------------------------------------------------

def build_goal_statement(start_block: str, target_block: str, peg_touching_a: bool) -> str:
    a_disp = start_block.replace("_", " ")
    b_disp = target_block.replace("_", " ")
    touching_peg = "touching" if peg_touching_a else "not touching"
    return f"The {a_disp} is touching the {b_disp}. The peg is {touching_peg} the {a_disp}."


# ---------------------------------------------------------------------------
# 1. Episode selection: 3 random episodes from the LoRA 750-episode val split,
#    filtered to genuine A-B touching AND peg-A touching at the final timestep.
# ---------------------------------------------------------------------------

with open(LORA_VAL_SPLIT_JSON) as f:
    lora_split = json.load(f)
val_files = lora_split["val_files"]
print(f"Loaded {len(val_files)}-episode val split from {LORA_VAL_SPLIT_JSON}\n")

qualifying = []
for fn in val_files:
    stem = fn[:-4]  # strip ".pkl"
    pkl_path = os.path.join(DATA_DIR, fn)
    with open(pkl_path, "rb") as f:
        ep = pickle.load(f)
    meta = ep["metadata"]
    a, b = meta["start_block"], meta["oracle_target_block"]
    block_states = ep["block_states"]
    peg_positions = ep["actual_peg_positions"]

    dist_ab_final = float(np.linalg.norm(block_states[-1][a] - block_states[-1][b]))
    dist_peg_final = float(np.linalg.norm(peg_positions[-1] - block_states[-1][a]))

    if dist_ab_final < TOUCH_THRESH and dist_peg_final < TOUCH_THRESH:
        qualifying.append({
            "stem": stem, "start_block": a, "target_block": b,
            "instruction": meta["instruction_str"],
            "dist_ab_final": dist_ab_final, "dist_peg_final": dist_peg_final,
        })

print(f"{len(qualifying)} / {len(val_files)} val episodes have genuine A-B AND peg-A "
      f"touching at the final timestep (dist < {TOUCH_THRESH}).\n")

rng = random.Random(EPISODE_SEED)
chosen = rng.sample(qualifying, N_EPISODES)

print(f"=== Chosen {N_EPISODES} episodes (seed={EPISODE_SEED}) ===")
for rec in chosen:
    goal_stmt = build_goal_statement(rec["start_block"], rec["target_block"], peg_touching_a=True)
    print(f"  {rec['stem']}")
    print(f"    instruction:        \"{rec['instruction']}\"")
    print(f"    final dist(A,B):    {rec['dist_ab_final']:.4f}  (< {TOUCH_THRESH} -> touching)")
    print(f"    final dist(A,peg):  {rec['dist_peg_final']:.4f}  (< {TOUCH_THRESH} -> touching)")
    print(f"    goal sentence:      \"{goal_stmt}\"")
print()

episode_stems = [rec["stem"] for rec in chosen]


# ---------------------------------------------------------------------------
# 2. Per-checkpoint evaluation
# ---------------------------------------------------------------------------

for tag, ckpt_path, kind in CHECKPOINTS:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    print(f"=== {tag} ({kind}) ===")
    print(f"Checkpoint: {ckpt_path}")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    if kind == "lora":
        apply_lora_to_visual_encoder(clip_model, r=16, alpha=32, dropout=0.2)
        clip_model = clip_model.to(DEVICE)
        load_lora_state_dict(clip_model, ckpt["lora_state_dict"])
        tokenizer = None
    else:  # "full"
        clip_model = clip_model.to(DEVICE)
        clip_model.load_state_dict(ckpt["clip_state_dict"])
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
    clip_model.eval()

    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()
    print(f"Model loaded (epoch {ckpt['epoch']}, step {ckpt.get('step', '?')})\n")

    for stem in episode_stems:
        pkl_path = os.path.join(DATA_DIR, stem + ".pkl")
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)
        meta = ep["metadata"]
        a, b = meta["start_block"], meta["oracle_target_block"]
        instruction = meta["instruction_str"]
        block_states = ep["block_states"]
        peg_positions = ep["actual_peg_positions"]

        frames_path = pkl_path.replace(".pkl", "_frames.npy")
        frames = np.load(frames_path, mmap_mode="r")  # (T, 180, 320, 3) uint8
        T = frames.shape[0]

        with torch.no_grad():
            pixel_values = torch.stack(
                [preprocess(Image.fromarray(np.asarray(frames[t]))) for t in range(T)]
            ).to(DEVICE)
            z_obs = clip_model.encode_image(pixel_values, normalize=True).cpu().numpy()  # (T, 768)

            if kind == "lora":
                z_sem = np.load(pkl_path.replace(".pkl", "_zsem.npz"))["z_sem"]
                goal = torch.tensor(z_sem[-1], dtype=torch.float32)
            else:
                peg_touch_final = bool(
                    np.linalg.norm(peg_positions[-1] - block_states[-1][a]) < TOUCH_THRESH
                )
                goal_stmt = build_goal_statement(a, b, peg_touching_a=peg_touch_final)
                tokens = tokenizer([goal_stmt]).to(DEVICE)
                goal = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0).cpu()

        timesteps, cosine_sims = [], []
        actions_zero = torch.zeros(1, HORIZON, 2, dtype=torch.float32)
        with torch.no_grad():
            for t in range(T - HORIZON):
                z_t = torch.tensor(z_obs[t], dtype=torch.float32).unsqueeze(0)
                z_pred = model(z_t, actions_zero).squeeze(0)
                cos_sim = F.cosine_similarity(z_pred.unsqueeze(0), goal.unsqueeze(0)).item()
                timesteps.append(t)
                cosine_sims.append(cos_sim)

        timesteps = np.array(timesteps, dtype=np.int64)
        cosine_sims = np.array(cosine_sims, dtype=np.float32)

        dist_ab = np.array(
            [np.linalg.norm(block_states[t][a] - block_states[t][b]) for t in timesteps],
            dtype=np.float32,
        )
        dist_peg = np.array(
            [np.linalg.norm(peg_positions[t] - block_states[t][a]) for t in timesteps],
            dtype=np.float32,
        )

        # ---- plot ----
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax1.plot(timesteps, cosine_sims, color="tab:blue", marker="o", markersize=3, label="cos_sim(z_pred, goal)")
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Cosine similarity to goal z_sem", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(timesteps, dist_ab, color="tab:red", marker="x", markersize=3, label="dist(A, B)")
        ax2.plot(timesteps, dist_peg, color="tab:purple", marker="^", markersize=3, linestyle="--", label="dist(A, peg)")
        ax2.axhline(TOUCH_THRESH, linestyle=":", color="gray", linewidth=1)
        ax2.set_ylabel("distance (world coords)", color="black")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)

        ax1.set_title(f"{stem} ({tag}, zero actions)\n\"{instruction}\"")
        fig.tight_layout()
        plot_path = os.path.join(OUT_DIR, f"{tag}_{stem}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)

        print(f"  {stem}: cos_sim first={cosine_sims[0]:.4f} last={cosine_sims[-1]:.4f} "
              f"min={cosine_sims.min():.4f} max={cosine_sims.max():.4f}  -> {plot_path}")
    print()

print(f"Done. 12 plots written to {OUT_DIR}")
