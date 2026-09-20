"""
Zero-action success-signal probe: for every one of the 300 frames already saved in
episode_0_full_finetune_wider_hl_epoch8_steps300.mp4, feed that frame's CLIP embedding
into the epoch-8 wider_hl dynamics model together with an all-zero (8,2) action chunk
(same "zero actions" convention as eval_3_expert_trajs_0_actions.py and the cos(zpred0,z*)
diagnostic added to mppi_rollout_clip_full_finetune.py) and report cos(z_pred, z_star).

The question this answers: does cos(z_pred_zero_action, z*) alone -- with no live MPPI
planning, just "what does the model think is true about the current frame if no action
is taken" -- behave like a usable success signal? Step 200 is a frame the user identified
(from ground-truth peg/block positions) as an actual success state (peg touching
green_cube AND green_cube touching blue_moon); steps 30 and 230 are non-success comparison
points.

No live simulation/pybullet env is used -- this reads frames directly out of the already-
saved rollout video.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/zero_action_success_signal_probe.py
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import imageio
from PIL import Image

MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(
    MAIN_PROJECT_DIR, "rollouts", "rollouts_wider_hl", "episode_0_full_finetune_wider_hl_epoch8_steps300.mp4"
)
CKPT_PATH = os.path.join(
    MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_finetuned_wider_hl", "epoch_008.pt"
)
OUT_NPZ = os.path.join(MAIN_PROJECT_DIR, "eval", "zero_action_success_signal_probe_epoch8_data.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HORIZON = 8
BATCH_SIZE = 32
HIGHLIGHT_STEPS = [30, 200, 230]

GOAL_TEXT = "the green cube is touching the blue moon. the peg is touching the green cube."


# ---------------------------------------------------------------------------
# Inline model classes (matches mppi_rollout_clip_full_finetune.py's wider_hl
# architecture exactly, to load checkpoints_clip_full_finetuned_wider_hl/epoch_008.pt)
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


def main():
    print(f"Checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}\n")

    print("Loading CLIP ViT-L/14 (datacomp_xl_s13b_b90k) + fine-tuned clip_state_dict ...")
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
        tokens = clip_tokenizer([GOAL_TEXT]).to(DEVICE)
        z_star = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)
    print(f"Goal: {GOAL_TEXT}\n")

    print(f"Reading frames from {VIDEO_PATH} ...")
    reader = imageio.get_reader(VIDEO_PATH)
    n_frames = reader.count_frames()
    print(f"  {n_frames} frames found\n")

    cos_zero_star = np.zeros(n_frames, dtype=np.float32)

    with torch.no_grad():
        for batch_start in range(0, n_frames, BATCH_SIZE):
            batch_steps = list(range(batch_start, min(batch_start + BATCH_SIZE, n_frames)))
            imgs = [clip_preprocess(Image.fromarray(reader.get_data(s))) for s in batch_steps]
            img_batch = torch.stack(imgs).to(DEVICE)                     # (B, 3, 224, 224)
            z_t_batch = F.normalize(clip_model.encode_image(img_batch), dim=-1)  # (B, 768)

            actions_zero = torch.zeros(len(batch_steps), HORIZON, 2, device=DEVICE)
            z_pred_batch = model(z_t_batch, actions_zero)                # (B, 768)
            cos_batch = (z_pred_batch * z_star.unsqueeze(0)).sum(dim=-1)  # (B,)

            for s, c in zip(batch_steps, cos_batch.cpu().numpy()):
                cos_zero_star[s] = c

            print(f"  processed steps {batch_steps[0]}-{batch_steps[-1]}")

    print(f"\n{'step':>4}  {'cos(zpred0,z*)':>15}")
    print("-" * 22)
    for step in range(n_frames):
        print(f"{step:>4}  {cos_zero_star[step]:>15.4f}")

    print(f"\n=== Highlighted comparison steps ===")
    for step in HIGHLIGHT_STEPS:
        print(f"  step {step:>3}: cos(zpred0,z*) = {cos_zero_star[step]:+.4f}")

    np.savez(OUT_NPZ, cos_zero_star=cos_zero_star)
    print(f"\nSaved -> {OUT_NPZ}")


if __name__ == "__main__":
    main()
