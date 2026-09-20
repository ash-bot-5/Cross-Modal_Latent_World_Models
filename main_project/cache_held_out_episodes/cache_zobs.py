import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import open_clip

DATA_DIR = "/home/ashwink/held_out_episode_5/demos/"
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-S13B-B90K"
)
model = model.to(device).eval()

pkl_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".pkl"))

for ep_i, pkl_file in enumerate(pkl_files):
    pkl_path  = os.path.join(DATA_DIR, pkl_file)
    zobs_path = pkl_path.replace(".pkl", "_zobs.npz")

    if os.path.exists(zobs_path):
        continue

    with open(pkl_path, "rb") as f:
        ep = pickle.load(f)

    frames = ep["frames"]  # list or (T, H, W, 3) uint8
    T = len(frames)

    imgs = torch.stack([preprocess(Image.fromarray(frames[t])) for t in range(T)])  # (T, 3, H, W)

    z_obs_parts = []
    with torch.no_grad():
        for start in range(0, T, BATCH_SIZE):
            batch = imgs[start : start + BATCH_SIZE].to(device)
            z = model.encode_image(batch)
            z = F.normalize(z, dim=-1)
            z_obs_parts.append(z.cpu().float().numpy())

    z_obs = np.concatenate(z_obs_parts, axis=0)  # (T, 768) float32
    np.savez(zobs_path, z_obs=z_obs)

    print(f"[{ep_i + 1}/{len(pkl_files)}] cached {pkl_file}")

print("Done.")
