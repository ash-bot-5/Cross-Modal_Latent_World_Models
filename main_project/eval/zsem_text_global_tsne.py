"""
Global t-SNE of text-only z_sem embeddings, colored by touching vs not touching.

z_sem = L2_normalize(mean(CLIP_text(stmt_AB), CLIP_text(stmt_BA)))
  where stmt_AB / stmt_BA are the two touching statements for each episode's final frame.

Model: OpenCLIP ViT-L/14, pretrained=datacomp_xl_s13b_b90k

Output: zsem_text_global_tsne.png

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/zsem_text_global_tsne.py
"""

import open_clip

import glob
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = "/home/ashwink/full_data_5000/langtable/demos/"
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
N_EPISODES = 1000
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load OpenCLIP
# ---------------------------------------------------------------------------
print(f"Loading OpenCLIP ViT-L/14 (datacomp_xl_s13b_b90k) on {DEVICE} ...")
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Statement conversion
# ---------------------------------------------------------------------------

def qa_to_statement(question: str, answer: bool) -> str:
    stmt = question
    if stmt.startswith("Is "):
        stmt = stmt[3:]
    stmt = stmt.rstrip("?")
    if stmt:
        stmt = stmt[0].lower() + stmt[1:]
    if answer:
        stmt = stmt.replace(" touching", " is touching", 1)
    else:
        stmt = stmt.replace(" touching", " is not touching", 1)
    return stmt

# ---------------------------------------------------------------------------
# Sample episodes
# ---------------------------------------------------------------------------
all_files     = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
sampled_files = random.sample(all_files, min(N_EPISODES, len(all_files)))
print(f"Sampled {len(sampled_files)} episodes from {len(all_files)} total.\n")

# ---------------------------------------------------------------------------
# Data collection loop
# ---------------------------------------------------------------------------
z_sems = []
labels = []

for pkl_path in tqdm(sampled_files, desc="Episodes"):
    try:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        qa_pairs = ep["task_relevant_qa"][-1]
        touching_qas = [qa for qa in qa_pairs if "touching" in qa["question"].lower()]

        if len(touching_qas) < 2:
            continue

        stmt_ab = qa_to_statement(touching_qas[0]["question"], touching_qas[0]["answer"])
        stmt_ba = qa_to_statement(touching_qas[1]["question"], touching_qas[1]["answer"])

        with torch.no_grad():
            z_both = F.normalize(
                model.encode_text(tokenizer([stmt_ab, stmt_ba]).to(DEVICE)), dim=-1
            )  # (2, D)
        z_avg = F.normalize(z_both.mean(dim=0), dim=-1)  # (D,)

        z_sems.append(z_avg.cpu().float().numpy())
        labels.append("touching" if touching_qas[0]["answer"] else "not touching")

    except Exception as e:
        print(f"Skipping {pkl_path}: {e}")
        continue

z_sems = np.array(z_sems)
labels = np.array(labels)

print(f"\nEpisodes collected: {len(z_sems)}")
print(f"  touching:     {(labels == 'touching').sum()}")
print(f"  not touching: {(labels == 'not touching').sum()}\n")

# ---------------------------------------------------------------------------
# t-SNE + plot
# ---------------------------------------------------------------------------
print("Running t-SNE ...")
coords = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(z_sems)

LABEL_COLORS = {"touching": "tab:blue", "not touching": "tab:orange"}

fig, ax = plt.subplots(figsize=(8, 6))
for lbl, color in LABEL_COLORS.items():
    m = labels == lbl
    ax.scatter(
        coords[m, 0], coords[m, 1],
        c=color, s=30, alpha=0.7, linewidths=0,
        label=f"{lbl} (n={m.sum()})",
    )
ax.set_title("t-SNE of z_sem (text only, avg, 1000 eps) — touching vs not touching")
ax.legend(fontsize=9)
plt.tight_layout()

out = os.path.join(OUT_DIR, "zsem_text_global_tsne.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved {out}")
