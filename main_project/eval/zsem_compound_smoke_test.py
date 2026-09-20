"""
Smoke test: compound string vs. averaged embeddings for multi-predicate z_sem.

For each episode, two z_sem vectors are computed:
  z_compound  — CLIP embed of "[touching stmt]. [peg stmt]." as a single string
  z_averaged  — L2_normalize((z_touching + z_peg) / 2)
                where z_touching = L2_normalize(mean(embed(AB), embed(BA)))
                and   z_peg      = L2_normalize(embed(peg_stmt))

Four t-SNE plots compare the two strategies side by side:
  zsem_compound_tsne_task_identity.png
  zsem_compound_tsne_touching.png
  zsem_averaged_tsne_task_identity.png
  zsem_averaged_tsne_touching.png

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/zsem_compound_smoke_test.py
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
# Data collection
# ---------------------------------------------------------------------------
z_compounds  = []
z_averageds  = []
task_ids     = []
touch_labels = []

for pkl_path in tqdm(sampled_files, desc="Episodes"):
    try:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        # --- Touching QA pair ---
        qa_pairs = ep["task_relevant_qa"][-1]
        touching_qas = [qa for qa in qa_pairs if "touching" in qa["question"].lower()]
        if len(touching_qas) < 2:
            continue

        stmt_ab = qa_to_statement(touching_qas[0]["question"], touching_qas[0]["answer"])
        stmt_ba = qa_to_statement(touching_qas[1]["question"], touching_qas[1]["answer"])

        # --- Peg QA ---
        peg_raw = ep["peg_touching_starting_block_qa"][-1]
        if isinstance(peg_raw, list):
            peg_raw = peg_raw[0]
        peg_stmt = qa_to_statement(peg_raw["question"], peg_raw["answer"])

        # --- Compound string ---
        compound_stmt = f"{stmt_ab}. {peg_stmt}."

        # --- Single batched CLIP call: [AB, BA, peg, compound] ---
        tokens = tokenizer([stmt_ab, stmt_ba, peg_stmt, compound_stmt]).to(DEVICE)
        with torch.no_grad():
            z_all = F.normalize(model.encode_text(tokens), dim=-1)  # (4, D)

        z_ab, z_ba, z_peg, z_comp = z_all[0], z_all[1], z_all[2], z_all[3]

        z_touching = F.normalize((z_ab + z_ba) / 2.0, dim=-1)
        z_compound = z_comp
        z_averaged = F.normalize((z_touching + z_peg) / 2.0, dim=-1)

        z_compounds.append(z_compound.cpu().float().numpy())
        z_averageds.append(z_averaged.cpu().float().numpy())

        sb   = ep["metadata"]["start_block"]
        tb   = ep["metadata"]["oracle_target_block"]
        task_ids.append(f"{sb} → {tb}")
        touch_labels.append("touching" if touching_qas[0]["answer"] else "not touching")

    except Exception as e:
        print(f"Skipping {pkl_path}: {e}")
        continue

z_compounds  = np.array(z_compounds)
z_averageds  = np.array(z_averageds)
task_ids     = np.array(task_ids)
touch_labels = np.array(touch_labels)

n_touch     = (touch_labels == "touching").sum()
n_not_touch = (touch_labels == "not touching").sum()
unique_tasks = sorted(set(task_ids))
print(f"\nEpisodes collected: {len(z_compounds)}")
print(f"  touching:     {n_touch}")
print(f"  not touching: {n_not_touch}")
print(f"  unique tasks: {len(unique_tasks)}\n")

# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------
print("Running t-SNE for compound embeddings ...")
coords_compound = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(z_compounds)

print("Running t-SNE for averaged embeddings ...")
coords_averaged = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(z_averageds)

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Color maps
# ---------------------------------------------------------------------------
cmap = plt.cm.gist_rainbow
task_colors = {
    t: cmap(i / max(len(unique_tasks) - 1, 1))
    for i, t in enumerate(unique_tasks)
}
LABEL_COLORS = {"touching": "tab:blue", "not touching": "tab:orange"}

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_task_identity(ax, coords, task_ids, title):
    for task in unique_tasks:
        m = task_ids == task
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=[task_colors[task]], s=30, alpha=0.8, linewidths=0,
                   label=task)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, framealpha=0.8)


def plot_touching(ax, coords, touch_labels, title):
    for lbl, color in LABEL_COLORS.items():
        m = touch_labels == lbl
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=color, s=30, alpha=0.7, linewidths=0,
                   label=f"{lbl} (n={m.sum()})")
    ax.set_title(title)
    ax.legend(fontsize=9)

# ---------------------------------------------------------------------------
# Four plots
# ---------------------------------------------------------------------------
PLOTS = [
    ("compound", coords_compound, task_ids,     "task_identity",
     "t-SNE z_compound — task identity (start → target)"),
    ("compound", coords_compound, touch_labels,  "touching",
     "t-SNE z_compound — touching vs not touching"),
    ("averaged", coords_averaged, task_ids,     "task_identity",
     "t-SNE z_averaged — task identity (start → target)"),
    ("averaged", coords_averaged, touch_labels,  "touching",
     "t-SNE z_averaged — touching vs not touching"),
]

for strategy, coords, color_data, plot_type, title in PLOTS:
    fig, ax = plt.subplots(figsize=(10 if plot_type == "task_identity" else 8, 6))
    if plot_type == "task_identity":
        plot_task_identity(ax, coords, color_data, title)
        out = os.path.join(OUT_DIR, f"zsem_{strategy}_tsne_task_identity.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
    else:
        plot_touching(ax, coords, color_data, title)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"zsem_{strategy}_tsne_touching.png")
        plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")
