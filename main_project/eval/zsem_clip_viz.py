"""
Compute z_sem latent vectors via CLIP late fusion and visualize with PCA and t-SNE.

z_sem = L2_normalize((z_image + z_text) / 2)
  z_image: CLIP image embedding of the final frame (L2-normalized)
  z_text:  mean of CLIP text embeddings for all touching-statement variants (L2-normalized)

Model: OpenCLIP ViT-L/14, pretrained=datacomp_xl_s13b_b90k

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/zsem_clip_viz.py
"""

import open_clip

import glob
import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR  = "/home/ashwink/full_data_5000/langtable/demos/"
OUT_DIR   = os.path.dirname(os.path.abspath(__file__))
N_EPISODES = 100
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
SEED      = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load OpenCLIP
# ---------------------------------------------------------------------------
print(f"Loading OpenCLIP ViT-L/14 (datacomp_xl_s13b_b90k) on {DEVICE} ...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Statement conversion
# ---------------------------------------------------------------------------

def qa_to_statement(question: str, answer: bool) -> str:
    """Convert a Q&A pair to a declarative statement for CLIP encoding."""
    stmt = question
    if stmt.startswith("Is "):
        stmt = stmt[3:]
    stmt = stmt.rstrip("?")
    if stmt:
        stmt = stmt[0].lower() + stmt[1:]
    if answer:
        return stmt
    else:
        parts = stmt.split(" ", 1)
        return parts[0] + " not " + parts[1] if len(parts) > 1 else parts[0]

# ---------------------------------------------------------------------------
# Sample episodes
# ---------------------------------------------------------------------------
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
sampled_files = random.sample(all_files, min(N_EPISODES, len(all_files)))
print(f"Sampled {len(sampled_files)} episodes from {len(all_files)} total.\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
z_sems        = []
labels        = []
start_blocks  = []
target_blocks = []
sanity_printed = False
statement_examples_printed = 0

for pkl_path in tqdm(sampled_files, desc="Episodes"):
    try:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        # --- Sanity check: raw task_relevant_qa from first episode ---
        if not sanity_printed:
            print("\n--- task_relevant_qa[0] from first loaded episode (raw) ---")
            print(ep["task_relevant_qa"][0])
            print()
            sanity_printed = True

        # --- Final frame and QA pairs ---
        final_frame = ep["frames"][-1]          # (H, W, 3) uint8
        qa_pairs    = ep["task_relevant_qa"][-1]

        # Keep only "touching" QA pairs
        touching_qas = [
            qa for qa in qa_pairs
            if "touching" in qa["question"].lower()
        ]

        if not touching_qas:
            continue

        # --- Statement conversion sanity check (first 3 episodes) ---
        if statement_examples_printed < 3:
            print(f"  Statement conversion examples ({os.path.basename(pkl_path)}):")
            for qa in touching_qas:
                stmt = qa_to_statement(qa["question"], qa["answer"])
                print(f"    Q: {qa['question']}  A: {qa['answer']}  → \"{stmt}\"")
            print()
            statement_examples_printed += 1

        # --- CLIP image embedding ---
        pil_img = Image.fromarray(final_frame)
        img_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            z_image = model.encode_image(img_tensor)
            z_image = F.normalize(z_image, dim=-1)          # (1, D)

        # --- CLIP text embeddings: average over touching statements, then normalize ---
        statements = [qa_to_statement(qa["question"], qa["answer"]) for qa in touching_qas]
        tokens = tokenizer(statements).to(DEVICE)
        with torch.no_grad():
            z_texts = model.encode_text(tokens)             # (N_stmts, D)
            z_text  = F.normalize(z_texts.mean(dim=0, keepdim=True), dim=-1)  # (1, D)

        # --- Late fusion ---
        z_sem = F.normalize((z_image + z_text) / 2.0, dim=-1)   # (1, D)

        z_sems.append(z_sem.cpu().float().numpy()[0])

        # Label: all touching_qas should agree (symmetric pair), use first
        label = "touching" if touching_qas[0]["answer"] else "not touching"
        labels.append(label)

        start_blocks.append(ep["metadata"]["start_block"])
        target_blocks.append(ep["metadata"]["oracle_target_block"])

    except Exception as e:
        print(f"Skipping {pkl_path}: {e}")
        continue

z_sems        = np.array(z_sems)
labels        = np.array(labels)
start_blocks  = np.array(start_blocks)
target_blocks = np.array(target_blocks)

# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
n_touch     = (labels == "touching").sum()
n_not_touch = (labels == "not touching").sum()
print(f"\nEpisodes collected: {len(z_sems)}")
print(f"  touching:     {n_touch}")
print(f"  not touching: {n_not_touch}\n")

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
COLOR_MAP = {"touching": "tab:blue", "not touching": "tab:orange"}

SHAPE_MARKERS = {
    "star":     "*",
    "cube":     "s",
    "moon":     "v",
    "pentagon": "D",
    "circle":   "o",
    "triangle": "^",
}

def get_marker(block_name: str) -> str:
    suffix = block_name.split("_")[-1]
    return SHAPE_MARKERS.get(suffix, "P")


def plot_with_objects(ax, coords, labels, start_blocks, target_blocks, title):
    """Scatter plot encoding: color=start_block, shape=target_block type, fill=label."""
    unique_starts  = sorted(set(start_blocks))
    unique_tshapes = sorted({b.split("_")[-1] for b in target_blocks})

    cmap = plt.cm.tab20
    start_color = {b: cmap(i / max(len(unique_starts) - 1, 1))
                   for i, b in enumerate(unique_starts)}

    for sb in unique_starts:
        for ts in unique_tshapes:
            for lbl, filled in [("touching", True), ("not touching", False)]:
                mask = (start_blocks == sb) & \
                       (np.array([b.split("_")[-1] for b in target_blocks]) == ts) & \
                       (labels == lbl)
                if not mask.any():
                    continue
                color  = start_color[sb]
                marker = SHAPE_MARKERS.get(ts, "P")
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    facecolors=color if filled else "none",
                    edgecolors=color,
                    marker=marker,
                    s=70, linewidths=1.5, alpha=0.85,
                )

    ax.set_title(title)

    # --- Legend: colors for start_block ---
    color_handles = [
        mpatches.Patch(color=start_color[b], label=b)
        for b in unique_starts
    ]
    # --- Legend: shapes for target block type ---
    shape_handles = [
        mlines.Line2D([], [], color="gray", marker=SHAPE_MARKERS.get(ts, "P"),
                      linestyle="None", markersize=8, label=f"target: {ts}")
        for ts in unique_tshapes
    ]
    # --- Legend: fill for label ---
    fill_handles = [
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=8, markerfacecolor="gray", label="touching"),
        mlines.Line2D([], [], color="gray", marker="o", linestyle="None",
                      markersize=8, markerfacecolor="none", markeredgewidth=1.5,
                      label="not touching"),
    ]

    legend = ax.legend(
        handles=color_handles + shape_handles + fill_handles,
        bbox_to_anchor=(1.05, 1), loc="upper left",
        fontsize=7, framealpha=0.8,
    )

def scatter_by_label(ax, coords, labels, title):
    for label, color in COLOR_MAP.items():
        mask = labels == label
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=f"{label} (n={mask.sum()})",
            alpha=0.7, s=30, linewidths=0,
        )
    ax.set_title(title)
    ax.legend()

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
pca = PCA(n_components=2, random_state=SEED)
z_pca = pca.fit_transform(z_sems)

fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_label(ax, z_pca, labels, "PCA of z_sem (CLIP ViT-L/14 late fusion)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.tight_layout()
out_pca = os.path.join(OUT_DIR, "zsem_viz_pca.png")
plt.savefig(out_pca, dpi=150)
plt.close()
print(f"Saved {out_pca}")

# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------
tsne  = TSNE(n_components=2, random_state=SEED, perplexity=30)
z_tsne = tsne.fit_transform(z_sems)

fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_label(ax, z_tsne, labels, "t-SNE of z_sem (CLIP ViT-L/14 late fusion)")
plt.tight_layout()
out_tsne = os.path.join(OUT_DIR, "zsem_viz_tsne.png")
plt.savefig(out_tsne, dpi=150)
plt.close()
print(f"Saved {out_tsne}")

# ---------------------------------------------------------------------------
# PCA — object encoding
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
plot_with_objects(
    ax, z_pca, labels, start_blocks, target_blocks,
    "PCA of z_sem — color: start block  |  shape: target block type  |  fill: touching",
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
out_pca_obj = os.path.join(OUT_DIR, "zsem_viz_pca_objects.png")
plt.savefig(out_pca_obj, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_pca_obj}")

# ---------------------------------------------------------------------------
# t-SNE — object encoding
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
plot_with_objects(
    ax, z_tsne, labels, start_blocks, target_blocks,
    "t-SNE of z_sem — color: start block  |  shape: target block type  |  fill: touching",
)
out_tsne_obj = os.path.join(OUT_DIR, "zsem_viz_tsne_objects.png")
plt.savefig(out_tsne_obj, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_tsne_obj}")
