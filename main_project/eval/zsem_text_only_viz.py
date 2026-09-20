"""
Compute z_sem latent vectors via CLIP text encoder only (no image) and visualize.

For each episode, three z_sem vectors are produced:
  AB  — CLIP text embedding of the A→B touching statement
  BA  — CLIP text embedding of the B→A touching statement
  avg — L2-normalized mean of AB and BA

Model: OpenCLIP ViT-L/14, pretrained=datacomp_xl_s13b_b90k

Active output plots:
  zsem_text_only_viz_tsne_single_pair.png  — all 5000 eps, green_star→blue_cube avg, color=label
  zsem_text_only_viz_tsne_task_identity.png — 500 eps, avg, color=task (start→target)

Commented-out plots (already saved):
  zsem_text_only_viz_pca_objects.png / tsne_objects.png
  zsem_text_only_viz_pca_types.png  / tsne_types.png

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/zsem_text_only_viz.py
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = "/home/ashwink/full_data_5000/langtable/demos/"
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
N_EPISODES = 500
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Load OpenCLIP (text encoder only — no image preprocessing needed)
# ---------------------------------------------------------------------------
print(f"Loading OpenCLIP ViT-L/14 (datacomp_xl_s13b_b90k) on {DEVICE} ...")
model, _, _ = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer("ViT-L-14")
print("Model loaded.\n")

# ---------------------------------------------------------------------------
# Block lookup tables (fixed across the full dataset)
# (commented out — used only by the _objects plots below)
# ---------------------------------------------------------------------------
# BLOCKS = [
#     "blue_cube", "blue_moon", "green_cube", "green_star",
#     "red_moon",  "red_pentagon", "yellow_pentagon", "yellow_star",
# ]
# BLOCK_COLORS = {b: plt.cm.tab10(i / 10) for i, b in enumerate(BLOCKS)}
# BLOCK_MARKERS = {
#     "blue_cube":       "o",
#     "blue_moon":       "v",
#     "green_cube":      "s",
#     "green_star":      "*",
#     "red_moon":        "^",
#     "red_pentagon":    "D",
#     "yellow_pentagon": "P",
#     "yellow_star":     "X",
# }
# BLOCK_MARKER_SIZES = {b: (120 if BLOCK_MARKERS[b] == "*" else 70) for b in BLOCKS}

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
# Sample episodes (500 for main loop / Plot 2)
# ---------------------------------------------------------------------------
all_files     = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
sampled_files = random.sample(all_files, min(N_EPISODES, len(all_files)))
print(f"Sampled {len(sampled_files)} episodes from {len(all_files)} total.\n")

# ---------------------------------------------------------------------------
# Main loop — three z_sem vectors per episode
# ---------------------------------------------------------------------------
z_sems        = []
labels        = []
embed_types   = []
start_blocks  = []
target_blocks = []

sanity_printed             = False
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

        qa_pairs = ep["task_relevant_qa"][-1]
        touching_qas = [
            qa for qa in qa_pairs
            if "touching" in qa["question"].lower()
        ]

        if len(touching_qas) < 2:
            continue

        stmt_ab = qa_to_statement(touching_qas[0]["question"], touching_qas[0]["answer"])
        stmt_ba = qa_to_statement(touching_qas[1]["question"], touching_qas[1]["answer"])

        # --- Statement conversion sanity check (first 3 episodes) ---
        if statement_examples_printed < 3:
            print(f"  Statements ({os.path.basename(pkl_path)}):")
            print(f"    AB: \"{stmt_ab}\"")
            print(f"    BA: \"{stmt_ba}\"")
            print()
            statement_examples_printed += 1

        tokens_ab = tokenizer([stmt_ab]).to(DEVICE)
        tokens_ba = tokenizer([stmt_ba]).to(DEVICE)
        with torch.no_grad():
            z_ab  = F.normalize(model.encode_text(tokens_ab), dim=-1)   # (1, D)
            z_ba  = F.normalize(model.encode_text(tokens_ba), dim=-1)   # (1, D)
        z_avg = F.normalize((z_ab + z_ba) / 2.0, dim=-1)               # (1, D)

        label = "touching" if touching_qas[0]["answer"] else "not touching"
        sb    = ep["metadata"]["start_block"]
        tb    = ep["metadata"]["oracle_target_block"]

        for vec, etype in [(z_ab, "AB"), (z_ba, "BA"), (z_avg, "avg")]:
            z_sems.append(vec.cpu().float().numpy()[0])
            labels.append(label)
            embed_types.append(etype)
            start_blocks.append(sb)
            target_blocks.append(tb)

    except Exception as e:
        print(f"Skipping {pkl_path}: {e}")
        continue

z_sems        = np.array(z_sems)
labels        = np.array(labels)
embed_types   = np.array(embed_types)
start_blocks  = np.array(start_blocks)
target_blocks = np.array(target_blocks)

# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
n_touch     = (labels[embed_types == "avg"] == "touching").sum()
n_not_touch = (labels[embed_types == "avg"] == "not touching").sum()
print(f"\nEpisodes collected: {(embed_types == 'avg').sum()}")
print(f"  touching:     {n_touch}")
print(f"  not touching: {n_not_touch}\n")

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Commented-out: original PCA/t-SNE + 4 plots (already saved)
# ---------------------------------------------------------------------------
# pca   = PCA(n_components=2, random_state=SEED)
# z_pca = pca.fit_transform(z_sems)
#
# tsne   = TSNE(n_components=2, random_state=SEED, perplexity=30)
# z_tsne = tsne.fit_transform(z_sems)
#
# LABEL_COLORS  = {"touching": "tab:blue", "not touching": "tab:orange"}
# ETYPE_MARKERS = {"AB": "o", "BA": "s", "avg": "*"}
# ETYPE_SIZES   = {"AB": 40,  "BA": 40,  "avg": 100}
#
# def plot_objects(ax, coords, labels, start_blocks, target_blocks, embed_types, title):
#     ... (see git history)
#
# def plot_types(ax, coords, labels, embed_types, title):
#     ... (see git history)
#
# for proj_name, coords in [("pca", z_pca), ("tsne", z_tsne)]:
#     ... plot_objects / plot_types calls ...

# ---------------------------------------------------------------------------
# Label colors used by the new plots
# ---------------------------------------------------------------------------
LABEL_COLORS = {"touching": "tab:blue", "not touching": "tab:orange"}

# ===========================================================================
# New Plot 1: single object pair zoom-in (all 5000 episodes)
# ===========================================================================
PAIR_START, PAIR_TARGET = "green_star", "blue_cube"
print(f"\nCollecting all episodes for {PAIR_START} → {PAIR_TARGET} ...")

z_pair   = []
lbl_pair = []

for pkl_path in tqdm(all_files, desc=f"Pair: {PAIR_START}→{PAIR_TARGET}"):
    try:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)
        meta = ep["metadata"]
        if meta["start_block"] != PAIR_START or meta["oracle_target_block"] != PAIR_TARGET:
            continue
        qa_pairs = ep["task_relevant_qa"][-1]
        touching_qas = [qa for qa in qa_pairs if "touching" in qa["question"].lower()]
        if len(touching_qas) < 2:
            continue
        stmt_ab = qa_to_statement(touching_qas[0]["question"], touching_qas[0]["answer"])
        stmt_ba = qa_to_statement(touching_qas[1]["question"], touching_qas[1]["answer"])
        with torch.no_grad():
            z_ab  = F.normalize(model.encode_text(tokenizer([stmt_ab]).to(DEVICE)), dim=-1)
            z_ba  = F.normalize(model.encode_text(tokenizer([stmt_ba]).to(DEVICE)), dim=-1)
        z_avg = F.normalize((z_ab + z_ba) / 2.0, dim=-1)
        z_pair.append(z_avg.cpu().float().numpy()[0])
        lbl_pair.append("touching" if touching_qas[0]["answer"] else "not touching")
    except Exception as e:
        print(f"Skipping {pkl_path}: {e}")
        continue

z_pair   = np.array(z_pair)
lbl_pair = np.array(lbl_pair)
n_touch_pair     = (lbl_pair == "touching").sum()
n_nottouch_pair  = (lbl_pair == "not touching").sum()
print(f"{PAIR_START}→{PAIR_TARGET}: {len(z_pair)} episodes "
      f"(touching={n_touch_pair}, not touching={n_nottouch_pair})")

if len(z_pair) >= 30:
    coords_pair = TSNE(
        n_components=2, random_state=SEED, perplexity=min(30, len(z_pair) - 1)
    ).fit_transform(z_pair)
    proj_label = "t-SNE"
else:
    coords_pair = PCA(n_components=2, random_state=SEED).fit_transform(z_pair)
    proj_label  = "PCA"

fig, ax = plt.subplots(figsize=(7, 5))
for lbl, color in LABEL_COLORS.items():
    m = lbl_pair == lbl
    ax.scatter(
        coords_pair[m, 0], coords_pair[m, 1],
        c=color, s=60, alpha=0.8, linewidths=0,
        label=f"{lbl} (n={m.sum()})",
    )
ax.set_title(f"{proj_label} of z_sem (text only) — {PAIR_START} → {PAIR_TARGET} (avg)")
ax.legend(fontsize=9)
plt.tight_layout()
out_pair = os.path.join(OUT_DIR, "zsem_text_only_viz_tsne_single_pair.png")
plt.savefig(out_pair, dpi=150)
plt.close()
print(f"Saved {out_pair}")

# ===========================================================================
# New Plot 2: task identity clustering (500-episode avg subset)
# ===========================================================================
mask_avg   = embed_types == "avg"
z_avg_all  = z_sems[mask_avg]
sb_avg     = start_blocks[mask_avg]
tb_avg     = target_blocks[mask_avg]
task_ids   = np.array([f"{s} → {t}" for s, t in zip(sb_avg, tb_avg)])

unique_tasks = sorted(set(task_ids))
cmap         = plt.cm.gist_rainbow
task_colors  = {
    t: cmap(i / max(len(unique_tasks) - 1, 1))
    for i, t in enumerate(unique_tasks)
}

print(f"\nRunning t-SNE for task identity plot ({len(z_avg_all)} avg points, "
      f"{len(unique_tasks)} unique tasks) ...")
coords_task = TSNE(n_components=2, random_state=SEED, perplexity=30).fit_transform(z_avg_all)

fig, ax = plt.subplots(figsize=(10, 6))
for task in unique_tasks:
    m = task_ids == task
    ax.scatter(
        coords_task[m, 0], coords_task[m, 1],
        c=[task_colors[task]], s=40, alpha=0.8, linewidths=0,
        label=task,
    )
ax.set_title("t-SNE of z_sem (text only, avg, 500 eps) — colored by task identity")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=6, framealpha=0.8)
out_task = os.path.join(OUT_DIR, "zsem_text_only_viz_tsne_task_identity.png")
plt.savefig(out_task, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_task}")
