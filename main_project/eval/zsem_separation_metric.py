"""
Quantify touching/not-touching separation for compound vs. averaged z_sem encodings.

For each (start_block, target_block) pair that has at least one touching and one
not-touching episode, compute the cosine distance between the touching centroid and
the not-touching centroid. Report mean, std, min, max across all qualifying pairs.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/zsem_separation_metric.py
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = "/home/ashwink/full_data_5000/langtable/demos/"
N_EPISODES = 2000
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
touch_labels = []
start_blocks = []
target_blocks = []

for pkl_path in tqdm(sampled_files, desc="Episodes"):
    try:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        # --- Touching QAs ---
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
        touch_labels.append("touching" if touching_qas[0]["answer"] else "not touching")
        start_blocks.append(ep["metadata"]["start_block"])
        target_blocks.append(ep["metadata"]["oracle_target_block"])

    except Exception as e:
        print(f"Skipping {pkl_path}: {e}")
        continue

z_compounds   = np.array(z_compounds)
z_averageds   = np.array(z_averageds)
touch_labels  = np.array(touch_labels)
start_blocks  = np.array(start_blocks)
target_blocks = np.array(target_blocks)

n_touch     = (touch_labels == "touching").sum()
n_not_touch = (touch_labels == "not touching").sum()
unique_tasks = sorted(set(zip(start_blocks, target_blocks)))
print(f"Episodes collected: {len(z_compounds)}")
print(f"  touching:     {n_touch}")
print(f"  not touching: {n_not_touch}")
print(f"  unique tasks: {len(unique_tasks)}\n")

# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_separation(z_vecs, touch_labels, start_blocks, target_blocks):
    pairs = sorted(set(zip(start_blocks, target_blocks)))
    distances = []
    for sb, tb in pairs:
        mask    = (start_blocks == sb) & (target_blocks == tb)
        mask_t  = mask & (touch_labels == "touching")
        mask_nt = mask & (touch_labels == "not touching")
        if mask_t.sum() == 0 or mask_nt.sum() == 0:
            continue
        c_t  = z_vecs[mask_t].mean(axis=0)
        c_nt = z_vecs[mask_nt].mean(axis=0)
        cos_sim  = np.dot(c_t, c_nt) / (np.linalg.norm(c_t) * np.linalg.norm(c_nt))
        distances.append(1.0 - cos_sim)
    d = np.array(distances)
    return {
        "mean": d.mean(), "std": d.std(),
        "min": d.min(), "max": d.max(),
        "n_qualifying": len(d), "n_total_pairs": len(pairs),
    }

r_compound = compute_separation(z_compounds,  touch_labels, start_blocks, target_blocks)
r_averaged = compute_separation(z_averageds,  touch_labels, start_blocks, target_blocks)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
W = 16  # column width

def fmt(v):
    return f"{v:.4f}"

def frac(r):
    return f"{r['n_qualifying']} / {r['n_total_pairs']}"

header  = f"{'':20s} {'compound':>{W}}  {'averaged':>{W}}"
divider = "━" * (20 + W * 2 + 4)
sep     = "─" * (20 + W * 2 + 4)

print("Separation metric: cosine distance between touching / not-touching centroids per task pair")
print(divider)
print(header)
print(sep)
for label, key in [("mean", "mean"), ("std", "std"), ("min", "min"), ("max", "max")]:
    print(f"{label:20s} {fmt(r_compound[key]):>{W}}  {fmt(r_averaged[key]):>{W}}")
print(f"{'qualifying pairs':20s} {frac(r_compound):>{W}}  {frac(r_averaged):>{W}}")
print(divider)
