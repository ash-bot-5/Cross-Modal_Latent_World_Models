"""
Cache compound z_sem vectors for all timesteps of all held-out episodes.

For each timestep t in each episode:
  compound_stmt = f"{stmt_ab}. {peg_stmt}."
  z_sem[t] = L2_normalize(CLIP_text_encode(compound_stmt))

where stmt_ab is the A→B touching statement and peg_stmt is the peg-touching statement.
All T compound strings per episode are encoded in a single CLIP forward pass.

Output: {DATA_DIR}/{stem}_zsem.npz with keys:
  z_sem        — (T, 768) float32
  touching     — (T,) bool
  peg_touching — (T,) bool

Safe to re-run: existing npz files are skipped.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/cache_held_out_episodes/cache_zsem_clip.py
"""

import open_clip

import glob
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/home/ashwink/held_out_episode_5/demos/"
OUT_DIR  = DATA_DIR
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEED     = 42

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
# Main loop
# ---------------------------------------------------------------------------
pkl_files  = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
n_total    = len(pkl_files)
n_processed = 0
n_skipped   = 0
n_failed    = 0

print(f"Found {n_total} pkl files in {DATA_DIR}\n")

for pkl_path in tqdm(pkl_files, desc="Episodes"):
    stem     = Path(pkl_path).stem
    out_path = os.path.join(OUT_DIR, stem + "_zsem.npz")

    if os.path.exists(out_path):
        n_skipped += 1
        continue

    try:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        T = len(ep["task_relevant_qa"])
        has_peg_qa = "peg_touching_starting_block_qa" in ep
        peg_qa_list = ep.get("peg_touching_starting_block_qa", [])

        # For episodes without peg_touching QA, derive from positions
        start_block = ep["metadata"]["start_block"]
        PEG_TOUCH_THRESH = 0.05

        all_stmts       = []
        touching_labels = []
        peg_labels      = []
        valid           = True

        for t in range(T):
            touching_qas = [
                qa for qa in ep["task_relevant_qa"][t]
                if "touching" in qa["question"].lower()
            ]
            if len(touching_qas) < 2:
                tqdm.write(f"WARN {stem} t={t}: fewer than 2 touching QAs. Skipping episode.")
                valid = False
                break

            stmt_ab = qa_to_statement(touching_qas[0]["question"], touching_qas[0]["answer"])

            if has_peg_qa and t < len(peg_qa_list):
                peg_raw = peg_qa_list[t]
                if isinstance(peg_raw, list):
                    peg_raw = peg_raw[0]
                peg_stmt = qa_to_statement(peg_raw["question"], peg_raw["answer"])
                peg_touching = bool(peg_raw["answer"])
            else:
                # Derive peg touching from positions
                peg_pos = ep["actual_peg_positions"][t]
                block_pos = ep["block_states"][t][start_block]
                dist = np.linalg.norm(peg_pos - block_pos)
                peg_touching = dist < PEG_TOUCH_THRESH
                start_disp = start_block.replace("_", " ")
                touching_word = "is touching" if peg_touching else "is not touching"
                peg_stmt = f"the peg {touching_word} the {start_disp}"

            compound_stmt = f"{stmt_ab}. {peg_stmt}."
            all_stmts.append(compound_stmt)
            touching_labels.append(bool(touching_qas[0]["answer"]))
            peg_labels.append(peg_touching)

        if not valid:
            n_failed += 1
            continue

        # Single batched CLIP call over all T timesteps
        tokens = tokenizer(all_stmts).to(DEVICE)   # (T, 77)
        with torch.no_grad():
            z = F.normalize(model.encode_text(tokens), dim=-1)  # (T, 768)
        z_sem = z.cpu().float().numpy()

        np.savez(
            out_path,
            z_sem        = z_sem,
            touching     = np.array(touching_labels, dtype=bool),
            peg_touching = np.array(peg_labels, dtype=bool),
        )
        n_processed += 1

    except Exception as e:
        tqdm.write(f"ERROR {stem}: {e}")
        n_failed += 1
        continue

print(f"\nDone. Processed: {n_processed} | Skipped (already existed): {n_skipped} | Failed: {n_failed}")
