"""
Smoke test for full_5000_cache_zsem.py — runs on the first 3 episodes (deterministic)
and prints verbose diagnostics so you can verify before running the full 5000.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/caching/smoke_test_cache_zsem.py
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = "/home/ashwink/full_data_5000/langtable/demos/"
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_zsem_out")
N_EPISODES = 3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
SEED       = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

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
# Statement conversion (identical to full_5000_cache_zsem.py)
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
# Process 3 episodes
# ---------------------------------------------------------------------------
pkl_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))[:N_EPISODES]
print(f"Smoke-testing on {len(pkl_files)} episodes:\n")

n_ok = 0

for pkl_path in pkl_files:
    stem     = Path(pkl_path).stem
    out_path = os.path.join(OUT_DIR, stem + "_zsem.npz")

    print(f"{'='*60}")
    print(f"Episode: {stem}")

    try:
        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        T  = len(ep["task_relevant_qa"])
        sb = ep["metadata"]["start_block"]
        tb = ep["metadata"]["oracle_target_block"]
        print(f"T={T}  start={sb}  target={tb}")

        peg_qa_list = ep["peg_touching_starting_block_qa"]
        if len(peg_qa_list) < T:
            print(f"  ERROR: peg_touching_starting_block_qa has {len(peg_qa_list)} entries, expected {T}. Skipping.")
            continue

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
                print(f"  ERROR t={t}: fewer than 2 touching QAs. Skipping episode.")
                valid = False
                break

            stmt_ab = qa_to_statement(touching_qas[0]["question"], touching_qas[0]["answer"])
            peg_raw = peg_qa_list[t]
            if isinstance(peg_raw, list):
                peg_raw = peg_raw[0]
            peg_stmt = qa_to_statement(peg_raw["question"], peg_raw["answer"])

            compound = f"{stmt_ab}. {peg_stmt}."
            all_stmts.append(compound)
            touching_labels.append(bool(touching_qas[0]["answer"]))
            peg_labels.append(bool(peg_raw["answer"]))

        if not valid:
            continue

        # Print sample compounds
        for label, t in [("t=0 ", 0), ("t=-1", T - 1)]:
            print(f"  {label}  compound: \"{all_stmts[t]}\"")

        # Batched CLIP call
        tokens = tokenizer(all_stmts).to(DEVICE)
        with torch.no_grad():
            z = F.normalize(model.encode_text(tokens), dim=-1)
        z_sem = z.cpu().float().numpy()

        # Shape + norm check
        norms = np.linalg.norm(z_sem[:3], axis=-1)
        print(f"  z_sem shape:  {z_sem.shape}  dtype: {z_sem.dtype}")
        print(f"  norms (t=0..2): [{', '.join(f'{v:.4f}' for v in norms)}]  ← should all be 1.0")

        # Label distributions
        n_touch    = sum(touching_labels)
        n_peg      = sum(peg_labels)
        print(f"  touching:     {n_touch} True / {T - n_touch} False")
        print(f"  peg_touching: {n_peg} True / {T - n_peg} False")

        # Save
        np.savez(
            out_path,
            z_sem        = z_sem,
            touching     = np.array(touching_labels, dtype=bool),
            peg_touching = np.array(peg_labels, dtype=bool),
        )

        # Round-trip check
        loaded = np.load(out_path)
        keys   = list(loaded.keys())
        zs     = loaded["z_sem"]
        tch    = loaded["touching"]
        peg    = loaded["peg_touching"]
        ok     = (zs.shape == z_sem.shape and tch.shape == (T,) and peg.shape == (T,))
        print(f"  Reloaded {stem}_zsem.npz — keys: {keys}")
        print(f"  z_sem: {zs.shape} {zs.dtype}   touching: {tch.shape} {tch.dtype}   peg_touching: {peg.shape} {peg.dtype}   {'✓' if ok else '✗ SHAPE MISMATCH'}")

        n_ok += 1

    except Exception as e:
        print(f"  ERROR: {e}")
        continue

print(f"\n{'='*60}")
print(f"Smoke test complete. {n_ok}/{N_EPISODES} episodes processed successfully.")
print(f"Output written to: {OUT_DIR}")
