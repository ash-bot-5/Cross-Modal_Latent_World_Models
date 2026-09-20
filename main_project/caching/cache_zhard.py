"""
Pre-cache hard negative CLIP text embeddings for all LangTable episodes.

For each episode, generates 448 false statements per timestep (56 ordered block
pairs (A,B) and (B,A) × 8 peg-block combos), deduplicates unique strings across
all timesteps, encodes with frozen CLIP, and saves as {stem}_zhard.npz with:
  - "unique_zsem": (N, 768) float32 — deduplicated encoded negatives
  - "neg_indices": (T, 448) int16  — per-timestep indices into unique_zsem,
                                     guaranteed false at that specific timestep

Both sentences in every string are false by construction:
  "The {A} is {touching_AB} the {B}. The peg is {touching_C} the {C}."
  touching_AB = "not touching" if dist(A,B) < 0.05  (they ARE touching)
              = "touching"     if dist(A,B) >= 0.05  (they are NOT touching)
  (same inverted logic for peg-C)

Usage:
    python cache_zhard.py             # cache all episodes
    python cache_zhard.py --smoke     # test 3 episodes, assert shapes + validity
    python cache_zhard.py --profile   # time 20 episodes, no disk writes
"""

import sys
import pickle
import time
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import open_clip

DATA_DIR     = "/home/ashwink/full_data_5000/langtable/demos/"
TOUCH_THRESH = 0.05
CLIP_BATCH   = 256


def load_clip(device: str):
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    return model, tokenizer


def build_hard_neg_data(ep: dict) -> tuple[list[str], np.ndarray]:
    """
    Single pass over all T timesteps.

    Returns:
        unique_strings: list of N unique false-statement strings (ordered by first occurrence)
        neg_indices: (T, 448) int16 — per-timestep indices into unique_strings,
                     guaranteed false at that specific timestep
    """
    block_states  = ep["block_states"]         # list[T] of dict: name → (2,)
    peg_positions = ep["actual_peg_positions"]  # (T, 2)
    T = len(block_states)

    # block_states keys include a 'peg' entry — exclude it to keep only the 8 block names
    block_names = sorted(k for k in block_states[0].keys() if k != 'peg')
    pairs       = list(permutations(block_names, 2))  # P(8,2) = 56 ordered pairs, (A,B) and (B,A)

    assert len(pairs) * len(block_names) == 448, \
        f"Expected 448 negs per timestep, got {len(pairs) * len(block_names)}"
    string_to_idx: dict[str, int] = {}
    unique_strings: list[str]     = []
    neg_indices = np.zeros((T, 448), dtype=np.int16)

    for t in range(T):
        bpos = block_states[t]
        peg  = peg_positions[t]
        col  = 0

        for (a, b) in pairs:
            dist_ab        = np.linalg.norm(bpos[a] - bpos[b])
            touching_ab    = "not touching" if dist_ab < TOUCH_THRESH else "touching"
            a_disp, b_disp = a.replace("_", " "), b.replace("_", " ")

            for c in block_names:
                dist_pc    = np.linalg.norm(peg - bpos[c])
                touching_c = "not touching" if dist_pc < TOUCH_THRESH else "touching"
                c_disp     = c.replace("_", " ")

                s = (f"The {a_disp} is {touching_ab} the {b_disp}. "
                     f"The peg is {touching_c} the {c_disp}.")

                if s not in string_to_idx:
                    string_to_idx[s] = len(unique_strings)
                    unique_strings.append(s)

                neg_indices[t, col] = string_to_idx[s]
                col += 1

    return unique_strings, neg_indices


def encode_strings(strings: list[str], model, tokenizer, device: str) -> np.ndarray:
    """Encode strings with CLIP in batches. Returns (N, 768) float32."""
    parts = []
    with torch.no_grad():
        for i in range(0, len(strings), CLIP_BATCH):
            tokens = tokenizer(strings[i : i + CLIP_BATCH]).to(device)
            z = F.normalize(model.encode_text(tokens), dim=-1)
            parts.append(z.cpu().float().numpy())
    return np.concatenate(parts, axis=0)


def get_pkl_files() -> list[Path]:
    return sorted(Path(DATA_DIR).glob("*.pkl"))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # --smoke: test on 3 episodes, write to disk, assert shapes + index validity
    # ------------------------------------------------------------------
    if "--smoke" in sys.argv:
        print(f"[smoke] Loading CLIP on {device}...")
        model, tokenizer = load_clip(device)

        done = 0
        for pkl_path in get_pkl_files():
            stem      = pkl_path.stem
            zsem_path = pkl_path.with_name(stem + "_zsem.npz")
            if not zsem_path.exists():
                continue

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)

            unique_strings, neg_indices = build_hard_neg_data(ep)
            vecs       = encode_strings(unique_strings, model, tokenizer, device)
            zhard_path = pkl_path.with_name(stem + "_zhard.npz")
            np.savez(str(zhard_path), unique_zsem=vecs, neg_indices=neg_indices)

            T_ep = len(ep["block_states"])
            loaded = np.load(str(zhard_path))
            assert loaded["neg_indices"].shape == (T_ep, 448), \
                f"Expected ({T_ep}, 448), got {loaded['neg_indices'].shape}"
            assert loaded["neg_indices"].dtype == np.int16, \
                f"Expected int16, got {loaded['neg_indices'].dtype}"
            N = loaded["unique_zsem"].shape[0]
            assert int(loaded["neg_indices"].max()) < N and int(loaded["neg_indices"].min()) >= 0, \
                "neg_indices out of range"
            print(f"  {stem}: unique_zsem={vecs.shape}, neg_indices={neg_indices.shape}")

            done += 1
            if done >= 3:
                break

        print("Smoke test passed.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # --profile: time 20 episodes, no disk writes
    # ------------------------------------------------------------------
    if "--profile" in sys.argv:
        print(f"[profile] Loading CLIP on {device}...")
        model, tokenizer = load_clip(device)

        unique_counts: list[int]   = []
        encode_times:  list[float] = []
        done = 0

        for pkl_path in get_pkl_files():
            stem      = pkl_path.stem
            zsem_path = pkl_path.with_name(stem + "_zsem.npz")
            if not zsem_path.exists():
                continue

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)

            unique_strings, neg_indices = build_hard_neg_data(ep)
            assert neg_indices.shape[1] == 448, f"Expected 448 negs per t, got {neg_indices.shape[1]}"
            unique_counts.append(len(unique_strings))

            t0 = time.perf_counter()
            encode_strings(unique_strings, model, tokenizer, device)
            encode_times.append(time.perf_counter() - t0)

            done += 1
            if done >= 20:
                break

        mean_unique = sum(unique_counts) / len(unique_counts)
        total_unique = sum(unique_counts)
        mean_encode  = sum(encode_times)  / len(encode_times)
        est_total_s  = mean_encode * 5000

        print(f"Mean unique strings per episode : {mean_unique:.1f}")
        print(f"Total unique strings (20 eps)   : {total_unique}")
        print(f"Mean encoding time per episode  : {mean_encode:.3f}s")
        print(f"Estimated total runtime (5000)  : {est_total_s / 60:.1f} min")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Default: cache all episodes
    # ------------------------------------------------------------------
    print(f"Loading CLIP on {device}...")
    model, tokenizer = load_clip(device)

    processed = 0
    for pkl_path in get_pkl_files():
        stem       = pkl_path.stem
        zsem_path  = pkl_path.with_name(stem + "_zsem.npz")
        zhard_path = pkl_path.with_name(stem + "_zhard.npz")

        if not zsem_path.exists():
            continue
        if zhard_path.exists():
            continue

        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        unique_strings, neg_indices = build_hard_neg_data(ep)
        vecs = encode_strings(unique_strings, model, tokenizer, device)
        np.savez(str(zhard_path), unique_zsem=vecs, neg_indices=neg_indices)

        processed += 1
        if processed % 100 == 0:
            print(f"[{processed}] cached {stem}")

    print(f"Done. Processed {processed} episodes.")
