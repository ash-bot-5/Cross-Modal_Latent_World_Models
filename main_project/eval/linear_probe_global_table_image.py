"""
Image-side linear-probe evaluation of the fine-tuned CLIP image embedder.

Mirrors linear_probe_global_table.py (the text-side probe), adapted for real frames:
GlobalStatementTable's 1792 synthetic compound statements have no image analogue (a
single real frame can't depict an arbitrary hypothetical peg-block relation), so
instead we sample real (episode, timestep) frames and compute their TRUE touching
labels directly from block/peg positions, using the exact TOUCH_THRESH=0.05 threshold
the model was trained against (dynamics_model_clip_full_finetune.py:79,190-201).

Two independent logistic regression probes, sharing the same per-checkpoint image
embeddings:
  1. A-B touching probe   -- identity-disjoint split grouped by pair_id
                             (frozenset({start_block, target_block}), as "a|b" string)
  2. Peg touching probe   -- identity-disjoint split grouped by peg_block_id
                             (= start_block, the only block the real peg relation is
                             ever about)

Each split holds out entire identity groups (not individual rows/episodes), so no
block-pair or peg-block identity present in val is ever also present in train.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/linear_probe_global_table_image.py
"""

import os
import json
import pickle
import random
import glob

import numpy as np
import torch
import open_clip
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
CKPT_DIR = "/home/ashwink/Documents/swms/main_project/training/checkpoints_clip_image_text_finetuned/"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINTS = ["pretrained", "epoch_000.pt", "epoch_001.pt", "epoch_008.pt"]

TOUCH_THRESH = 0.05  # matches dynamics_model_clip_full_finetune.py exactly

N_EPISODES = 400
TIMESTEP_STRIDE = 2
EPISODE_SAMPLE_SEED = 0

SPLIT_SEED_PROBE1 = 42   # held-out A-B pair identities
SPLIT_SEED_PROBE2 = 123  # held-out peg-block identities
VAL_PAIR_FRAC = 0.2      # ~20% of discovered pair identities
VAL_PEG_FRAC = 0.25      # ~25% of discovered peg-block identities

BATCH_SIZE = 64  # smaller than the text probe's 256 -- image forward passes are heavier

DATA_CACHE_PATH = os.path.join(OUT_DIR, "linear_probe_image_data.npz")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# 1. Data collection (checkpoint-independent) -- real frames + position-derived labels
# ---------------------------------------------------------------------------
def collect_frame_data():
    if os.path.exists(DATA_CACHE_PATH):
        print(f"Loading cached frame data from {DATA_CACHE_PATH}")
        d = np.load(DATA_CACHE_PATH, allow_pickle=False)
        return (
            d["frames"],
            d["ab_label"].astype(bool),
            d["peg_label"].astype(bool),
            d["pair_id"],
            d["peg_block_id"],
            d["episode_id"],
        )

    print(f"Sampling {N_EPISODES} qualifying episodes (>=1 A-B touching timestep each) from {DATA_DIR} ...")
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.pkl")))
    # Full deterministic shuffle (not just N_EPISODES) -- episodes that never reach an
    # A-B touching state (e.g. failures) are skipped, so we may need to scan past more
    # than N_EPISODES candidates to find N_EPISODES qualifying ones.
    shuffled_files = random.Random(EPISODE_SAMPLE_SEED).sample(all_files, len(all_files))

    frames, ab_labels, peg_labels, pair_ids, peg_block_ids, episode_ids = [], [], [], [], [], []
    n_accepted = 0
    n_scanned = 0
    n_skipped_no_touch = 0

    for pkl_path in shuffled_files:
        if n_accepted >= N_EPISODES:
            break
        n_scanned += 1

        with open(pkl_path, "rb") as f:
            ep = pickle.load(f)

        meta = ep["metadata"]
        start_block = meta["start_block"]
        target_block = meta["oracle_target_block"]

        T = len(ep["frames"])
        block_states = ep["block_states"]
        peg_positions = ep["actual_peg_positions"]

        # Full-trajectory A-B touching state (every timestep, not just the strided ones)
        # so we know up front whether this episode ever reaches touching at all.
        ab_touch_full = np.array(
            [np.linalg.norm(block_states[t][start_block] - block_states[t][target_block]) < TOUCH_THRESH
             for t in range(T)]
        )
        if not ab_touch_full.any():
            n_skipped_no_touch += 1
            continue

        n_accepted += 1
        pair_id = "|".join(sorted([start_block, target_block]))

        strided_ts = list(range(0, T, TIMESTEP_STRIDE))
        # Guarantee this episode contributes >=1 touching row even if the stride
        # happens to step over its (possibly brief) touching window.
        if not any(ab_touch_full[t] for t in strided_ts):
            strided_ts = sorted(set(strided_ts) | {int(np.nonzero(ab_touch_full)[0][0])})

        for t in strided_ts:
            bpos = block_states[t]
            peg = peg_positions[t]
            peg_dist = np.linalg.norm(peg - bpos[start_block])

            frames.append(ep["frames"][t])
            ab_labels.append(bool(ab_touch_full[t]))
            peg_labels.append(bool(peg_dist < TOUCH_THRESH))
            pair_ids.append(pair_id)
            peg_block_ids.append(start_block)
            episode_ids.append(os.path.basename(pkl_path)[:-4])

        if n_accepted % 50 == 0:
            print(f"  [{n_accepted}/{N_EPISODES}] qualifying episodes accepted "
                  f"({n_scanned} scanned, {n_skipped_no_touch} skipped for no touching), "
                  f"{len(frames)} rows so far")

    print(f"Done scanning: accepted {n_accepted} episodes out of {n_scanned} candidates scanned "
          f"({n_skipped_no_touch} skipped for never reaching an A-B touching state).")

    frames = np.stack(frames, axis=0)  # (N, 180, 320, 3) uint8
    ab_labels = np.array(ab_labels, dtype=bool)
    peg_labels = np.array(peg_labels, dtype=bool)
    pair_ids = np.array(pair_ids, dtype="<U64")
    peg_block_ids = np.array(peg_block_ids, dtype="<U64")
    episode_ids = np.array(episode_ids, dtype="<U64")

    print(f"Collected {len(frames)} rows total. Caching to {DATA_CACHE_PATH} ...")
    np.savez(
        DATA_CACHE_PATH,
        frames=frames,
        ab_label=ab_labels,
        peg_label=peg_labels,
        pair_id=pair_ids,
        peg_block_id=peg_block_ids,
        episode_id=episode_ids,
    )
    return frames, ab_labels, peg_labels, pair_ids, peg_block_ids, episode_ids


frames, ab_label, peg_label, pair_id_arr, peg_block_id_arr, episode_id_arr = collect_frame_data()
print(f"\nTotal rows: {len(frames)}")
print(f"  ab_label positive (touching): {ab_label.sum()} / {len(ab_label)}")
print(f"  peg_label positive (touching): {peg_label.sum()} / {len(peg_label)}\n")

# ---------------------------------------------------------------------------
# 2. Identity-disjoint splits (independent partitions, grouped by identity tag)
# ---------------------------------------------------------------------------
unique_pairs = sorted(set(pair_id_arr.tolist()))
n_val_pairs = max(1, round(VAL_PAIR_FRAC * len(unique_pairs)))
val_pairs = set(random.Random(SPLIT_SEED_PROBE1).sample(unique_pairs, n_val_pairs))
val1_mask = np.isin(pair_id_arr, list(val_pairs))
train1_mask = ~val1_mask

unique_peg_blocks = sorted(set(peg_block_id_arr.tolist()))
n_val_peg_blocks = max(1, round(VAL_PEG_FRAC * len(unique_peg_blocks)))
val_peg_blocks = set(random.Random(SPLIT_SEED_PROBE2).sample(unique_peg_blocks, n_val_peg_blocks))
val2_mask = np.isin(peg_block_id_arr, list(val_peg_blocks))
train2_mask = ~val2_mask

print("[Probe 1: A-B touching] identity-disjoint split (grouped by pair_id)")
print(f"  discovered pair identities: {len(unique_pairs)}, held out: {sorted(val_pairs)}")
print(f"  train pairs: {len(unique_pairs) - n_val_pairs}, val pairs: {n_val_pairs}, overlap: 0 (by construction)")
print(f"  n_train rows: {train1_mask.sum()} (pos={ab_label[train1_mask].sum()}), "
      f"n_val rows: {val1_mask.sum()} (pos={ab_label[val1_mask].sum()})\n")

print("[Probe 2: peg touching] identity-disjoint split (grouped by peg_block_id)")
print(f"  discovered peg-block identities: {len(unique_peg_blocks)}, held out: {sorted(val_peg_blocks)}")
print(f"  train peg blocks: {len(unique_peg_blocks) - n_val_peg_blocks}, val peg blocks: {n_val_peg_blocks}, overlap: 0 (by construction)")
print(f"  n_train rows: {train2_mask.sum()} (pos={peg_label[train2_mask].sum()}), "
      f"n_val rows: {val2_mask.sum()} (pos={peg_label[val2_mask].sum()})\n")


# ---------------------------------------------------------------------------
# 3. Per-checkpoint embedding + two probes
# ---------------------------------------------------------------------------
def encode_all_images(model, preprocess, frames: np.ndarray) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in range(0, len(frames), BATCH_SIZE):
            batch_frames = frames[i : i + BATCH_SIZE]
            imgs = torch.stack([preprocess(Image.fromarray(f)) for f in batch_frames]).to(DEVICE)
            z = model.encode_image(imgs)
            z = z / z.norm(dim=-1, keepdim=True)
            embs.append(z.cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def fit_probe_and_report(X, y, train_mask, val_mask, ckpt_name, probe_name, out_dir):
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    margins = clf.decision_function(X_val)
    val_auc = roc_auc_score(y_val, margins)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(margins[y_val], bins=40, alpha=0.6, label=f"touching (n={y_val.sum()})", color="tab:blue")
    ax.hist(margins[~y_val], bins=40, alpha=0.6, label=f"not touching (n={(~y_val).sum()})", color="tab:orange")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("signed margin (w . x + b)")
    ax.set_ylabel("count")
    ax.set_title(f"{ckpt_name} -- {probe_name} (image)\nval_acc={val_acc:.3f}, val_auc={val_auc:.3f}")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{ckpt_name}_{probe_name}_image_margin_histogram.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return {
        "checkpoint": ckpt_name,
        "probe": probe_name,
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "out_path": out_path,
    }


results = []

for ckpt_name in CHECKPOINTS:
    print(f"=== {ckpt_name} ===")
    emb_cache_path = os.path.join(OUT_DIR, f"linear_probe_global_table_image_{ckpt_name}_embeddings.npz")

    if os.path.exists(emb_cache_path):
        print(f"  Loading cached embeddings from {emb_cache_path}")
        X = np.load(emb_cache_path)["embeddings"]
    else:
        print("  Loading CLIP model ...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
        )
        if ckpt_name != "pretrained":
            ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["clip_state_dict"])
            print(f"  Loaded clip_state_dict from {ckpt_path} (epoch={ckpt.get('epoch')})")
        model = model.to(DEVICE).eval()

        print(f"  Encoding {len(frames)} frames ...")
        X = encode_all_images(model, preprocess, frames)
        np.savez(emb_cache_path, embeddings=X)

        del model
        torch.cuda.empty_cache()

    r1 = fit_probe_and_report(X, ab_label, train1_mask, val1_mask, ckpt_name, "probe_ab", OUT_DIR)
    r2 = fit_probe_and_report(X, peg_label, train2_mask, val2_mask, ckpt_name, "probe_peg", OUT_DIR)
    results.append(r1)
    results.append(r2)

    print(f"  [probe_ab ] train_acc={r1['train_acc']:.3f} val_acc={r1['val_acc']:.3f} val_auc={r1['val_auc']:.3f}")
    print(f"  [probe_peg] train_acc={r2['train_acc']:.3f} val_acc={r2['val_acc']:.3f} val_auc={r2['val_auc']:.3f}")
    print()

# ---------------------------------------------------------------------------
# 4. Final summary
# ---------------------------------------------------------------------------
print("=" * 88)
print(f"{'checkpoint':<14} {'probe':<10} {'n_train':>8} {'n_val':>6} {'train_acc':>10} {'val_acc':>8} {'val_auc':>8}")
print("-" * 88)
for r in results:
    print(
        f"{r['checkpoint']:<14} {r['probe']:<10} {r['n_train']:>8} {r['n_val']:>6} "
        f"{r['train_acc']:>10.3f} {r['val_acc']:>8.3f} {r['val_auc']:>8.3f}"
    )
print("=" * 88)

results_path = os.path.join(OUT_DIR, "linear_probe_global_table_image_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults table saved to {results_path}")
