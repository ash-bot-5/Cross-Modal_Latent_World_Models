"""
Linear-probe evaluation of the fine-tuned CLIP text embedder (shortcut-learning check).

Reuses the exact GlobalStatementTable enumeration from
dynamics_model_clip_full_finetune.py (1792 compound statements: 56 ordered block
pairs x 2 A-B touching states x 8 peg blocks x 2 peg-touching states) -- the same
statement space the model already trains against. For each checkpoint's fine-tuned
CLIP text tower, encodes all 1792 statements, then fits two independent logistic
regression probes:

  1. A-B touching probe   (does the text embedding linearly separate the A-B clause's
                            touching state?)
  2. Peg touching probe   (same, for the peg clause's touching state)

Both probes use identity-disjoint train/val splits (held-out block PAIRS for probe 1,
held-out peg BLOCK identities for probe 2) so val accuracy can't be explained by the
probe having seen the exact same object identities at train time -- if accuracy stays
high under this split, that's real evidence of a linearly-decodable relation rather
than UMAP-visible-but-not-truly-separable object-identity shortcuts.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/linear_probe_global_table.py
"""

import os
import json
import pickle
import random
from itertools import permutations

import numpy as np
import torch
import open_clip
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

SPLIT_SEED_PROBE1 = 42   # held-out A-B pair identities
SPLIT_SEED_PROBE2 = 123  # held-out peg block identities
N_VAL_PAIRS = 11         # ~20% of 56
N_VAL_PEG_BLOCKS = 2     # ~25% of 8

BATCH_SIZE = 256

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Inline-copied from dynamics_model_clip_full_finetune.py (lines 92-159) --
# avoids importing that module, which pulls in langtable_dataset/dynamics_model/
# dynamics_model_clip_lora_finetune/open_clip/wandb at top level.
# ---------------------------------------------------------------------------


def get_block_names(data_dir: str) -> list[str]:
    """Derive the 8 fixed block names from real data, same convention as
    cache_zhard.py's `sorted(k for k in block_states[0].keys() if k != 'peg')` --
    block_states dicts also carry a 'peg' key, so it must be filtered out explicitly
    rather than assumed absent."""
    pkl_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl"))
    with open(os.path.join(data_dir, pkl_files[0]), "rb") as f:
        ep = pickle.load(f)
    names = sorted(k for k in ep["block_states"][0].keys() if k != "peg")
    assert len(names) == 8, f"expected 8 block names, got {len(names)}: {names}"
    return names


class GlobalStatementTable:
    """Canonical, deterministic enumeration of all 1792 possible compound statements.
    No episode data needed beyond the fixed 8 block names."""

    def __init__(self, block_names: list[str]):
        self.block_names = block_names
        self.n_blocks = len(block_names)
        self.pairs = list(permutations(block_names, 2))  # 56 ordered pairs
        self.n_pairs = len(self.pairs)
        assert self.n_pairs == 56 and self.n_blocks == 8

        self.pair_to_idx = {pair: i for i, pair in enumerate(self.pairs)}
        self.block_to_idx = {b: i for i, b in enumerate(block_names)}
        self.pair_block_idx = list(permutations(range(self.n_blocks), 2))

        self.size = self.n_pairs * 2 * self.n_blocks * 2  # 1792
        self.strings = self._build_strings()
        self.ab_state = (np.arange(self.size) // (self.n_blocks * 2)) % 2
        self.pair_idx_per_entry = np.arange(self.size) // (4 * self.n_blocks)

    def _build_strings(self) -> list[str]:
        strings: list[str | None] = [None] * self.size
        for pair_idx, (a, b) in enumerate(self.pairs):
            a_disp, b_disp = a.replace("_", " "), b.replace("_", " ")
            for ab_state in (0, 1):  # 0 = not touching, 1 = touching
                touching_ab = "touching" if ab_state else "not touching"
                for peg_idx, c in enumerate(self.block_names):
                    c_disp = c.replace("_", " ")
                    for peg_state in (0, 1):
                        touching_peg = "touching" if peg_state else "not touching"
                        idx = self.index(pair_idx, ab_state, peg_idx, peg_state)
                        strings[idx] = (
                            f"The {a_disp} is {touching_ab} the {b_disp}. "
                            f"The peg is {touching_peg} the {c_disp}."
                        )
        assert all(s is not None for s in strings)
        return strings  # type: ignore[return-value]

    def index(self, pair_idx: int, ab_state: int, peg_idx: int, peg_state: int) -> int:
        return ((pair_idx * 2 + ab_state) * self.n_blocks + peg_idx) * 2 + peg_state


# ---------------------------------------------------------------------------
# Build the table + derived label/identity arrays (checkpoint-independent)
# ---------------------------------------------------------------------------
print("Building GlobalStatementTable ...")
block_names = get_block_names(DATA_DIR)
table = GlobalStatementTable(block_names)
print(f"  {table.size} statements, {table.n_pairs} ordered pairs, {table.n_blocks} blocks\n")

ab_label = table.ab_state.astype(bool)                          # probe 1 target
peg_label = (np.arange(table.size) % 2).astype(bool)             # probe 2 target
pair_idx_per_entry = table.pair_idx_per_entry                     # probe 1 identity axis
peg_block_idx_per_entry = (np.arange(table.size) // 2) % table.n_blocks  # probe 2 identity axis

# ---------------------------------------------------------------------------
# Identity-disjoint splits
# ---------------------------------------------------------------------------
val_pair_ids = set(random.Random(SPLIT_SEED_PROBE1).sample(range(table.n_pairs), N_VAL_PAIRS))
train1_mask = ~np.isin(pair_idx_per_entry, list(val_pair_ids))
val1_mask = ~train1_mask

val_peg_block_ids = set(random.Random(SPLIT_SEED_PROBE2).sample(range(table.n_blocks), N_VAL_PEG_BLOCKS))
train2_mask = ~np.isin(peg_block_idx_per_entry, list(val_peg_block_ids))
val2_mask = ~train2_mask

print("[Probe 1: A-B touching] identity-disjoint split")
print(f"  train pairs: {table.n_pairs - N_VAL_PAIRS}, val pairs: {N_VAL_PAIRS}, overlap: 0 (by construction)")
print(f"  n_train rows: {train1_mask.sum()}, n_val rows: {val1_mask.sum()}\n")

print("[Probe 2: peg touching] identity-disjoint split")
print(f"  train peg blocks: {table.n_blocks - N_VAL_PEG_BLOCKS}, val peg blocks: {N_VAL_PEG_BLOCKS}, overlap: 0 (by construction)")
print(f"  n_train rows: {train2_mask.sum()}, n_val rows: {val2_mask.sum()}\n")


# ---------------------------------------------------------------------------
# Per-checkpoint embedding + probes
# ---------------------------------------------------------------------------
def encode_all_statements(model, tokenizer, statements: list[str]) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in range(0, len(statements), BATCH_SIZE):
            batch = statements[i : i + BATCH_SIZE]
            tokens = tokenizer(batch).to(DEVICE)
            z = model.encode_text(tokens)
            z = z / z.norm(dim=-1, keepdim=True)
            embs.append(z.cpu().float().numpy())
    return np.concatenate(embs, axis=0)


def fit_probe_and_report(X, y, train_mask, val_mask, ckpt_name, probe_name, out_dir):
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    clf = LogisticRegression(max_iter=1000)
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
    ax.set_title(f"{ckpt_name} -- {probe_name}\nval_acc={val_acc:.3f}, val_auc={val_auc:.3f}")
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{ckpt_name}_{probe_name}_margin_histogram.png")
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
tokenizer = open_clip.get_tokenizer("ViT-L-14")

for ckpt_name in CHECKPOINTS:
    print(f"=== {ckpt_name} ===")
    emb_cache_path = os.path.join(OUT_DIR, f"linear_probe_global_table_{ckpt_name}_embeddings.npz")

    if os.path.exists(emb_cache_path):
        print(f"  Loading cached embeddings from {emb_cache_path}")
        X = np.load(emb_cache_path)["embeddings"]
    else:
        print("  Loading CLIP model ...")
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
        )
        if ckpt_name != "pretrained":
            ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["clip_state_dict"])
            print(f"  Loaded clip_state_dict from {ckpt_path} (epoch={ckpt.get('epoch')})")
        model = model.to(DEVICE).eval()

        print(f"  Encoding {table.size} statements ...")
        X = encode_all_statements(model, tokenizer, table.strings)
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
# Final summary
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

results_path = os.path.join(OUT_DIR, "linear_probe_global_table_results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults table saved to {results_path}")
