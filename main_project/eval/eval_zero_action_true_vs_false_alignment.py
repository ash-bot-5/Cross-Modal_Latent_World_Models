"""
Generalization test: does the checkpoint's zero-action prediction z_pred = model(z_t, zeros(1,8,2))
sit closer (higher cosine similarity) to the CLIP embedding of a TRUE touching statement than
to the same statement with BOTH clauses fully inverted (guaranteed FALSE), across every one of
the 448 ordered (A, B, C) triples -- 8*7 ordered block pairs (A,B) x 8 peg-target blocks C, no
restriction that C equals A or B -- on 5 random validation episodes?

Ground-truth touching (block-block and peg-block alike) reuses this project's existing
convention from main_project/caching/cache_zhard.py: dist(pos_a, pos_b) < TOUCH_THRESH (0.05).
Statement template is identical to cache_zhard.py's direct-construction convention:
  "The {a} is {touching_ab} the {b}. The peg is {touching_c} the {c}."
cache_zhard.py always builds the fully-inverted (guaranteed false) version of this template;
here we build BOTH the accurate (TRUE) version and the fully-inverted (FALSE) version for
every triple, at every timestep.

Key efficiency point: each clause has only 2 possible wordings ("touching" / "not touching"),
so the full vocabulary of distinct sentence strings is fixed and small -- built as an 8x8x8x2x2
= 2048-slot tensor (a==b diagonal slots included for simplicity but never queried; the 448 valid
a!=b triples x 4 polarity combos = 1792 of those 2048 are the ones actually used) -- independent
of timestep or episode. All of it is CLIP-text-encoded ONCE up front and reused via boolean-
indexed lookup for every timestep of every episode, rather than re-encoding text in the hot loop.

Since this checkpoint jointly fine-tuned BOTH CLIP towers, both the statement vocabulary and the
per-frame image embeddings are computed with THIS checkpoint's own loaded CLIP weights (not the
frozen encoder used for cached _zobs.npz), matching the rationale already used in
mppi_rollout_clip_full_finetune.py and zero_action_success_signal_probe.py.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_zero_action_true_vs_false_alignment.py
"""

import os
import json
import random
import pickle
import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image

CHECKPOINT_EPOCH = 15

MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_PATH = os.path.join(
    MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_finetuned_wider_hl", f"epoch_{CHECKPOINT_EPOCH:03d}.pt"
)
SPLIT_JSON = os.path.join(
    MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_finetuned_wider_hl", "train_val_split.json"
)
DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOUCH_THRESH = 0.05          # shared block-block / peg-block threshold, matches cache_zhard.py
HORIZON = 8
N_EPISODES = 5
SAMPLE_SEED = 123            # reproducible episode sample, matches this project's SAMPLE_SEED convention
CLIP_TEXT_BATCH = 256
CLIP_IMAGE_BATCH = 32


# ---------------------------------------------------------------------------
# Inline model classes (matches mppi_rollout_clip_full_finetune.py's wider_hl
# architecture exactly, to load checkpoints_clip_full_finetuned_wider_hl/epoch_008.pt)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * hidden + beta


class ActionEncoderConv(nn.Module):
    def __init__(self, action_dim: int = 2, horizon: int = 8,
                 conv_channels: int = 64, kernel_size: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(action_dim, conv_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.act = nn.Mish()
        self.out = nn.Linear(conv_channels * horizon, out_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv(actions.transpose(1, 2)))
        return self.out(h.flatten(1))


class LatentDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = ActionEncoderConv()
        self.fc1 = nn.Linear(768, 1024);  self.ln1 = nn.LayerNorm(1024)
        self.film1 = FiLMLayer(256, 1024); self.act1 = nn.Mish()
        self.fc2 = nn.Linear(1024, 1024);  self.ln2 = nn.LayerNorm(1024)
        self.film2 = FiLMLayer(256, 1024); self.act2 = nn.Mish()
        self.fc_out = nn.Linear(1024, 768); self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        global_cond = self.action_encoder(actions)
        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)


def build_statement_vocab(block_names, clip_model, clip_tokenizer, log):
    """Encode all 8*8*8*2*2=1792 possible statement strings once. Returns a (8,8,8,2,2,768)
    torch tensor on DEVICE indexed [a_idx, b_idx, c_idx, polarity_ab, polarity_c] (bool as 0/1).
    a_idx==b_idx slots are built (harmless placeholder text) but never queried by the caller."""
    n = len(block_names)
    strings = []
    index_of = {}
    for a_idx in range(n):
        a_disp = block_names[a_idx].replace("_", " ")
        for b_idx in range(n):
            b_disp = block_names[b_idx].replace("_", " ")
            for c_idx in range(n):
                c_disp = block_names[c_idx].replace("_", " ")
                for pol_ab in (True, False):
                    touching_ab = "touching" if pol_ab else "not touching"
                    for pol_c in (True, False):
                        touching_c = "touching" if pol_c else "not touching"
                        s = (f"The {a_disp} is {touching_ab} the {b_disp}. "
                             f"The peg is {touching_c} the {c_disp}.")
                        index_of[(a_idx, b_idx, c_idx, pol_ab, pol_c)] = len(strings)
                        strings.append(s)

    log.info(f"Encoding {len(strings)} unique statement strings with CLIP text tower ...")
    parts = []
    with torch.no_grad():
        for i in range(0, len(strings), CLIP_TEXT_BATCH):
            tokens = clip_tokenizer(strings[i:i + CLIP_TEXT_BATCH]).to(DEVICE)
            z = F.normalize(clip_model.encode_text(tokens), dim=-1)
            parts.append(z)
    all_embeds = torch.cat(parts, dim=0)  # (1792, 768)

    vocab = torch.zeros(n, n, n, 2, 2, 768, device=DEVICE)
    for (a_idx, b_idx, c_idx, pol_ab, pol_c), flat_idx in index_of.items():
        vocab[a_idx, b_idx, c_idx, int(pol_ab), int(pol_c)] = all_embeds[flat_idx]
    return vocab


def compute_stats(arr: np.ndarray) -> dict:
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(
        NOTEBOOKS_DIR, f"eval_zero_action_true_vs_false_alignment_epoch{CHECKPOINT_EPOCH}_{timestamp}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    log = logging.getLogger("eval_true_vs_false")
    log.info(f"Logging to {log_path}")
    log.info(f"Device: {DEVICE}\n")

    # -----------------------------------------------------------------------
    # 1. Checkpoint + CLIP (both towers, jointly fine-tuned) + dynamics model
    # -----------------------------------------------------------------------
    log.info(f"Checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    log.info(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}\n")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model = LatentDynamicsModel().to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    # -----------------------------------------------------------------------
    # 2. Pick 5 random validation episodes
    # -----------------------------------------------------------------------
    with open(SPLIT_JSON) as f:
        split = json.load(f)
    val_files = split["val_files"]
    sampled = random.Random(SAMPLE_SEED).sample(val_files, N_EPISODES)
    log.info(f"Sampled {N_EPISODES} val episodes (seed={SAMPLE_SEED}) out of {len(val_files)}:")
    for fn in sampled:
        log.info(f"  {fn}")
    log.info("")

    # -----------------------------------------------------------------------
    # 3. Load first episode to get the canonical block name list, then build the
    # fixed 1792-string statement vocabulary once (shared across all episodes/timesteps).
    # -----------------------------------------------------------------------
    episodes = {}
    for fn in sampled:
        with open(os.path.join(DATA_DIR, fn), "rb") as f:
            episodes[fn] = pickle.load(f)

    block_names = sorted(k for k in episodes[sampled[0]]["block_states"][0].keys() if k != "peg")
    n_blocks = len(block_names)
    log.info(f"Block names ({n_blocks}): {block_names}\n")

    vocab = build_statement_vocab(block_names, clip_model, clip_tokenizer, log)
    log.info(f"Vocab tensor shape: {tuple(vocab.shape)}\n")

    # Fixed (A,B,C) index grids and the a!=b validity mask -- identical every episode/timestep.
    idx = torch.arange(n_blocks)
    A, B, C = torch.meshgrid(idx, idx, idx, indexing="ij")   # each (8,8,8)
    valid_mask = (A != B).reshape(-1)                        # (512,) bool -> keeps 448 True
    A_flat, B_flat, C_flat = A.reshape(-1), B.reshape(-1), C.reshape(-1)
    assert int(valid_mask.sum()) == n_blocks * (n_blocks - 1) * n_blocks == 448

    # -----------------------------------------------------------------------
    # 4. Per-episode evaluation
    # -----------------------------------------------------------------------
    per_episode_true = {}
    per_episode_false = {}

    for fn in sampled:
        ep = episodes[fn]
        frames = ep["frames"]
        block_states = ep["block_states"]
        peg_positions = ep["actual_peg_positions"]
        T = len(frames)

        log.info(f"=== Episode {fn}  (T={T}) ===")

        # ---- Batch-encode all T frames with THIS checkpoint's own CLIP image tower,
        # then batch the zero-action dynamics-model forward pass. ----
        z_pred_list = []
        with torch.no_grad():
            for start in range(0, T, CLIP_IMAGE_BATCH):
                batch_frames = frames[start:start + CLIP_IMAGE_BATCH]
                imgs = torch.stack([clip_preprocess(Image.fromarray(f)) for f in batch_frames]).to(DEVICE)
                z_t_batch = F.normalize(clip_model.encode_image(imgs), dim=-1)          # (B, 768)
                actions_zero = torch.zeros(len(batch_frames), HORIZON, 2, device=DEVICE)
                z_pred_batch = model(z_t_batch, actions_zero)                            # (B, 768)
                z_pred_list.append(z_pred_batch)
        z_pred_all = torch.cat(z_pred_list, dim=0)  # (T, 768)

        # ---- Ground truth + gather + cosine, per timestep ----
        pos = np.stack([np.stack([block_states[t][name] for name in block_names]) for t in range(T)])
        # pos: (T, 8, 2)
        dist_matrix = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)  # (T, 8, 8)
        touch_matrix = dist_matrix < TOUCH_THRESH                                        # (T, 8, 8)
        peg_pos = np.asarray(peg_positions)[:T]                                          # (T, 2)
        dist_peg = np.linalg.norm(pos - peg_pos[:, None, :], axis=-1)                    # (T, 8)
        peg_touch = dist_peg < TOUCH_THRESH                                              # (T, 8)

        ep_true_vals = []
        ep_false_vals = []

        with torch.no_grad():
            for t in range(T):
                truth_ab = torch.from_numpy(touch_matrix[t][A_flat.numpy(), B_flat.numpy()])  # (512,) bool
                truth_c = torch.from_numpy(peg_touch[t][C_flat.numpy()])                       # (512,) bool

                true_emb = vocab[A_flat, B_flat, C_flat, truth_ab.long(), truth_c.long()]       # (512,768)
                false_emb = vocab[A_flat, B_flat, C_flat, (~truth_ab).long(), (~truth_c).long()]  # (512,768)

                true_emb = true_emb[valid_mask].to(DEVICE)    # (448,768)
                false_emb = false_emb[valid_mask].to(DEVICE)  # (448,768)

                z_p = z_pred_all[t].unsqueeze(0)  # (1,768)
                cos_true = (true_emb * z_p).sum(dim=-1)    # (448,)
                cos_false = (false_emb * z_p).sum(dim=-1)  # (448,)

                ep_true_vals.append(cos_true.cpu().numpy())
                ep_false_vals.append(cos_false.cpu().numpy())

        ep_true_arr = np.concatenate(ep_true_vals)    # (T*448,)
        ep_false_arr = np.concatenate(ep_false_vals)  # (T*448,)
        per_episode_true[fn] = ep_true_arr
        per_episode_false[fn] = ep_false_arr

        true_stats = compute_stats(ep_true_arr)
        false_stats = compute_stats(ep_false_arr)
        win_rate = float((ep_true_arr > ep_false_arr).mean())
        mean_margin = float((ep_true_arr - ep_false_arr).mean())

        log.info(f"  TRUE  statements: count={true_stats['count']:>7}  mean={true_stats['mean']:+.4f}  "
                  f"std={true_stats['std']:.4f}  min={true_stats['min']:+.4f}  max={true_stats['max']:+.4f}")
        log.info(f"  FALSE statements: count={false_stats['count']:>7}  mean={false_stats['mean']:+.4f}  "
                  f"std={false_stats['std']:.4f}  min={false_stats['min']:+.4f}  max={false_stats['max']:+.4f}")
        log.info(f"  win_rate (cos_true > cos_false): {win_rate:.4f}")
        log.info(f"  mean margin (cos_true - cos_false): {mean_margin:+.4f}\n")

    # -----------------------------------------------------------------------
    # 5. Aggregate across all 5 episodes
    # -----------------------------------------------------------------------
    all_true = np.concatenate([per_episode_true[fn] for fn in sampled])
    all_false = np.concatenate([per_episode_false[fn] for fn in sampled])
    overall_true_stats = compute_stats(all_true)
    overall_false_stats = compute_stats(all_false)
    overall_win_rate = float((all_true > all_false).mean())
    overall_margin = float((all_true - all_false).mean())

    log.info("=== Summary across all 5 episodes ===")
    log.info(f"{'episode':<40} {'true_mean':>10} {'false_mean':>10} {'win_rate':>9} {'margin':>9}")
    for fn in sampled:
        t_arr, f_arr = per_episode_true[fn], per_episode_false[fn]
        wr = float((t_arr > f_arr).mean())
        mg = float((t_arr - f_arr).mean())
        log.info(f"{fn:<40} {t_arr.mean():>10.4f} {f_arr.mean():>10.4f} {wr:>9.4f} {mg:>9.4f}")

    log.info("")
    log.info(f"Pooled TRUE  statements: count={overall_true_stats['count']:>8}  mean={overall_true_stats['mean']:+.4f}  "
              f"std={overall_true_stats['std']:.4f}  min={overall_true_stats['min']:+.4f}  max={overall_true_stats['max']:+.4f}")
    log.info(f"Pooled FALSE statements: count={overall_false_stats['count']:>8}  mean={overall_false_stats['mean']:+.4f}  "
              f"std={overall_false_stats['std']:.4f}  min={overall_false_stats['min']:+.4f}  max={overall_false_stats['max']:+.4f}")
    log.info(f"Overall win_rate (cos_true > cos_false): {overall_win_rate:.4f}")
    log.info(f"Overall mean margin (cos_true - cos_false): {overall_margin:+.4f}")
    log.info(f"\nFull log written to {log_path}")


if __name__ == "__main__":
    main()
