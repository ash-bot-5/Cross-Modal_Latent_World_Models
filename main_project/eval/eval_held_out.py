"""
Per-timestep cosine distance eval on held-out episode 0.

For each timestep t (up to T - HORIZON):
  z_pred = model(z_obs[t], actions[t:t+HORIZON])
  d_pos     = 1 - cos(z_pred, z_sem[t+HORIZON])   # target future state
  d_current = 1 - cos(z_pred, z_sem[t])            # current state (naive baseline)
  d_neg     = 1 - cos(z_pred, random_hard_neg)     # a guaranteed-false hard negative
  correct   = d_pos < d_neg

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_held_out.py
"""

import os
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Model classes (copied inline to avoid importing dynamics_model.py, which
# triggers open_clip + LangTableDataset at module level)
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


class LatentDynamicsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(16, 256),
            nn.Mish(),
            nn.Linear(256, 256),
        )
        self.fc1 = nn.Linear(768, 512)
        self.ln1 = nn.LayerNorm(512)
        self.film1 = FiLMLayer(256, 512)
        self.act1 = nn.Mish()
        self.fc2 = nn.Linear(512, 512)
        self.ln2 = nn.LayerNorm(512)
        self.film2 = FiLMLayer(256, 512)
        self.act2 = nn.Mish()
        self.fc_out = nn.Linear(512, 768)
        self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        global_cond = self.action_encoder(actions.flatten(1))
        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)


# ---------------------------------------------------------------------------
# Sentence reconstruction — mirrors cache_held_out_episodes/cache_zsem_clip.py's
# qa_to_statement + compound_stmt logic exactly (the script that actually built
# this held-out episode's _zsem.npz), so this is the literal string CLIP-encoded
# into z_sem[t] — the npz only stores the embedding, not the source text. Held-out
# episode pkls lack `peg_touching_starting_block_qa` (unlike the main 5000-episode
# dataset), so the peg statement is derived from raw positions instead.
# ---------------------------------------------------------------------------

PEG_TOUCH_THRESH = 0.05


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


def sentence_at(t: int, ep: dict) -> str:
    touching_qas = [qa for qa in ep["task_relevant_qa"][t] if "touching" in qa["question"].lower()]
    stmt_ab = qa_to_statement(touching_qas[0]["question"], touching_qas[0]["answer"])

    peg_qa_list = ep.get("peg_touching_starting_block_qa", [])
    if peg_qa_list and t < len(peg_qa_list):
        peg_raw = peg_qa_list[t]
        if isinstance(peg_raw, list):
            peg_raw = peg_raw[0]
        peg_stmt = qa_to_statement(peg_raw["question"], peg_raw["answer"])
    else:
        start_block = ep["metadata"]["start_block"]
        peg_pos = ep["actual_peg_positions"][t]
        block_pos = ep["block_states"][t][start_block]
        peg_touching = np.linalg.norm(peg_pos - block_pos) < PEG_TOUCH_THRESH
        touching_word = "is touching" if peg_touching else "is not touching"
        peg_stmt = f"the peg {touching_word} the {start_block.replace('_', ' ')}"

    return f"{stmt_ab}. {peg_stmt}."


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR   = os.path.join(REPO_ROOT, "checkpoints")
TRAIN_DIR  = "/home/ashwink/full_data_5000/langtable/demos/"
HELD_DIR   = "/home/ashwink/held_out_episode_5/demos/"
STEM       = "blocktoblock_2  _success"
HORIZON    = 8
DEVICE     = "cpu"  # eval is fast enough on CPU; avoids competing with training GPU

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. Find latest checkpoint by mtime
# ---------------------------------------------------------------------------
ckpt_files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
if not ckpt_files:
    raise FileNotFoundError(f"No .pt files in {CKPT_DIR}")
ckpt_name = max(ckpt_files, key=lambda f: os.path.getmtime(os.path.join(CKPT_DIR, f)))
ckpt_path = os.path.join(CKPT_DIR, ckpt_name)
print(f"Checkpoint: {ckpt_path}")

# ---------------------------------------------------------------------------
# 2. Action stats from training pkls (actions only — no frames loaded)
# ---------------------------------------------------------------------------
print(f"Computing action stats from {TRAIN_DIR} ...")
all_actions = []
pkl_files = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl"))
for pkl_file in pkl_files:
    with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
        ep = pickle.load(f)
    all_actions.append(np.array(ep["actions"], dtype=np.float32))
all_actions = np.concatenate(all_actions, axis=0)
lo = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)
hi = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)
print(f"  {len(pkl_files)} episodes  |  action shape: {all_actions.shape}  |  lo={lo.tolist()}  hi={hi.tolist()}")

# ---------------------------------------------------------------------------
# 3. Load model
# ---------------------------------------------------------------------------
ckpt = torch.load(ckpt_path, map_location=DEVICE)
model = LatentDynamicsModel().to(DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Model loaded (epoch {ckpt['epoch']}, step {ckpt.get('step', '?')})\n")

# ---------------------------------------------------------------------------
# 4. Load held-out episode 0
# ---------------------------------------------------------------------------
with open(os.path.join(HELD_DIR, STEM + ".pkl"), "rb") as f:
    ep = pickle.load(f)
actions_raw = ep["actions"]  # list of T-1 arrays, each (2,)

z_obs = torch.tensor(
    np.load(os.path.join(HELD_DIR, STEM + "_zobs.npz"))["z_obs"], dtype=torch.float32
)
z_sem = torch.tensor(
    np.load(os.path.join(HELD_DIR, STEM + "_zsem.npz"))["z_sem"], dtype=torch.float32
)
zhard       = np.load(os.path.join(HELD_DIR, STEM + "_zhard.npz"))
unique_zsem = torch.tensor(zhard["unique_zsem"], dtype=torch.float32)
neg_indices = zhard["neg_indices"]  # (T, 224) int16

T = z_obs.shape[0]
print(f"Episode: {STEM}  |  T={T}  |  eval timesteps: {T - HORIZON}")
print(f"{'t':>4}  {'d_pos':>8}  {'d_cur':>8}  {'d_neg':>8}  {'correct':>8}  {'sem_chgd':>9}  sentence(t)")
print("-" * 58)

# ---------------------------------------------------------------------------
# 5. Per-timestep eval
# ---------------------------------------------------------------------------
correct = total = correct_sem_changed = total_sem_changed = 0

with torch.no_grad():
    for t in range(T - HORIZON):
        z_t = z_obs[t].unsqueeze(0)

        chunk = torch.tensor(
            np.array(actions_raw[t : t + HORIZON], dtype=np.float32)
        )  # (HORIZON, 2)
        actions_norm = ((chunk - lo) / (hi - lo) * 2.0 - 1.0).unsqueeze(0)  # (1, HORIZON, 2)

        z_pred = model(z_t, actions_norm).squeeze(0)  # (768,)

        z_future  = z_sem[t + HORIZON]
        z_current = z_sem[t]

        # One random hard negative for this timestep
        idx_t  = neg_indices[t].astype(np.int64)           # (224,) int64
        rnd    = np.random.randint(len(idx_t))
        z_neg  = unique_zsem[idx_t[rnd]]                   # (768,)

        d_pos     = 1.0 - F.cosine_similarity(z_pred.unsqueeze(0), z_future.unsqueeze(0)).item()
        d_current = 1.0 - F.cosine_similarity(z_pred.unsqueeze(0), z_current.unsqueeze(0)).item()
        d_neg     = 1.0 - F.cosine_similarity(z_pred.unsqueeze(0), z_neg.unsqueeze(0)).item()

        sem_changed = (1.0 - F.cosine_similarity(z_future.unsqueeze(0), z_current.unsqueeze(0)).item()) > 1e-4
        is_correct  = d_pos < d_neg

        total   += 1
        correct += int(is_correct)
        if sem_changed:
            total_sem_changed   += 1
            correct_sem_changed += int(is_correct)

        sentence_t = sentence_at(t, ep)
        print(f"{t:>4}  {d_pos:>8.4f}  {d_current:>8.4f}  {d_neg:>8.4f}  {str(is_correct):>8}  {str(sem_changed):>9}  {sentence_t}")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print("-" * 58)
print(f"\nSummary: {correct}/{total} correct (d_pos < d_neg)")
print(f"  sem_changed=True subset: {correct_sem_changed}/{total_sem_changed}")
