"""
Pre-compute and cache z_sem latent vectors for held-out episodes.

z_sem = outputs.hidden_states[-1][:, -1, :] from PaliGemmaWMForConditionalGeneration,
        run with actions=None and suffix=answer so the model sees (image, Q, A) only.
        Shape per vector: (2048,) float32.

Output: one .npz per episode in OUT_DIR, keyed by QA category, shapes:
    task_relevant_qa              (T, 4,     2048)
    task_relevant_qa_wrong_answer (T, 4,     2048)
    task_irrelevant_qa            (T, N_irr, 2048)  — reflexive pairs removed
    task_completely_irrelevant_qa (T, N_cmp, 2048)  — reflexive pairs removed

Run:
    python cache_held_out_episodes/cache_zsem.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from swm.paligemma_wm import PaliGemmaWMForConditionalGeneration, PaliGemmaWMProcessor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR   = "/home/ashwink/held_out_episode_5/demos/"
OUT_DIR    = "/home/ashwink/held_out_episode_5/demos/"
CHECKPOINT = "jacob3333/paligemma-3b-mix-224-wm"
DEVICE     = "cuda"
BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_reflexive(question: str) -> bool:
    """True when subject and object block are identical (e.g. 'Is the X close to the X?')."""
    q = question.lower().rstrip('?')
    for sep in (' close to ', ' touching '):
        if sep in q:
            before, after = q.split(sep, 1)
            subj = before.removeprefix('is the ').strip()
            obj  = after.removeprefix('the ').strip()
            return subj == obj
    return False


def compute_zsem_batch(processor, model, images, questions, answers):
    """
    images:    List[PIL.Image]
    questions: List[str]
    answers:   List[str]  — "yes" or "no"
    Returns:   np.ndarray, shape (B, 2048), float32
    """
    inputs = processor(
        text=questions,
        images=images,
        actions=None,
        suffix=answers,
        return_tensors="pt",
        padding="longest",
        tokenize_newline_separately=False,
    ).to(DEVICE, dtype=torch.float16)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[-1]: (B, seq_len, 2048) — last transformer layer
    # [:, -1, :]: last token position (end of answer suffix)
    return outputs.hidden_states[-1][:, -1, :].cpu().float().numpy()


def process_episode(pkl_path, out_path, processor, model):
    """
    Load one pkl, compute z_sem for every (timestep, QA pair), save .npz.
    Returns the total number of vectors written.
    """
    with open(pkl_path, 'rb') as f:
        ep = pickle.load(f)

    frames = ep['frames']  # list of T arrays (H, W, 3) uint8
    T = len(frames)

    # -----------------------------------------------------------------------
    # Determine per-category counts from timestep 0 (constant within episode)
    # -----------------------------------------------------------------------
    n_irr = sum(1 for qa in ep['task_irrelevant_qa'][0]
                if not is_reflexive(qa['question']))
    n_cmp = sum(1 for qa in ep['task_completely_irrelevant_qa'][0]
                if not is_reflexive(qa['question']))

    arr_rel   = np.zeros((T, 4,     2048), dtype=np.float32)
    arr_wrong = np.zeros((T, 4,     2048), dtype=np.float32)
    arr_irr   = np.zeros((T, n_irr, 2048), dtype=np.float32)
    arr_cmp   = np.zeros((T, n_cmp, 2048), dtype=np.float32)

    # -----------------------------------------------------------------------
    # Build flat item list: (category, t, qi, question, answer_str)
    # -----------------------------------------------------------------------
    items = []
    for t in range(T):
        for qi, qa in enumerate(ep['task_relevant_qa'][t]):
            items.append(('rel', t, qi, qa['question'], 'yes' if qa['answer'] else 'no'))

        for qi, qa in enumerate(ep['task_relevant_qa_wrong_answer'][t]):
            items.append(('wrong', t, qi, qa['question'], 'yes' if qa['answer'] else 'no'))

        qi_irr = 0
        for qa in ep['task_irrelevant_qa'][t]:
            if not is_reflexive(qa['question']):
                items.append(('irr', t, qi_irr, qa['question'], 'yes' if qa['answer'] else 'no'))
                qi_irr += 1

        qi_cmp = 0
        for qa in ep['task_completely_irrelevant_qa'][t]:
            if not is_reflexive(qa['question']):
                items.append(('cmp', t, qi_cmp, qa['question'], 'yes' if qa['answer'] else 'no'))
                qi_cmp += 1

    # -----------------------------------------------------------------------
    # Process in batches
    # -----------------------------------------------------------------------
    for start in range(0, len(items), BATCH_SIZE):
        chunk = items[start: start + BATCH_SIZE]
        pil_images = [Image.fromarray(frames[cat_t_qi[1]]) for cat_t_qi in chunk]
        questions  = [c[3] for c in chunk]
        answers    = [c[4] for c in chunk]

        zs = compute_zsem_batch(processor, model, pil_images, questions, answers)

        for idx, (cat, t, qi, _, _) in enumerate(chunk):
            if cat == 'rel':
                arr_rel[t, qi]   = zs[idx]
            elif cat == 'wrong':
                arr_wrong[t, qi] = zs[idx]
            elif cat == 'irr':
                arr_irr[t, qi]   = zs[idx]
            elif cat == 'cmp':
                arr_cmp[t, qi]   = zs[idx]

    np.savez(
        out_path,
        task_relevant_qa=arr_rel,
        task_relevant_qa_wrong_answer=arr_wrong,
        task_irrelevant_qa=arr_irr,
        task_completely_irrelevant_qa=arr_cmp,
    )
    return len(items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading processor and model from {CHECKPOINT} ...")
    processor = PaliGemmaWMProcessor.from_pretrained(CHECKPOINT)
    model = PaliGemmaWMForConditionalGeneration.from_pretrained(
        CHECKPOINT, dtype=torch.float16
    ).to(DEVICE)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print("Model loaded.\n")

    pkl_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.pkl')))
    total = len(pkl_files)
    print(f"Found {total} episodes in {DATA_DIR}")

    episodes_done  = 0
    total_vectors  = 0
    t0 = time.time()

    for i, pkl_path in enumerate(pkl_files):
        stem     = Path(pkl_path).stem
        out_path = os.path.join(OUT_DIR, stem + '_zsem.npz')

        if os.path.exists(out_path):
            print(f"[{i+1}/{total}] {stem}.pkl  SKIP (already cached)")
            continue

        n_vecs = process_episode(pkl_path, out_path, processor, model)

        episodes_done += 1
        total_vectors += n_vecs
        elapsed = time.time() - t0
        time_per_ep = elapsed / episodes_done
        eta = (total - i - 1) * time_per_ep
        print(
            f"[{i+1}/{total}] {stem}.pkl  |  "
            f"elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s  "
            f"(vectors so far: {total_vectors:,})"
        )

    print(f"\nDone. Processed {episodes_done} episodes. "
          f"Total z_sem vectors cached: {total_vectors:,}.")


if __name__ == "__main__":
    main()
