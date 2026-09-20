"""
Final-timestep touching statistics over the full 5000-episode dataset.

For each episode, at the final timestep:
  A touching B    = task_relevant_qa[-1]'s first "touching" QA's answer
                     (the episode's own start_block -> oracle_target_block pair)
  peg touching A  = peg_touching_starting_block_qa[-1]'s answer

Reuses the exact same QA-extraction logic as full_5000_cache_zsem.py (which
defines "touching"/"peg_touching" for this dataset), just applied to the
final timestep only, reading directly from the .pkl (no CLIP/GPU needed).

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/dataset_final_touching_stats.py
"""

import os
import pickle
import re

DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "5000dataset_statistics.txt"
)

STEM_RE = re.compile(r"blocktoblock_(\d+)_(success|failure)")


def episode_number(stem: str) -> int:
    m = STEM_RE.match(stem)
    return int(m.group(1))


def classify_episode(ep: dict) -> tuple[bool, bool] | None:
    """Returns (a_touching_b, peg_touching_a) at the final timestep, or None if
    the episode can't be classified (mirrors full_5000_cache_zsem.py's own
    skip conditions)."""
    task_qa = ep.get("task_relevant_qa")
    peg_qa_list = ep.get("peg_touching_starting_block_qa")
    if not task_qa or not peg_qa_list:
        return None

    t = len(task_qa) - 1
    if t < 0 or t >= len(peg_qa_list):
        return None

    touching_qas = [qa for qa in task_qa[t] if "touching" in qa["question"].lower()]
    if len(touching_qas) < 2:
        return None

    peg_raw = peg_qa_list[t]
    if isinstance(peg_raw, list):
        if not peg_raw:
            return None
        peg_raw = peg_raw[0]

    a_touching_b = bool(touching_qas[0]["answer"])
    peg_touching_a = bool(peg_raw["answer"])
    return a_touching_b, peg_touching_a


def main():
    pkl_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".pkl"))
    print(f"Found {len(pkl_files)} pkl files in {DATA_DIR}")

    # group key -> list of episode numbers
    group1 = []  # A touching B AND peg touching A
    group2 = []  # A NOT touching B, peg touching A
    group3 = []  # A NOT touching B, peg NOT touching A
    group4 = []  # A touching B, peg NOT touching A
    skipped = []  # (stem, reason)

    for pkl_file in pkl_files:
        stem = pkl_file[:-4]
        pkl_path = os.path.join(DATA_DIR, pkl_file)
        try:
            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)
        except Exception as e:
            skipped.append((stem, f"failed to load pkl: {e}"))
            continue

        result = classify_episode(ep)
        if result is None:
            skipped.append((stem, "missing/malformed touching QA data"))
            continue

        a_touching_b, peg_touching_a = result
        ep_num = episode_number(stem)

        if a_touching_b and peg_touching_a:
            group1.append(ep_num)
        elif (not a_touching_b) and peg_touching_a:
            group2.append(ep_num)
        elif (not a_touching_b) and (not peg_touching_a):
            group3.append(ep_num)
        else:  # a_touching_b and not peg_touching_a
            group4.append(ep_num)

    group1.sort()
    group2.sort()
    group3.sort()
    group4.sort()

    n_total = len(pkl_files)
    n_classified = len(group1) + len(group2) + len(group3) + len(group4)
    n_skipped = len(skipped)

    def pct(n: int) -> float:
        return 100.0 * n / n_classified if n_classified else 0.0

    lines = []
    lines.append(f"Final-timestep touching statistics — {DATA_DIR}")
    lines.append(f"Total episodes found: {n_total}")
    lines.append(f"Successfully classified: {n_classified}  |  Skipped/failed: {n_skipped} (see below)")
    lines.append("")
    lines.append(f"Group 1 — A touching B AND peg touching A: {len(group1)} ({pct(len(group1)):.2f}%)")
    lines.append(f"Group 2 — A NOT touching B, peg touching A: {len(group2)} ({pct(len(group2)):.2f}%)")
    lines.append(f"Group 3 — A NOT touching B, peg NOT touching A: {len(group3)} ({pct(len(group3)):.2f}%)")
    lines.append(f"Group 4 — A touching B, peg NOT touching A: {len(group4)} ({pct(len(group4)):.2f}%)")
    lines.append("")
    lines.append("--- Group 1 episode numbers (A touching B AND peg touching A) ---")
    lines.append(", ".join(str(n) for n in group1))
    lines.append("")
    lines.append("--- Group 2 episode numbers (A NOT touching B, peg touching A) ---")
    lines.append(", ".join(str(n) for n in group2))
    lines.append("")
    lines.append("--- Group 3 episode numbers (A NOT touching B, peg NOT touching A) ---")
    lines.append(", ".join(str(n) for n in group3))
    lines.append("")
    lines.append("--- Group 4 episode numbers (A touching B, peg NOT touching A) ---")
    lines.append(", ".join(str(n) for n in group4))
    lines.append("")
    lines.append("--- Skipped/failed episodes (couldn't classify) ---")
    if skipped:
        for stem, reason in skipped:
            lines.append(f"{stem}: {reason}")
    else:
        lines.append("(none)")

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Group 1 (A touching B AND peg touching A): {len(group1)}/{n_classified} ({pct(len(group1)):.2f}%)")
    print(f"Wrote statistics to {OUT_PATH}")


if __name__ == "__main__":
    main()
