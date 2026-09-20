"""
Save raw rendered frames (as labeled PNGs) reconstructed from a rollout's saved pybullet states.

Companion utility to oracle_vs_mppi_charts.py: for every t in [--t_start, --t_end] (default
0..30, i.e. the first 31 timesteps), restores the exact pybullet state at t from
{stem}_pybullet_states.pkl (inner.set_pybullet_state) and saves env.get_frame() as a PNG -- a
lossless re-render, unlike extracting frames from the rollout's saved mp4 (H.264-compressed).

Each PNG is labeled with its timestep both in the filename (frame_t{NNN:03d}.png -- distinct from
oracle_vs_mppi_charts.py's own t{NNN:03d}.png chart files in the same directory, so nothing
collides) and burned into the image itself as small corner text ("t=<N>").

block_combo for env construction is read from the co-located {stem}_oracle_chunks.pkl's
meta.start_block/target_block (must match whatever combo actually produced pybullet_states.pkl,
since inner.set_pybullet_state() restores objects by obj_id -- see oracle_vs_mppi_charts.py's
docstring for why matching construction matters); pass --block_a/--block_b to override if that
file isn't present alongside this rollout.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/dump_pybullet_state_frames.py \\
        --rollout_dir main_project/rollouts_5_per/expert+subopt_iters1000_exec8_with_pkls/epoch11 \\
        --rollout_idx 0
"""

import os
import re
import sys
import pickle
import argparse

from PIL import ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="epoch directory holding {stem}_pybullet_states.pkl, e.g. "
                             ".../rollouts_5_per/expert+subopt_iters1000_exec8_with_pkls/epoch11")
    parser.add_argument("--rollout_idx", type=int, default=0)
    parser.add_argument("--t_start", type=int, default=0)
    parser.add_argument("--t_end", type=int, default=30,
                        help="default 30 -> t_start..t_end inclusive = first 31 timesteps")
    parser.add_argument("--block_a", type=str, default=None,
                        help="override start_block instead of reading it from the co-located "
                             "oracle_chunks.pkl")
    parser.add_argument("--block_b", type=str, default=None,
                        help="override target_block instead of reading it from the co-located "
                             "oracle_chunks.pkl")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="default: MPPI_charts/oracle_vs_mppi/<parent>_<epoch>_rollout<idx>/ "
                             "-- the same folder oracle_vs_mppi_charts.py writes its charts to")
    args = parser.parse_args()

    rollout_dir = os.path.abspath(args.rollout_dir.rstrip("/"))
    rollout_idx = args.rollout_idx
    epoch_dirname = os.path.basename(rollout_dir)
    m = re.search(r"epoch(\d+)$", epoch_dirname)
    if not m:
        raise ValueError(
            f"Could not parse an epoch number from --rollout_dir's basename {epoch_dirname!r} "
            f"-- expected something like '.../epoch11'."
        )
    rollout_epoch = int(m.group(1))
    stem = f"epoch{rollout_epoch}_rollout{rollout_idx}"

    states_path = os.path.join(rollout_dir, f"{stem}_pybullet_states.pkl")
    if not os.path.exists(states_path):
        raise FileNotFoundError(states_path)
    print(f"Loading pybullet states from {states_path} ...")
    with open(states_path, "rb") as f:
        pybullet_states = pickle.load(f)
    n_states = len(pybullet_states)
    print(f"  {n_states} states available")

    block_a, block_b = args.block_a, args.block_b
    if block_a is None or block_b is None:
        oracle_path = os.path.join(rollout_dir, f"{stem}_oracle_chunks.pkl")
        if not os.path.exists(oracle_path):
            raise FileNotFoundError(
                f"{oracle_path} not found and --block_a/--block_b not given -- need one or the "
                f"other to know which two blocks this rollout used."
            )
        with open(oracle_path, "rb") as f:
            meta = pickle.load(f)["meta"]
        block_a = block_a or meta["start_block"]
        block_b = block_b or meta["target_block"]
    print(f"block_combo = [{block_a}, {block_b}]")

    if args.t_start < 0 or args.t_end < args.t_start:
        raise ValueError(f"invalid range --t_start={args.t_start} --t_end={args.t_end}")
    if args.t_end >= n_states:
        raise ValueError(f"--t_end={args.t_end} >= n_states={n_states}")

    parent_dirname = os.path.basename(os.path.dirname(rollout_dir))
    out_dir = args.out_dir or os.path.join(
        MAIN_PROJECT_DIR, "MPPI_charts", "oracle_vs_mppi",
        f"{parent_dirname}_{epoch_dirname}_rollout{rollout_idx}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output -> {out_dir}")

    print("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env({"ood": False, "block_combo": [block_a, block_b]}, seed=42)
    inner = env.env
    print("Env ready.")

    n_saved = 0
    for t in range(args.t_start, args.t_end + 1):
        inner.set_pybullet_state(pybullet_states[t])
        frame = env.get_frame().convert("RGB")

        draw = ImageDraw.Draw(frame)
        label = f"t={t}"
        x, y = 4, 4
        # Black outline + white fill so the label stays legible over any background color.
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((x + dx, y + dy), label, fill="black")
        draw.text((x, y), label, fill="white")

        out_path = os.path.join(out_dir, f"frame_t{t:03d}.png")
        frame.save(out_path)
        n_saved += 1

    print(f"Saved {n_saved} frames -> {out_dir}")


if __name__ == "__main__":
    main()
