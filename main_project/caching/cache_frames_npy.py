"""
Pre-cache raw episode frames as memory-mappable .npy files.

ep["frames"] is a list of T separate (180,320,3) uint8 ndarrays inside each
episode's .pkl. LangTableDataset-style eager loading (`self.frames.append(ep["frames"])`)
keeps every episode's full frame list resident in RAM for the dataset's entire
lifetime — fine for small subsets, but ~50GB across the full 5000-episode set,
which OOMs. This script stacks each episode's frames into a single (T,180,320,3)
uint8 array and saves it as a plain (uncompressed) .npy file, so downstream code
can do `np.load(path, mmap_mode="r")[t]` to read a single frame's bytes on demand
without ever materializing the whole episode in memory. Plain .npy (not .npz) is
required specifically because .npz archives don't support mmap_mode on read.

Idempotent — skips episodes whose {stem}_frames.npy already exists.

Usage:
    python cache_frames_npy.py             # cache all episodes
    python cache_frames_npy.py --smoke     # process 3 episodes, verify correctness
"""

import sys
import os
import pickle

import numpy as np

DATA_DIR = "/home/ashwink/full_data_5000/langtable/demos/"


def get_pkl_files() -> list[str]:
    return sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".pkl"))


def cache_one(pkl_file: str) -> str | None:
    pkl_path = os.path.join(DATA_DIR, pkl_file)
    frames_path = pkl_path.replace(".pkl", "_frames.npy")
    if os.path.exists(frames_path):
        return None

    with open(pkl_path, "rb") as f:
        ep = pickle.load(f)
    frames = np.stack(ep["frames"], axis=0)  # (T,180,320,3) uint8
    np.save(frames_path, frames)
    return frames_path


if __name__ == "__main__":
    pkl_files = get_pkl_files()

    if "--smoke" in sys.argv:
        done = 0
        for pkl_file in pkl_files:
            pkl_path = os.path.join(DATA_DIR, pkl_file)
            frames_path = pkl_path.replace(".pkl", "_frames.npy")

            with open(pkl_path, "rb") as f:
                ep = pickle.load(f)
            expected = ep["frames"]

            out_path = cache_one(pkl_file)
            if out_path is None:
                out_path = frames_path  # already cached from a prior run

            mmap = np.load(out_path, mmap_mode="r")
            assert mmap.shape == (len(expected), 180, 320, 3), \
                f"{pkl_file}: shape mismatch {mmap.shape} vs expected ({len(expected)},180,320,3)"
            assert mmap.dtype == np.uint8
            for t in (0, len(expected) // 2, len(expected) - 1):
                assert np.array_equal(np.asarray(mmap[t]), expected[t]), \
                    f"{pkl_file}: frame {t} mismatch between .npy cache and original .pkl"
            print(f"  {pkl_file}: shape={mmap.shape}, spot-checked frames match original pkl")

            done += 1
            if done >= 3:
                break
        print("Smoke test passed.")
        sys.exit(0)

    processed = 0
    for pkl_file in pkl_files:
        if cache_one(pkl_file) is not None:
            processed += 1
            if processed % 200 == 0:
                print(f"[{processed}] cached {pkl_file}")

    print(f"Done. Processed {processed} episodes (skipped {len(pkl_files) - processed} already cached).")
