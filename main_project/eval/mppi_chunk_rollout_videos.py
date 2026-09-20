"""
Replay the cached epoch-11 MPPI chunks in the real LangTable sim and record one .mp4 each,
to check whether the model's predicted cos_sim(z_pred, z_goal) actually tracks what the chunk
does physically.

mppi_global_or_local_min.py scores chunks in latent space only -- it never steps the real env.
This script closes that loop: it takes the 5 top MPPI-pass winners (run_chunks_raw) plus the
near-perfect reference chunk (ref_chunk_raw) cached in
    MPPI_charts/min_determination/expert+subopt-model/epoch11_mppi_5runs_data.npz
and replays each one, action-for-action, from the SAME fixed probe state those chunks were
planned against (peg -> green_cube -> blue_moon, pre-aligned collinear, peg starting already
TOUCHING green_cube). Six videos result -- watch whether the high-cos_sim chunks visibly push
green_cube into blue_moon and the reference chunk (highest cos_sim) does so most cleanly.

The probe state must match mppi_global_or_local_min.py exactly, including its two-stage peg
placement -- see the PEG_PARK_OFFSET comment below. The asserts against the npz's cached
green_xy/blue_xy/peg_xy enforce that; if that script's probe geometry changes, this one fails
loudly rather than silently replaying chunks from a state they were never planned against.

Output: every invocation writes into a FRESH timestamped subdirectory of
    MPPI_charts/min_determination/expert+subopt-model/rollouts/
named epoch{N}_pegtouch{offset}_{YYYYmmdd_HHMMSS}/, so nothing is ever overwritten and each
batch of videos is self-labeling (which epoch, which probe geometry, when it was produced).
Older videos -- including the pre-touching-peg batch sitting directly in rollouts/ -- are left
exactly where they are.

No model / CLIP / MPPI here: the chunks and their cos_sims are already cached, and the cached
run_chunks_raw / ref_chunk_raw are in RAW action units -- exactly what env.step() consumes
(same denormalization mppi_rollout_clip_full_finetune.py applies before env.step). So this only
loads the npz, rebuilds the env at the probe state, and replays the actions.

Each chunk is just 8 sim steps, so after the 8 actions we add a few zero-action SETTLE_STEPS
(so the physics settles and the real outcome is visible) and freeze the final frame, purely for
watchability -- the zero-action tail is clearly post-chunk, not part of the planned action.

Subprocess-per-chunk (like mppi_rollout_clip_full_finetune.py): pybullet supports only one
physics client per process, so each of the 6 rollouts builds its own env in its own process.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_chunk_rollout_videos.py
"""

import os
import re
import sys
import argparse
import datetime
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import imageio

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"
HORIZON = 8

MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "min_determination", "expert+subopt-model")
EPOCH = 11
DATA_NPZ = os.path.join(DATA_DIR, f"epoch{EPOCH}_mppi_5runs_data.npz")
ROLLOUT_ROOT = os.path.join(DATA_DIR, "rollouts")

FPS = 6              # playback speed
SETTLE_STEPS = 10    # zero-action steps after the chunk so the physics settles (post-chunk)
FREEZE_FRAMES = 6    # duplicate the final frame this many times so the end state is readable

# --- Probe-state geometry: must stay identical to mppi_global_or_local_min.py ---------------
# Inline-copied rather than imported on purpose: that module is a top-level script (no __main__
# guard), so importing it would load all 5000 training pkls, build an env and run every MPPI
# pass as an import side effect. Same reason the eval scripts inline-copy the model classes.
TOUCH_THRESH = 0.05         # peg/block touching threshold -- matches TOUCHING_DISTANCE_THRESHOLD
                            # in language_table/environments/lang_table_for_demo.py
PEG_START_OFFSET = 0.045    # final peg standoff from BLOCK_A along away_from_blue_direction;
                            # < TOUCH_THRESH, so the peg starts the replay already TOUCHING BLOCK_A

# Two-stage approach, load-bearing -- do NOT collapse into a single teleport.
# _set_robot_target_effector_pose only sets a TARGET; the arm drives toward it during
# _step_simulation_to_stabilize(). Commanding the final touching-range pose straight from the
# arm's rest pose makes that path sweep through BLOCK_A and launch it (measured: single teleport
# to 0.045 displaces BLOCK_A ~0.06 and inflates dist(A,B) from 0.199 to ~0.24). Worse here than
# in the planning script: the green/blue asserts below run BEFORE the teleport, so a corrupted
# start state would not be caught and you'd get six plausible-looking but meaningless videos.
# Parking first at PEG_PARK_OFFSET -- same axis, clear of the block -- disturbs nothing.
PEG_PARK_OFFSET = 0.10


def _load_chunks():
    """Return a list of (label, chunk_raw (8,2) float32, cos_sim float) -- the 5 epoch-11 MPPI
    winners followed by the near-perfect reference chunk."""
    data = np.load(DATA_NPZ)
    run_chunks = data["run_chunks_raw"].astype(np.float32)   # (5, 8, 2)
    run_cos = data["run_cos_sim"].astype(float)              # (5,)
    ref_chunk = data["ref_chunk_raw"].astype(np.float32)     # (8, 2)
    ref_cos = float(data["ref_cos_sim"])
    entries = []
    for i in range(run_chunks.shape[0]):
        entries.append((f"run{i + 1}", run_chunks[i], float(run_cos[i])))
    entries.append(("ref", ref_chunk, ref_cos))
    return entries, data


def run_one(chunk_idx: int, out_dir: str) -> None:
    """Build one env at the probe state, replay chunk `chunk_idx`, save one .mp4 into `out_dir`.
    Process-local (env/pybullet) -- safe to call exactly once per process. `out_dir` is passed in
    from the orchestrator rather than derived here so all chunks of one invocation land in the
    same timestamped directory."""
    entries, data = _load_chunks()
    label, chunk_raw, cos_sim = entries[chunk_idx]

    # Probe-state references stored when the chunk was generated -- used to assert the fresh
    # env reproduces the identical starting state the chunk was planned against.
    npz_green = data["green_xy"]
    npz_blue = data["blue_xy"]
    npz_peg = data["peg_xy"]

    print(f"=== chunk {chunk_idx}: {label}  cos_sim={cos_sim:.4f} ===")

    # --- Env + peg-teleport probe state (verbatim from mppi_rollout_clip_full_finetune.py) ---
    from swm.utils.envs import get_lang_table_env
    from language_table.environments.utils.pose3d import Pose3d
    from scipy.spatial.transform import Rotation as R

    env = get_lang_table_env(
        {"ood": False, "block_combo": [BLOCK_A, BLOCK_B]},
        seed=42,
    )
    inner = env.env
    blocks = inner.get_block_states()
    green_xy = blocks[BLOCK_A]
    blue_xy = blocks[BLOCK_B]

    away_from_blue_direction = (green_xy - blue_xy) / np.linalg.norm(green_xy - blue_xy)
    peg_xy = green_xy + away_from_blue_direction * PEG_START_OFFSET

    # Consistency: this rollout must start from the exact state the chunk was planned against.
    assert np.allclose(green_xy, npz_green, atol=1e-5), (
        f"{BLOCK_A} {green_xy} != cached {npz_green} -- table arrangement diverged from the npz"
    )
    assert np.allclose(blue_xy, npz_blue, atol=1e-5), (
        f"{BLOCK_B} {blue_xy} != cached {npz_blue} -- table arrangement diverged from the npz"
    )
    assert np.allclose(peg_xy, npz_peg, atol=1e-5), (
        f"peg {peg_xy} != cached {npz_peg} -- probe geometry diverged from the npz. If the npz "
        f"was regenerated with a different PEG_START_OFFSET, update this script to match."
    )

    def _move_peg_to(xy):
        inner._set_robot_target_effector_pose(Pose3d(
            rotation=R.from_rotvec([0, np.pi, 0]),
            translation=np.array([xy[0], xy[1], 0.145]),
        ))
        inner._step_simulation_to_stabilize()

    _move_peg_to(green_xy + away_from_blue_direction * PEG_PARK_OFFSET)  # stage 1: park clear
    _move_peg_to(peg_xy)                                                # stage 2: into contact range

    # Verify the REALIZED start state, not the commanded one: IK settles a fraction of a mm short
    # of target, and a regression in the staged approach would show up here as a displaced BLOCK_A.
    blocks_after = inner.get_block_states()
    green_after = blocks_after[BLOCK_A]
    peg_after = np.asarray(inner._robot.forward_kinematics().translation[:2], dtype=np.float64)
    green_moved = np.linalg.norm(green_after - green_xy)
    dist_peg_green = np.linalg.norm(peg_after - green_after)
    assert green_moved < 1e-3, (
        f"{BLOCK_A} moved {green_moved:.6f} while placing the peg -- the staged approach is "
        f"supposed to leave it untouched, so this replay would not start from the planned state"
    )
    assert dist_peg_green < TOUCH_THRESH, (
        f"realized peg start is {dist_peg_green:.4f} from {BLOCK_A}, not < {TOUCH_THRESH} -- "
        f"peg is not touching {BLOCK_A} at the start of the replay"
    )
    print(f"  probe state verified: {BLOCK_A}={green_xy}  {BLOCK_B}={blue_xy}  peg={peg_xy}\n"
          f"  realized: peg={peg_after}  dist(peg,{BLOCK_A})={dist_peg_green:.4f} (touching)  "
          f"{BLOCK_A} moved {green_moved:.6f}")

    # --- Replay the chunk, action-for-action, capturing a frame after each step ---
    frames = [np.array(env.get_frame())]  # initial frame before any action
    for i in range(HORIZON):
        env.step(chunk_raw[i])             # rows are raw actions -- feed straight to env.step
        frames.append(np.array(env.get_frame()))

    # Zero-action settle tail (post-chunk) so the physics comes to rest and the real outcome
    # is visible, then freeze the final frame for readability.
    for _ in range(SETTLE_STEPS):
        env.step(np.zeros(2, dtype=np.float32))
        frames.append(np.array(env.get_frame()))
    frames.extend([frames[-1]] * FREEZE_FRAMES)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"epoch{EPOCH}_{label}_cos{cos_sim:.4f}.mp4")
    # Never clobber: out_dir is timestamped per invocation, so an existing file here means two
    # chunks resolved to the same name (a bug), not a legitimate re-run.
    assert not os.path.exists(out_path), f"refusing to overwrite existing video {out_path}"
    imageio.mimsave(out_path, frames, fps=FPS)
    print(f"  saved {len(frames)} frames -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunk_idx", type=int, default=None,
        help="internal: replay a single chunk (0-4 = MPPI winners, 5 = reference) in this process",
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="internal: timestamped output dir chosen by the orchestrator and shared by all chunks",
    )
    args = parser.parse_args()

    if args.chunk_idx is not None:
        assert args.out_dir, "--chunk_idx requires --out_dir (set by the orchestrator)"
        run_one(args.chunk_idx, args.out_dir)
        return

    # Fresh timestamped output dir per invocation: nothing is ever overwritten, and the name
    # records which epoch's chunks these are and what probe geometry produced them.
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        ROLLOUT_ROOT,
        f"epoch{EPOCH}_pegtouch{int(round(PEG_START_OFFSET * 1000)):03d}_{stamp}",
    )
    os.makedirs(out_dir, exist_ok=False)  # timestamped -- collision means something is wrong

    # Orchestrator: one subprocess per chunk (pybullet allows only one physics client/process).
    entries, _ = _load_chunks()
    for idx in range(len(entries)):
        subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--chunk_idx", str(idx), "--out_dir", out_dir],
            check=True,
        )
    print(f"\nDone. {len(entries)} videos saved under {out_dir}")
    print(f"Pre-existing videos elsewhere under {ROLLOUT_ROOT} were left untouched.")


if __name__ == "__main__":
    main()
