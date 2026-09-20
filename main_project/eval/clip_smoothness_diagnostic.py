"""
CLIP embedding smoothness diagnostic — no MLP in the loop.

Ground-truth check on the raw cached CLIP image embeddings (z_obs / z_cur) for each episode in
DATA_DIR: do they move smoothly frame-to-frame, or are they already noisy before the latent
dynamics MLP ever touches them? For each episode, computes and plots:

  1. cosine_sim(z_cur(t), z_cur(t-1))  — local step-to-step smoothness
  2. cosine_sim(z_cur(t), z_cur(0))    — drift away from the start frame over the rollout
  3. ||z_cur(t) - z_cur(t-1)||         — L2 step size

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/clip_smoothness_diagnostic.py
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR      = "/home/ashwink/3_expert_trajs/"
OUT_DIR       = os.path.join(DATA_DIR, "clip_smoothness_diagnostic")
MONOTONIC_TOL = 1e-3  # tolerance for flagging an increase in sim_to_start as a violation

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Discover episodes (non-recursive glob — skips _mlp_eval.npz in subdirectories)
# ---------------------------------------------------------------------------
zobs_paths = sorted(glob.glob(os.path.join(DATA_DIR, "*_zobs.npz")))
stems = [os.path.basename(p)[: -len("_zobs.npz")] for p in zobs_paths]

print(f"Found {len(stems)} episode(s) in {DATA_DIR}: {stems}")
print("Processing all of them (small enough set that no sampling is needed).\n")

# ---------------------------------------------------------------------------
# Per-episode diagnostic
# ---------------------------------------------------------------------------
for stem, zobs_path in zip(stems, zobs_paths):
    z_obs = np.load(zobs_path)["z_obs"]  # (T, 768) float32, unit-norm rows
    T = z_obs.shape[0]

    consecutive_sim = np.sum(z_obs[1:] * z_obs[:-1], axis=-1)      # (T-1,), t = 1..T-1
    sim_to_start    = np.sum(z_obs * z_obs[0:1], axis=-1)          # (T,),   t = 0..T-1
    step_l2         = np.linalg.norm(z_obs[1:] - z_obs[:-1], axis=-1)  # (T-1,), t = 1..T-1

    t_consec = np.arange(1, T)
    t_start  = np.arange(0, T)

    # Monotonic violations: sim_to_start increasing (drifting back toward start) by > tol
    diffs = sim_to_start[2:] - sim_to_start[1:-1]  # aligned to t = 2..T-1
    violation_mask = diffs > MONOTONIC_TOL
    n_violations = int(violation_mask.sum())
    n_candidates = len(diffs)

    # ---- summary ----
    print(f"{stem}  (T={T})")
    print(f"  consecutive_sim: mean={consecutive_sim.mean():.4f}  "
          f"std={consecutive_sim.std():.4f}  min={consecutive_sim.min():.4f}")
    print(f"  sim_to_start:    mean={sim_to_start.mean():.4f}  final={sim_to_start[-1]:.4f}")
    print(f"  monotonic violations (sim_to_start increases by >{MONOTONIC_TOL}): "
          f"{n_violations}/{n_candidates}")

    # ---- plot ----
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    axes[0].plot(t_consec, consecutive_sim, marker="o", markersize=3)
    axes[0].set_ylabel("cos(z(t), z(t-1))")
    axes[0].set_title("Consecutive-step cosine similarity")

    axes[1].plot(t_start, sim_to_start, marker="o", markersize=3, color="tab:orange")
    axes[1].set_ylabel("cos(z(t), z(0))")
    axes[1].set_title("Drift from start frame")

    axes[2].plot(t_consec, step_l2, marker="o", markersize=3, color="tab:green")
    axes[2].set_ylabel("||z(t) - z(t-1)||")
    axes[2].set_xlabel("Timestep")
    axes[2].set_title("Consecutive-step L2 distance")

    fig.suptitle(f"CLIP embedding smoothness — {stem}")
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, f"{stem}_clip_smoothness.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"  plot: {plot_path}\n")
