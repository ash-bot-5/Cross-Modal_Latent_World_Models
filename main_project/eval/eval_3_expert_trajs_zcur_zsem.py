"""
Model-free diagnostic: cos(z_cur, z_sem_goal) over time for 3 expert trajectories
(episodes 0, 1, 3), plus an overlay against the MLP's cached predicted embedding.

For each timestep t (full range 0..T-1):
  cos_sim      = cos(z_obs[t], z_sem[-1])   # fixed goal: z_sem at the episode's final timestep

No dynamics model, checkpoint, or action data is used here — only the cached z_obs/z_sem
CLIP embeddings for each episode.

A second, overlay plot per episode additionally loads the MLP's cached predicted embedding
(z_pred) from cached_MLP_output_data/{stem}_mlp_eval.npz (written by eval_3_expert_trajs.py)
and plots cos(z_pred, z_sem_goal) alongside cos(z_cur, z_sem_goal). z_pred at conditioning
timestep t is a forecast of the embedding after an 8-step action chunk (i.e. z_obs[t+HORIZON]),
so its cosine-sim curve is shifted by +HORIZON on the x-axis to align with the timestep it is
actually predicting.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/eval_3_expert_trajs_zcur_zsem.py
"""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR    = "/home/ashwink/3_expert_trajs/"
MLP_CACHE_DIR = os.path.join(DATA_DIR, "cached_MLP_output_data")
CACHE_DIR   = os.path.join(DATA_DIR, "cached_data_zcur_zsem")
STEMS       = ["blocktoblock_0_success", "blocktoblock_1_success", "blocktoblock_3_success"]
EMBED_DIM   = 768

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------
for stem in STEMS:
    with open(os.path.join(DATA_DIR, stem + ".pkl"), "rb") as f:
        ep = pickle.load(f)
    instruction = ep["metadata"]["instruction_str"]

    z_obs = np.load(os.path.join(DATA_DIR, stem + "_zobs.npz"))["z_obs"]  # (T, 768)
    z_sem = np.load(os.path.join(DATA_DIR, stem + "_zsem.npz"))["z_sem"]  # (T, 768)

    assert z_obs.shape[-1] == EMBED_DIM, (
        f"{stem}: expected z_obs dim {EMBED_DIM}, got {z_obs.shape[-1]} (shape={z_obs.shape})"
    )
    assert z_sem.shape[-1] == EMBED_DIM, (
        f"{stem}: expected z_sem dim {EMBED_DIM}, got {z_sem.shape[-1]} (shape={z_sem.shape})"
    )

    T = z_obs.shape[0]
    goal = z_sem[-1]  # fixed goal embedding

    timesteps = np.arange(T, dtype=np.int64)
    cosine_sim = z_obs @ goal  # (T,) — rows are unit-norm, so dot product == cosine similarity

    # ---- cache ----
    cache_path = os.path.join(CACHE_DIR, f"{stem}_zcur_zsem.npz")
    np.savez(cache_path, timesteps=timesteps, cosine_sim=cosine_sim)

    # ---- base plot ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(timesteps, cosine_sim, marker="o", markersize=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cosine similarity to goal z_sem")
    ax.set_title(f"{stem}\n\"{instruction}\"")
    plt.tight_layout()
    plot_path = os.path.join(CACHE_DIR, f"{stem}_cosine_sim.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"{stem}  (T={T}, instruction=\"{instruction}\")")
    print(f"  cos_sim: first={cosine_sim[0]:.4f}  last={cosine_sim[-1]:.4f}  "
          f"min={cosine_sim.min():.4f}  max={cosine_sim.max():.4f}  mean={cosine_sim.mean():.4f}")
    print(f"  cached: {cache_path}")
    print(f"  plot:   {plot_path}")

    # -----------------------------------------------------------------------
    # Overlay plot: z_pred (cached MLP output) vs z_cur, both against the same goal
    # -----------------------------------------------------------------------
    mlp_cache_path = os.path.join(MLP_CACHE_DIR, f"{stem}_mlp_eval.npz")
    if not os.path.exists(mlp_cache_path):
        raise FileNotFoundError(
            f"Missing cached MLP predictions for {stem}: {mlp_cache_path}\n"
            f"Run eval_3_expert_trajs.py first to generate this cache."
        )
    mlp_cache = np.load(mlp_cache_path)
    timesteps_cond = mlp_cache["timesteps"]  # (N,) conditioning timestep t
    mlp_output = mlp_cache["mlp_output"]     # (N, 768)

    assert mlp_output.shape[-1] == EMBED_DIM, (
        f"{stem}: expected mlp_output dim {EMBED_DIM}, got {mlp_output.shape[-1]} "
        f"(shape={mlp_output.shape})"
    )

    # Defensive normalization (mlp_output is already unit-norm from the model's F.normalize,
    # but don't assume that holds for every future cache).
    z_pred_normed = mlp_output / np.linalg.norm(mlp_output, axis=-1, keepdims=True)
    cosine_sim_pred = z_pred_normed @ goal  # (N,)

    # z_pred at conditioning timestep t forecasts the embedding after an 8-step action chunk,
    # i.e. it approximates z_obs[t + HORIZON] — shift the x-axis accordingly.
    N = len(timesteps_cond)
    HORIZON = T - N
    target_timesteps = timesteps_cond + HORIZON
    assert target_timesteps.max() == T - 1, (
        f"{stem}: expected shifted target_timesteps to end at T-1={T - 1}, "
        f"got {target_timesteps.max()} (HORIZON={HORIZON}, N={N}, T={T})"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(timesteps, cosine_sim, marker="o", markersize=3, label="z_cur → goal")
    ax.plot(target_timesteps, cosine_sim_pred, marker="o", markersize=3,
            label="z_pred → goal (t+HORIZON)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Cosine similarity to goal z_sem")
    ax.set_title(f"{stem}\n\"{instruction}\"")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    overlay_plot_path = os.path.join(CACHE_DIR, f"{stem}_cosine_sim_overlay.png")
    plt.savefig(overlay_plot_path, dpi=150)
    plt.close()

    print(f"  cos_sim_pred: mean={cosine_sim_pred.mean():.4f}  "
          f"(vs z_cur mean={cosine_sim.mean():.4f})  HORIZON={HORIZON}")
    print(f"  overlay plot: {overlay_plot_path}\n")
