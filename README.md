# Cross-Modal Latent World Models

This repo's **active project** lives in [`main_project/`](main_project/): a lightweight
latent dynamics model for LangTable block-pushing that predicts a future CLIP embedding
from a current-frame CLIP embedding and a chunk of proposed actions, trained contrastively
against CLIP text embeddings of scene descriptions. An MPPI planner uses the model to pick
actions that drive the predicted embedding toward a goal-text embedding.

Everything outside `main_project/` (`swm/`, `scripts/evaluate_swm_hydra.py`, `configs/`) is
an **earlier project** this one builds on — a fine-tuned PaliGemma VLM used as a yes/no
reward model to rerank diffusion-policy action samples. It's kept because `main_project/`
reuses pieces of it (the LangTable env wrapper, the PaliGemma model code, the episode `.pkl`
schema).

**Start here:**
- [`CLAUDE.md`](CLAUDE.md) — full technical writeup of both projects: data pipeline,
  model architecture, training, planning, evaluation scripts, file layout.
- [`ARTIFACTS.md`](ARTIFACTS.md) — map of the large on-disk directories (checkpoints, wandb
  logs, rollout videos, diagnostic charts) that aren't tracked in git, including which
  experiments are active/unfinished vs. dead/superseded.

## Setup

```bash
uv sync
source .venv/bin/activate
bash ckpts/download_checkpoints.sh   # downloads the four HuggingFace checkpoints for the legacy swm/ reward model
```

Requires Python 3.10. The `language-table` and `ogbench` dependencies are pinned to forked
repos (see `pyproject.toml` `[tool.uv.sources]`).

## Running the active project (`main_project/`)

See `CLAUDE.md` for the full data-caching → training → planning/eval flow. The rest of
this section is about picking up exactly where the project currently stands.

### The current-best model

**`main_project/checkpoints/clip_full_finetuned_subopt_h16/`** — epochs 0–12, `latest.pt` =
epoch 12 (Aug 6). This is the most-trained checkpoint in the repo and the right one to build
on. It was trained by:

**`main_project/training/dynamics_model_clip_full_finetune_combined_horizon16.py`**
— full fine-tuning of **both** the CLIP ViT-L/14 image and text towers, jointly with a
fresh FiLM-MLP dynamics model (defined in `dynamics_model.py`'s `LatentDynamicsModel`,
hidden dim 1024, Conv1d action encoder), action-chunk horizon **16**, trained on the
**combined** expert (5000-ep `langtable/demos/`) + suboptimal (1675-ep
`langtable_switch/demos/`) data.

Its caching dependency is **not** the `_zobs.npz`/`_zsem.npz`/`_zhard.npz` pipeline
described elsewhere in `CLAUDE.md` — because it fine-tunes CLIP's text tower itself
(re-encoding statements live from a fixed 1792-entry statement table every step), it only
needs pre-cached raw-frame `.npy` files:

```bash
python main_project/caching/cache_frames_npy.py           # expert set
python main_project/caching/cache_frames_npy_switch.py    # suboptimal set
```

To resume training, note the checkpoint directory was consolidated during a 2026-09-20
cleanup (see `ARTIFACTS.md`) and no longer matches the script's own default — pass both
flags explicitly:

```bash
python main_project/training/dynamics_model_clip_full_finetune_combined_horizon16.py \
    --train \
    --resume_from main_project/checkpoints/clip_full_finetuned_subopt_h16/latest.pt \
    --checkpoint_dir main_project/checkpoints/clip_full_finetuned_subopt_h16
```

It also uses a 2-GPU-rebalanced training step (`main_project/training/gpu_split_joint_step.py`,
imported and on by default — pass `--legacy_even_split` to disable) that's worth reusing
as-is for any future full-fine-tune run.

**Not the current direction:** the PaliGemma+LoRA dynamics model
(`main_project/training/paligemma_finetune.py`, `clip_lora.py`, and the archived
`dynamics_model_clip_lora_finetune*.py` copies in `main_project/archive/`) — Chuning
directed full fine-tuning instead. Don't start there unless that's changed again.

### All models trained, oldest to newest

| Checkpoint dir | Trained by | Architecture | Data | Date |
|---|---|---|---|---|
| `main_project/checkpoints/` | `dynamics_model.py` | Frozen CLIP embeddings, hidden dim 512, flatten-MLP action encoder | expert only, horizon 8 | Jul 9 |
| `main_project/checkpoints_upweighted/` | `dynamics_model_with_upweighting.py` | Same as above + upweighted loss | expert only, horizon 8 | Jul 9 |
| `main_project/training/checkpoints_clip_image_finetuned/` | *(archived)* `dynamics_model_clip_lora_finetune.py` | LoRA on CLIP image tower only, text frozen | expert only, horizon 8 | Jul 14 |
| `main_project/training/checkpoints_clip_image_text_finetuned/` | `dynamics_model_clip_full_finetune.py` | Full fine-tune of both CLIP towers, hidden dim 1024 | expert only, horizon 8 | Jul 16 |
| `main_project/training/checkpoints_clip_full_finetuned_wider_hl/` | *unknown — exact launch script/data not recoverable from the checkpoint* | Full CLIP fine-tune, hidden dim 1024, Conv1d action encoder (confirmed by inspecting the checkpoint's tensor shapes directly — matches current `dynamics_model.py` exactly) | expert only (presumed), horizon 8 | Jul 18 |
| `main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data/` | `dynamics_model_clip_full_finetune_combined.py` | Full CLIP fine-tune, hidden dim 1024 | **combined** expert + suboptimal, horizon 8 | Jul 23–31 |
| **`main_project/checkpoints/clip_full_finetuned_subopt_h16/`** | `dynamics_model_clip_full_finetune_combined_horizon16.py` | Same as above | combined expert + suboptimal, **horizon 16** | **Aug 4–6 (current best)** |

`main_project/checkpoints_upweighted/`, the two A/B `eval_results_*` dirs, and everything
dead/superseded (`OLD/`, the LoRA scripts, the empty `checkpoints_clip_full_finetuned_combined/`
stub) are catalogued in full in `ARTIFACTS.md` — not repeated here.

### Evaluating the current (h16) model

Most-recently-run first (cross-referenced against their output artifact timestamps, since
file mtimes were reset by the 2026-09-20 rename pass):

1. `main_project/eval/mppi_iteration1_h16_direction_alignment.py` — Aug 6, the last thing run
2. `main_project/eval/linear_probe_epoch11_h16_green_cube_confound.py` — Aug 5
3. `main_project/eval/linear_probe_epoch11_h16_snapshot.py` — Aug 5
4. `main_project/eval/mppi_obstacle_avoidance_probe.py`, `mppi_obstacle_expert_chunk_probe.py` (+ `_h8.py`), `mppi_temp_samples_sweep.py`, `mppi_rollout_random_scenes.py`, `mppi_rollout_5_per_checkpoint.py` — earlier passes in the same h16 sweep

### Where those evaluations' outputs already live

- `main_project/linear_probe_MLP_output_charts/` and `main_project/MPPI_charts/` — the
  active/unfinished diagnostic threads; see `ARTIFACTS.md` for the open question each one
  was chasing
- `main_project/rollouts_5_per/` — per-config MPPI rollout videos + result JSON, one subdir
  per swept hyperparameter combination

### Caching/training scripts worth repurposing for the next model

- `main_project/caching/cache_frames_npy.py` + `cache_frames_npy_switch.py` — the caching
  step the current model lineage actually depends on (not `full_5000_cache_zsem.py`/
  `cache_zhard.py`, which only the older frozen-CLIP `dynamics_model.py` path needs)
- `main_project/training/dynamics_model_clip_full_finetune_combined_horizon16.py` — copy/
  extend this (or its horizon-8 sibling `dynamics_model_clip_full_finetune_combined.py`) as
  the starting point for a new full-fine-tune run
- `main_project/training/gpu_split_joint_step.py` — reusable as-is for 2-GPU training speedup
- `main_project/training/dynamics_model.py` — defines the shared `LatentDynamicsModel`/
  `FiLMLayer`/`ActionEncoderConv` classes every full-fine-tune script imports; edit here to
  change the dynamics-model architecture itself

## Running the legacy `swm/` reward-model evaluation

```bash
python scripts/evaluate_swm_hydra.py --config-name eval_gradient_full
```

Configs are in `configs/`:

- `eval_base_diffusion_lt.yaml` — LangTable, expert diffusion only (no SWM planning)
- `eval_base_ogbench.yaml` — OGBench, expert diffusion only
- `eval_gradient_full.yaml` — LangTable, gradient-based planning with SWM reward
- `eval_gradient_full_ogbench.yaml` — OGBench, gradient-based planning with SWM reward

Each run writes `video.mp4`, `heat_map_*.png`, `goal.txt`, `reward.pkl`, and appends to
`results.txt` under `<root_save_path>/<name>/<planning_name>/<block_a>_<block_b>/<seed>/`.
