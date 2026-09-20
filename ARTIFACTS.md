# ARTIFACTS.md

Map of the non-code artifacts in this repo (checkpoints, eval outputs, rollouts, wandb
logs) — what each is, what state it's in, and (importantly) which of it is **not** in git.

**Git policy (see `.gitignore`):** everything is tracked *except* (a) model-weight files
(`*.pt`, `*.pth`, `*.bin`, `*.ckpt`), excluded by type regardless of size, and (b) seven
individual non-weight files too big for GitHub (100MB/file hard limit), listed explicitly
below. Charts, JSON/CSV, small `.npz` caches, rollout videos, `.log` files, and wandb run
logs are all tracked (~2,600 files, ~1.7GB, none over 50MB). Only the excluded files need
to move separately from `git clone`/`git push` (rsync, S3, etc.).

## Exact paths of everything NOT in git

Repo root is **`/home/ashwink/Documents/swms/`**; every relative path used elsewhere in this
doc (and in `CLAUDE.md`/`README.md`) hangs off it. Verified with `git status --ignored` on
2026-09-20. `.venv/`, `uv.lock`, and `.claude/` are also ignored but are ordinary tooling
state, not research artifacts, so they're omitted.

### Model-weight files (`*.pt`) — ~280GB, the reason for the policy

Only the `.pt` files in these directories are excluded; small sibling files
(`train_val_split.json`, `epoch_metrics.jsonl`, `step_metrics.jsonl`) **are** tracked.

| Absolute path | `.pt` files | Size | What it is |
|---|---|---|---|
| `/home/ashwink/Documents/swms/main_project/checkpoints/clip_full_finetuned_subopt_h16/` | 14 | 72.5G | **current-best model** (epochs 0-12 + `latest.pt`) |
| `/home/ashwink/Documents/swms/main_project/checkpoints/` (loose files at this level) | 28 | 0.6G | original frozen-CLIP baseline (hidden dim 512) |
| `/home/ashwink/Documents/swms/main_project/checkpoints_upweighted/` | 28 | 0.6G | upweighted-loss A/B companion to the baseline |
| `/home/ashwink/Documents/swms/main_project/training/checkpoints_clip_full_finetuned_wider_hl/` | 17 | 88.0G | full-CLIP fine-tune, launch script unknown (see `CLAUDE.md`) |
| `/home/ashwink/Documents/swms/main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data/` | 13 | 67.3G | horizon-8 combined-data sibling of the current best |
| `/home/ashwink/Documents/swms/main_project/training/checkpoints_clip_image_text_finetuned/` | 10 | 51.5G | first full-CLIP fine-tune run (expert data only) |
| `/home/ashwink/Documents/swms/main_project/training/checkpoints_clip_image_finetuned/` | 15 | 0.9G | LoRA image-tower-only run, deprioritized |
| `/home/ashwink/Documents/swms/main_project/archive/OLD/checkpoints_OLD/` | 36 | 0.7G | dead, earliest drafts (epochs to 199) |
| `/home/ashwink/Documents/swms/main_project/archive/OLD/checkpoints_OLD_notebooks_partial_snapshot/` | 21 | 0.4G | dead, partial snapshot of the same run |

### Oversized non-weight files (exceed GitHub's 100MB/file limit)

| Absolute path | Size | Regenerable by |
|---|---|---|
| `/home/ashwink/Documents/swms/main_project/eval/linear_probe_image_data.npz` | 2.1G | `linear_probe_global_table_image.py` / `mlp_probe_global_table_image.py` |
| `/home/ashwink/Documents/swms/main_project/linear_probe_MLP_output_charts/ground_truth_data.npz` | 623M | `linear_probe_MLP_output.py` |
| `/home/ashwink/Documents/swms/main_project/linear_probe_MLP_output_charts/ground_truth_data_h16.npz` | 605M | `linear_probe_epoch11_h16_snapshot.py` |
| `/home/ashwink/Documents/swms/main_project/linear_probe_MLP_output_charts/ground_truth_data_h16_green_cube_start.npz` | 654M | `linear_probe_epoch11_h16_green_cube_confound.py` |
| `/home/ashwink/Documents/swms/main_project/linear_probe_MLP_output_charts/ground_truth_data_h16_green_cube_to_blue_moon.npz` | 256M | `linear_probe_epoch11_h16_green_cube_confound.py` |
| `/home/ashwink/Documents/swms/main_project/linear_probe_MLP_output_charts/ground_truth_data_h16_green_cube_to_blue_moon_ts100.npz` | 577M | `linear_probe_epoch11_h16_green_cube_confound.py` |
| `/home/ashwink/Documents/swms/main_project/wandb_archive/run-20260701_121452-0tjfay81/run-0tjfay81.wandb` | 113M | not regenerable (raw wandb run log; the run's `files/` metadata is tracked) |

(Script attributions verified against each script's cache-path constants. The linear-probe
`.npz` files are cached ground-truth data that those scripts recompute if absent.)

### Tracked, but lives at these absolute paths on this machine

Everything else under `/home/ashwink/Documents/swms/` is tracked — including the larger
output directories `main_project/rollouts_5_per/` (1.2GB), `main_project/training/*_diagnostics/`,
`main_project/MPPI_charts/`, `main_project/eval_results_*/`, `main_project/eval/` (loose
charts/CSVs/JSON/logs and subdirectories), `main_project/archive/` (non-weight files),
`main_project/notebooks/`, and all four `wandb_archive/` locations.

External data this repo's scripts read but which was never part of the repo at all
(already absolute, listed for completeness): `/home/ashwink/full_data_5000/langtable/demos/`
(5000-episode expert set), `/home/ashwink/full_data_5000/langtable_switch/demos/`
(1675-episode suboptimal set), `/home/ashwink/held_out_episode_5/demos/` (held-out set).

## Active / most recent work

**`main_project/checkpoints/clip_full_finetuned_subopt_h16/`** — the newest, most-trained
checkpoint in the repo (epochs 0–12, `latest.pt` = epoch 12, Aug 6). This run's checkpoints
were originally split across two directories (epochs 0–4 landed in
`main_project/training/`, epochs 5–12 landed at the repo's top level) because the training
script computes its output path from `__file__`'s parent, and it looks like training was
resumed from a different working directory partway through. Consolidated into one directory
2026-09-20; `epoch_metrics.jsonl`/`step_metrics.jsonl` were concatenated from both halves.
**If you resume training from a script that derives its output dir from `__file__`, always
launch it the same way (same cwd) each time, or you'll fragment the run again.**

**`main_project/linear_probe_MLP_output_charts/`** (3.2G) — active, unfinished diagnostic.
Open question: *can a linear probe predict the ground-truth `dist(A, B)` or `dist(A, peg)`,
or classify whether `A,B` / `A,peg` are touching (all at timestep t+8), given the model's
predicted latent `z_pred` at different training epochs?* Subdirs are keyed by checkpoint
epoch/horizon (e.g. `epoch11_h16_snapshot`, `epoch11_h16_block_confound`).

**`main_project/MPPI_charts/`** (20M) — active, unfinished diagnostic, tied to the same
epoch11/h16 checkpoint. Open question: *how good a planning signal does the dynamics model
provide — how smooth and accurate is the MPPI cost landscape it induces?* Subdirs:
`iteration1_direction_alignment`, `min_determination` (num-samples/temperature sweeps),
`obstacle_avoidance*`, `oracle_vs_mppi`.

**`main_project/rollouts/`, `main_project/rollouts_5_per/`** (414M) — MPPI rollout videos
and per-episode result JSON from many swept configurations (temperature, sample count,
execution horizon, obstacle-avoidance probes). Naming encodes the sweep params, e.g.
`expert+subopt_h16_temp0.1_evals1000_K2000_exec16/`.

## Documented, kept-as-is experiment

**`main_project/checkpoints_upweighted/`** + **`main_project/eval_results_upweighted/`** +
**`main_project/eval_results_non-upweighted/`** — a paired A/B experiment from Jul 9
(same day as the baseline `main_project/checkpoints/` run), comparing a loss-upweighting
variant against the non-upweighted baseline. Kept in place per project owner — may still
inform hyperparameter decisions on the current h16 run.

## Deprioritized

**`main_project/eval_results_clip_lora_epoch8/`** — eval snapshot of the PaliGemma+LoRA
dynamics model at epoch 8 (Jul 14). The LoRA approach (`main_project/training/paligemma_finetune.py`,
also described in `CLAUDE.md`) has been **deprioritized — Chuning's direction is full
fine-tuning instead of LoRA** for the PaliGemma-based dynamics model. Treat anything LoRA-related
as historical context, not the active path, unless that direction changes again.

## `main_project/archive/` — dead/superseded, kept for reference only

- **`OLD/`** — earliest drafts of the dynamics model and dataset loader (author's own `OLD`
  naming), with `checkpoints_OLD/` (epochs up to 199) and a partial earlier snapshot
  `checkpoints_OLD_notebooks_partial_snapshot/` (epochs up to 099, found loose in
  `main_project/notebooks/` and folded in here — not a full merge, just consolidated
  alongside the more complete copy since both are already dead).
- **`checkpoints_clip_full_finetuned_combined/`** — never held actual weights, just a stray
  `train_val_split.json`. Looks like an aborted run's leftover directory.
- **`clip_lora_dynamics_model_deprioritized/`** — two non-identical copies of
  `dynamics_model_clip_lora_finetune.py` (one from `main_project/notebooks/`, one from
  `main_project/training/`) that were found living in two places with diverging content —
  the `training/` copy has mmap-based frame loading, crash-recovery, and `--resume_from`
  that the `notebooks/` copy lacks, despite the `notebooks/` copy having a *later* mtime.
  Neither was deleted since it's unclear which lineage is authoritative and the whole LoRA
  direction is deprioritized anyway (see above) — if LoRA work resumes, diff these first.

## wandb run archives (four separate locations — not duplicates of each other)

`wandb_archive/` (top-level, 1 run, Aug 4), `main_project/wandb_archive/` (4 runs, Jun 30–Jul),
`main_project/training/wandb_archive/` (6 runs, Jul 14+), `main_project/notebooks/wandb_latest/`
(4 runs, Jun 30) each hold **different, non-overlapping wandb runs**. They exist in four
places for the same reason the h16 checkpoint run got split: whichever script logged to
`wandb/` was run from a different working directory each time, and the live `wandb/` dir
got manually archived to a same-named sibling `wandb_archive/` after each batch of runs. Not
worth physically merging (risk of corrupting wandb's internal run metadata for low benefit) —
just be aware there are four when searching for a specific run by date.

## Elsewhere in the repo root

- **`notebooks/`** (top-level, 1.8M) — unrelated to `main_project/notebooks/` despite the
  identical name; holds one exploratory notebook (`data_visualization.ipynb`) and its HTML
  export. Confusing pairing purely because of the shared name — no action taken beyond
  noting it here.
- **`reference_repos_MPPI/`** — vendored reference code (`PLDM`, `pytorch_mppi`,
  `CEM_reference_code`), not project code. Already covered in `CLAUDE.md`.

## Known pre-existing bug (found during this cleanup, not fixed)

`main_project/eval/mppi_rollout.py` references a checkpoint path
`checkpoints_non-upweighted/best.pt` that does not exist anywhere in the repo — likely meant
to be the baseline `main_project/checkpoints/` run. Whoever picks this back up should check
whether that code path is actually exercised before relying on it.
