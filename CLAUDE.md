# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

This repo's active research lives in **`main_project/`**: a lightweight **latent dynamics model** for LangTable block-pushing that predicts a future **CLIP embedding** from a current-frame CLIP embedding and a chunk of proposed actions, trained contrastively to align with CLIP text embeddings of scene descriptions. An MPPI planner uses the model to pick actions that drive the predicted embedding toward a goal-text embedding. This directly replaces the reward-model approach below with something much cheaper to run at planning time (a small MLP or a single fine-tuned VLM forward pass, vs. re-scoring every candidate trajectory against a full VLM).

Everything **outside `main_project/`** (`swm/`, `scripts/evaluate_swm_hydra.py`, `configs/`) is the **earlier project** this one builds on and is secondary/reference at this point: a fine-tuned PaliGemma VLM used as a yes/no reward model to rerank diffusion-policy action samples. It's kept because `main_project/` reuses pieces of it (`swm.utils.envs.get_lang_table_env`, `swm.paligemma_wm`, the LangTable episode `.pkl` schema).

## Setup

```bash
uv sync
source .venv/bin/activate
bash ckpts/download_checkpoints.sh   # downloads the four HuggingFace checkpoints for the legacy swm/ reward model
```

Requires Python 3.10. The `language-table` and `ogbench` dependencies are pinned to forked repos (see `pyproject.toml` `[tool.uv.sources]`). `peft` (LoRA) is a dependency for `main_project/training/paligemma_finetune.py`.

---

## `main_project/` — CLIP-latent dynamics model + MPPI planning

### High-level flow

```
Caching (offline, once per episode):
  cache_zobs.py   → {stem}_zobs.npz   z_obs (T, 768)  CLIP image embedding per frame
  full_5000_cache_zsem.py /
  cache_zsem_clip.py → {stem}_zsem.npz  z_sem (T, 768)  CLIP text embedding of a compound
                                          "A touching B. peg touching C." statement per frame
  cache_zhard.py  → {stem}_zhard.npz  z_hard  224 guaranteed-false statement embeddings/timestep,
                                       deduplicated per episode, for InfoNCE hard negatives

Training — frozen-CLIP lineage (dynamics_model.py, dynamics_model_with_upweighting.py):
  LangTableDataset assembles (z_t, action_chunk, z_sem_target, z_hard_neg) tuples
  → LatentDynamicsModel predicts z_pred
  → InfoNCE loss pulls z_pred toward z_sem_target, away from the 224 hard negatives

Training — full-CLIP-fine-tune lineage (dynamics_model_clip_full_finetune*.py — the
current-best model's lineage, see below):
  A per-script Dataset assembles (raw frame, action_chunk, statement-table index) tuples
  from pre-cached {stem}_frames.npy (no cached z_sem/z_hard)
  → CLIP image+text towers (jointly fine-tuned) + LatentDynamicsModel predict z_pred and
    re-encode the target/negative statements live, every step
  → same InfoNCE loss, against the 1792-entry global statement table instead of the
    224 cached hard negatives

Training — PaliGemma+LoRA lineage (paligemma_finetune.py — deprioritized):
  → PaliGemmaLatentDynamics predicts z_pred directly from pixels
  → same InfoNCE loss against cached z_sem_target / z_hard_neg

Planning / eval (mppi_rollout.py):
  MPPI (planning/mppi_core.py) samples action chunks, rolls them through the trained
  dynamics model in latent space (no rendering), scores by cos-distance to a goal
  text embedding z_star, and executes the best chunk in the real LangTable env.
```

The diagram above is the shared conceptual skeleton (InfoNCE + MPPI). In practice there are
**two separate caching/training lineages** that don't interoperate — which one a given
training script needs depends entirely on whether it keeps CLIP frozen or fine-tunes it:

- **Frozen-CLIP lineage** (original): CLIP is never trained; `LangTableDataset` reads
  pre-cached `_zobs.npz`/`_zsem.npz`/`_zhard.npz`. Used by `dynamics_model.py` and
  `dynamics_model_with_upweighting.py`.
- **Full-CLIP-fine-tune lineage** (current direction, as of 2026-09-20 — see
  "Where Ashwin left off" in `README.md`): both CLIP towers train jointly with the MLP, so
  training needs raw pixels, not frozen embeddings. Each script in this lineage defines its
  own dataset class (not `LangTableDataset`) that reads pre-cached `{stem}_frames.npy` files
  (via `cache_frames_npy.py`/`cache_frames_npy_switch.py`) and re-encodes text statements
  live every step from a fixed combinatorial statement table — **no `_zsem.npz`/`_zhard.npz`
  dependency at all**. Used by `dynamics_model_clip_full_finetune.py` and its two
  descendants (see below). This is the lineage that produced the current best checkpoint.

There is also a third, **deprioritized** LoRA lineage (`clip_lora.py`,
`dynamics_model_clip_lora_finetune.py`, `paligemma_finetune.py`) — Chuning directed full
fine-tuning instead of LoRA, so treat anything LoRA-related as historical context unless
told that direction changed again.

### Dynamics model variants

**Lineage A — frozen CLIP embeddings**

**1. `main_project/training/dynamics_model.py`** — defines the shared architecture classes
(`FiLMLayer`, `ActionEncoderConv`, `LatentDynamicsModel`) that every full-CLIP-fine-tune
script in Lineage B also imports, so changes here propagate to all of them. Current
`LatentDynamicsModel`: an action chunk (shape `(horizon, 2)`, `horizon` defaults to 8 but is
an explicit constructor param — the horizon-16 full-finetune script passes 16) is encoded by
`ActionEncoderConv` (a `Conv1d`-based encoder that treats the 2 action dims as channels and
the horizon as the sequence axis, so neighboring timesteps can interact via the convolution
kernel before flattening — unlike a flatten-first `Linear`, which discards step order
entirely) to a 256-dim conditioning vector, which FiLM-modulates a 2-layer MLP with
**hidden dim 1024** (not 512 — an earlier baseline run used 512, see the checkpoint table
under "Checkpoints & run artifacts") mapping `z_t (768,) → z_pred (768,)`. Output is
L2-normalized. `FiLMLayer` does `(1+gamma)*hidden + beta` from a zero-initialized linear
projection of the conditioning vector (so it starts as identity).

`InfoNCELoss`: hard-negatives-only contrastive loss (no in-batch negatives) — `z_pred` vs.
one positive (`z_sem_target`) and 224 hard negatives, temperature 0.1, cross-entropy with
target index 0.

Trained via `dynamics_model.py --train` (wandb-logged, `AdamW`, batch 256, 200 epochs,
checkpoints every 5 epochs to `main_project/checkpoints/epoch_NNN.pt` + `latest.pt`,
resumable via `resume_from`). This is the original baseline — `main_project/checkpoints/`
(the loose `epoch_*.pt` files directly in that directory, not the `clip_full_finetuned_subopt_h16/`
subdirectory) is its output, hidden dim **512** with a flatten-first `Linear` action encoder
(an earlier architecture than the 1024/Conv1d version `dynamics_model.py` has evolved to
since — the file has changed under the baseline checkpoint without a re-run).

**2. `main_project/training/dynamics_model_with_upweighting.py`** — same architecture and
frozen-CLIP data pipeline as `dynamics_model.py`, plus a per-sample loss weight: any
`(z_t, action_chunk)` sample whose ground-truth touching/state actually *changes* by
`t + HORIZON` gets `changed_weight=3.0` (vs. `1.0` for samples where nothing changes),
addressing class imbalance toward "boring" static transitions. Validation loss stays
unweighted so it's comparable to the baseline run's val loss. Output:
`main_project/checkpoints_upweighted/` (`--checkpoint_dir` default). A same-day (Jul 9)
paired A/B run against the baseline — see `ARTIFACTS.md`.

**Lineage B — full CLIP fine-tuning (current direction)**

**3. `main_project/training/dynamics_model_clip_full_finetune.py`** — full fine-tuning of
**both** the CLIP ViT-L/14 image and text towers, jointly with a fresh FiLM-MLP dynamics
model (imports `LatentDynamicsModel` from `dynamics_model.py`). Motivation (from the
docstring): the LoRA image-only variant left the CLIP *text* tower frozen, under suspicion
that frozen text embeddings collapse "A touching B" / "A not touching B" into near-identical
vectors, capping InfoNCE separation. Full fine-tuning (not LoRA) was chosen because CLIP
ViT-L/14 is only ~428M params — full `AdamW` optimizer state fits in a single 3090 Ti's
24GB, so LoRA's memory savings aren't needed. Key design points: a **global statement
table** of all 1792 possible compound statements (56 ordered block pairs × 2 states × 8
peg-target blocks × 2 states) is tokenized once and re-encoded fresh every training step —
every sample's positive/negative is a pure integer index into it, no per-batch string
handling; both `A→B` and `B→A` phrasings of the positive are trained every timestep
(touching is symmetric); one block pair (`blue_moon`↔`red_pentagon`, forward direction) is
held out from train entirely so a text-side margin probe can distinguish genuine
generalization from memorization. Data: 5000-episode expert set only, horizon 8. Output:
`main_project/training/checkpoints_clip_image_text_finetuned/` (`--checkpoint_dir` default).

**4. `main_project/training/dynamics_model_clip_full_finetune_combined.py`** — identical to
script 3 except: training data now **combines** the 5000-episode expert set
(`langtable/demos/`) with a 1675-episode "suboptimal" set (`langtable_switch/demos/`, mostly
failure trajectories), each split independently via the same `train_val_split()` (random
750/5000 expert, random 15%/1675 suboptimal, logged separately in `train_val_split.json`);
the held-out-pair mechanism is removed entirely (not compatible with the second data
source); suboptimal episodes use a different positive/negative statement rule (no single
coherent task-relevant pair, so `A` = block currently closest to the peg, `B` = block
currently closest to `A`, recomputed fresh every timestep) while expert-episode logic is
unchanged; diagnostics (margin probe, progress-curve plots) run on expert val episodes only.
No `_zsem.npz` dependency anywhere (the one place the original script read one — the
touching label for probe episodes — is recomputed directly from `block_states`/
`actual_peg_positions`). Horizon 8. Output:
`main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data/`.

**5. `main_project/training/dynamics_model_clip_full_finetune_combined_horizon16.py`** —
**the current best model.** Identical to script 4 except `HORIZON = 16` and
`LatentDynamicsModel(horizon=16)`; checkpoint/diagnostics/wandb run names are distinct so
this run's artifacts never collide with the horizon-8 run's. Output directory default is
`checkpoints_clip_full_fine-tuned_with_subopt_data_h16` — as of the 2026-09-20 cleanup this
checkpoint now lives at **`main_project/checkpoints/clip_full_finetuned_subopt_h16/`**
(consolidated from two directories a mid-run cwd change had split it across; pass
`--checkpoint_dir` explicitly to resume, since the script's own default no longer matches —
see `README.md`'s "Resuming this project" section for the exact resume command). Epochs
0–12 trained so far, `latest.pt` = epoch 12.

Scripts 3–5 all import and use, by default, a **2-GPU-rebalanced training step**:
**`main_project/training/gpu_split_joint_step.py`** (`ImageTextJointStep`,
`calibrate_split`, `run_split_step`). Problem it fixes: naively wrapping only the image
tower + MLP in `nn.DataParallel(device_ids=[0,1])` (even 50/50 image-batch split) then
calling `encode_text()` on the full statement table separately afterward leaves GPU 1 idle
for the entire text phase — measured only 1.29× speedup over single-GPU. Fix: give
`device_ids[1]` a smaller image shard plus the full text tower, sized via a one-time startup
timing probe (`calibrate_split`) so both GPUs finish each step at about the same time. Also
fixes a silent precision bug: `torch.nn.parallel.parallel_apply` only propagates a bool
(`torch.is_autocast_enabled()`) into per-device worker threads, not the actual dtype, so
each worker silently defaults to fp16 instead of the intended bf16 — this module re-enters
`torch.autocast(dtype=torch.bfloat16)` explicitly inside itself to work around that. Pass
`--legacy_even_split` to any of scripts 3–5 to disable and use the old (slower, fp16-bugged)
path instead. **`main_project/training/test_gpu_split_equivalence.py`** is a standalone
numerical-equivalence check for this module (splits vs. single-GPU-bf16 reference), not a
training entrypoint itself.

**Checkpoint of unknown provenance:** `main_project/training/checkpoints_clip_full_finetuned_wider_hl/`
(Jul 18) — referenced by roughly a dozen eval/diagnostic scripts as the "expert-only_model"
comparison baseline, but no training script in the current codebase defaults to this
checkpoint directory name, and the checkpoint itself doesn't record its own launch command.
Inspecting its saved tensor shapes directly (`fc1.weight (1024, 768)`, Conv1d
`action_encoder.conv.weight (64, 2, 3)`) confirms its architecture is exactly what
`dynamics_model.py`'s `LatentDynamicsModel` implements today (hidden dim 1024, Conv1d
action encoder) plus a full CLIP fine-tune (`clip_state_dict` present in the checkpoint) —
so architecturally it's a same-shape sibling of scripts 3–5, just from before the "combined
expert+suboptimal data" and "horizon 16" changes existed. Treat as reference-only; don't
assume it's reproducible from any single current script.

**Lineage C — PaliGemma + LoRA (deprioritized)**

**`main_project/training/paligemma_finetune.py`** — replaces the FiLM-MLP with a fine-tuned
PaliGemmaWM that predicts the future latent directly from pixels rather than a
pre-computed CLIP embedding. `PaliGemmaLatentDynamics` (subclasses `PaliGemmaWMModel`)
builds `[image tokens][action tokens][rand_n]` — a single learnable `rand_n` readout token
appended at the end — runs it through the backbone with bidirectional attention (no causal
suffix), reads out `rand_n`'s final hidden state, and projects it to 768-dim to match the
same `z_sem` space. Trained with the identical hard-negative InfoNCE loss so the two
dynamics models are directly comparable. LoRA (`r=16`, target modules restricted to
`language_model.*.(q_proj|v_proj)` — scoped away from SigLIP's vision tower, which shares
those submodule names) makes this tractable on 2×RTX 3090 Ti; `rand_n`, `projection_head`,
and `action_projector` are separately unfrozen and trained at full precision alongside the
LoRA adapter. Checkpoints save the LoRA adapter (`peft`'s `save_pretrained`) plus
`extra_state.pt` (`rand_n`, projection head, action projector, optimizer state) per epoch.

**`main_project/training/clip_lora.py`** — LoRA building blocks for open_clip's ViT-L/14
image encoder (adapted from a CLIP-LoRA reference repo, reusing only the "split the fused
`nn.MultiheadAttention.in_proj_weight` into per-projection Linears, then LoRA-wrap them"
mechanism). Only the image tower (`clip_model.visual.transformer.resblocks`) is ever
touched; the text tower is left frozen. `LoRALinear` is a pure functional forward (no
in-place merge/unmerge of the base weight, unlike the reference repo's version — unsafe
under autograd and risky with `nn.DataParallel`).

**`dynamics_model_clip_lora_finetune.py`** — the LoRA-image-tower-only dynamics model
training script itself. Two non-identical copies were found during the 2026-09-20 cleanup
(one in `main_project/notebooks/`, one in `main_project/training/`, with meaningfully
different content — see `ARTIFACTS.md`) and archived unreconciled to
`main_project/archive/clip_lora_dynamics_model_deprioritized/`. Its output was
`main_project/training/checkpoints_clip_image_finetuned/` (despite the misleading name —
several eval scripts' comments confirm this is "LoRA: image tower only, text frozen").

### Data caching pipeline

**Frozen-CLIP lineage (dynamics_model.py, dynamics_model_with_upweighting.py) needs the
first three rows below.** Run against `/home/ashwink/full_data_5000/langtable/demos/` (main
5000-episode set) or `/home/ashwink/held_out_episode_5/demos/` (held-out set, separate
scripts of the same name under `main_project/cache_held_out_episodes/`). All are idempotent
— skip episodes whose output `.npz`/`.npy` already exists.

| Script | Output | Notes |
|---|---|---|
| `cache_zobs.py` | `{stem}_zobs.npz` → `z_obs (T,768)` | CLIP ViT-L/14 (`hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-S13B-B90K`) image encoder, L2-normalized |
| `full_5000_cache_zsem.py` / `cache_held_out_episodes/cache_zsem_clip.py` | `{stem}_zsem.npz` → `z_sem (T,768)`, `touching (T,)` bool, `peg_touching (T,)` bool | CLIP text encoder (`ViT-L-14`, `datacomp_xl_s13b_b90k`) on a compound `"{A touching B stmt}. {peg touching stmt}."`, one batched forward pass per episode. Held-out variant derives the peg statement from raw positions when `peg_touching_starting_block_qa` is absent from the episode dict. |
| `cache_zhard.py` | `{stem}_zhard.npz` → `unique_zsem (N,768)`, `neg_indices (T,224) int16` | For every timestep, generates 224 guaranteed-false `"A {touching status} B. peg {touching status} C."` strings (28 block pairs × 8 peg-block combos, inverted-truth by construction), dedupes across the episode, encodes uniques with CLIP text, and stores per-timestep indices into the dedup table. `--smoke` / `--profile` flags for quick validation/timing. |

**Full-CLIP-fine-tune lineage (dynamics_model_clip_full_finetune*.py, the current-best
model's lineage) needs only this row instead** — it never reads `_zobs.npz`/`_zsem.npz`/
`_zhard.npz` at all, since CLIP itself is being trained and statements are re-encoded live:

| Script | Output | Notes |
|---|---|---|
| `cache_frames_npy.py` | `{stem}_frames.npy` → `(T,180,320,3)` uint8, memory-mappable | `ep["frames"]` is normally a list of `T` separate uint8 arrays inside each episode's `.pkl`; eager-loading that (as `LangTableDataset` does) keeps every episode's full frame list resident in RAM — fine for a subset, but ~50GB across the full 5000-episode set, which OOMs. This script stacks each episode's frames into one array and saves it as a **plain, uncompressed** `.npy` (not `.npz` — `.npz` archives don't support `mmap_mode` on read), so downstream code does `np.load(path, mmap_mode="r")[t]` to read one frame's bytes on demand without ever materializing the whole episode. Idempotent; `--smoke` processes 3 episodes. |
| `cache_frames_npy_switch.py` | same, for `/home/ashwink/full_data_5000/langtable_switch/demos/` | The 1675-episode "suboptimal" (mostly failure) set combined into training by `dynamics_model_clip_full_finetune_combined*.py`. |

`main_project/cache_held_out_episodes/` mirrors the frozen-CLIP-lineage scripts pointed at
`/home/ashwink/held_out_episode_5/demos/` instead of the main 5000-episode dir.

**Note on naming collision:** `scripts/cache_zsem.py` (top-level, outside `main_project/`) also produces a `_zsem.npz` file, but it is unrelated — it computes 2048-dim PaliGemma hidden states (`hidden_states[-1][:, -1, :]`) per QA pair from the legacy reward model, keyed by QA category (`task_relevant_qa`, etc.). `main_project`'s `z_sem` is a single 768-dim CLIP text embedding per timestep. Don't conflate the two — they live in different directories and are consumed by different pipelines. `scripts/cache_zsem.py` is also under active revision (see `variable_qa_num_plan.md` at the repo root, which plans a multi-QA-pair `num_qa_pairs` extension to it).

### `LangTableDataset` (`main_project/langtable_dataset.py`)

**Used only by the frozen-CLIP lineage** (`dynamics_model.py`,
`dynamics_model_with_upweighting.py`). The full-CLIP-fine-tune lineage
(`dynamics_model_clip_full_finetune*.py`) each define their own dataset class inline instead
— they need raw pixels (via pre-cached `{stem}_frames.npy`, `mmap_mode="r"`) and re-encode
text statements live every step, so `LangTableDataset`'s cached-`z_sem`/`z_hard`-only
interface doesn't fit their needs.

Loads all `.pkl` episodes from `data_dir`, requiring matching `_zsem.npz` and `_zhard.npz` (episode skipped with a warning if either is missing). `HORIZON = 8`. If `use_cached_zobs=True` (default), also requires `_zobs.npz`; if `False`, CLIP loads onto `device` and encodes frames on the fly in `__getitem__` (GPU-resident CLIP means `DataLoader` must use `num_workers=0`).

Builds a flat index of `(episode_idx, t)` for `t` in `range(T - HORIZON)`. `__getitem__` returns:
```python
{
    "z_t":          (768,)       # z_obs[t] (cached or freshly encoded)
    "actions":      (8, 2)       # actions[t:t+HORIZON], normalized to [-1, 1] via per-dataset min/max
    "z_sem_target": (768,)       # z_sem[t + HORIZON] — the *future* semantic state, not z_sem[t]
    "z_hard_neg":   (224, 768)   # this timestep's 224 hard negatives, gathered via neg_indices → unique_zsem
}
```
Action normalization stats (`action_stats["min"/"max"]`) are computed once across all loaded episodes.

**Important for anything that consumes a trained model's predictions:** `z_pred = model(z_obs[t], actions[t:t+HORIZON])` is a forecast of `z_obs[t + HORIZON]`, not of `z_obs[t]`. Any diagnostic aligning `z_pred` against a per-timestep ground truth must shift the x-axis by `+HORIZON`.

### Planning (`main_project/planning/mppi_core.py`)

Standalone `MPPI` class implementing Algorithm 2 from Williams et al. 2017 ("Information Theoretic MPC for Model-Based RL"). Fully decoupled from any specific dynamics model or environment — `dynamics_fn(state, action) -> state` and `cost_fn(terminal_state) -> cost` are opaque callables, state shape is never inspected. Diagonal-covariance noise (`noise_sigma: (nu,)`, not a full covariance matrix), terminal cost only, warm-started nominal sequence `U` shifted *after* the action is extracted each `command()` call. Ported/adapted from `reference_repos_MPPI/pytorch_mppi` (vendored external reference code — see below) with those simplifications.

`main_project/eval/mppi_rollout.py` wires this to the trained FiLM-MLP: `dynamics_fn` calls the model on batches of `(K, 768)` latents and `(K, 16)` normalized action chunks (`horizon=1` at the MPPI level since the model itself already predicts 8 real env-steps ahead); `cost_fn` is `1 - cos(z_pred, z_star)` where `z_star` is the CLIP text embedding of a goal statement like `"the green cube is touching the blue moon. the peg is touching the green cube."`. Each step also logs dynamics-model debug diagnostics (prediction accuracy vs. realized next state, pairwise cosine collapse across the K candidate rollouts) to a timestamped log file, and saves the episode as an `.mp4` under `main_project/rollouts/`.

### Evaluation / diagnostics (`main_project/eval/`)

~60 scripts, mostly one-off diagnostics rather than a maintained test suite. Most follow the
same pattern: find the target checkpoint(s) explicitly by path (or latest-by-mtime in a
given checkpoint dir), inline-copy the model classes to avoid triggering `open_clip`/dataset
imports at module load time, cache results to `.npz`, save `.png`(s). Grouped by theme
below; see `README.md`'s "Evaluating the current (h16) model" for which of these were most
recently run, and `ARTIFACTS.md` for where their output artifacts (charts/videos/JSON) live.

**Held-out / ranking accuracy (frozen-CLIP baseline)**

| Script | Checks |
|---|---|
| `eval_held_out.py` | Per-timestep accuracy on a held-out episode: `d_pos = 1 - cos(z_pred, z_sem[t+HORIZON])` vs. `d_neg = 1 - cos(z_pred, a random hard negative)`; `correct = d_pos < d_neg`. Also reports accuracy restricted to timesteps where the semantic label actually changed (`sem_changed`), to separate "model tracks a static scene" from "model tracks real transitions." |
| `eval_train_vs_heldout_ranking.py` | InfoNCE ranking accuracy, train sample vs. 3 held-out episodes, under two hard-negative conventions (`neg@t` from `neg_indices[t]` vs. a fixed alternative) |
| `eval_checkpoint_comparison_table.py` | Compares 5 checkpoints (pre-bugfix old model, non-upweighted best/latest, upweighted best/latest) on train vs. held-out InfoNCE ranking accuracy |
| `eval_zero_action_true_vs_false_alignment.py` | Generalization test: does `model(z_t, zeros(1,8,2))` sit closer to a TRUE touching statement's embedding than to the same statement with both clauses inverted, across all 448 ordered `(A,B,C)` triples? |

**Expert-trajectory walkthroughs (baseline model, +LoRA variants)**

| Script | Checks |
|---|---|
| `eval_3_expert_trajs.py` | `cos(z_pred, z_sem[-1])` over time on the 8 val-set episodes divisible by 125 |
| `eval_3_expert_trajs_0_actions.py` | Same, all-zero action chunk — isolates whether the model reads touching-state off the image embedding alone, ignoring actions |
| `eval_3_expert_trajs_fake_zsem.py` | Same, plus cosine sim to a fixed clearly-false statement, to sanity check InfoNCE separation |
| `eval_3_expert_trajs_zcur_zsem.py` | Model-free: `cos(z_obs[t], z_sem[-1])` directly (no dynamics model), overlaid against cached `z_pred` from `eval_3_expert_trajs.py`, correctly time-shifted by `+HORIZON` |
| `eval_3_expert_trajs_clip_lora.py`, `eval_3_expert_trajs_0_actions_clip_lora.py`, `eval_3_expert_trajs_fake_zsem_clip_lora.py` | The same trio of checks, for the CLIP-LoRA (Lineage C, deprioritized) epoch-8 checkpoint instead |

**CLIP fine-tuning experiment comparisons (Lineage B/C checkpoints)**

| Script | Checks |
|---|---|
| `eval_finetuned_checkpoints_cosine_sim.py` | Cosine-sim diagnostic plots across the 4 CLIP fine-tuning-experiment checkpoints, all-zero actions |
| `eval_finetuned_checkpoints_cosine_sim_expert_actions.py` | Same, with real expert actions instead of zero-actions |
| `eval_action_sensitivity.py` | Does varying the candidate action chunk (fixed `z_t`, fixed goal `z_star`) actually move `cos(z_pred, z_star)`? This is the exact property MPPI planning depends on — none of the other comparison metrics test it directly |
| `eval_model_comparison_clip_lora_vs_upweighted.py` / `_100ep.py` | Quantifies CLIP-LoRA (epoch 8) vs. upweighted-baseline FiLM-MLP on planning-relevant terms: does `z_pred` reliably rank the true goal above decoys, and does `cos(z_pred, z_star)` move sensibly with the action; `_100ep` variant subsamples 100 held-out episodes for speed and covers 4 models in one run |
| `text_embedding_similarity_matrix.py` | Pairwise cosine-sim matrix for 6 fixed statements, using the `wider_hl` epoch-8 fine-tuned CLIP **text** tower — probes the fine-tuned text embedding geometry directly, no dynamics model or images involved |
| `zero_action_success_signal_probe.py` | Feeds each of 300 saved rollout frames' CLIP embedding + an all-zero action chunk into the `wider_hl` epoch-8 model, same "zero actions" convention as the baseline's `eval_3_expert_trajs_0_actions.py` |
| `epoch11_frame200_cosine_sim.py` (+ a companion `epoch11_frame208_cosine_sim_summary.txt`) | Single-frame sanity check for epoch 11 of the horizon-8 combined-subopt checkpoint: does the real CLIP image embedding sit closer to the true scene statement than to all 448 guaranteed-false ones? |

**Linear/nonlinear probes (active diagnostic thread — see `ARTIFACTS.md` for the open question)**

| Script | Checks |
|---|---|
| `linear_probe_global_table.py` | Shortcut-learning check on the fine-tuned CLIP **text** tower alone: linear-probes the 1792-entry global statement table's embeddings |
| `linear_probe_global_table_image.py` | Image-side analogue, adapted for real frames (no synthetic-statement equivalent on the image side) |
| `mlp_probe_global_table_image.py` | Nonlinear (1-hidden-layer MLP) version of the image-side probe — tests whether the linear probe's weak/moderate separability (val AUC 0.69–0.85) is a genuine representational limit or just a linear-readout limit |
| `linear_probe_MLP_output.py` | Probes the **dynamics model's** forecast `z_pred = model(z_t, a_t:t+8)` itself (not just the CLIP embedder) — can a probe recover ground-truth distances/touching-state from `z_pred`? Sweeps `wider_hl` and horizon-8 combined-subopt checkpoints across epochs |
| `linear_probe_epoch11_snapshot.py` | Single-epoch snapshot of the above, epoch 11 of the horizon-8 combined-subopt checkpoint |
| `linear_probe_epoch11_h16_snapshot.py` | Same, epoch 11 of the **h16** checkpoint (current model) |
| `linear_probe_epoch11_h16_green_cube_confound.py` | Block-identity confound check on the h16 epoch-11 probes above — is the probe actually reading geometry, or just memorizing which block is `green_cube`? |
| `linear_probe_dist_ridge_refit.py` | Narrow refit of only the 2 distance-regression probes using `RidgeCV` instead of unregularized `LinearRegression` |

**MPPI planning-landscape diagnostics (active thread — the other half of the open question in `ARTIFACTS.md`)**

| Script | Checks |
|---|---|
| `cost_landscape_heatmap.py` | 6×6 grid of candidate `(dx,dy)` actions (one fixed action repeated for all 8 steps) → cosine-sim-to-goal heatmap, `wider_hl` epochs 0/1/8, fixed start state |
| `cost_landscape_realistic_chunks.py` | Follow-up to the above using thousands of **real, unmodified** 8-step action chunks sampled from the dataset (magnitude-decile-stratified) instead of a synthetic constant-repeat action — the synthetic probe's implied per-step magnitude turned out roughly 2× a real chunk's, a shape the model essentially never saw in training |
| `cost_landscape_mppi_selected.py` | MPPI-planning companion to the above: runs full MPPI passes across epochs 5–11 of the horizon-8 combined-subopt checkpoint |
| `mppi_global_or_local_min.py` | Global-vs-local-minima probe built off `cost_landscape_mppi_selected.py` — does the planner converge to one basin, or does it vary by seed? |
| `mppi_temperature_sweep.py` | Softmax-temperature sweep on the epoch-11 5-seed MPPI probe (motivated by high seed-to-seed volatility observed at the shipped `mppi_temperature=0.01`) |
| `mppi_temp_samples_sweep.py` | Generalized temperature-**or**-num-samples (K) sweep for any checkpoint, extended to also target h16 (16-step chunks), not just the hardcoded h8/epoch11 default |
| `mppi_iteration_sweep.py` | Planner-iteration-count sweep (each `command()` call does `n_planning_itrs` refits + 1 initial eval; shipped default is 9 → 10 dynamics evaluations per call) |
| `mppi_iteration1_h16_direction_alignment.py` | **h16-specific, most recently run.** On the very first MPPI iteration — before CEM refinement can bias the candidate population toward a previous pass's winners — does the model already score chunks that push loosely toward the goal? |
| `mppi_obstacle_avoidance_probe.py` | Obstacle scenario: peg starts *between* `green_cube` and `blue_moon`, nearly touching `green_cube` (the mirror image of every other fixed-scene probe here, which place the peg on the far side for a clear straight push) |
| `mppi_obstacle_expert_chunk_probe.py` (h16) / `_h8.py` | Does the model rate a physically-correct detour-and-push chunk *above* the chunks its own MPPI loop actually converges to? |
| `mppi_chunk_rollout_videos.py` | Replays cached epoch-11 MPPI chunks in the real sim and records one `.mp4` each, checking whether predicted `cos_sim(z_pred, z_goal)` tracks what the chunk does physically |
| `oracle_vs_mppi_charts.py` | Per-timestep chart comparing oracle-policy proposals vs. fresh MPPI-probe proposals vs. what the peg actually did, all recomputed from the exact real pybullet state at that timestep of a closed-loop rollout |
| `dump_pybullet_state_frames.py` | Utility companion to `oracle_vs_mppi_charts.py`: renders labeled PNG frames from a rollout's saved pybullet states |

**Live/batch MPPI rollouts**

| Script | Checks |
|---|---|
| `mppi_rollout.py` | Original live MPPI loop debugging the FiLM `LatentDynamicsModel`'s prediction quality itself (not the optimizer), frozen-CLIP baseline |
| `mppi_rollout_clip_lora.py` | Same, CLIP-LoRA model (image encoder LoRA-loaded from the checkpoint's `lora_state_dict`) |
| `mppi_rollout_clip_full_finetune.py` | Same, fully fine-tuned (no LoRA) model, epoch-8-only with extended diagnostics |
| `mppi_rollout_clip_full_finetune_changing_goal.py` | Same sweep (epochs 6–15, `wider_hl`), two-stage goal variant |
| `mppi_rollout_5_per_checkpoint.py` | Batch: 5 independent rollouts per epoch checkpoint, for every checkpoint in a given directory, fixed scene, success/failure summary |
| `mppi_rollout_random_scenes.py` | Same, but each rollout gets its own randomly-sampled `(start_block, target_block)` pair and randomly-placed peg/blocks instead of the fixed `(green_cube, blue_moon)` scene |

**Dataset-level / raw-embedding diagnostics (no dynamics model)**

| Script | Checks |
|---|---|
| `clip_smoothness_diagnostic.py` | Ground-truth check on raw `z_obs` alone: step-to-step cosine smoothness, drift from the start frame, L2 step size |
| `dataset_final_touching_stats.py` | Final-timestep touching statistics over the full 5000-episode dataset |
| `zsem_clip_viz.py`, `zsem_text_only_viz.py`, `zsem_text_global_tsne.py`, `zsem_separation_metric.py`, `zsem_compound_smoke_test.py` | t-SNE / separation-metric visualizations of the `z_sem` embedding space (by task identity, by touching label) |
| `smoke_test_cache_zsem.py` (`main_project/caching/`) | Smoke test for the `z_sem` caching logic |

### Checkpoints & run artifacts

**For the full, current-as-of-2026-09-20 inventory of every checkpoint/artifact directory
— what's active, what's a kept comparison experiment, what's dead — see `README.md`'s
"Resuming this project" section and `ARTIFACTS.md`.** Summary of the checkpoint-relevant
pieces:

- `main_project/checkpoints/` — the original frozen-CLIP baseline's `epoch_NNN.pt`s +
  `latest.pt` directly at this level, **plus** a subdirectory,
  `main_project/checkpoints/clip_full_finetuned_subopt_h16/`, holding the current-best
  model (Lineage B, script 5 above). That subdirectory's checkpoints were originally split
  across two locations by a mid-run working-directory change and were consolidated here
  during the 2026-09-20 cleanup — see `ARTIFACTS.md` for the story and
  `README.md` for the exact resume command (the training script's own `--checkpoint_dir`
  default no longer matches this path, so it must be passed explicitly).
- `main_project/training/checkpoints_*/` — one directory per Lineage B/C training run
  (`checkpoints_clip_image_finetuned` [LoRA], `checkpoints_clip_image_text_finetuned`,
  `checkpoints_clip_full_finetuned_wider_hl`,
  `checkpoints_clip_full_fine-tuned_with_subopt_data`), named by each script's own
  `--checkpoint_dir` default. See the "Dynamics model variants" table above for which
  script produced which.
- `main_project/checkpoints_upweighted/` — the upweighted-loss A/B companion to the
  baseline, script 2 above.
- `main_project/training/checkpoints/paligemma_dynamics/` — documented default output dir
  for `paligemma_finetune.py`, but **does not exist anywhere on disk** — this script does
  not appear to have ever been run to completion (or its output was since removed).
- `main_project/archive/` — dead/superseded material kept for reference only (earliest
  drafts, an empty aborted-run stub, the two divergent LoRA script copies). Full breakdown
  in `ARTIFACTS.md`.
- Wandb run logs live in **four separate archived locations** (`wandb_archive/` at the repo
  root, `main_project/wandb_archive/`, `main_project/training/wandb_archive/`,
  `main_project/notebooks/wandb_latest/`) holding different, non-overlapping runs — not
  duplicates of each other; see `ARTIFACTS.md` for why there are four.
- `main_project/train.log` — plain-text epoch/loss log from a frozen-CLIP baseline training run.
- `main_project/rollouts/` — `.mp4` outputs from `mppi_rollout.py`.
  `main_project/rollouts_5_per/` holds the much larger set of per-swept-config batch
  rollout outputs from the Lineage B eval scripts (5 rollouts per checkpoint × many
  hyperparameter sweeps — see `ARTIFACTS.md`).
- Git policy: everything above is tracked **except** model-weight files (`*.pt` etc., excluded by
  type — ~280GB) and seven individually oversized files (six `.npz` caches + one `.wandb` log).
  Exact paths of everything excluded are in `ARTIFACTS.md`; the checkpoints must be moved
  separately (rsync/S3) since they will not arrive via `git clone`.

---

## Legacy: `swm/` — PaliGemma VLM reward model

The earlier project. Given a robot action sequence and a natural-language yes/no question about task success ("Is the red cube touching the blue moon?"), a fine-tuned PaliGemma VLM outputs a `yes`/`no` probability; a planner uses these probabilities to select the best action sequence from a diffusion policy's candidates.

### Running evaluation

```bash
python scripts/evaluate_swm_hydra.py --config-name eval_gradient_full
```

Configs in `configs/`:
- `eval_base_diffusion_lt.yaml` — LangTable, expert diffusion only (no SWM planning)
- `eval_base_ogbench.yaml` — OGBench, expert diffusion only
- `eval_gradient_full.yaml` — LangTable, gradient-based planning with SWM reward
- `eval_gradient_full_ogbench.yaml` — OGBench, gradient-based planning with SWM reward

Each run writes `video.mp4`, `heat_map_*.png`, `goal.txt`, `reward.pkl`, and appends to `results.txt` under `<root_save_path>/<name>/<planning_name>/<block_a>_<block_b>/<seed>/`. Runs are checkpointed: if `reward.pkl` already exists for a seed, it is loaded and the run is skipped.

### High-level flow

```
eval() in swm/evaluation.py
  ├── env (LangTableEnv / OGBenchEnv) wraps the simulator
  ├── goal_generator (e.g. PushBlocksTogetherGoal / StackBlocksGoal)
  │     └── .get_questions() → list of (question_str, "yes"/"no", weight)
  ├── diffusion_model (DiffusionPolicy) — samples candidate action trajectories
  └── get_plan() in swm/planning_algos.py
        ├── sample_initial_actions() from diffusion or uniform noise
        ├── model.get_probabilistic_rewards_wm() via SWMRewardDataset + DataLoader
        └── returns best_action_seq selected by argmax weighted reward
```

### Key modules

| Module | Role |
|---|---|
| `swm/paligemma_wm/` | Custom PaliGemma variant with an action projector; HuggingFace-compatible |
| `swm/semantic_world_model.py` | `SWMGradModel` — loads the model and exposes `get_scores()` / `get_probabilistic_rewards_wm()` |
| `swm/planning_algos.py` | Three planning modes: base sampling, MPPI, gradient ascent (SGD on action tensors) |
| `swm/evaluation.py` | Single-episode evaluation loop |
| `swm/utils/envs.py` | `LangTableEnv` and `OGBenchEnv` wrappers around the simulators; `get_lang_table_env()` is also reused directly by `main_project/eval/mppi_rollout.py` |
| `swm/utils/goal_generators.py` | Per-task `BaseGoalGenerator` subclasses that produce questions and check termination |
| `swm/utils/dataset.py` | `SWMRewardDataset` (batched reward scoring) and `DiffusionDataset` (diffusion policy training data) |
| `swm/diffusion_policy/` | DDPM diffusion policy: vision encoder + noise prediction network |
| `swm/constants.py` | `ANSWER_OPTIONS = ["yes", "no"]` |

### PaliGemmaWM model

The world model is a PaliGemma VLM extended with an **action projector** (`PaliGemmaWMMultiModalActionProjector`): a single linear layer that maps raw action vectors (dim = `action_input_dim`) into the LM token embedding space. During the forward pass the input sequence is structured as:

```
[image tokens × image_seq_length] [action tokens × num_actions] [BOS] [question text] [\n]
```

Action tokens are placeholder tokens (`<action>`) in the text; their embeddings are overwritten by the projected action features. Actions padded beyond the sequence are filled with `inf` and filtered out in the model forward pass. The model uses bidirectional attention over the prompt prefix (image + action + question) and causal attention only over the generated suffix.

Inference extracts the logit at the **last token position** (before generation), applies softmax, and reads the probabilities at the `yes` and `no` token IDs.

`main_project/training/paligemma_finetune.py` reuses `PaliGemmaWMModel` and `PaliGemmaWMProcessor` from this package but bypasses the question/answer text path entirely — see above.

### Data pipeline for the diffusion policy (`DiffusionDataset`)

The diffusion policy is trained on offline trajectory data stored as `.pkl` files. Each file is a dict:

```python
{
    "frames": np.ndarray,   # shape (T, H, W, 3), dtype uint8
    "actions": np.ndarray,  # shape (T, action_dim), dtype float32
}
```

Loading (`swm/utils/dataset.py:DiffusionDataset`):
1. All `.pkl` files in `data_folder_path` are loaded and concatenated.
2. Each trajectory is padded at the end by `pad_length` (default 12) frames — last frame is tiled, actions are zero-padded.
3. A flat index of `(frame_tuple, (ac_start, ac_stop))` pairs is built. `frame_tuple` is a tuple of `obs_horizon` frame indices for the observation stack.
4. All images are stored as `torch.Tensor` of shape `(total_frames, 3, H, W)` (channels-first).
5. Actions are normalized to `[-1, 1]` per-dimension using per-dataset min/max stats (stored in `self.stats`).

Each `__getitem__` returns:
```python
{
    "image":  torch.Tensor,   # shape (obs_horizon, 3, H, W), float32 in [0, 1]
    "action": torch.Tensor,   # shape (horizon, action_dim), float32 in [-1, 1]
}
```

Action normalization uses `normalize_data` / `unnormalize_data` (linear min-max to `[-1, 1]`). The stats dict is `{"action": {"min": Tensor, "max": Tensor}}`.

### Data pipeline for reward scoring (`SWMRewardDataset`)

Used only at inference time inside `get_probabilistic_rewards_wm()`. It is constructed fresh each planning step with:
- A single PIL image (the current observation)
- An action sequence array of shape `(num_samples, pred_horizon, action_dim)`
- A list of `(question_str, desired_token, weight)` tuples

The dataset enumerates the Cartesian product of `(question_idx, action_idx, horizon_step)` where `horizon_step` steps in `action_skip` increments up to `pred_horizon`. Each item is a prefix of the action sequence up to `h_step`: `action_seq[a_idx, :h_step]`.

`collate_fn` returns lists (not stacked tensors) for images and actions since action sequences have variable length (different `h_step` values). The processor handles padding internally via `pad_sequence(..., padding_value=inf)`.

Scores are written into a `(num_questions, num_samples, pred_horizon)` reward array. Final weighted reward sums across questions and horizon steps to produce a scalar per sample.

### Planning modes (controlled by config flags)

| Flag | Mode | Description |
|---|---|---|
| `expert_diffusion=True` | Expert baseline | Runs diffusion policy with no SWM reranking |
| `diffusion=True, mppi=False, gradient=False` | Base diffusion | Single diffusion sample, scored but not reranked |
| `mppi=True` | MPPI | Iterative resampling with exponential trajectory weighting |
| `gradient=True` | Gradient ascent | SGD on continuous action tensor; requires `gradient=True` in `get_probabilistic_rewards_wm` |

### Environments

Both environments expose the same `BaseEnv` interface. The key method for visualization is `project_actions_to_camera_frame()`, which reprojects 3D action trajectories into pixel coordinates for heat-map overlays.

- **LangTable**: 2D tabletop block-pushing (PyBullet + TensorFlow Agents); `action_dim=2`; `scale_factor=0.03`
- **OGBench**: 3D cube-stacking with a UR5e arm (MuJoCo); `action_dim=5`; `scale_factor=1.0`; uses `visual-cube-quadruple-v0`

### Goal generators

Each `BaseGoalGenerator` subclass couples a reward function to a question template. The `get_questions()` method returns a list of `(text, answer, weight)` tuples. For `StackBlocksGoal` (OGBench) there is an internal multi-step state machine (`self.step`): step 0 asks "is the robot grasping X?" and transitions to step 1 once `yes_prob > 0.9`, then switches to stacking questions.

### Working with PaliGemma

When working with PaliGemma in any way, please read `/swm/paligemma_wm/PALIGEMMA_CONTEXT.md` first, to understand the schema of the files in that folder.

### Vendored reference code

`reference_repos_MPPI/` (`PLDM`, `pytorch_mppi`) is pasted in from other repos as reference material for `main_project/planning/mppi_core.py` — not part of this project's own code, don't treat changes there as project work.

### LangTable episode pkl schema

Demo episodes are stored as `.pkl` files in `/home/ashwink/full_data_5000/langtable/demos/` (held-out set: `/home/ashwink/held_out_episode_5/demos/`). Filename pattern: `blocktoblock_{id}_{success|failure}.pkl`. Loading with `pickle.load` requires the `language_table` package (available in the swms `.venv`; not available in RoboTwin). This schema is shared by both `swm/` and `main_project/` — all the caching scripts above read directly from these `.pkl` files.

Each file is a **dict** with the following keys:

```python
{
    "metadata": {
        "reward_name":              str,   # e.g. "blocktoblock"
        "instruction_str":          str,   # e.g. "put the green star close to the green cube"
        "start_block":              str,   # e.g. "green_star"  — block being pushed
        "oracle_target_block":      str,   # e.g. "green_cube"  — destination block
        "oracle_target_translation": None,
        "target_absolute_location":  None,
        "target_relative_location":  None,
    },
    "initial_state":    dict,              # simulator initial state
    "success":          bool,
    "actions":          list,              # length T-1
    "frames":           list,              # length T; each element: np.ndarray (180, 320, 3) uint8 RGB
    "actual_peg_positions": np.ndarray,   # shape (T, 2), float32 — peg (x, y) in world coords at each timestep
    "block_states":     list,              # length T; each element: dict mapping block_name → np.array([x, y], float32)
    "task_relevant_qa":              list, # length T; each element: list of QA dicts (4 per timestep)
    "task_relevant_qa_wrong_answer": list, # length T; same structure, wrong answers
    "task_irrelevant_qa":            list, # length T; variable count per timestep1
    "task_completely_irrelevant_qa": list, # length T; variable count per timestep
    "dist_btwn_relevant_blocks":     list, # length T
    "peg_touching_starting_block_qa": list,  # absent in some held-out episodes — derive from positions instead (see cache_zsem_clip.py, eval_held_out.py)
}
```

**Accessing block and peg positions:**
```python
# Peg position at timestep t — shape (2,) float32, world-frame (x, y)
peg_xy = ep["actual_peg_positions"][t]

# All block positions at timestep t — dict: block_name → np.array([x, y], float32)
block_pos_t = ep["block_states"][t]          # e.g. {"green_star": array([0.12, -0.03]), ...}

# Position of a specific block at timestep t
green_star_xy = ep["block_states"][t]["green_star"]

# Position of the task-relevant blocks (mover and target)
mover  = ep["metadata"]["start_block"]        # e.g. "green_star"
target = ep["metadata"]["oracle_target_block"] # e.g. "green_cube"
mover_xy  = ep["block_states"][t][mover]
target_xy = ep["block_states"][t][target]
```

Block name keys follow the pattern `"{color}_{shape}"` (e.g. `"blue_cube"`, `"red_moon"`). All 8 blocks are present in every `block_states` dict. Coordinates are 2D world-frame floats in the same space as `actual_peg_positions`.

**QA dict format** (used in all four `*_qa` lists):
```python
{"question": str, "answer": bool}
# e.g. {"question": "Is the green star touching the blue cube?", "answer": True}
```

Question format is always `"Is the [color] [shape] touching/close to the [color] [shape]?"`.
Touching questions come in symmetric pairs: A→B and B→A with the same answer (touching is symmetric).

**Blocks**: All 8 blocks are present in every episode. The 8 fixed identities (color_shape):

| | cube | moon | pentagon | star |
|---|---|---|---|---|
| blue | ✓ | ✓ | — | — |
| green | ✓ | — | — | ✓ |
| red | — | ✓ | ✓ | — |
| yellow | — | — | ✓ | ✓ |

Not a full 4×4 cross-product — each color appears with exactly 2 shapes. ~625 episodes per (start, target) pair across 5000 total episodes.

**Accessing the final frame and QA**:
```python
final_frame = ep["frames"][-1]           # (180, 320, 3) uint8
final_qa    = ep["task_relevant_qa"][-1] # list of 4 QA dicts
```
