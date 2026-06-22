# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Evaluation code for **Semantic World Models (SWM)** — a fine-tuned PaliGemma-based vision-language model conditioned on robot action sequences. The model acts as a reward signal: given an image, a proposed action sequence, and a natural-language question about task success ("Is the red cube touching the blue moon?"), it outputs a `yes`/`no` probability. A planner uses these probabilities to select the best action sequence.

## Setup

```bash
uv sync
source .venv/bin/activate
bash ckpts/download_checkpoints.sh   # downloads all four HuggingFace checkpoints
```

Requires Python 3.10. The `language-table` and `ogbench` dependencies are pinned to forked repos (see `pyproject.toml` `[tool.uv.sources]`).

## Running evaluation

```bash
python scripts/evaluate_swm_hydra.py --config-name eval_gradient_full
```

Configs in `configs/`:
- `eval_base_diffusion_lt.yaml` — LangTable, expert diffusion only (no SWM planning)
- `eval_base_ogbench.yaml` — OGBench, expert diffusion only
- `eval_gradient_full.yaml` — LangTable, gradient-based planning with SWM reward
- `eval_gradient_full_ogbench.yaml` — OGBench, gradient-based planning with SWM reward

Each run writes `video.mp4`, `heat_map_*.png`, `goal.txt`, `reward.pkl`, and appends to `results.txt` under `<root_save_path>/<name>/<planning_name>/<block_a>_<block_b>/<seed>/`. Runs are checkpointed: if `reward.pkl` already exists for a seed, it is loaded and the run is skipped.

## Architecture

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
| `swm/utils/envs.py` | `LangTableEnv` and `OGBenchEnv` wrappers around the simulators |
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
