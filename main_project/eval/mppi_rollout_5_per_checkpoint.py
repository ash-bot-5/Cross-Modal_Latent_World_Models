"""
Batch MPPI rollout evaluation: 5 independent rollouts per epoch checkpoint, for every
checkpoint found in a given checkpoint directory, with a success/failure summary.

Duplicated from mppi_rollout_clip_full_finetune.py. Everything about the MPPI planning
loop itself is UNCHANGED from that script: CLIP loading, LatentDynamicsModel/FiLMLayer/
ActionEncoderConv class defs, action-stats computation, get_lang_table_env(..., seed=42),
the peg-teleport block (forcing peg -> BLOCK_A -> BLOCK_B collinear, just past BLOCK_A),
the dynamics_fn/cost_fn closures, and the MPPI(...) hyperparameters (K=512, pred_horizon=8,
n_planning_itrs=9, max_action_value=1.0, warm_start_std=0.1).

pred_horizon is no longer a hardcoded 8: it's auto-detected per checkpoint from
action_encoder.out.weight's shape in the loaded mlp_state_dict (out_features / conv_channels),
so this script also runs horizon-16 checkpoints (e.g. checkpoints_clip_full_fine-tuned_with_subopt_data_h16/)
without any CLI flag -- LatentDynamicsModel/ActionEncoderConv both take a horizon= arg now.

Three of those are now CLI-configurable, all defaulting to the original behaviour so an
unflagged invocation still reproduces every existing result under rollouts_5_per/:

  --planner           mppi (default) or cem. "mppi" is planning/mppi_core.py's MPPI, which is
                      really a Gaussian CEM with SOFTMAX weighting over all K samples; "cem" is
                      planning/cem_core.py's CEM, a textbook top-Ne elite truncation. Every
                      shared hyperparameter (K=512, pred_horizon=8, n_planning_itrs=9,
                      max_action_value=1.0, warm_start_std=0.1) is passed identically to both,
                      so the two arms are compute-matched and differ only in the refit rule.
                      --num_elites sets Ne (default: elite_frac=0.1, i.e. Ne=51 at K=512);
                      it is ignored by --planner mppi, as --mppi_temperature is by --planner cem.
                      --tag is REQUIRED with --planner cem (see the guard in main()).
  --cold_start        calls planner.reset() before EVERY planning call, so each plan draws a
                      fresh Uniform(-1,1) population with no memory of the previous chunk --
                      true open-loop chunk planning. Makes --warm_start_std inert. Beware that
                      --n_execute 8 does NOT achieve this: rolling an 8-row chunk by 8 is the
                      identity and zeroes every row, so the warm start becomes
                      Normal(0, warm_start_std) -- a measured std of 0.100 against 0.577 for a
                      cold draw, i.e. the TIGHTEST do-nothing prior of any --n_execute value
                      rather than the absence of a prior. --tag is REQUIRED.
  --evals             total dynamics evaluations per planning call; the planner receives
                      n_planning_itrs = evals - 1. Default 10 as before (the shipped
                      n_planning_itrs=9). Same units as eval/mppi_iteration_sweep.py's --evals,
                      so a number means the same thing in both scripts. This is the dominant
                      cost of a rollout -- ~0.6 ms per evaluation at K=512, so --evals 10000 is
                      ~6 s per planning call and ~30 min per 300-step rollout, versus ~3 s per
                      rollout at the default. The orchestrator prints an up-front estimate when
                      the total exceeds ~15 min. --tag is REQUIRED when non-default.
  --warm_start_std    std of the population sampled around the shifted previous chunk on every
                      planning call after the first; default 0.1 as before. Both planners accept
                      it. Worth knowing: a cold-start draw is Uniform(-1,1), std 0.577, so 0.1
                      makes the warm-started population ~5.8x narrower before any refit, and 299
                      of 300 planning calls in a rollout are warm-started. eval/
                      mppi_temperature_sweep.py cannot measure this -- it calls command() once on
                      a fresh instance, so the warm-start branch never runs there. --tag is
                      REQUIRED when non-default.
  --n_execute         how many sub-steps of each action-horizon-step chunk to execute open-loop before
                      re-planning; default 1 as before. --max_steps stays denominated in ENV
                      steps, so --n_execute 4 runs max_steps//4 plans over the same simulator
                      time and remains comparable with the n_execute=1 arms. Both planners
                      accept it -- each is constructed with n_execute so its warm start shifts
                      forward by exactly what was executed rather than a hardcoded 1. --tag is
                      REQUIRED whenever it exceeds 1.
  --mppi_temperature  CEM softmax temperature, default 0.01 as before. The temperature sweeps
                      under MPPI_charts/min_determination/expert+subopt-model/temp_sweep{,_fine}/
                      favour 0.1 for open-loop single-chunk planning; --mppi_temperature 0.1
                      tests whether that carries over to closed-loop rollouts.
  --num_samples       number of candidate action-chunk samples per planning call (K). Default 512,
                      the value every existing rollouts_5_per/ result used. Both planners accept
                      it. Dominant cost driver alongside --evals -- doubling it roughly doubles
                      per-evaluation wall-clock. --tag is REQUIRED when non-default.
  --seed_base         default unset, i.e. the original behaviour: the env seed and peg placement
                      are already fully deterministic every rollout, while MPPI's own CEM
                      sampling draws from the ambient (unseeded) RNG, so the N rollouts per
                      checkpoint naturally explore different actions -- at the cost of not being
                      reproducible. Setting it seeds rollout i with seed_base + i, which keeps
                      the rollouts different from EACH OTHER while making each one individually
                      repeatable.

What's new relative to the source script:
  - Takes --ckpt_dir on the CLI instead of a hardcoded CHECKPOINTS list; discovers every
    epoch_*.pt in that directory and runs N_ROLLOUTS (default 5) rollouts per epoch.
  - Maps the checkpoint dir's basename to an output subfolder name (CKPT_DIR_TO_SUBFOLDER)
    and writes everything under main_project/rollouts_5_per/<subfolder>/epoch<N>/. --tag appends a
    suffix to <subfolder>, which is REQUIRED when changing a planner hyperparameter: the
    already-done check below skips any rollout whose _result.json exists, so re-running into a
    populated directory silently does nothing rather than overwriting.
  - --epochs restricts the run to specific epochs instead of every checkpoint in the directory.
  - Each rollout's mp4 is named epoch<N>_rollout<i>_<success|failure>.mp4, where success is
    dist(BLOCK_A, BLOCK_B) <= SUCCESS_DIST_THRESHOLD at ANY point in the episode.
  - Each rollout also writes a small *_result.json (epoch, rollout_idx, min_dist_a_b, success)
    into the same epoch dir, and a per-rollout .log (same per-step diagnostic table as the
    source script) into the same epoch dir (instead of the script's own directory) -- this
    generates 55-80 rollouts' worth of logs, which would otherwise clutter main_project/eval/.
  - Each rollout also writes *_pybullet_states.pkl: a list of inner.get_pybullet_state() dicts
    (robots/robot_end_effectors/objects/start_block/etc., the same structure documented in
    CLAUDE.md's LangTable episode pkl schema section), one entry per element of the saved mp4's
    frame list -- states[i] is the full sim state at the moment frame i was captured, so a frame
    index into the video is also a valid index into this file. Nothing upstream of this script
    (the LangTable demo .pkl episodes, cache_z*.py, the diffusion/reward-model dataset loaders)
    caches full pybullet state anywhere; only 2D block/peg positions and rendered RGB frames were
    ever persisted before this. Confirmed picklable (plain dataclasses, no live pybullet-client
    references) and ~10KB/frame, so a 300-step rollout is ~3MB.
  - After all rollouts for --ckpt_dir finish, aggregates all *_result.json files under
    the subfolder into one summary.txt in that subfolder, broken down by epoch.

Run (one invocation per checkpoint directory):
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_rollout_5_per_checkpoint.py --ckpt_dir main_project/training/checkpoints_clip_full_finetuned_wider_hl
    python main_project/eval/mppi_rollout_5_per_checkpoint.py --ckpt_dir main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data

Single-epoch run at a non-default temperature (writes to expert+subopt_temp0.1/, leaving the
existing temp=0.01 results in expert+subopt/ untouched):
    python main_project/eval/mppi_rollout_5_per_checkpoint.py \
        --ckpt_dir main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data \
        --epochs 11 --mppi_temperature 0.1 --tag temp0.1 --seed_base 123

Same single-epoch probe under the true-CEM planner (writes to expert+subopt_cem/):
    python main_project/eval/mppi_rollout_5_per_checkpoint.py \
        --ckpt_dir main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data \
        --epochs 11 --planner cem --tag cem --seed_base 123

Each rollout runs in its own subprocess (this script re-invokes itself with --ckpt_path/
--epoch/--rollout_idx/--out_dir) rather than looping in-process, because the LangTable env
wraps a pybullet physics client and pybullet only supports one such client per process --
constructing a second env in the same process after the first would be unsafe/undefined.
"""

import sys
import os
import re
import glob
import json
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import imageio

from planning.mppi_core import MPPI
from planning.cem_core import CEM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_A   = "green_cube"
BLOCK_B   = "blue_moon"
MAX_STEPS_DEFAULT = 300
N_ROLLOUTS_DEFAULT = 5
K_DEFAULT = 512   # num_samples default
SUCCESS_DIST_THRESHOLD = 0.05

# CEM softmax temperature. 0.01 is the original hardcoded value that every existing result in
# rollouts_5_per/ was generated with, so it stays the default. The temperature sweeps
# (MPPI_charts/min_determination/expert+subopt-model/temp_sweep{,_fine}/) favour 0.1 on
# open-loop single-chunk planning -- pass --mppi_temperature 0.1 to test that in closed loop.
MPPI_TEMPERATURE_DEFAULT = 0.01

# Per-dimension std used to sample each command() call's initial population around the shifted
# previous chunk. 0.1 is the value every existing rollouts_5_per/ result used, so it stays the
# default. Note a cold-start population is Uniform(-1,1), std 0.577 -- so at 0.1 the warm-started
# population is already ~5.8x narrower before any refit runs, and since 299 of 300 planning calls
# in a rollout are warm-started, this sets the exploration width for essentially the whole
# episode. eval/mppi_temperature_sweep.py cannot measure it: it calls command() once on a fresh
# instance, so prev_best_action is always None and the warm-start branch never executes.
WARM_START_STD_DEFAULT = 0.1

# Total dynamics evaluations per planning call = n_planning_itrs + 1 (1 initial evaluation plus
# n_planning_itrs refits). 10 -- i.e. the shipped n_planning_itrs=9 -- is what every existing
# rollouts_5_per/ result used, so it stays the default. Expressed as total evaluations rather
# than n_planning_itrs to match eval/mppi_iteration_sweep.py's --evals, so the same number means
# the same thing in both scripts. NOTE this is the dominant cost of a rollout: measured ~0.6 ms
# per evaluation at K=512, so 10000 evals is ~6 s per planning call and ~30 min per 300-step
# rollout, versus ~3 s per rollout at the default 10.
EVALS_DEFAULT = 10

TRAIN_DIR    = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLOUTS_ROOT = os.path.join(MAIN_PROJECT_DIR, "rollouts_5_per")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Maps a --ckpt_dir's basename to the output subfolder name under ROLLOUTS_ROOT.
CKPT_DIR_TO_SUBFOLDER = {
    "checkpoints_clip_full_fine-tuned_with_subopt_data": "expert+subopt",
    "checkpoints_clip_full_finetuned_wider_hl": "expert-only",
    # Same basename appears at both main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data_h16/
    # (epoch_000-004) and the repo-root checkpoints_clip_full_fine-tuned_with_subopt_data_h16/ (epoch_005-012)
    # -- training was cancelled and resumed into a new dir. One entry here (keyed by basename) resolves
    # both --ckpt_dir paths to the same output subfolder so their results aggregate into one summary.
    "checkpoints_clip_full_fine-tuned_with_subopt_data_h16": "expert+subopt_h16",
}

# ---------------------------------------------------------------------------
# Inline model classes (importing dynamics_model.py triggers LangTableDataset
# + open_clip at module level — copy classes here to stay lightweight)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, out_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return (1 + gamma) * hidden + beta


class ActionEncoderConv(nn.Module):
    """Conv1d action encoder: treats the 2 action dims as channels and the 8 timesteps as
    the sequence axis, so neighboring timesteps can interact via the kernel before the
    temporal axis is collapsed by flattening (unlike a flatten-first Linear, which discards
    step order entirely)."""

    def __init__(self, action_dim: int = 2, horizon: int = 8,
                 conv_channels: int = 64, kernel_size: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Conv1d(action_dim, conv_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2)
        self.act = nn.Mish()
        self.out = nn.Linear(conv_channels * horizon, out_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, horizon, action_dim) -> (B, action_dim, horizon) for Conv1d
        h = self.act(self.conv(actions.transpose(1, 2)))  # (B, conv_channels, horizon)
        return self.out(h.flatten(1))                      # (B, out_dim)


class LatentDynamicsModel(nn.Module):
    def __init__(self, horizon: int = 8):
        super().__init__()
        self.action_encoder = ActionEncoderConv(horizon=horizon)
        self.fc1 = nn.Linear(768, 1024);  self.ln1 = nn.LayerNorm(1024)
        self.film1 = FiLMLayer(256, 1024); self.act1 = nn.Mish()
        self.fc2 = nn.Linear(1024, 1024);  self.ln2 = nn.LayerNorm(1024)
        self.film2 = FiLMLayer(256, 1024); self.act2 = nn.Mish()
        self.fc_out = nn.Linear(1024, 768); self.ln_out = nn.LayerNorm(768)

    def forward(self, z_t: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        # actions: (B, horizon, 2)
        global_cond = self.action_encoder(actions)
        h = self.act1(self.film1(self.ln1(self.fc1(z_t)), global_cond))
        h = self.act2(self.film2(self.ln2(self.fc2(h)), global_cond))
        return F.normalize(self.ln_out(self.fc_out(h)), dim=-1)


def run_rollout(ckpt_path: str, epoch: int, rollout_idx: int, out_dir: str,
                 max_steps: int, mppi_temperature: float = MPPI_TEMPERATURE_DEFAULT,
                 seed: int | None = None, planner_name: str = "mppi",
                 num_elites: int | None = None, n_execute: int = 1,
                 warm_start_std: float = WARM_START_STD_DEFAULT,
                 n_evals: int = EVALS_DEFAULT, cold_start: bool = False,
                 num_samples: int = K_DEFAULT) -> None:
    """Runs one full planning rollout with the given checkpoint and saves an .mp4 + result
    json + log into out_dir. Everything below is process-local (CLIP, env, model, planner) —
    safe to call exactly once per process.

    planner_name: "mppi" (default) uses planning/mppi_core.py's softmax-weighted MPPI;
        "cem" uses planning/cem_core.py's top-Ne elite CEM. Only the refit rule differs --
        every shared hyperparameter is passed identically to both below.
    num_elites: CEM only. None uses cem_core's elite_frac=0.1 default (Ne=51 at K=512).
    n_execute: how many sub-steps of each action-horizon-step chunk to execute open-loop before
        re-planning. max_steps stays denominated in ENV steps, so n_execute=4 runs
        max_steps//4 plans and the episode covers the same simulator time as n_execute=1.
        Passed to the planner too, so its warm start shifts by the same amount.
    warm_start_std: per-dimension std for the population sampled around the shifted previous
        chunk on every command() call after the first. Both planners take it. Defaults to 0.1,
        the value every existing rollouts_5_per/ result used.
    n_evals: total dynamics evaluations per planning call; the planner receives
        n_planning_itrs = n_evals - 1. Same meaning as eval/mppi_iteration_sweep.py's --evals.
        This dominates rollout wall-clock -- see EVALS_DEFAULT's note.
    cold_start: when True, planner.reset() is called before EVERY planning call, so each plan
        draws a fresh Uniform(-1,1) population (std 0.577) with no memory of the previous chunk.
        Makes warm_start_std inert, since the warm-start branch is never taken. Note that
        n_execute=8 does NOT achieve this on its own: rolling an 8-row chunk by 8 is the identity
        and zeroes every row, so the warm start becomes Normal(0, warm_start_std) -- measured std
        0.100 vs 0.577 cold, i.e. the tightest do-nothing prior of any n_execute, not the absence
        of one.
    mppi_temperature: CEM softmax temperature, MPPI planner only. Defaults to the original
        hardcoded 0.01 so unflagged runs reproduce every existing result in rollouts_5_per/.
    num_samples: number of candidate action-chunk samples per planning call (K). Both planners
        take it. Defaults to K_DEFAULT (512), the value every existing rollouts_5_per/ result
        used.
    seed: when not None, seeds torch's global RNG for this rollout. The env reset and peg
        placement are already deterministic, so MPPI's CEM sampling is the only source of
        randomness -- seeding it makes THIS rollout individually reproducible without making
        the N rollouts identical to each other (the orchestrator passes seed_base + rollout_idx,
        so each still explores differently). Default None preserves the original ambient-RNG
        behaviour that the existing baselines were generated under."""

    os.makedirs(out_dir, exist_ok=True)
    stem = f"epoch{epoch}_rollout{rollout_idx}"

    if seed is not None:
        torch.manual_seed(seed)

    # -----------------------------------------------------------------------
    # Logging — mirror to stdout and a persistent, timestamped log file so the
    # full per-step record survives regardless of how the script is launched.
    # -----------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"{stem}_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,  # reset handlers in case this module is re-entered
    )
    log = logging.getLogger(f"mppi_rollout.{stem}")
    log.info(f"Checkpoint dir epoch: {epoch}  Rollout: {rollout_idx}")
    log.info(f"Logging to {log_path}")

    # -----------------------------------------------------------------------
    # 1. Fixed goal text, held constant for the entire rollout (no stage-switching).
    # -----------------------------------------------------------------------
    goal_text = (
        f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
        f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
    )
    log.info(f"Goal (fixed, steps 0-{max_steps - 1}): {goal_text}\n")

    # -----------------------------------------------------------------------
    # 2. Checkpoint + CLIP (image + text encoder), both towers fully fine-tuned --
    # loaded as a plain full state dict, no LoRA adapter involved.
    # -----------------------------------------------------------------------
    log.info(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    log.info(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}\n")

    # Action horizon is a property of the checkpoint's own architecture (how many action-chunk
    # sub-steps the action_encoder was trained on), not a free planning choice -- detect it from
    # the saved weights instead of trusting a CLI flag, so this always matches the checkpoint.
    # action_encoder.out.weight: (256, conv_channels * horizon), conv_channels=64 (ActionEncoderConv default)
    action_horizon = ckpt["mlp_state_dict"]["action_encoder.out.weight"].shape[1] // 64
    log.info(f"Detected action horizon from checkpoint: {action_horizon}\n")

    log.info("Loading CLIP ViT-L/14 (datacomp_xl_s13b_b90k) + fine-tuned clip_state_dict ...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    # z_star uses THIS checkpoint's own fine-tuned text tower (both towers were
    # fine-tuned jointly here, so the goal embedding must come from the same model
    # instance -- a separately-loaded frozen CLIP text tower would not match this
    # checkpoint's fine-tuned embedding space).
    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_star = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)
    log.info(f"z_star shape: {z_star.shape}\n")

    # -----------------------------------------------------------------------
    # 3. Load LatentDynamicsModel
    # -----------------------------------------------------------------------
    model = LatentDynamicsModel(horizon=action_horizon).to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    # -----------------------------------------------------------------------
    # 4. Action stats from training pkls (for denormalization before env.step)
    # -----------------------------------------------------------------------
    log.info(f"Loading action stats from {TRAIN_DIR} ...")
    all_actions = []
    for pkl_file in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl")):
        with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
            ep = pickle.load(f)
        all_actions.append(np.array(ep["actions"], dtype=np.float32))
    all_actions = np.concatenate(all_actions, axis=0)
    lo = torch.tensor(all_actions.min(axis=0), dtype=torch.float32)  # (2,)
    hi = torch.tensor(all_actions.max(axis=0), dtype=torch.float32)  # (2,)
    log.info(f"  lo={lo.tolist()}  hi={hi.tolist()}\n")

    # -----------------------------------------------------------------------
    # 5. Init LangTable env
    # -----------------------------------------------------------------------
    log.info("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env(
        {"ood": False, "block_combo": [BLOCK_A, BLOCK_B]},
        seed=42,
    )
    log.info("Env ready.\n")

    # -----------------------------------------------------------------------
    # 5b. Force the peg to start collinear with, and just past, BLOCK_A —
    # i.e. peg -> green_cube -> blue_moon in a straight line, so a single push
    # accomplishes the task instead of needing to navigate first.
    # (Unchanged from mppi_rollout_clip_full_finetune.py, per instruction.)
    # -----------------------------------------------------------------------
    from language_table.environments.utils.pose3d import Pose3d
    from scipy.spatial.transform import Rotation as R

    inner = env.env  # GymWrapper.__getattr__ proxies unknown attrs straight through to the
                     # inner LanguageTableDataGeneration, so no third .env is needed/exists
                     # (verified directly: env.env.env raises AttributeError, env.env works)
    blocks = inner.get_block_states()
    green_xy = blocks[BLOCK_A]
    blue_xy = blocks[BLOCK_B]

    away_from_blue_direction = (green_xy - blue_xy) / np.linalg.norm(green_xy - blue_xy)
    peg_xy = green_xy + away_from_blue_direction * 0.06  # continue PAST green_cube, away from
                                                          # blue_moon — not between them

    pose = Pose3d(
        rotation=R.from_rotvec([0, np.pi, 0]),  # same fixed downward-facing rotation the
                                                 # internal random reset uses for the effector
        translation=np.array([peg_xy[0], peg_xy[1], 0.145]),  # 0.145 = EFFECTOR_HEIGHT
    )
    inner._set_robot_target_effector_pose(pose)
    inner._step_simulation_to_stabilize()

    dist_peg_green = np.linalg.norm(peg_xy - green_xy)
    dist_peg_blue = np.linalg.norm(peg_xy - blue_xy)
    dist_green_blue = np.linalg.norm(green_xy - blue_xy)
    log.info(f"Forced peg start: {peg_xy}  ({BLOCK_A}={green_xy}, {BLOCK_B}={blue_xy})")
    log.info(f"  dist(peg,{BLOCK_A})={dist_peg_green:.4f}  dist(peg,{BLOCK_B})={dist_peg_blue:.4f}  "
             f"dist({BLOCK_A},{BLOCK_B})={dist_green_blue:.4f}\n")
    assert dist_peg_blue > dist_green_blue, (
        f"peg ended up between {BLOCK_A} and {BLOCK_B}, not on the outside past {BLOCK_A} — geometry bug"
    )

    # -----------------------------------------------------------------------
    # 6. MPPI dynamics_fn and cost_fn
    # -----------------------------------------------------------------------

    # dynamics_fn is called once per CEM iteration; stash its output from the
    # last call so we can inspect the model's predicted z_pred batch right
    # after planner.command() returns, instead of discarding it.
    _last_z_pred_batch = {"value": None}

    def dynamics_fn(state, action):
        # state:  (K, 768)              — batch of latent obs
        # action: (K, action_horizon, 2) — normalized action chunks
        z_pred_batch = model(state, action)   # (K, 768)
        _last_z_pred_batch["value"] = z_pred_batch.detach()
        return z_pred_batch

    def cost_fn(terminal_state):
        # terminal_state: (K, 768), z_star: (768,)
        return 1.0 - (terminal_state * z_star).sum(dim=-1)           # (K,)

    # -----------------------------------------------------------------------
    # 7. Init the planner. Both options share every hyperparameter below and both
    # warm-start each call after the first from a shifted version of the prior best
    # chunk, so an mppi-vs-cem A/B isolates exactly one variable: how the sampling
    # distribution is refit (softmax over all K samples vs. hard top-Ne truncation).
    #
    #   mppi -- ported from swm/planning_algos.py's plan_model_mppi: a Gaussian CEM
    #           with softmax reward-weighting.
    #   cem  -- planning/cem_core.py: textbook CEM, elite mean/std refit. return_mode
    #           is deliberately left at its "best" default; "mean" would make the last
    #           dynamics_fn call a batch of 1, collapsing the cos(predK,z*) min/mean/max
    #           diagnostic column below into a single repeated value.
    # -----------------------------------------------------------------------
    if planner_name == "cem":
        planner = CEM(
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            pred_horizon=action_horizon,
            action_dim=2,
            num_samples=num_samples,
            n_planning_itrs=n_evals - 1,
            device=DEVICE,
            num_elites=num_elites,   # None -> elite_frac=0.1 -> Ne=0.1*num_samples
            max_action_value=1.0,
            warm_start_std=warm_start_std,
            n_execute=n_execute,     # keeps the warm-start shift aligned with what we execute
        )
        log.info(f"CEM: K={num_samples}  num_elites={planner.num_elites}  "
                 f"evals={n_evals} (n_planning_itrs={planner.n_planning_itrs})  "
                 f"max_action_value=1.0  warm_start_std={planner.warm_start_std}  "
                 f"n_execute={planner.n_execute}  "
                 f"return_mode={planner.return_mode}  min_std={planner.min_std}  "
                 f"seed={'ambient RNG' if seed is None else seed}\n")
    else:
        planner = MPPI(
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            pred_horizon=action_horizon,
            action_dim=2,
            num_samples=num_samples,
            mppi_temperature=mppi_temperature,
            n_planning_itrs=n_evals - 1,
            device=DEVICE,
            max_action_value=1.0,
            warm_start_std=warm_start_std,
            n_execute=n_execute,     # keeps the warm-start shift aligned with what we execute
        )
        log.info(f"MPPI: K={num_samples}  mppi_temperature={mppi_temperature}  "
                 f"evals={n_evals} (n_planning_itrs={planner.n_planning_itrs})  "
                 f"max_action_value=1.0  warm_start_std={planner.warm_start_std}  "
                 f"n_execute={planner.n_execute}  "
                 f"seed={'ambient RNG' if seed is None else seed}\n")

    # -----------------------------------------------------------------------
    # 8. Rollout loop — dynamics-model debug diagnostics every planning step
    #
    # The planner returns an action_horizon-sub-step chunk but the caller chooses how many of
    # those sub-steps to actually execute before re-planning (n_execute). max_steps stays
    # denominated in ENV steps, so every arm covers the same amount of simulator time
    # regardless of n_execute -- n_execute=4 does 75 plans x 4 actions, not 300 plans.
    # -----------------------------------------------------------------------
    frames = []
    pybullet_states = []   # inner.get_pybullet_state() dict per frame, index-aligned with frames
    dist_a_b_history = []

    def world_distances():
        """Ground-truth world-frame distances straight from the simulator (not CLIP-based):
        peg-to-BLOCK_A and BLOCK_A-to-BLOCK_B. Same calls as the peg-forcing setup above
        (inner.get_block_states(), inner._robot.forward_kinematics()), re-queried live."""
        blocks_now = inner.get_block_states()
        peg_now = np.array(inner._robot.forward_kinematics().translation[:2], dtype=np.float32)
        return (float(np.linalg.norm(peg_now - blocks_now[BLOCK_A])),
                float(np.linalg.norm(blocks_now[BLOCK_A] - blocks_now[BLOCK_B])))

    n_plans = max_steps // n_execute
    log.info(f"Execution cadence: n_execute={n_execute} -> {n_plans} plans x {n_execute} "
             f"actions = {n_plans * n_execute} env steps (max_steps={max_steps})")
    if cold_start:
        warm_desc = ("DISABLED (cold_start) -- planner.reset() before every plan, so each draws "
                     "a fresh Uniform(-1,1) population (std 0.577); warm_start_std is inert")
    else:
        warm_desc = f"enabled, std={warm_start_std}, shifted forward by n_execute={n_execute}"
    log.info(f"Warm start: {warm_desc}\n")

    log.info(
        f"{'step':>4}  {'cos(zt,z*)':>10}  {'cos(zt,zt-1)':>12}  |  "
        f"{'cos(predK,z*) min/mean/max':>27}  {'cos(pred_sel,z*)':>16}  {'cos(zpred0,z*)':>15}  {'||a_raw||':>9}  |  "
        f"{'dist(peg,A)':>11}  {'dist(A,B)':>10}"
    )
    log.info("-" * 175)

    with torch.no_grad():
        z_prev = None

        # Sample the starting state once; every subsequent sample happens after an env
        # step inside the inner loop, so dist_a_b_history covers EVERY env step. Sampling
        # only once per plan would let the success criterion (min over the episode) miss a
        # close approach that occurred on one of the intermediate, un-sampled sub-steps.
        dist_a_b_history.append(world_distances()[1])

        for plan_idx in range(n_plans):
            step = plan_idx * n_execute      # env-step index, so the column stays comparable
                                             # across runs with different n_execute

            # Encode current observation
            pil_frame = env.get_frame()                                           # PIL Image
            rgb_np    = np.array(pil_frame)                                       # (H, W, 3)
            frames.append(rgb_np)
            pybullet_states.append(inner.get_pybullet_state())

            img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
            z_t   = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)

            # ---- Real / ground truth ----
            cos_t_star = (z_t * z_star).sum().item()
            # NOTE at n_execute > 1 this compares frames n_execute env steps apart, not
            # adjacent ones, so it reads lower than in an n_execute=1 run by construction.
            cos_t_prev = (z_t * z_prev).sum().item() if z_prev is not None else float("nan")

            dist_peg_a_now, dist_a_b_now = world_distances()

            # Plan — returns (action_horizon, 2) normalized action chunk. With cold_start, discard the
            # warm-start state first so this plan searches a fresh Uniform(-1,1) population
            # instead of a narrow band around the previous chunk.
            if cold_start:
                planner.reset()
            action_chunk = planner.command(z_t)

            # ---- Model-internal stats from the K candidate rollout (last planner iter) ----
            z_pred_K = _last_z_pred_batch["value"]                       # (K, 768)
            cos_predK_star = (z_pred_K * z_star.unsqueeze(0)).sum(dim=-1)  # (K,)

            # Model's prediction for the specific action chunk actually chosen
            z_pred_selected = model(z_t.unsqueeze(0), action_chunk.unsqueeze(0)).squeeze(0)
            cos_pred_selected_star = (z_pred_selected * z_star).sum().item()

            # ---- Zero-action baseline: what does the model predict if fed an all-zero
            # action chunk instead of the planned one? Isolates how much of the model's
            # goal-similarity signal comes from the image embedding z_t alone vs. the
            # chosen action -- same probe as eval_3_expert_trajs_0_actions.py. ----
            actions_zero = torch.zeros(1, action_horizon, 2, device=DEVICE)
            z_pred_zero = model(z_t.unsqueeze(0), actions_zero).squeeze(0)
            cos_zero_star = (z_pred_zero * z_star).sum().item()

            # Denormalize the sub-steps we are about to execute back to raw action space.
            # ||a_raw|| logs the FIRST sub-step's norm regardless of n_execute, so the
            # column stays directly comparable with every existing rollouts_5_per/ run.
            chunk_norm = action_chunk[:n_execute].cpu().float()                 # (n_execute, 2)
            chunk_raw  = (chunk_norm + 1.0) / 2.0 * (hi - lo) + lo              # (n_execute, 2)
            a_raw_norm = chunk_raw[0].norm().item()

            log.info(
                f"{step:>4}  {cos_t_star:>10.4f}  {cos_t_prev:>12.4f}  |  "
                f"{cos_predK_star.min().item():>8.4f}/{cos_predK_star.mean().item():>8.4f}/{cos_predK_star.max().item():>8.4f}  "
                f"{cos_pred_selected_star:>16.4f}  {cos_zero_star:>15.4f}  {a_raw_norm:>9.4f}  |  "
                f"{dist_peg_a_now:>11.4f}  {dist_a_b_now:>10.4f}"
            )

            # Execute n_execute sub-steps open-loop before re-planning. One frame and one
            # dist sample per ENV step, so the mp4 stays real-time and the success
            # criterion sees every intermediate state.
            for j in range(n_execute):
                if j > 0:
                    frames.append(np.array(env.get_frame()))
                    pybullet_states.append(inner.get_pybullet_state())
                env.step(chunk_raw[j].numpy())
                dist_a_b_history.append(world_distances()[1])

            z_prev = z_t

    # -----------------------------------------------------------------------
    # 9. Success/failure + save video, result json
    # -----------------------------------------------------------------------
    min_dist_a_b = float(min(dist_a_b_history))
    success = min_dist_a_b <= SUCCESS_DIST_THRESHOLD
    outcome = "success" if success else "failure"

    out_path = os.path.join(out_dir, f"{stem}_{outcome}.mp4")
    imageio.mimsave(out_path, frames, fps=10)
    log.info(f"\nSaved {len(frames)} frames -> {out_path}")

    assert len(pybullet_states) == len(frames), (
        f"pybullet_states ({len(pybullet_states)}) and frames ({len(frames)}) length mismatch -- "
        f"they must stay index-aligned so a frame number is also a valid states.pkl index"
    )
    states_path = os.path.join(out_dir, f"{stem}_pybullet_states.pkl")
    with open(states_path, "wb") as f:
        pickle.dump(pybullet_states, f)
    log.info(f"Saved {len(pybullet_states)} pybullet states -> {states_path}")

    log.info(f"min dist({BLOCK_A},{BLOCK_B}) over episode = {min_dist_a_b:.4f}  ->  {outcome.upper()}")
    log.info(f"Full log written to {log_path}")

    result_path = os.path.join(out_dir, f"{stem}_result.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "epoch": epoch,
                "rollout_idx": rollout_idx,
                "min_dist_a_b": min_dist_a_b,
                "success": success,
                "video_path": out_path,
                "pybullet_states_path": states_path,
            },
            f,
            indent=2,
        )
    log.info(f"Result json -> {result_path}")


def discover_epoch_checkpoints(ckpt_dir: str) -> list:
    """Returns a sorted list of (epoch:int, ckpt_path:str) for every epoch_*.pt in ckpt_dir."""
    pattern = os.path.join(ckpt_dir, "epoch_*.pt")
    results = []
    for path in glob.glob(pattern):
        m = re.search(r"epoch_(\d+)\.pt$", os.path.basename(path))
        if m:
            results.append((int(m.group(1)), path))
    results.sort(key=lambda t: t[0])
    return results


def write_summary(subfolder_dir: str, subfolder_name: str, ckpt_dir: str) -> None:
    """Aggregates every *_result.json under subfolder_dir into one summary.txt, broken
    down by epoch."""
    result_paths = sorted(
        glob.glob(os.path.join(subfolder_dir, "epoch*", "*_result.json"))
    )
    by_epoch = {}
    for rp in result_paths:
        with open(rp) as f:
            r = json.load(f)
        by_epoch.setdefault(r["epoch"], []).append(r)

    lines = []
    lines.append(f"Summary: {subfolder_name} ({ckpt_dir})")
    lines.append(
        f"Success criterion: dist({BLOCK_A}, {BLOCK_B}) <= {SUCCESS_DIST_THRESHOLD} "
        f"at any point in the episode"
    )
    lines.append("=" * 80)
    lines.append("")

    total_success = 0
    total_count = 0
    for epoch in sorted(by_epoch.keys()):
        rollouts = sorted(by_epoch[epoch], key=lambda r: r["rollout_idx"])
        n_success = sum(1 for r in rollouts if r["success"])
        n_total = len(rollouts)
        total_success += n_success
        total_count += n_total
        pct = 100.0 * n_success / n_total if n_total else float("nan")
        lines.append(f"Epoch {epoch}: {n_success}/{n_total} success ({pct:.1f}%)")
        for r in rollouts:
            outcome = "SUCCESS" if r["success"] else "FAILURE"
            lines.append(
                f"  rollout {r['rollout_idx']}: {outcome}  min_dist={r['min_dist_a_b']:.4f}"
            )
        lines.append("")

    overall_pct = 100.0 * total_success / total_count if total_count else float("nan")
    lines.append(f"Overall: {total_success}/{total_count} success ({overall_pct:.1f}%)")

    summary_path = os.path.join(subfolder_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir", type=str, default=None,
        help="checkpoint directory to sweep (e.g. "
             "main_project/training/checkpoints_clip_full_finetuned_wider_hl); runs "
             "--n_rollouts rollouts for every epoch_*.pt found in it",
    )
    parser.add_argument("--n_rollouts", type=int, default=N_ROLLOUTS_DEFAULT)
    parser.add_argument("--max_steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument(
        "--epochs", type=int, nargs="+", default=None,
        help="restrict to these epoch numbers (e.g. --epochs 11). Default: every epoch_*.pt "
             "found in --ckpt_dir.",
    )
    parser.add_argument(
        "--planner", type=str, choices=["mppi", "cem"], default="mppi",
        help="which planner to run: 'mppi' (default) = planning/mppi_core.py's softmax-weighted "
             "Gaussian CEM, the algorithm every existing rollouts_5_per/ result used; 'cem' = "
             "planning/cem_core.py's textbook top-Ne elite CEM. All shared hyperparameters are "
             "identical between the two, so the arms differ only in the refit rule. --tag is "
             "required with 'cem'.",
    )
    parser.add_argument(
        "--num_elites", type=int, default=None,
        help="CEM only (ignored by --planner mppi): elite count Ne. Default None uses "
             "cem_core's elite_frac=0.1, i.e. Ne=51 at K=512.",
    )
    parser.add_argument(
        "--cold_start", action="store_true",
        help="call planner.reset() before EVERY planning call, so each plan draws a fresh "
             "Uniform(-1,1) population (std 0.577) with no memory of the previous chunk -- true "
             "open-loop chunk planning. Makes --warm_start_std inert. Note --n_execute 8 does NOT "
             "do this: rolling an 8-row chunk by 8 zeroes every row, leaving Normal(0, "
             "warm_start_std), a measured std of 0.100 vs 0.577 cold -- the tightest do-nothing "
             "prior of any --n_execute, not the absence of one. Requires --tag.",
    )
    parser.add_argument(
        "--evals", type=int, default=EVALS_DEFAULT,
        help=f"total dynamics evaluations per planning call; the planner gets "
             f"n_planning_itrs = evals - 1 (default {EVALS_DEFAULT}, the value every existing "
             f"rollouts_5_per/ result used). Same meaning as eval/mppi_iteration_sweep.py's "
             f"--evals. Accepted by both planners. THIS DOMINATES WALL-CLOCK: ~0.6 ms per "
             f"evaluation at K=512, so --evals 10000 is ~6 s per planning call, ~30 min per "
             f"300-step rollout. Requires --tag when non-default.",
    )
    parser.add_argument(
        "--warm_start_std", type=float, default=WARM_START_STD_DEFAULT,
        help=f"per-dimension std of the population sampled around the shifted previous chunk on "
             f"every planning call after the first (default {WARM_START_STD_DEFAULT}, the value "
             f"every existing rollouts_5_per/ result used). Accepted by both planners. A "
             f"cold-start draw is Uniform(-1,1), std 0.577, so values approaching that make the "
             f"warm start as wide as a fresh draw. Requires --tag when non-default.",
    )
    parser.add_argument(
        "--n_execute", type=int, default=1,
        help="how many sub-steps of each action-horizon-step chunk to execute open-loop before re-planning "
             "(default 1, the value every existing rollouts_5_per/ result used). --max_steps "
             "stays in ENV steps, so --n_execute 4 runs max_steps//4 plans over the same "
             "simulator time. Supported by both planners -- each receives n_execute so its "
             "warm start shifts by the same amount the caller actually executed. Must divide "
             "--max_steps evenly, and requires --tag.",
    )
    parser.add_argument(
        "--mppi_temperature", type=float, default=MPPI_TEMPERATURE_DEFAULT,
        help=f"MPPI planner only (ignored by --planner cem): CEM softmax temperature "
             f"(default {MPPI_TEMPERATURE_DEFAULT}, the value all existing rollouts_5_per/ "
             f"results used).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=K_DEFAULT,
        help=f"number of candidate action-chunk samples per planning call (default "
             f"{K_DEFAULT}, the value every existing rollouts_5_per/ result used). Both "
             f"planners accept it. Requires --tag when non-default.",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="output subfolder suffix, e.g. --tag temp0.1 writes to expert+subopt_temp0.1/ "
             "instead of expert+subopt/. Use it whenever changing a planner hyperparameter, so "
             "the new run neither overwrites earlier results nor is skipped by the "
             "already-done check below.",
    )
    parser.add_argument(
        "--seed_base", type=int, default=None,
        help="if set, rollout i is seeded with seed_base + i, making each rollout individually "
             "reproducible (the rollouts still differ from one another). Default: unseeded, "
             "matching how the existing baselines were generated.",
    )

    # Internal worker-mode flags (used by subprocess re-invocations of this script).
    parser.add_argument("--ckpt_path", type=str, default=None,
                         help="internal: run a single rollout for this checkpoint")
    parser.add_argument("--epoch", type=int, default=None, help="internal: epoch number")
    parser.add_argument("--rollout_idx", type=int, default=None, help="internal: rollout index")
    parser.add_argument("--out_dir", type=str, default=None, help="internal: output directory")
    parser.add_argument("--seed", type=int, default=None, help="internal: this rollout's seed")
    args = parser.parse_args()

    if args.ckpt_path is not None:
        # Worker mode: run exactly one rollout in this process.
        run_rollout(args.ckpt_path, args.epoch, args.rollout_idx, args.out_dir, args.max_steps,
                    mppi_temperature=args.mppi_temperature, seed=args.seed,
                    planner_name=args.planner, num_elites=args.num_elites,
                    n_execute=args.n_execute, warm_start_std=args.warm_start_std,
                    n_evals=args.evals, cold_start=args.cold_start,
                    num_samples=args.num_samples)
        return

    # Orchestrator mode.
    if args.ckpt_dir is None:
        parser.error("--ckpt_dir is required in orchestrator mode")

    ckpt_dir = os.path.abspath(args.ckpt_dir)
    basename = os.path.basename(os.path.normpath(ckpt_dir))
    subfolder_name = CKPT_DIR_TO_SUBFOLDER.get(basename)
    if subfolder_name is None:
        raise ValueError(
            f"Unrecognized --ckpt_dir basename '{basename}'. Known dirs: "
            f"{list(CKPT_DIR_TO_SUBFOLDER.keys())}. Add it to CKPT_DIR_TO_SUBFOLDER "
            f"if this is a new checkpoint directory."
        )

    epoch_checkpoints = discover_epoch_checkpoints(ckpt_dir)
    if not epoch_checkpoints:
        raise ValueError(f"No epoch_*.pt files found in {ckpt_dir}")

    if args.epochs is not None:
        wanted = set(args.epochs)
        found = {e for e, _ in epoch_checkpoints}
        missing = wanted - found
        if missing:
            raise ValueError(
                f"--epochs requested {sorted(missing)}, which have no epoch_*.pt in {ckpt_dir}. "
                f"Available: {sorted(found)}"
            )
        epoch_checkpoints = [(e, p) for e, p in epoch_checkpoints if e in wanted]

    # Changing the planner is the strongest form of "changed a planner hyperparameter", and
    # without a tag it is silently WRONG rather than merely messy: the output dir would be the
    # untagged one, whose _result.json files already exist, so the already-done check below
    # would skip every rollout and write_summary() would then re-report the old MPPI numbers as
    # though they were this CEM run. Fail loudly instead.
    if args.n_execute < 1:
        parser.error(f"--n_execute must be >= 1, got {args.n_execute}")
    if args.evals < 1:
        parser.error(f"--evals must be >= 1 (1 = initial evaluation only), got {args.evals}")
    if args.num_samples < 1:
        parser.error(f"--num_samples must be >= 1, got {args.num_samples}")
    if args.max_steps % args.n_execute != 0:
        parser.error(
            f"--max_steps {args.max_steps} is not divisible by --n_execute {args.n_execute}, "
            f"so the episode would cover {args.max_steps // args.n_execute * args.n_execute} "
            f"env steps instead of {args.max_steps} and would not be comparable with the "
            f"other arms. Pick a divisible pair."
        )

    changed = [d for d in (
        f"--planner {args.planner}" if args.planner != "mppi" else None,
        f"--n_execute {args.n_execute}" if args.n_execute > 1 else None,
        f"--evals {args.evals}" if args.evals != EVALS_DEFAULT else None,
        "--cold_start" if args.cold_start else None,
        f"--warm_start_std {args.warm_start_std}"
        if args.warm_start_std != WARM_START_STD_DEFAULT else None,
        f"--mppi_temperature {args.mppi_temperature}"
        if args.planner == "mppi" and args.mppi_temperature != MPPI_TEMPERATURE_DEFAULT else None,
        f"--num_samples {args.num_samples}" if args.num_samples != K_DEFAULT else None,
    ) if d]
    if changed and not args.tag:
        parser.error(
            f"{', '.join(changed)} changes a planner hyperparameter, which requires --tag. "
            f"Without it this run would write into the untagged results directory, where every "
            f"rollout would be skipped as already-done and the summary would report the existing "
            f"baseline results instead of this run's."
        )

    # A tag keeps a hyperparameter-changed run out of the untagged results' directory. Without
    # it, the already-done check below would skip every rollout whose _result.json exists and
    # the run would silently do nothing.
    if args.tag:
        subfolder_name = f"{subfolder_name}_{args.tag}"
    subfolder_dir = os.path.join(ROLLOUTS_ROOT, subfolder_name)
    os.makedirs(subfolder_dir, exist_ok=True)

    print(f"=== {subfolder_name}: {len(epoch_checkpoints)} epochs x {args.n_rollouts} "
          f"rollouts each, from {ckpt_dir} ===")
    seeds_desc = ("ambient RNG" if args.seed_base is None
                  else str([args.seed_base + i for i in range(args.n_rollouts)]))
    planner_desc = (f"planner=cem  num_elites={args.num_elites if args.num_elites is not None else 'elite_frac=0.1'}"
                    if args.planner == "cem"
                    else f"planner=mppi  mppi_temperature={args.mppi_temperature}")
    n_plans_total = (args.max_steps // args.n_execute) * len(epoch_checkpoints) * args.n_rollouts
    warm_desc = ("warm_start=COLD (reset every plan)" if args.cold_start
                 else f"warm_start_std={args.warm_start_std}")
    print(f"    {planner_desc}  num_samples={args.num_samples}  {warm_desc}  "
          f"evals={args.evals} (n_planning_itrs={args.evals - 1})  "
          f"max_steps={args.max_steps}  n_execute={args.n_execute} "
          f"({args.max_steps // args.n_execute} plans/rollout)  seeds={seeds_desc}")
    # Wall-clock warning: at K=512 one evaluation costs ~0.6 ms, so the planning budget is the
    # whole cost of a large-evals run. Scale that baseline by num_samples/K_DEFAULT since batch
    # size (K) is the dominant driver of per-eval cost. Print an estimate up front rather than
    # letting someone discover mid-run that they launched a multi-hour job.
    per_eval_sec = 6e-4 * (args.num_samples / K_DEFAULT)
    est_hours = n_plans_total * args.evals * per_eval_sec / 3600.0
    if est_hours >= 0.25:
        print(f"    ESTIMATED PLANNING TIME: ~{est_hours:.1f} h "
              f"({n_plans_total} plans x {args.evals} evals x ~{per_eval_sec * 1000:.2f} ms "
              f"[scaled from ~0.6 ms/eval @ K={K_DEFAULT}]), excluding per-rollout CLIP/pkl "
              f"load. Ctrl-C now if that is not what you intended.")
    print(f"    output -> {subfolder_dir}")

    for epoch, ckpt_path in epoch_checkpoints:
        epoch_dir = os.path.join(subfolder_dir, f"epoch{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        for rollout_idx in range(args.n_rollouts):
            result_path = os.path.join(epoch_dir, f"epoch{epoch}_rollout{rollout_idx}_result.json")
            if os.path.exists(result_path):
                print(f"--- epoch {epoch} rollout {rollout_idx}: already done, skipping "
                      f"(found {result_path}) ---")
                continue
            print(f"--- epoch {epoch} rollout {rollout_idx} ({ckpt_path}) ---")
            cmd = [
                sys.executable, os.path.abspath(__file__),
                "--ckpt_path", ckpt_path,
                "--epoch", str(epoch),
                "--rollout_idx", str(rollout_idx),
                "--out_dir", epoch_dir,
                "--max_steps", str(args.max_steps),
                "--mppi_temperature", str(args.mppi_temperature),
                "--planner", args.planner,
                "--n_execute", str(args.n_execute),
                "--warm_start_std", str(args.warm_start_std),
                "--evals", str(args.evals),
                "--num_samples", str(args.num_samples),
            ]
            if args.cold_start:
                cmd += ["--cold_start"]      # bare store_true flag, not a key/value pair
            if args.num_elites is not None:
                cmd += ["--num_elites", str(args.num_elites)]
            if args.seed_base is not None:
                cmd += ["--seed", str(args.seed_base + rollout_idx)]
            subprocess.run(cmd, check=True)

    write_summary(subfolder_dir, subfolder_name, ckpt_dir)
    print(f"Done. Rollouts saved under {subfolder_dir}")


if __name__ == "__main__":
    main()
