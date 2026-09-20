"""
Batch MPPI rollout evaluation with RANDOM starting scenes: N independent rollouts per epoch
checkpoint, each starting from its own randomly-sampled (start_block, target_block) pair and
randomly-placed peg/blocks, instead of the fixed (green_cube, blue_moon) + forced near-contact
peg placement used by mppi_rollout_5_per_checkpoint.py (the "source script", copied verbatim
here as the starting point).

Everything about the MPPI/CEM planning loop, CLIP loading, LatentDynamicsModel/FiLMLayer/
ActionEncoderConv class defs (including the source script's already-existing per-checkpoint
horizon auto-detection -- action_encoder.out.weight's shape tells you whether a checkpoint was
trained on 8-step or 16-step action chunks, so this also runs horizon-16 checkpoints with no
extra flag), action-stats computation, the dynamics_fn/cost_fn closures, both planner
constructions, the per-step diagnostic logging table, mp4/pybullet-state saving, the
already-done skip check, and every CLI flag is UNCHANGED from the source script. See that
script's docstring for what each flag does.

What's different here (the actual point of this script):

  - The source script's section "5b" (force the peg to a specific staged position just past
    BLOCK_A, collinear with BLOCK_B) is DELETED. LanguageTable.reset() -- called once inside
    get_lang_table_env() -- already does a fully random, SAFE peg + block placement on its own:
    _reset_poses_randomly() (language_table/environments/language_table.py) parks every block
    off-table, places the peg via a direct pose command onto an EMPTY table (no sweep-through-
    block risk -- see [[project_langtable_peg_teleport_staged]] for why a teleport INTO contact
    range is unsafe; this isn't that), then rejection-samples each block's pose against both
    other-block and arm-distance thresholds before placing it directly (no physics sweep at
    all). So genuinely random + safe scene generation was already built into the env; we just
    have to stop overriding it.
  - block_combo (passed into LanguageTableDataGeneration's constructor) only ever reaches
    reward_factory, which is always NoOpReward here -- its reset() ignores block_combo and its
    reward() always returns (0, False). So block_combo has zero effect on env mechanics; no
    change needed there.
  - --seed_base now DEFAULTS TO 0 instead of None/ambient, and is load-bearing for the SCENE,
    not just planner noise: rollout i's seed = seed_base + i drives THREE things --
    get_lang_table_env(..., seed=seed) (block/peg placement, was hardcoded 42 in the source
    script), a local random.Random(seed) instance (picks this rollout's (start_block,
    target_block) pair from whichever 8 blocks the seeded reset actually placed), and
    torch.manual_seed(seed) (MPPI/CEM sampling, as in the source script). One seed fully
    specifies one reproducible scene + task + plan draw.
  - goal_text, world_distances(), and the success criterion generalize from hardcoded
    BLOCK_A/BLOCK_B to this rollout's sampled start_block/target_block. No minimum-separation
    filter on the pair -- fully random, so some scenes may start already close together.
  - Each *_result.json gains start_block, target_block, goal_text, seed. write_summary() prints
    each rollout's block pair + goal text next to its success/min_dist line.

Run (one invocation per checkpoint directory):
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_rollout_random_scenes.py \\
        --ckpt_dir main_project/training/checkpoints_clip_full_fine-tuned_with_subopt_data --epochs 8

Each rollout runs in its own subprocess (re-invokes this script with --ckpt_path/--epoch/
--rollout_idx/--out_dir), same reason as the source script: pybullet only supports one physics
client per process.
"""

import sys
import os
import re
import glob
import json
import random
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
MAX_STEPS_DEFAULT = 300
N_ROLLOUTS_DEFAULT = 10   # this script's whole point is scene diversity, so default to more
                          # than the source script's 5
K_DEFAULT = 512   # num_samples default
SUCCESS_DIST_THRESHOLD = 0.05

MPPI_TEMPERATURE_DEFAULT = 0.01
WARM_START_STD_DEFAULT = 0.1
EVALS_DEFAULT = 10

TRAIN_DIR    = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLOUTS_ROOT = os.path.join(MAIN_PROJECT_DIR, "rollouts_5_per")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# Maps a --ckpt_dir's basename to the output subfolder name under ROLLOUTS_ROOT. Copied
# verbatim from the source script, including the h16 basename-collision comment: the same
# basename lives at two physical paths because training was cancelled and resumed elsewhere,
# and both are meant to aggregate into one summary.
CKPT_DIR_TO_SUBFOLDER = {
    "checkpoints_clip_full_fine-tuned_with_subopt_data": "expert+subopt",
    "checkpoints_clip_full_finetuned_wider_hl": "expert-only",
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
    """Conv1d action encoder: treats the 2 action dims as channels and the horizon timesteps
    as the sequence axis, so neighboring timesteps can interact via the kernel before the
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
                 seed: int = 0, planner_name: str = "mppi",
                 num_elites: int | None = None, n_execute: int = 1,
                 warm_start_std: float = WARM_START_STD_DEFAULT,
                 n_evals: int = EVALS_DEFAULT, cold_start: bool = False,
                 num_samples: int = K_DEFAULT) -> None:
    """Runs one full planning rollout FROM A RANDOM SCENE with the given checkpoint and saves
    an .mp4 + result json + log into out_dir. Everything below is process-local (CLIP, env,
    model, planner) — safe to call exactly once per process.

    seed: drives THREE things (see module docstring) -- the env's block/peg placement, the
        (start_block, target_block) pair choice, and torch's MPPI/CEM sampling RNG. Unlike the
        source script, this is never None: one seed fully specifies one reproducible scene +
        task + plan draw, and the orchestrator always passes seed_base + rollout_idx.

    See mppi_rollout_5_per_checkpoint.py's run_rollout docstring for every other parameter --
    planner_name, num_elites, n_execute, warm_start_std, n_evals, cold_start, mppi_temperature,
    num_samples are all unchanged in meaning."""

    os.makedirs(out_dir, exist_ok=True)
    stem = f"epoch{epoch}_rollout{rollout_idx}"

    torch.manual_seed(seed)
    rng = random.Random(seed)   # dedicated to picking (start_block, target_block) below

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
    log.info(f"Checkpoint dir epoch: {epoch}  Rollout: {rollout_idx}  Seed: {seed}")
    log.info(f"Logging to {log_path}\n")

    # -----------------------------------------------------------------------
    # 1. Checkpoint + CLIP (image + text encoder), both towers fully fine-tuned --
    # loaded as a plain full state dict, no LoRA adapter involved. goal_text/z_star are
    # deferred until after the env is constructed and a random block pair is chosen (step 4b
    # below), since they depend on which blocks the random scene actually placed.
    # -----------------------------------------------------------------------
    log.info(f"Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    log.info(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}\n")

    # Action horizon is a property of the checkpoint's own architecture (how many action-chunk
    # sub-steps the action_encoder was trained on), not a free planning choice -- detect it from
    # the saved weights instead of trusting a CLI flag, so this always matches the checkpoint.
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

    # -----------------------------------------------------------------------
    # 2. Load LatentDynamicsModel
    # -----------------------------------------------------------------------
    model = LatentDynamicsModel(horizon=action_horizon).to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    # -----------------------------------------------------------------------
    # 3. Action stats from training pkls (for denormalization before env.step)
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
    # 4. Init LangTable env -- seeded per-rollout, so each of the N rollouts gets its own
    # random block/peg layout via LanguageTable.reset()'s own _reset_poses_randomly() (already
    # safe: peg is placed on an empty table, blocks are then rejection-sampled against both
    # each other and the peg -- see module docstring). block_combo is passed for signature
    # compatibility only; NoOpReward ignores it, so it has no effect on the placed layout.
    # -----------------------------------------------------------------------
    log.info(f"Initializing LangTable env (seed={seed}) ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env(
        {"ood": False, "block_combo": None},
        seed=seed,
    )
    log.info("Env ready.\n")

    inner = env.env  # GymWrapper.__getattr__ proxies unknown attrs straight through to the
                     # inner LanguageTableDataGeneration, so no third .env is needed/exists
                     # (verified directly: env.env.env raises AttributeError, env.env works)

    # -----------------------------------------------------------------------
    # 4b. Random (start_block, target_block), chosen from whichever 8 blocks the seeded reset
    # actually placed -- NOT forced into any particular geometry. No minimum-separation filter:
    # fully random, so some scenes may start already close together.
    # -----------------------------------------------------------------------
    blocks_now = inner.get_block_states()
    block_names = sorted(k for k in blocks_now if k != "peg")
    start_block, target_block = rng.sample(block_names, 2)

    start_xy = blocks_now[start_block]
    target_xy = blocks_now[target_block]
    peg_xy = blocks_now["peg"]
    log.info(f"Random scene: start_block={start_block}  target_block={target_block}")
    log.info(f"  {start_block}={start_xy}  {target_block}={target_xy}  peg={peg_xy}")
    log.info(f"  dist(peg,{start_block})={np.linalg.norm(peg_xy - start_xy):.4f}  "
             f"dist({start_block},{target_block})={np.linalg.norm(start_xy - target_xy):.4f}\n")

    goal_text = (
        f"the {start_block.replace('_', ' ')} is touching the {target_block.replace('_', ' ')}. "
        f"the peg is touching the {start_block.replace('_', ' ')}."
    )
    log.info(f"Goal (fixed, steps 0-{max_steps - 1}): {goal_text}")

    # z_star uses THIS checkpoint's own fine-tuned text tower (both towers were fine-tuned
    # jointly here, so the goal embedding must come from the same model instance -- a
    # separately-loaded frozen CLIP text tower would not match this checkpoint's fine-tuned
    # embedding space).
    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_star = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)
    log.info(f"z_star shape: {z_star.shape}\n")

    # -----------------------------------------------------------------------
    # 5. MPPI dynamics_fn and cost_fn
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
    # 6. Init the planner. Both options share every hyperparameter below and both
    # warm-start each call after the first from a shifted version of the prior best
    # chunk, so an mppi-vs-cem A/B isolates exactly one variable: how the sampling
    # distribution is refit (softmax over all K samples vs. hard top-Ne truncation).
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
                 f"return_mode={planner.return_mode}  min_std={planner.min_std}  seed={seed}\n")
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
                 f"n_execute={planner.n_execute}  seed={seed}\n")

    # -----------------------------------------------------------------------
    # 7. Rollout loop — dynamics-model debug diagnostics every planning step
    #
    # The planner returns an action_horizon-sub-step chunk but the caller chooses how many of
    # those sub-steps to actually execute before re-planning (n_execute). max_steps stays
    # denominated in ENV steps, so every arm covers the same amount of simulator time
    # regardless of n_execute -- n_execute=4 does 75 plans x 4 actions, not 300 plans.
    # -----------------------------------------------------------------------
    frames = []
    pybullet_states = []   # inner.get_pybullet_state() dict per frame, index-aligned with frames
    dist_start_target_history = []

    def world_distances():
        """Ground-truth world-frame distances straight from the simulator (not CLIP-based):
        peg-to-start_block and start_block-to-target_block, re-queried live."""
        blocks_now = inner.get_block_states()
        peg_now = np.array(inner._robot.forward_kinematics().translation[:2], dtype=np.float32)
        return (float(np.linalg.norm(peg_now - blocks_now[start_block])),
                float(np.linalg.norm(blocks_now[start_block] - blocks_now[target_block])))

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
        f"{'dist(peg,start)':>15}  {'dist(start,target)':>18}"
    )
    log.info("-" * 175)

    with torch.no_grad():
        z_prev = None

        # Sample the starting state once; every subsequent sample happens after an env
        # step inside the inner loop, so dist_start_target_history covers EVERY env step.
        # Sampling only once per plan would let the success criterion (min over the episode)
        # miss a close approach that occurred on one of the intermediate, un-sampled sub-steps.
        dist_start_target_history.append(world_distances()[1])

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

            dist_peg_start_now, dist_start_target_now = world_distances()

            # Plan — returns (action_horizon, 2) normalized action chunk. With cold_start,
            # discard the warm-start state first so this plan searches a fresh Uniform(-1,1)
            # population instead of a narrow band around the previous chunk.
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
                f"{dist_peg_start_now:>15.4f}  {dist_start_target_now:>18.4f}"
            )

            # Execute n_execute sub-steps open-loop before re-planning. One frame and one
            # dist sample per ENV step, so the mp4 stays real-time and the success
            # criterion sees every intermediate state.
            for j in range(n_execute):
                if j > 0:
                    frames.append(np.array(env.get_frame()))
                    pybullet_states.append(inner.get_pybullet_state())
                env.step(chunk_raw[j].numpy())
                dist_start_target_history.append(world_distances()[1])

            z_prev = z_t

    # -----------------------------------------------------------------------
    # 8. Success/failure + save video, result json
    # -----------------------------------------------------------------------
    min_dist_a_b = float(min(dist_start_target_history))
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

    log.info(f"min dist({start_block},{target_block}) over episode = {min_dist_a_b:.4f}  ->  "
             f"{outcome.upper()}")
    log.info(f"Full log written to {log_path}")

    result_path = os.path.join(out_dir, f"{stem}_result.json")
    with open(result_path, "w") as f:
        json.dump(
            {
                "epoch": epoch,
                "rollout_idx": rollout_idx,
                "seed": seed,
                "start_block": start_block,
                "target_block": target_block,
                "goal_text": goal_text,
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
    down by epoch, including each rollout's randomly-sampled task description."""
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
        f"Success criterion: dist(start_block, target_block) <= {SUCCESS_DIST_THRESHOLD} "
        f"at any point in the episode (block pair is randomized per rollout -- see below)"
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
                f"  rollout {r['rollout_idx']} (seed={r.get('seed', '?')}): {outcome}  "
                f"min_dist={r['min_dist_a_b']:.4f}  "
                f"{r.get('start_block', '?')} -> {r.get('target_block', '?')}"
            )
            if r.get("goal_text"):
                lines.append(f"    goal: {r['goal_text']}")
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
             "--n_rollouts random-scene rollouts for every epoch_*.pt found in it",
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
             "Gaussian CEM; 'cem' = planning/cem_core.py's textbook top-Ne elite CEM. --tag is "
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
             "open-loop chunk planning. Makes --warm_start_std inert. Requires --tag.",
    )
    parser.add_argument(
        "--evals", type=int, default=EVALS_DEFAULT,
        help=f"total dynamics evaluations per planning call; the planner gets "
             f"n_planning_itrs = evals - 1 (default {EVALS_DEFAULT}). Accepted by both "
             f"planners. THIS DOMINATES WALL-CLOCK. Requires --tag when non-default.",
    )
    parser.add_argument(
        "--warm_start_std", type=float, default=WARM_START_STD_DEFAULT,
        help=f"per-dimension std of the population sampled around the shifted previous chunk on "
             f"every planning call after the first (default {WARM_START_STD_DEFAULT}). Requires "
             f"--tag when non-default.",
    )
    parser.add_argument(
        "--n_execute", type=int, default=1,
        help="how many sub-steps of each action-horizon-step chunk to execute open-loop before "
             "re-planning (default 1). --max_steps stays in ENV steps, so --n_execute 4 runs "
             "max_steps//4 plans over the same simulator time. Must divide --max_steps evenly, "
             "and requires --tag.",
    )
    parser.add_argument(
        "--mppi_temperature", type=float, default=MPPI_TEMPERATURE_DEFAULT,
        help=f"MPPI planner only (ignored by --planner cem): CEM softmax temperature "
             f"(default {MPPI_TEMPERATURE_DEFAULT}).",
    )
    parser.add_argument(
        "--num_samples", type=int, default=K_DEFAULT,
        help=f"number of candidate action-chunk samples per planning call (default "
             f"{K_DEFAULT}). Both planners accept it. Requires --tag when non-default.",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="output subfolder suffix, e.g. --tag temp0.1 writes to expert+subopt_temp0.1/ "
             "instead of expert+subopt/. Use it whenever changing a planner hyperparameter, so "
             "the new run neither overwrites earlier results nor is skipped by the "
             "already-done check below.",
    )
    parser.add_argument(
        "--seed_base", type=int, default=0,
        help="rollout i is seeded with seed_base + i (default 0). Unlike the fixed-scene "
             "source script, this is load-bearing here -- it drives the random scene (block/peg "
             "placement) and the (start_block, target_block) choice, not just planner noise. A "
             "different --seed_base draws a different set of --n_rollouts scenes.",
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

    # Changing a planner hyperparameter without a tag is silently WRONG rather than merely
    # messy: the output dir would be the untagged one, whose _result.json files already exist,
    # so the already-done check below would skip every rollout. Fail loudly instead.
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

    if args.tag:
        subfolder_name = f"{subfolder_name}_{args.tag}"
    subfolder_dir = os.path.join(ROLLOUTS_ROOT, subfolder_name)
    os.makedirs(subfolder_dir, exist_ok=True)

    print(f"=== {subfolder_name}: {len(epoch_checkpoints)} epochs x {args.n_rollouts} "
          f"random-scene rollouts each, from {ckpt_dir} ===")
    seeds_desc = str([args.seed_base + i for i in range(args.n_rollouts)])
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
                "--seed", str(args.seed_base + rollout_idx),
            ]
            if args.cold_start:
                cmd += ["--cold_start"]      # bare store_true flag, not a key/value pair
            if args.num_elites is not None:
                cmd += ["--num_elites", str(args.num_elites)]
            subprocess.run(cmd, check=True)

    write_summary(subfolder_dir, subfolder_name, ckpt_dir)
    print(f"Done. Rollouts saved under {subfolder_dir}")


if __name__ == "__main__":
    main()
