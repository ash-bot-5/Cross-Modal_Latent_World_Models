"""
Answers a specific question about the h16 model (checkpoints_clip_full_fine-tuned_with_subopt_data_h16,
the HORIZON=16 action-chunk variant): on the very FIRST MPPI planning iteration -- before any
CEM refinement has a chance to bias the candidate population toward whatever looked good on a
previous pass -- does the dynamics model already score action chunks that push loosely toward
green_cube lower-cost (better) than chunks that push in some other direction?

Uses the same fixed probe scene as every other MPPI_charts/ script in this directory: peg forced
collinear with green_cube and blue_moon (just past green_cube, on the far side from blue_moon),
same seed=42 block_combo=[green_cube, blue_moon] env, same single-stage "touching" goal text. See
[[project_langtable_peg_teleport_staged]] -- offset 0.06 is the pre-existing convention verified
safe with a single (non-staged) teleport.

The "iteration 1, before CEM refinement bias" requirement pins down an easy-to-get-wrong MPPI
hyperparameter: n_planning_itrs=0, NOT 1. Tracing planning/mppi_core.py's MPPI.command() loop --
`for itr in range(n_planning_itrs + 1): ... is_terminal = itr == n_planning_itrs; if is_terminal:
break (no resample)` -- with n_planning_itrs=1, itr=0 is NOT terminal, so the loop resamples a
brand-new population from a softmax-refit distribution before terminating at itr=1: last_actions/
last_returns would already be one CEM refit-resample cycle removed from the raw first population.
With n_planning_itrs=0, the single itr=0 pass IS terminal immediately (no resample), so
last_actions/last_returns are exactly the raw Uniform(-max_action_value, max_action_value) draw,
scored once, with zero CEM bias -- literally "iteration 1" in this codebase's loop structure.

Direction-alignment scoring: for every one of the K raw candidates, compute
dir_align = dot(chunk's net-displacement unit vector, unit vector from the peg's actual
post-teleport position toward green_cube), then bucket into "toward green_cube"
(dir_align > --dir_align_threshold, default 0.5) vs "other". Reports both a continuous view
(scatter of dir_align vs cost, colored by bucket) and a bucketed comparison (mean cost per
bucket with a Welch's t-test + Cohen's d), since a single scan over the raw K=512 first-iteration
population is a much stronger claim than eyeballing MPPI's top-5 final picks the way
cost_landscape_mppi_selected.py does.

No real env execution -- like every other probe in this directory, this only ever calls
mppi.command() once against the one fixed probe latent; env.step() is never called.

Run:
    source /home/ashwink/Documents/swms/.venv/bin/activate
    python main_project/eval/mppi_iteration1_h16_direction_alignment.py --epoch 11
"""

import os
import sys
import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
import open_clip
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planning.mppi_core import MPPI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BLOCK_A = "green_cube"
BLOCK_B = "blue_moon"

TRAIN_DIR = "/home/ashwink/full_data_5000/langtable/demos/"
MAIN_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(MAIN_PROJECT_DIR)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# epoch 5-12 live at repo root, epoch 0-4 live under main_project/training/ -- same split
# mppi_rollout_5_per_checkpoint.py / mppi_obstacle_avoidance_probe.py already rely on.
H16_CKPT_DIRS = [
    os.path.join(REPO_ROOT, "checkpoints_clip_full_fine-tuned_with_subopt_data_h16"),
    os.path.join(MAIN_PROJECT_DIR, "training", "checkpoints_clip_full_fine-tuned_with_subopt_data_h16"),
]

EXPECTED_GREEN_XY = np.array([0.40339102, 0.18967466])
EXPECTED_BLUE_XY = np.array([0.32039625, 0.00902261])

MPPI_TEMPERATURE = 0.01
MAX_ACTION_VALUE = 1.0
WARM_START_STD = 0.1  # passed for API completeness; never exercised -- fresh MPPI instance,
                       # single command() call, prev_best_action stays None throughout.

# Validated categorical palette (dataviz skill, checked via scripts/validate_palette.js --
# same pair already used for a 2-series comparison in mppi_obstacle_avoidance_probe.py /
# mppi_temp_samples_sweep.py).
TOWARD_COLOR = "#2a78d6"
OTHER_COLOR = "#eb6834"
INK = "#333333"
INK_MUTED = "#888888"

# Block-icon colors + touch threshold, verbatim from mppi_obstacle_avoidance_probe.py --
# deliberately muted relative to the viridis trajectory coloring so an icon never reads as
# "another trajectory."
ICON_GREEN = "#3a7d3a"
ICON_BLUE = "#4d6f99"
TOUCH_THRESH = 0.05  # matches TOUCHING_DISTANCE_THRESHOLD, see [[project_langtable_peg_teleport_staged]]
RANK1_COLOR = "gold"


# ---------------------------------------------------------------------------
# Inline model classes -- horizon-parametrized (mppi_rollout_5_per_checkpoint.py's pattern),
# copied rather than imported to avoid triggering dynamics_model.py's LangTableDataset/
# open_clip module-level side effects, per this directory's established convention.
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


def resolve_h16_checkpoint(epoch: int) -> str:
    candidates = [os.path.join(d, f"epoch_{epoch:03d}.pt") for d in H16_CKPT_DIRS]
    for path in candidates:
        if os.path.exists(path):
            return path
    available = {}
    for d in H16_CKPT_DIRS:
        if os.path.isdir(d):
            available[d] = sorted(f for f in os.listdir(d) if f.startswith("epoch_"))
    raise FileNotFoundError(
        f"No epoch_{epoch:03d}.pt found in either h16 checkpoint dir. Checked: {candidates}. "
        f"Available: {available}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epoch", type=int, default=11,
                         help="h16 checkpoint epoch to load (default: 11, matching the epoch used "
                              "in linear_probe_epoch11_h16_snapshot.py / green_cube_confound.py).")
    parser.add_argument("--num_samples", type=int, default=512, help="K, MPPI population size.")
    parser.add_argument("--seed", type=int, default=123,
                         help="torch.manual_seed value, set immediately before the single "
                              "mppi.command() call.")
    parser.add_argument("--dir_align_threshold", type=float, default=0.5,
                         help="dir_align cutoff for the 'toward green_cube' bucket.")
    parser.add_argument("--n_display_trajectories", type=int, default=50,
                         help="how many of the K candidates to draw as individual lines in the "
                              "trajectory-map chart (plotting all K -- e.g. 512 -- renders as an "
                              "unreadable mass; a stratified subsample stays legible while still "
                              "showing the full cos_sim range within each direction bucket).")
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(MAIN_PROJECT_DIR, "MPPI_charts", "iteration1_direction_alignment"),
                         help="output root; results land under <out_dir>/epoch<N>/.")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, f"epoch{args.epoch}")
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Load checkpoint, confirm it's actually an h16 (horizon=16) checkpoint.
    # -----------------------------------------------------------------------
    ckpt_path = resolve_h16_checkpoint(args.epoch)
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    print(f"  epoch={ckpt['epoch']}  step={ckpt.get('step', '?')}")

    # action_encoder.out.weight: (256, conv_channels * horizon), conv_channels=64 (default).
    action_horizon = ckpt["mlp_state_dict"]["action_encoder.out.weight"].shape[1] // 64
    assert action_horizon == 16, (
        f"Expected an h16 (horizon=16) checkpoint, but {ckpt_path} has action_horizon="
        f"{action_horizon}. Wrong checkpoint directory?"
    )
    print(f"Confirmed action_horizon={action_horizon}\n")

    # -----------------------------------------------------------------------
    # 2. Action normalization stats (denormalize MPPI's [-1,1] output back to raw units).
    # -----------------------------------------------------------------------
    print(f"Loading action stats from {TRAIN_DIR} ...")
    all_actions = []
    for pkl_file in sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith(".pkl")):
        with open(os.path.join(TRAIN_DIR, pkl_file), "rb") as f:
            ep = pickle.load(f)
        all_actions.append(np.array(ep["actions"], dtype=np.float32))
    all_actions = np.concatenate(all_actions, axis=0)
    action_lo = all_actions.min(axis=0)  # (2,)
    action_hi = all_actions.max(axis=0)  # (2,)
    print(f"Action normalization stats: lo={action_lo.tolist()} hi={action_hi.tolist()}\n")

    # -----------------------------------------------------------------------
    # 3. Fixed probe scene: env + peg-collinear teleport, verbatim from
    #    mppi_rollout_clip_full_finetune.py / cost_landscape_mppi_selected.py.
    # -----------------------------------------------------------------------
    print("Initializing LangTable env ...")
    from swm.utils.envs import get_lang_table_env
    env = get_lang_table_env({"ood": False, "block_combo": [BLOCK_A, BLOCK_B]}, seed=42)
    print("Env ready.")

    from language_table.environments.utils.pose3d import Pose3d
    from scipy.spatial.transform import Rotation as R

    inner = env.env
    blocks = inner.get_block_states()
    green_xy = blocks[BLOCK_A]
    blue_xy = blocks[BLOCK_B]

    assert np.allclose(green_xy, EXPECTED_GREEN_XY, atol=1e-5), (
        f"{BLOCK_A} position {green_xy} does not match expected {EXPECTED_GREEN_XY} "
        f"-- table arrangement has diverged."
    )
    assert np.allclose(blue_xy, EXPECTED_BLUE_XY, atol=1e-5), (
        f"{BLOCK_B} position {blue_xy} does not match expected {EXPECTED_BLUE_XY} "
        f"-- table arrangement has diverged."
    )
    print(f"Block arrangement verified: {BLOCK_A}={green_xy}, {BLOCK_B}={blue_xy}")

    away_from_blue_direction = (green_xy - blue_xy) / np.linalg.norm(green_xy - blue_xy)
    peg_xy_target = green_xy + away_from_blue_direction * 0.06

    pose = Pose3d(
        rotation=R.from_rotvec([0, np.pi, 0]),
        translation=np.array([peg_xy_target[0], peg_xy_target[1], 0.145]),
    )
    inner._set_robot_target_effector_pose(pose)
    inner._step_simulation_to_stabilize()

    dist_peg_blue = np.linalg.norm(peg_xy_target - blue_xy)
    dist_green_blue = np.linalg.norm(green_xy - blue_xy)
    assert dist_peg_blue > dist_green_blue, (
        f"peg ended up between {BLOCK_A} and {BLOCK_B}, not on the outside past {BLOCK_A} -- geometry bug"
    )

    # Direction toward green_cube, computed from the peg's ACTUAL post-stabilize position
    # (IK settles a fraction of a mm short of the commanded target -- see
    # [[project_langtable_peg_teleport_staged]]), not the commanded peg_xy_target.
    peg_xy_actual = np.array(inner._robot.forward_kinematics().translation[:2], dtype=np.float32)
    direction_to_green_cube = (green_xy - peg_xy_actual) / np.linalg.norm(green_xy - peg_xy_actual)

    # Sanity check: for this teleport geometry (peg parked on the blue->green axis, just past
    # green), direction_to_green_cube should coincide with the green->blue unit vector every
    # other script in this directory calls `correct_direction`. Compared by ANGLE, not raw
    # np.allclose on vector components -- IK settles a fraction of a mm short of the commanded
    # target (see [[project_langtable_peg_teleport_staged]]), which perturbs peg_xy_actual off
    # the exact blue-green axis and so perturbs this unit vector by a degree or so even under
    # correct geometry; a raw component atol=1e-3 fails on that expected noise (measured ~0.6 deg
    # in practice). A real geometry bug (peg on the wrong side, teleport failed, etc.) would show
    # up as tens of degrees, not fractions of one, so 5 deg is a tight-but-safe bound.
    correct_direction = (blue_xy - green_xy) / np.linalg.norm(blue_xy - green_xy)
    cos_angle = float(np.clip(np.dot(direction_to_green_cube, correct_direction), -1.0, 1.0))
    angle_deviation_deg = np.degrees(np.arccos(cos_angle))
    assert angle_deviation_deg < 5.0, (
        f"direction_to_green_cube {direction_to_green_cube} deviates {angle_deviation_deg:.2f} deg "
        f"from correct_direction {correct_direction} (expected <5 deg from IK settle-short noise) "
        f"-- peg geometry unexpected."
    )
    print(f"Peg actual position: {peg_xy_actual}")
    print(f"direction_to_green_cube: {direction_to_green_cube}  "
          f"({angle_deviation_deg:.3f} deg off correct_direction, within expected IK settle noise)\n")

    pil_frame = env.get_frame()
    frame_path = os.path.join(out_dir, "first_frame.png")
    pil_frame.save(frame_path)
    print(f"Saved first frame -> {frame_path}")

    goal_text = (
        f"the {BLOCK_A.replace('_', ' ')} is touching the {BLOCK_B.replace('_', ' ')}. "
        f"the peg is touching the {BLOCK_A.replace('_', ' ')}."
    )
    print(f"Goal: {goal_text}\n")

    # -----------------------------------------------------------------------
    # 4. CLIP + dynamics model, both loaded from this checkpoint's OWN fine-tuned towers --
    #    a fine-tuned text tower invalidates any separately-loaded frozen-CLIP z_sem cache.
    # -----------------------------------------------------------------------
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    clip_model = clip_model.to(DEVICE)
    clip_model.load_state_dict(ckpt["clip_state_dict"])
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer("ViT-L-14")

    model = LatentDynamicsModel(horizon=action_horizon).to(DEVICE)
    model.load_state_dict(ckpt["mlp_state_dict"])
    model.eval()

    with torch.no_grad():
        tokens = clip_tokenizer([goal_text]).to(DEVICE)
        z_star = F.normalize(clip_model.encode_text(tokens), dim=-1).squeeze(0)  # (768,)

        img_t = clip_preprocess(pil_frame).unsqueeze(0).to(DEVICE)
        z_t = F.normalize(clip_model.encode_image(img_t), dim=-1).squeeze(0)  # (768,)

    def dynamics_fn(state, action):
        # state:  (K, 768)
        # action: (K, action_horizon, 2) normalized action chunks
        with torch.no_grad():
            return model(state, action)

    def cost_fn(terminal_state):
        # terminal_state: (K, 768) -> (K,)
        return 1.0 - (terminal_state * z_star).sum(dim=-1)

    # -----------------------------------------------------------------------
    # 5. MPPI, n_planning_itrs=0 -- see module docstring for the loop-trace reasoning: this
    #    is the setting that makes last_actions/last_returns hold the RAW pre-CEM-refit
    #    Uniform(-1,1) population, scored exactly once, with zero resample bias.
    # -----------------------------------------------------------------------
    mppi = MPPI(
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        pred_horizon=action_horizon,
        action_dim=2,
        num_samples=args.num_samples,
        mppi_temperature=MPPI_TEMPERATURE,
        n_planning_itrs=0,
        device=DEVICE,
        max_action_value=MAX_ACTION_VALUE,
        warm_start_std=WARM_START_STD,
    )

    torch.manual_seed(args.seed)
    with torch.no_grad():
        command_action = mppi.command(z_t)  # (action_horizon, 2) normalized

    last_actions = mppi.last_actions   # (K, action_horizon, 2) normalized
    last_returns = mppi.last_returns   # (K,)
    best_idx = int(torch.argmax(last_returns))
    assert last_actions[best_idx].allclose(command_action), (
        "argmax(last_returns) does not index the chunk command() returned -- MPPI internals changed?"
    )

    cost = (-last_returns).cpu().numpy()          # (K,)  lower = better, matches cost_fn's units
    cos_sim = (last_returns + 1.0).cpu().numpy()  # (K,)  cost_fn is exactly 1 - cos_sim

    # -----------------------------------------------------------------------
    # 6. Direction-alignment scoring, vectorized over all K candidates.
    # -----------------------------------------------------------------------
    chunks_norm = last_actions.cpu().numpy()  # (K, action_horizon, 2)
    chunks_raw = (chunks_norm + 1.0) / 2.0 * (action_hi - action_lo) + action_lo  # (K, action_horizon, 2)
    disp = chunks_raw.sum(axis=1)  # (K, 2) net displacement per candidate
    disp_mag = np.linalg.norm(disp, axis=1)
    disp_unit = disp / (disp_mag[:, None] + 1e-8)
    dir_align = disp_unit @ direction_to_green_cube  # (K,)

    toward_mask = dir_align > args.dir_align_threshold
    other_mask = ~toward_mask
    n_toward = int(toward_mask.sum())
    n_other = int(other_mask.sum())

    cost_toward = cost[toward_mask]
    cost_other = cost[other_mask]
    cos_sim_toward = cos_sim[toward_mask]
    cos_sim_other = cos_sim[other_mask]

    # -----------------------------------------------------------------------
    # 7. Statistics.
    # -----------------------------------------------------------------------
    if n_toward > 1 and n_other > 1:
        t_stat, p_value = scipy_stats.ttest_ind(cost_toward, cost_other, equal_var=False)
        pooled_std = np.sqrt((cost_toward.var(ddof=1) + cost_other.var(ddof=1)) / 2.0)
        cohens_d = (cost_toward.mean() - cost_other.mean()) / (pooled_std + 1e-12)
    else:
        t_stat, p_value, cohens_d = float("nan"), float("nan"), float("nan")

    summary_lines = [
        f"epoch={args.epoch}  ckpt={ckpt_path}",
        f"seed={args.seed}  K={args.num_samples}  n_planning_itrs=0 (raw pre-CEM population)",
        f"dir_align_threshold={args.dir_align_threshold}",
        "",
        f"toward green_cube bucket: n={n_toward}  "
        f"mean_cost={cost_toward.mean():.5f}  median_cost={np.median(cost_toward):.5f}  "
        f"std_cost={cost_toward.std():.5f}",
        f"other bucket:             n={n_other}  "
        f"mean_cost={cost_other.mean():.5f}  median_cost={np.median(cost_other):.5f}  "
        f"std_cost={cost_other.std():.5f}",
        "",
        f"toward green_cube bucket: mean_cos_sim={cos_sim_toward.mean():.5f}",
        f"other bucket:             mean_cos_sim={cos_sim_other.mean():.5f}",
        "",
        f"Welch's t-test on cost (toward vs other): t={t_stat:.4f}  p={p_value:.6g}",
        f"Cohen's d (toward vs other, cost): {cohens_d:.4f}",
        "",
        f"Correlation(dir_align, cost) across all K={args.num_samples}: "
        f"{np.corrcoef(dir_align, cost)[0, 1]:.4f}",
    ]
    if p_value < 0.05 and cost_toward.mean() < cost_other.mean():
        summary_lines.append(
            "\n=> toward-green-cube bucket has SIGNIFICANTLY LOWER cost (p<0.05): "
            "model correctly ranks green-cube-directed chunks better at iteration 1."
        )
    elif p_value < 0.05:
        summary_lines.append(
            "\n=> difference is significant (p<0.05) but toward-green-cube bucket does NOT have "
            "lower cost -- model does not favor green-cube-directed chunks at iteration 1."
        )
    else:
        summary_lines.append(
            "\n=> no significant difference between buckets (p>=0.05) at iteration 1."
        )

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text + "\n")

    summary_path = os.path.join(out_dir, f"epoch{args.epoch}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"Saved stats summary -> {summary_path}")

    # -----------------------------------------------------------------------
    # 8. Plot 1 -- scatter: dir_align (x) vs cost (y), all K raw candidates, colored by bucket.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(dir_align[other_mask], cost[other_mask], s=14, color=OTHER_COLOR, alpha=0.6,
               label=f"other (n={n_other})", zorder=2)
    ax.scatter(dir_align[toward_mask], cost[toward_mask], s=14, color=TOWARD_COLOR, alpha=0.6,
               label=f"toward green_cube (n={n_toward})", zorder=3)
    ax.axvline(args.dir_align_threshold, color=INK, linewidth=1, linestyle="--", alpha=0.6,
               label=f"threshold={args.dir_align_threshold}")
    ax.set_xlabel("dir_align  (dot(chunk displacement, direction to green_cube))")
    ax.set_ylabel("cost  (1 - cos_sim to goal)")
    ax.set_title(
        f"epoch{args.epoch} h16 model -- iteration-1 candidates (K={args.num_samples}, "
        f"n_planning_itrs=0)\ncost vs. direction alignment toward green_cube"
    )
    ax.legend(loc="best", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    scatter_path = os.path.join(out_dir, f"epoch{args.epoch}_dir_align_vs_cost_scatter.png")
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter -> {scatter_path}")

    # -----------------------------------------------------------------------
    # 9. Plot 2 -- bar: mean cost per bucket with SEM error bars.
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.5, 6))
    means = [cost_toward.mean(), cost_other.mean()]
    sems = [
        cost_toward.std(ddof=1) / np.sqrt(n_toward) if n_toward > 1 else 0.0,
        cost_other.std(ddof=1) / np.sqrt(n_other) if n_other > 1 else 0.0,
    ]
    labels = [f"toward green_cube\n(n={n_toward})", f"other\n(n={n_other})"]
    colors = [TOWARD_COLOR, OTHER_COLOR]

    ax.bar(labels, means, yerr=sems, capsize=6, color=colors, width=0.6)
    ax.set_ylabel("mean cost  (1 - cos_sim to goal)")
    ax.set_title(
        f"epoch{args.epoch} h16 model -- mean cost by direction bucket\n"
        f"Welch t-test p={p_value:.4g}, Cohen's d={cohens_d:.3f}"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    bar_path = os.path.join(out_dir, f"epoch{args.epoch}_bucket_cost_comparison.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved bucket comparison -> {bar_path}")

    # -----------------------------------------------------------------------
    # 10. Plot 3 -- MPPI-style trajectory map: a STRATIFIED SUBSAMPLE of the K candidates drawn
    # as cumulative delta end-effector-position paths from the peg's start (origin), colored by
    # CONTINUOUS predicted cos_sim (viridis + colorbar), with green_cube/blue_moon icons and the
    # MPPI-selected best candidate highlighted -- matches the visual style of the existing
    # MPPI_charts/ trajectory charts (mppi_obstacle_avoidance_probe.py's icon convention,
    # cost_landscape_realistic_chunks.py's continuous-coloring convention).
    #
    # Plotting all K=512 raw candidates renders as an unreadable mass (every candidate is a near-
    # straight line from the origin, so 512 overlapping thin strokes just blur into a purple/green
    # smear). Instead, sample --n_display_trajectories total, split evenly between the two
    # direction buckets and, WITHIN each bucket, spread evenly across that bucket's own sorted
    # cos_sim range (via np.linspace over rank, not a random draw) -- so the displayed subset still
    # honestly represents each bucket's full cos_sim spread (best case through worst case), rather
    # than random luck over- or under-representing one end.
    # -----------------------------------------------------------------------
    green_rel = green_xy - peg_xy_actual
    blue_rel = blue_xy - peg_xy_actual

    # (K, action_horizon, 2) raw chunks -> (K, action_horizon+1, 2) cumulative paths from origin,
    # vectorized (reuses chunks_raw already computed for dir_align above).
    K_actual = chunks_raw.shape[0]
    paths = np.concatenate(
        [np.zeros((K_actual, 1, 2), dtype=chunks_raw.dtype), np.cumsum(chunks_raw, axis=1)],
        axis=1,
    )  # (K, action_horizon+1, 2)

    def stratified_sample(mask: np.ndarray, n: int) -> np.ndarray:
        """Indices (into the full K population) of up to n candidates from `mask`, evenly spread
        across that subset's own sorted cos_sim range."""
        bucket_idx = np.where(mask)[0]
        if len(bucket_idx) == 0 or n <= 0:
            return np.array([], dtype=int)
        n = min(n, len(bucket_idx))
        order = bucket_idx[np.argsort(cos_sim[bucket_idx])]  # ascending cos_sim within bucket
        pick = np.linspace(0, len(order) - 1, n).round().astype(int)
        return order[pick]

    n_toward_display = args.n_display_trajectories // 2
    n_other_display = args.n_display_trajectories - n_toward_display
    display_idx = np.concatenate([
        stratified_sample(toward_mask, n_toward_display),
        stratified_sample(other_mask, n_other_display),
    ])
    print(f"Trajectory map: displaying {len(display_idx)} of K={K_actual} candidates "
          f"({n_toward_display} toward / {n_other_display} other, stratified by cos_sim).")

    cmap = matplotlib.colormaps["viridis"]
    color_norm = mcolors.Normalize(vmin=float(cos_sim.min()), vmax=float(cos_sim.max()))

    all_axis_points = [
        paths[display_idx].reshape(-1, 2),
        paths[best_idx].reshape(-1, 2),
        green_rel.reshape(1, 2),
        blue_rel.reshape(1, 2),
        np.zeros((1, 2)),
    ]
    axis_half = float(np.abs(np.concatenate(all_axis_points, axis=0)).max() * 1.2)
    axis_lim = (-axis_half, axis_half)

    fig, ax = plt.subplots(figsize=(9, 9))

    for idx in display_idx:
        ax.plot(paths[idx, :, 0], paths[idx, :, 1], color=cmap(color_norm(cos_sim[idx])),
                 alpha=0.75, linewidth=1.6, zorder=1)

    # Highlight MPPI's actual iteration-1 pick (argmax of last_returns) on top of the population.
    best_path = paths[best_idx]
    ax.plot(best_path[:, 0], best_path[:, 1], color=RANK1_COLOR, linewidth=3.0, zorder=5,
             label=f"MPPI pick (cos_sim={cos_sim[best_idx]:.4f})")
    ax.annotate(
        "", xy=(best_path[-1, 0], best_path[-1, 1]), xytext=(best_path[-2, 0], best_path[-2, 1]),
        arrowprops=dict(facecolor=RANK1_COLOR, edgecolor=RANK1_COLOR, width=2, headwidth=10),
        zorder=6,
    )
    ax.text(best_path[-1, 0], best_path[-1, 1], "  MPPI pick", color=INK, fontsize=10,
            fontweight="bold", va="center", zorder=6)

    # Block icons + peg-start marker, relative to the peg's actual post-teleport position.
    ax.scatter([green_rel[0]], [green_rel[1]], marker="s", s=260, color=ICON_GREEN,
               edgecolors=INK, linewidths=1.0, zorder=10)
    ax.add_patch(Circle(tuple(green_rel), TOUCH_THRESH, fill=False, linestyle="--",
                         edgecolor=ICON_GREEN, alpha=0.7, linewidth=1.3, zorder=9))
    ax.text(green_rel[0], green_rel[1] - TOUCH_THRESH - 0.008, "green_cube",
            color=ICON_GREEN, fontsize=9, fontweight="bold", ha="center", va="top", zorder=10)

    ax.scatter([blue_rel[0]], [blue_rel[1]], marker="*", s=420, color=ICON_BLUE,
               edgecolors=INK, linewidths=1.0, zorder=10)
    ax.text(blue_rel[0], blue_rel[1] - 0.014, "blue_moon",
            color=ICON_BLUE, fontsize=9, fontweight="bold", ha="center", va="top", zorder=10)

    ax.scatter([0.0], [0.0], marker="P", s=140, color=INK, zorder=10)
    ax.text(0.0, 0.008, "peg start", color=INK, fontsize=8, ha="center", va="bottom", zorder=10)

    sm = mcm.ScalarMappable(norm=color_norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="predicted cosine similarity to goal")

    ax.set_xlim(axis_lim)
    ax.set_ylim(axis_lim)
    ax.set_aspect("equal")
    ax.grid(True, color=INK_MUTED, alpha=0.2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("delta end-effector x (from peg start)")
    ax.set_ylabel("delta end-effector y (from peg start)")
    ax.set_title(
        f"epoch{args.epoch} h16 model -- iteration-1 candidates (K={args.num_samples}, "
        f"n_planning_itrs=0)\ncolored by predicted cosine similarity to goal"
    )
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    trajectory_map_path = os.path.join(out_dir, f"epoch{args.epoch}_trajectory_map.png")
    plt.savefig(trajectory_map_path, dpi=150)
    plt.close()
    print(f"Saved trajectory map -> {trajectory_map_path}")

    # -----------------------------------------------------------------------
    # 11. Save raw data.
    # -----------------------------------------------------------------------
    data_path = os.path.join(out_dir, f"epoch{args.epoch}_data.npz")
    np.savez(
        data_path,
        last_actions_normalized=chunks_norm,
        last_returns=last_returns.cpu().numpy(),
        cost=cost,
        cos_sim=cos_sim,
        dir_align=dir_align,
        toward_mask=toward_mask,
        direction_to_green_cube=direction_to_green_cube,
        green_xy=green_xy,
        blue_xy=blue_xy,
        peg_xy_actual=peg_xy_actual,
        epoch=args.epoch,
        seed=args.seed,
        dir_align_threshold=args.dir_align_threshold,
        t_stat=t_stat,
        p_value=p_value,
        cohens_d=cohens_d,
    )
    print(f"Saved data -> {data_path}")
    print(f"\nDone. Outputs under {out_dir}")


if __name__ == "__main__":
    main()
