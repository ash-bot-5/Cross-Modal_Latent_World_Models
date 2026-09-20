"""
Standalone CEM (Cross-Entropy Method) controller -- a faithful port of
reference_repos_MPPI/CEM_reference_code/cem.py's evalGaussian loop, built as a
drop-in sibling of planning/mppi_core.py's MPPI class.

This is textbook CEM: each iteration samples a population from a per-dimension
Gaussian, sorts by cost, keeps the top Ne "elites", and refits the Gaussian's
mean and std from the elite set ALONE -- a hard 0/1 weighting. Selection is
rank-based, so only the ORDERING of costs matters, never their magnitudes.

Why this file exists (the difference from mppi_core.MPPI):
  mppi_core.MPPI is not MPPI. It is a Gaussian CEM whose refit uses a SOFTMAX
  weighting over all K samples -- MPPI's weight rule grafted onto CEM's
  iterated distribution-fitting loop. That hybrid couples the cost SCALE to the
  exploration width, which neither parent algorithm does: real MPPI holds its
  covariance fixed, and real CEM floors its std at the elite spread. Measured
  consequence on this project's epoch-11 probe (K=512, 9 refinement iters), where
  the initial uniform draw has std 0.577:

      mppi_temperature=0.001  ->  final refit std 4.6e-04  (1250x shrink)  cos_sim -0.034
      mppi_temperature=0.01   ->  final refit std 1.5e-02  (  37x shrink)  cos_sim  0.587
      mppi_temperature=0.1    ->  final refit std 1.5e-01  ( 3.9x shrink)  cos_sim  0.656

  In closed-loop rollouts at the production temperature 0.01, the final
  population's cosine spread is 5x narrower than at 0.1 (median 0.030 vs 0.147
  over 1500 planning steps) and the executed sub-step is ~20% of the available
  action magnitude -- i.e. the planner had largely stopped exploring.

  Under hard top-Ne truncation that coupling does not exist: the refit std is
  the elite set's own spread, which cannot be driven to zero by a change in cost
  units. This class exists to test the CLIP-latent dynamics model under that
  regime instead.

Controlled-comparison design: EVERYTHING except the refit rule is kept identical
to mppi_core.MPPI, so an A/B between the two isolates exactly one variable --
    - same uniform initial population (bit-identical under the same seed, see
      _sample_initial_actions),
    - same warm-start-from-shifted-previous-chunk across real env steps,
    - same selection rule (argmax of the final population -- which under hard
      truncation IS the rank-1 elite, so it is simultaneously CEM-native),
    - same number of population evaluations per command() call (n_planning_itrs + 1),
    - same +1e-9 std epsilon and same ddof=0 std convention.

Divergences from CEM_reference_code/cem.py, all deliberate and reviewed:
  1. command() returns the rank-1 elite, not mu (cem.py:61). mu is never
     evaluated and, on a bounded and possibly multimodal cosine cost, can
     average across modes into a chunk that scores below every elite. Returning
     a population member also keeps argmax(last_returns) pointing at the
     returned chunk, which three eval/ scripts assert. return_mode="mean"
     restores the reference behaviour.
  2. 10 refits, not maxits=500 (cem.py:5). Each iteration here is 512 dynamics
     forward passes and the planner runs once per env step; 10 exactly
     compute-matches mppi_core.MPPI, which is what makes the A/B fair. THIS IS
     CEM STOPPED EARLY -- a poor result is partly a statement about budget, not
     purely about the algorithm. Raise n_planning_itrs to measure the ceiling.
  3. Warm start across command() calls. The reference is a one-shot optimizer
     with no environment and no such notion; this is inherited verbatim from
     mppi_core.py so it is not a second difference between the two arms. It has
     no effect on single-command() probes (prev_best_action is None there).
  4. Uniform initial population rather than the reference's Gaussian
     mu=0/sigma=init_scale (cem.py:66-67). Buys bit-identical RNG parity with
     MPPI, and avoids sigma=1 piling ~32% of its mass on a [-1,1] boundary.
     Affects only the first of the 10 populations; init_mode="gaussian" restores
     the reference path. NOTE this is unrelated to the reference's
     sampleMethod="Uniform", which is a different algorithm entirely (it tracks
     an elite bounding box rather than mean/std -- see cem.py:92-96 -- and is
     not ported here).
  5. std epsilon 1e-9 rather than the reference's 1e-17 (cem.py:80), for parity
     with mppi_core.py:204. In fp32 at |mu| ~ 0.5 both round to exactly mu, so
     there is no behavioural difference.
  6. reset(), the diagnostics block, and min_std have no reference counterpart.
     min_std defaults to 0.0, which IS the reference behaviour.

Faithful despite looking different: unbiased=False matches numpy's ddof=0
default that cem.py:73 relies on (torch's default is the divergence, and NaNs at
Ne=1); topk(Ne) yields the same elite set as sort-then-slice (cem.py:110-113);
refits == evals matches the reference's loop structure, whose last refit also
never resamples; Ne=51 of K=512 is the same 10% as the reference's 10 of 100;
always clamping equals the reference's clip when both bounds are given
(cem.py:81-82); (T, nu) action chunks are bookkeeping over the reference's flat
(d,) with d = T*nu; and expanding current_state to K rows is the reference's
np.repeat(instr, N) pattern (cem.py:106-108).

Not implemented, deliberately: PETS-style momentum smoothing of the mean
(mu <- alpha*mu_old + (1-alpha)*mu_new). It would be a SECOND difference from
mppi_core.MPPI and would muddy the comparison this file exists to make.
"""

import warnings
from typing import Callable

import torch


class CEM:
    """
    Gaussian Cross-Entropy Method planner with hard top-Ne elite truncation,
    ported from CEM_reference_code/cem.py's evalGaussian.

    Tensor shape conventions (identical to mppi_core.MPPI):
        current_state:  (*state_shape)                 -- single state, opaque shape
        actions:        (K, pred_horizon, action_dim)  -- candidate action chunks
        batched state:  (K, *state_shape)              -- K copies for parallel rollout
        returns:        (K,)
        elites:         (Ne, pred_horizon, action_dim)
        mean / std:     (pred_horizon, action_dim)     -- per-dimension, as the reference
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        pred_horizon: int,
        action_dim: int,
        num_samples: int,
        n_planning_itrs: int,
        device,
        num_elites: int | None = None,
        elite_frac: float = 0.1,
        max_action_value: float = 1.0,
        warm_start_std: float = 0.1,
        n_execute: int = 1,
        init_mode: str = "uniform",
        init_std: float = 0.5,
        return_mode: str = "best",
        min_std: float = 0.0,
        std_eps: float = 1e-9,
        collect_diagnostics: bool = False,
    ):
        """
        Args:
            dynamics_fn:      (state: Tensor(K, *), action: Tensor(K, pred_horizon, action_dim))
                               -> Tensor(K, *). Called once per CEM iteration with the full
                               candidate population. State shape is never inspected.
            cost_fn:          (terminal_state: Tensor(K, *)) -> Tensor(K,). Lower = better.
                               Negated internally into `returns` (higher = better) so that
                               `returns == -cost` -- the identity mppi_core.MPPI also upholds
                               and that eval/ scripts rely on to recover cosine similarity.
            pred_horizon:     T -- number of real sub-steps packed into one action chunk
                               (matches the dynamics model's own HORIZON).
            action_dim:       nu -- per-sub-step action dimensionality (e.g. 2 for (dx, dy)).
            num_samples:      K -- population size sampled per CEM iteration.
            n_planning_itrs:  same name and meaning as mppi_core.MPPI's, so the two are
                               compute-matched: total population evaluations per command()
                               call is n_planning_itrs + 1 in BOTH classes.
            device:           torch device for all internal tensors.
            num_elites:       Ne -- exact elite count. Takes precedence over elite_frac when
                               given, so a sweep can pin integers rather than rounded fractions.
            elite_frac:       used only when num_elites is None. 0.1 is what both references
                               agree on (cem.py is 10/100, PETS is 40/400); at K=512 it
                               resolves to Ne=51.
            max_action_value: symmetric clamp bound. Defaults to 1.0 since this project's
                               actions are normalized to [-1, 1].
            warm_start_std:   per-dimension std used to sample the initial population around
                               the shifted previous chunk, for every command() call after the
                               first (or after reset()). Identical to mppi_core.MPPI's.
            n_execute:        how many sub-steps of the returned chunk the CALLER executes
                               before calling command() again. Only affects the warm start,
                               which shifts the previous chunk forward by exactly this much --
                               seeding from a chunk shifted by 1 when 4 sub-steps were actually
                               executed would misalign the warm start by 3 timesteps. Must be
                               1 <= n_execute <= pred_horizon; at n_execute == pred_horizon the
                               whole chunk has been consumed, so the warm start degenerates to
                               all-zeros (correctly -- no information about the executed chunk
                               remains). Defaults to 1, matching mppi_core.MPPI, which has no
                               such parameter and always shifts by 1.
            init_mode:        "uniform" (default) draws the very first population from
                               Uniform(-max_action_value, max_action_value), bit-identical to
                               mppi_core.py:147-150. "gaussian" uses the reference's
                               mu=0/sigma=init_std init instead.
            init_std:         sigma for init_mode="gaussian". 0.5 rather than the reference's
                               1.0 because actions are clamped to [-1, 1] and sigma=1 would
                               pile ~32% of its mass on the boundary; 0.5 is the closest
                               non-saturating match to Uniform(-1,1)'s std of 0.577.
            return_mode:      "best" (default) returns the rank-1 elite -- a chunk that was
                               actually evaluated, and the same selection rule mppi_core.MPPI
                               uses. "mean" returns the terminal elite mean, faithful to
                               cem.py:61; it costs one extra batch-of-1 dynamics call to score
                               the returned chunk, and it BREAKS the argmax(last_returns)
                               contract by design (see command()).
            min_std:          anti-collapse floor applied to the refit std. Defaults to 0.0,
                               which is the reference behaviour and keeps the open question --
                               does hard truncation collapse the way softmax weighting did? --
                               honestly measurable. Set it (e.g. 0.05, ~9% of the initial
                               uniform std) only to deliberately run a floored arm.
            std_eps:          additive epsilon keeping the Normal scale strictly positive.
            collect_diagnostics: when True, command() records per-refit statistics into
                               self.last_diagnostics (see command()). Purely observational --
                               reads tensors the algorithm already computed, consumes no RNG,
                               and changes no numerics -- so leaving it False (the default)
                               costs nothing.
        """
        if init_mode not in ("uniform", "gaussian"):
            raise ValueError(f"init_mode must be 'uniform' or 'gaussian', got {init_mode!r}")
        if not (1 <= n_execute <= pred_horizon):
            raise ValueError(
                f"n_execute must satisfy 1 <= n_execute <= pred_horizon, got n_execute="
                f"{n_execute} with pred_horizon={pred_horizon}"
            )
        if return_mode not in ("best", "mean"):
            raise ValueError(f"return_mode must be 'best' or 'mean', got {return_mode!r}")

        # Resolve Ne once, here, so it can never drift between command() calls.
        ne = num_elites if num_elites is not None else int(round(elite_frac * num_samples))
        if not (1 <= ne <= num_samples):
            raise ValueError(
                f"num_elites must satisfy 1 <= Ne <= num_samples, got Ne={ne} with "
                f"num_samples={num_samples} "
                f"(from {'num_elites' if num_elites is not None else f'elite_frac={elite_frac}'})"
            )
        if ne == 1 and min_std <= 0.0:
            warnings.warn(
                "num_elites=1 with min_std=0: the refit std of a single elite is exactly 0, so "
                "the sampler freezes onto that one chunk for every remaining iteration and the "
                "result reflects the best member of the INITIAL draw. This is a legitimate "
                "degenerate arm, but it is almost certainly not what you want as a baseline.",
                stacklevel=2,
            )

        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.K = num_samples
        self.num_elites = ne
        self.n_planning_itrs = n_planning_itrs
        self.device = device
        self.max_action_value = max_action_value
        self.warm_start_std = warm_start_std
        self.n_execute = n_execute
        self.init_mode = init_mode
        self.init_std = init_std
        self.return_mode = return_mode
        self.min_std = min_std
        self.std_eps = std_eps
        self.collect_diagnostics = collect_diagnostics

        # Final-population state, published after each command() call. last_returns /
        # last_actions carry the same contract as mppi_core.MPPI's.
        self.last_actions: torch.Tensor | None = None
        self.last_returns: torch.Tensor | None = None
        self.last_elite_idx: torch.Tensor | None = None
        self.last_elite_returns: torch.Tensor | None = None

        # Terminal refit, i.e. the distribution CEM converged to.
        self.last_mean: torch.Tensor | None = None
        self.last_std: torch.Tensor | None = None
        self.last_elite_std: torch.Tensor | None = None

        # What command() actually returned, and its measured score. Mode-agnostic --
        # new consumers should read these rather than re-deriving via argmax.
        self.last_command_action: torch.Tensor | None = None
        self.last_command_return: torch.Tensor | None = None
        # Score of last_mean. Only populated in return_mode="mean" (where it is the
        # returned chunk's score); None in "best" mode, where evaluating the mean would
        # be an extra dynamics call bought for nothing.
        self.last_mean_return: torch.Tensor | None = None

        # Per-refit diagnostics from the most recent command() call -- only populated when
        # collect_diagnostics=True, else stays None.
        self.last_diagnostics: list[dict] | None = None

        # Warm-start state: the chunk command() returned last call, or None for the very
        # first call (or after reset()).
        self.prev_best_action: torch.Tensor | None = None

    def reset(self):
        """Clear warm-start state -- call at an episode boundary so a new rollout
        doesn't get seeded from a previous, unrelated episode's last chunk."""
        self.prev_best_action = None

    def _sample_initial_actions(self) -> torch.Tensor:
        """The very first population of each command() call.

        With init_mode="uniform" and no warm-start state, this is the EXPRESSION from
        mppi_core.py:147-150 verbatim -- torch.empty(...).uniform_() rather than
        torch.rand or torch.distributions.Uniform, which consume the global generator
        differently. That verbatim-ness is what makes the two planners draw a
        bit-identical first population under the same torch.manual_seed, so an A/B
        between them provably diverges only at the first refit.
        """
        if self.prev_best_action is None:
            if self.init_mode == "uniform":
                return (
                    torch.empty(self.K, self.pred_horizon, self.action_dim, device=self.device).uniform_(-1.0, 1.0)
                    * self.max_action_value
                )
            # init_mode == "gaussian": the reference's mu=0 / sigma=init_scale init
            # (cem.py:66-67), clamped to the action bounds as cem.py:81-82 does.
            mean = torch.zeros(self.pred_horizon, self.action_dim, device=self.device)
            std = torch.full_like(mean, self.init_std)
            actions = torch.distributions.Normal(mean, std).sample((self.K,))
            return torch.clamp(actions, -self.max_action_value, self.max_action_value)

        # Drop the just-executed sub-steps, shift the rest forward, and append that many
        # fresh zero-action sub-steps (the u_init=0 convention). The shift is n_execute,
        # not 1: the caller may execute several sub-steps of the chunk per plan, and
        # shifting by less would seed the next search from actions the robot has already
        # performed. At n_execute == 1 this is identical to mppi_core.py:154-155.
        shifted = torch.roll(self.prev_best_action, -self.n_execute, dims=0)
        shifted[-self.n_execute:] = 0.0

        actions = torch.distributions.Normal(shifted, self.warm_start_std).sample((self.K,))
        return torch.clamp(actions, -self.max_action_value, self.max_action_value)

    def _evaluate(self, current_state: torch.Tensor, actions: torch.Tensor,
                  batch: int | None = None) -> torch.Tensor:
        """Roll a population through dynamics_fn + cost_fn once, return `returns` (n,)
        (higher = better -- negated cost).

        `batch` defaults to K; it exists so return_mode="mean" can score the single
        terminal mean without allocating K copies of the state. current_state's shape
        is expanded, never inspected -- the opaque-state contract is preserved.
        """
        n = self.K if batch is None else batch
        state = current_state.unsqueeze(0).expand(n, *current_state.shape).clone()
        state = self.dynamics_fn(state, actions)
        cost = self.cost_fn(state)
        return -cost

    def command(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Run one full CEM planning pass and return a single action chunk.

        Args:
            current_state: (*state_shape,) -- single current state. Expanded to K
                           copies internally for batched rollout.

        Returns:
            action chunk: (pred_horizon, action_dim). In return_mode="best" (default)
                          this is the rank-1 elite of the final population -- which,
                          because truncation is a hard top-Ne cut, is exactly
                          argmax(last_returns), so `last_actions[last_returns.argmax()]`
                          is the returned chunk. In return_mode="mean" it is the terminal
                          elite mean, which is NOT a population member and for which that
                          identity does NOT hold; read last_command_action instead.

        Loop structure: 1 initial evaluation, then n_planning_itrs x (refit -> resample ->
        evaluate), then one terminal refit that does not resample. That is
        n_planning_itrs + 1 population evaluations -- identical to mppi_core.MPPI, so the
        two are compute-matched -- and n_planning_itrs + 1 refits, matching the reference's
        structure where refits == evals and the final mu is returned without resampling.
        return_mode="mean" adds one batch-of-1 evaluation on top (+0.02% at K=512).

        Side effects: sets last_actions / last_returns / last_elite_idx /
        last_elite_returns / last_mean / last_std / last_elite_std / last_command_action /
        last_command_return / last_mean_return / prev_best_action, and -- only when
        collect_diagnostics=True -- last_diagnostics, one dict per refit (so
        n_planning_itrs + 1 entries, one more than MPPI's; the extra terminal entry is the
        distribution CEM actually converged to). Fields:
            itr                  refit index
            resampled            False on the terminal refit, whose (mean, std) never
                                 sampled anything
            num_elites           Ne, constant -- carried so saved diagnostics self-document
            std_mean/min/max     the sampling std actually used. THIS is the collapse
                                 metric: it is what moved from 0.577 to 0.0154 (and to
                                 4.6e-4 at temperature 0.001) in mppi_core.MPPI.
            elite_std_mean       pre-floor elite spread; equals std_mean when min_std == 0
            return_spread        max - min over the full population (same definition as
                                 mppi_core's): is the cost landscape discriminative at all
            elite_return_spread  the same within the elite set; -> 0 means the elites have
                                 become degenerate
            elite_threshold      the Ne-th best return -- where truncation actually bites
            elite_return_mean    mean return over the elite set
            best_return          rank-1 return; its trace across iterations is progress
            mean_abs_mean        mean |mu|; approaching max_action_value signals the
                                 solution is saturating against the clamp

        There is deliberately NO `ess` field. Under hard top-Ne truncation the weights are
        uniform over the elite set, so 1/sum(w^2) is IDENTICALLY Ne -- a constant, not a
        measurement. Emitting it anyway would let it be printed, averaged and charted as
        though it meant something, which is precisely the failure that motivated this file:
        in mppi_core.MPPI, ESS reached 458-489 of 512 at the MOST collapsed settings,
        because a collapsed population has near-tied returns, which makes the softmax
        uniform, which maximises ESS. std_mean is the replacement. A consumer written
        against MPPI's diagnostics will raise KeyError here, which is the correct failure.
        """
        if self.collect_diagnostics:
            self.last_diagnostics = []

        actions = self._sample_initial_actions()
        actions = torch.clamp(actions, -self.max_action_value, self.max_action_value)
        returns = self._evaluate(current_state, actions)

        for itr in range(self.n_planning_itrs + 1):
            # topk returns indices in DESCENDING value order, so elite_idx[0] is the
            # global argmax of returns -- the property that makes return_mode="best"
            # both CEM-native (it is the rank-1 elite) and a drop-in for MPPI's argmax.
            elite_idx = torch.topk(returns, self.num_elites, largest=True).indices  # (Ne,)
            elites = actions[elite_idx]                                             # (Ne, T, nu)
            elite_returns = returns[elite_idx]                                      # (Ne,)

            mean = elites.mean(dim=0)                                               # (T, nu)
            # unbiased=False is ddof=0, matching numpy's .std() default that cem.py:73
            # relies on. torch's default (unbiased=True) would inflate sigma by
            # sqrt(Ne/(Ne-1)) AND return NaN at Ne=1, which would then silently poison
            # every remaining iteration through Normal(mean, nan).
            elite_std = elites.std(dim=0, unbiased=False)                           # (T, nu)
            # Additive epsilon, not clamp(min=eps) -- matches mppi_core.py:204 in the
            # collapse regime, where the two differ.
            std = torch.clamp(elite_std, min=self.min_std) + self.std_eps            # (T, nu)

            is_terminal = itr == self.n_planning_itrs

            if self.collect_diagnostics:
                # Recorded BEFORE resampling, so these describe the population that
                # produced this refit. Read-only -- no RNG consumed, no numerics touched.
                self.last_diagnostics.append(dict(
                    itr=itr,
                    resampled=not is_terminal,
                    num_elites=self.num_elites,
                    std_mean=float(std.mean().item()),
                    std_min=float(std.min().item()),
                    std_max=float(std.max().item()),
                    elite_std_mean=float(elite_std.mean().item()),
                    return_spread=float((returns.max() - returns.min()).item()),
                    elite_return_spread=float((elite_returns.max() - elite_returns.min()).item()),
                    elite_threshold=float(elite_returns.min().item()),
                    elite_return_mean=float(elite_returns.mean().item()),
                    best_return=float(elite_returns.max().item()),
                    mean_abs_mean=float(mean.abs().mean().item()),
                ))

            if is_terminal:
                break

            actions = torch.distributions.Normal(mean, std).sample((self.K,))       # (K, T, nu)
            actions = torch.clamp(actions, -self.max_action_value, self.max_action_value)
            returns = self._evaluate(current_state, actions)

        if self.return_mode == "best":
            best_idx = int(elite_idx[0])
            command_action = actions[best_idx]
            command_return = returns[best_idx]
            mean_return = None
        else:
            # mean is the average of already-clamped points, so by convexity it is
            # provably within [-A, A]; the clamp is a free invariant, not a correction.
            command_action = torch.clamp(mean, -self.max_action_value, self.max_action_value)
            # Score it for real rather than reporting a number for a chunk that was never
            # evaluated. One batch-of-1 dynamics call.
            mean_return = self._evaluate(current_state, command_action.unsqueeze(0), batch=1)[0]
            command_return = mean_return

        self.last_actions = actions
        self.last_returns = returns
        self.last_elite_idx = elite_idx
        self.last_elite_returns = elite_returns
        self.last_mean = mean
        self.last_std = std
        self.last_elite_std = elite_std
        self.last_command_action = command_action
        self.last_command_return = command_return
        self.last_mean_return = mean_return

        # Warm-start from the chunk actually executed. In "best" mode this is the argmax,
        # bit-identical to mppi_core.MPPI; in "mean" mode it correctly seeds from the mean
        # that was executed rather than from a candidate that was discarded.
        self.prev_best_action = command_action.detach().clone()

        return command_action


# ---------------------------------------------------------------------------
# Sanity-check demo -- trivial 2D point-mass, no dependencies beyond torch.
# Mirrors mppi_core.py's demo so the two can be run side by side.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    goal = torch.tensor([1.0, 1.0], device=device)

    # state_{t+1} = state_t + sum(action chunk); mirrors this project's "one chunk ->
    # one terminal state" dynamics model, which never exposes intermediate sub-steps.
    def dynamics_fn(state, action):
        # state: (n, 2), action: (n, T, 2) -> (n, 2)
        return state + action.sum(dim=1)

    def cost_fn(terminal_state):
        # terminal_state: (n, 2) -> (n,)
        return ((terminal_state - goal) ** 2).sum(dim=-1)

    cem = CEM(
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        pred_horizon=5,
        action_dim=2,
        num_samples=500,
        n_planning_itrs=9,
        device=device,
        elite_frac=0.1,          # -> Ne = 50
        max_action_value=0.5,
        collect_diagnostics=True,
    )
    print(f"CEM: K={cem.K}  Ne={cem.num_elites}  n_planning_itrs={cem.n_planning_itrs}  "
          f"return_mode={cem.return_mode!r}  min_std={cem.min_std}")

    state = torch.zeros(2, device=device)
    print(f"\n{'step':>4}  {'state':>26}  {'dist':>8}")
    print("-" * 46)
    for i in range(20):
        action_chunk = cem.command(state)    # (pred_horizon, action_dim)
        action = action_chunk[0]             # execute only the first real sub-step
        state = state + action
        dist = (state - goal).norm().item()
        print(f"{i:>4}  {str([round(x, 4) for x in state.tolist()]):>26}  {dist:>8.4f}")

    print("\nTerminal-refit diagnostics from the last command() call:")
    print("   itr:  " + " ".join(f"{d['itr']:>8d}" for d in cem.last_diagnostics))
    print("   std:  " + " ".join(f"{d['std_mean']:>8.4f}" for d in cem.last_diagnostics))
    print("  best:  " + " ".join(f"{d['best_return']:>8.4f}" for d in cem.last_diagnostics))

    # -- Assertion 1: the drop-in guarantee ---------------------------------
    # Under hard top-Ne truncation the rank-1 elite IS argmax(returns), so consumers
    # that index last_actions by argmax(last_returns) get exactly what command() returned.
    assert int(cem.last_elite_idx[0]) == int(torch.argmax(cem.last_returns)), \
        "elite_idx[0] is not argmax(last_returns) -- the MPPI drop-in contract is broken"
    assert torch.allclose(cem.last_actions[torch.argmax(cem.last_returns)],
                          cem.last_command_action), \
        "argmax(last_returns) does not index the chunk command() returned"
    print("\n[1/3] OK  argmax(last_returns) indexes the returned chunk")

    # -- Assertion 2: the RNG-parity guarantee ------------------------------
    # Same seed -> bit-identical first population as mppi_core.MPPI, so an A/B between
    # the two planners provably diverges only at the first refit. Safe to import:
    # mppi_core's demo is __main__-guarded.
    from mppi_core import MPPI

    mppi = MPPI(
        dynamics_fn=dynamics_fn,
        cost_fn=cost_fn,
        pred_horizon=5,
        action_dim=2,
        num_samples=500,
        mppi_temperature=0.1,
        n_planning_itrs=9,
        device=device,
        max_action_value=0.5,
    )
    # cem has run 20 command() calls above, so its warm-start state is populated and it
    # would take the shifted-previous-chunk branch. reset() puts it back on the cold-start
    # path, which is the one being compared (and exercises reset() while we are here).
    cem.reset()
    assert cem.prev_best_action is None, "reset() did not clear warm-start state"

    torch.manual_seed(123)
    mppi_first = mppi._sample_initial_actions()
    torch.manual_seed(123)
    cem_first = cem._sample_initial_actions()
    assert torch.equal(mppi_first, cem_first), \
        "initial populations differ -- the CEM/MPPI comparison would not be controlled"
    print("[2/3] OK  initial population is bit-identical to mppi_core.MPPI under one seed")

    # -- Assertion 3: the unbiased=False guarantee --------------------------
    # torch's default std(unbiased=True) would be NaN for a single elite, which would
    # propagate silently through Normal(mean, nan) and destroy the plan.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # the Ne=1 freeze warning is expected here
        degenerate = CEM(
            dynamics_fn=dynamics_fn,
            cost_fn=cost_fn,
            pred_horizon=5,
            action_dim=2,
            num_samples=500,
            n_planning_itrs=9,
            device=device,
            num_elites=1,
            max_action_value=0.5,
        )
    degenerate.command(torch.zeros(2, device=device))
    assert torch.isfinite(degenerate.last_std).all(), \
        "Ne=1 produced a non-finite std -- std(unbiased=False) is not being used"
    assert torch.isfinite(degenerate.last_command_action).all(), \
        "Ne=1 produced a non-finite action chunk"
    print(f"[3/3] OK  Ne=1 gives a finite std ({float(degenerate.last_std.mean()):.2e}), not NaN")

    print("\nAll assertions passed.")
