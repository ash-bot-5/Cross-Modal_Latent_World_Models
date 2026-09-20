"""
Standalone MPPI (Model Predictive Path Integral) controller.

In practice this is NOT a literal implementation of Williams et al. 2017 Algorithm 2 --
it is a Gaussian CEM whose per-iteration refit uses SOFTMAX weighting over all K samples
(Eq. 19-style temperature weighting, no noise-perturbation / importance-sampling-correction
formulation), ported from swm/planning_algos.py's plan_model_mppi and adapted into a
standalone, environment-agnostic, warm-started class. See planning/cem_core.py's module
docstring for the measured consequence of that softmax weighting on this project's
CLIP-latent dynamics model: mppi_temperature couples the cost SCALE to the refit std,
so a low temperature can collapse exploration width over successive planning calls.

Fully decoupled from any specific dynamics model, environment, or goal representation:
    dynamics_fn: (state: Tensor(K, *state_shape), action: Tensor(K, pred_horizon, action_dim))
                 -> Tensor(K, *state_shape). Called ONCE per planning iteration with the
                 whole candidate population -- pred_horizon sub-steps are consumed
                 internally by the dynamics model in a single forward pass (this project's
                 FiLM-MLP already predicts pred_horizon real env-steps ahead per call), not
                 looped over sub-step-by-sub-step here.
    cost_fn:     (terminal_state: Tensor(K, *state_shape)) -> Tensor(K,). Lower = better.
                 Negated internally into `returns` (higher = better) -- the identity
                 `returns == -cost` that planning/cem_core.py's CEM also upholds and that
                 several eval/ scripts rely on to recover cosine similarity from `returns`.

planning/cem_core.py's CEM is a drop-in sibling built to A/B against this class: EVERYTHING
except the refit rule (softmax weighting here vs. hard top-Ne elite truncation there) is
kept identical -- same uniform initial population, same warm-start-from-shifted-previous-
chunk across real env steps (shifted by n_execute, freed slots filled with zero), same
selection rule (argmax of the final population), same n_planning_itrs + 1 population
evaluations per command() call. See that file's docstring for the measured collapse
comparison between the two refit rules.
"""

from typing import Callable

import torch


class MPPI:
    """
    Model Predictive Path Integral controller (softmax-weighted Gaussian CEM -- see
    module docstring).

    Tensor shape conventions:
        current_state:  (*state_shape)                 -- single state, opaque shape
        actions:        (K, pred_horizon, action_dim)  -- candidate action chunks
        batched state:  (K, *state_shape)               -- K copies for parallel rollout
        returns:        (K,)
        mean / std:     (pred_horizon, action_dim)      -- per-dimension
    """

    def __init__(
        self,
        dynamics_fn: Callable,
        cost_fn: Callable,
        pred_horizon: int,
        action_dim: int,
        num_samples: int,
        mppi_temperature: float,
        n_planning_itrs: int,
        device,
        max_action_value: float = 1.0,
        warm_start_std: float = 0.1,
        n_execute: int = 1,
        collect_diagnostics: bool = False,
    ):
        """
        Args:
            dynamics_fn:      (state: Tensor(K, *), action: Tensor(K, pred_horizon, action_dim))
                               -> Tensor(K, *). Called once per planning iteration with the
                               full candidate population. State shape is never inspected.
            cost_fn:          (terminal_state: Tensor(K, *)) -> Tensor(K,). Lower = better.
                               Negated internally into `returns` (higher = better).
            pred_horizon:     T -- number of real sub-steps packed into one action chunk
                               (matches the dynamics model's own HORIZON).
            action_dim:       nu -- per-sub-step action dimensionality (e.g. 2 for (dx, dy)).
            num_samples:      K -- population size sampled per planning iteration.
            mppi_temperature: softmax temperature applied to `returns` when computing refit
                               weights. Low -> greedy/sharp weighting (can collapse the refit
                               std over successive calls); high -> near-uniform weighting.
            n_planning_itrs:  number of (refit -> resample -> evaluate) cycles per command()
                               call, plus one initial evaluation and one terminal refit that
                               does not resample -- n_planning_itrs + 1 population
                               evaluations total, compute-matched with planning/cem_core.py's
                               CEM.
            device:           torch device for all internal tensors.
            max_action_value: symmetric clamp bound. Defaults to 1.0 since this project's
                               actions are normalized to [-1, 1].
            warm_start_std:   per-dimension std used to sample the initial population around
                               the shifted previous chunk, for every command() call after the
                               first (or after reset()). The very first call (or the first
                               after reset()) instead draws from Uniform(-max_action_value,
                               max_action_value), ignoring this value entirely.
            n_execute:        how many sub-steps of the returned chunk the CALLER executes
                               before calling command() again. Only affects the warm start,
                               which shifts the previous chunk forward by exactly this much.
                               Must be 1 <= n_execute <= pred_horizon; at n_execute ==
                               pred_horizon the whole chunk has been consumed, so the warm
                               start degenerates to all-zeros (correctly -- no information
                               about the executed chunk remains). Defaults to 1.
            collect_diagnostics: when True, command() records per-refit statistics into
                               self.last_diagnostics: itr, resampled, ess (effective sample
                               size = 1/sum(weights**2)), std_mean/min/max, return_spread,
                               best_return, mean_abs_mean. Purely observational -- reads
                               tensors the algorithm already computed, consumes no RNG, and
                               changes no numerics -- so leaving it False (the default)
                               costs nothing.
        """
        if not (1 <= n_execute <= pred_horizon):
            raise ValueError(
                f"n_execute must satisfy 1 <= n_execute <= pred_horizon, got n_execute="
                f"{n_execute} with pred_horizon={pred_horizon}"
            )

        self.dynamics_fn = dynamics_fn
        self.cost_fn = cost_fn
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.K = num_samples
        self.mppi_temperature = mppi_temperature
        self.n_planning_itrs = n_planning_itrs
        self.device = device
        self.max_action_value = max_action_value
        self.warm_start_std = warm_start_std
        self.n_execute = n_execute
        self.collect_diagnostics = collect_diagnostics

        # Final-population state, published after each command() call. argmax(last_returns)
        # indexes the chunk command() returned -- several eval/ scripts assert this.
        self.last_actions: torch.Tensor | None = None
        self.last_returns: torch.Tensor | None = None

        # Per-refit diagnostics from the most recent command() call -- only populated when
        # collect_diagnostics=True, else stays None. n_planning_itrs + 1 entries (the extra
        # terminal entry is the distribution the population actually converged to).
        self.last_diagnostics: list[dict] | None = None

        # Warm-start state: the chunk command() returned last call, or None for the very
        # first call (or after reset()).
        self.prev_best_action: torch.Tensor | None = None

    def reset(self):
        """Clear warm-start state -- call at an episode boundary so a new rollout
        doesn't get seeded from a previous, unrelated episode's last chunk."""
        self.prev_best_action = None

    def _sample_initial_actions(self) -> torch.Tensor:
        """The very first population of each command() call. With no warm-start state,
        this is Uniform(-max_action_value, max_action_value), drawn via
        torch.empty(...).uniform_() rather than torch.rand or torch.distributions.Uniform,
        which consume the global RNG differently -- planning/cem_core.py's CEM mirrors this
        expression verbatim so the two planners draw a bit-identical first population under
        the same torch.manual_seed."""
        if self.prev_best_action is None:
            return (
                torch.empty(self.K, self.pred_horizon, self.action_dim, device=self.device).uniform_(-1.0, 1.0)
                * self.max_action_value
            )

        # Drop the just-executed sub-steps, shift the rest forward, and append that many
        # fresh zero-action sub-steps (the u_init=0 convention). The shift is n_execute, not
        # 1: the caller may execute several sub-steps of the chunk per plan, and shifting by
        # less would seed the next search from actions the robot has already performed.
        shifted = torch.roll(self.prev_best_action, -self.n_execute, dims=0)
        shifted[-self.n_execute:] = 0.0

        actions = torch.distributions.Normal(shifted, self.warm_start_std).sample((self.K,))
        return torch.clamp(actions, -self.max_action_value, self.max_action_value)

    def _evaluate(self, current_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll a population through dynamics_fn + cost_fn once, return `returns` (K,)
        (higher = better -- negated cost). current_state's shape is expanded, never
        inspected -- the opaque-state contract is preserved."""
        state = current_state.unsqueeze(0).expand(self.K, *current_state.shape).clone()
        state = self.dynamics_fn(state, actions)
        cost = self.cost_fn(state)
        return -cost

    def command(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Run one full planning pass and return a single action chunk.

        Args:
            current_state: (*state_shape,) -- single current state. Expanded to K copies
                           internally for batched rollout.

        Returns:
            action chunk: (pred_horizon, action_dim) -- the rank-1 member (by `returns`) of
                          the final population. argmax(last_returns) indexes exactly this
                          chunk, i.e. last_actions[last_returns.argmax()] == the returned
                          chunk.

        Loop structure: 1 initial evaluation, then n_planning_itrs x (softmax refit ->
        resample -> evaluate), then one terminal refit that does not resample -- matching
        planning/cem_core.py's CEM so the two are compute-matched population-evaluation for
        population-evaluation.
        """
        if self.collect_diagnostics:
            self.last_diagnostics = []

        actions = self._sample_initial_actions()
        actions = torch.clamp(actions, -self.max_action_value, self.max_action_value)
        returns = self._evaluate(current_state, actions)

        for itr in range(self.n_planning_itrs + 1):
            # Softmax reward-weighting (Williams et al. Eq. 19-style temperature weight,
            # not the noise-perturbation IS-correction formulation).
            weights = torch.softmax(returns / self.mppi_temperature, dim=0)              # (K,)
            mean = torch.sum(weights[:, None, None] * actions, dim=0)                     # (T, nu)
            std = torch.sqrt(
                torch.sum(weights[:, None, None] * (actions - mean) ** 2, dim=0)
            ) + 1e-9                                                                       # (T, nu)

            is_terminal = itr == self.n_planning_itrs

            if self.collect_diagnostics:
                # Recorded BEFORE resampling, so these describe the population that
                # produced this refit. Read-only -- no RNG consumed, no numerics touched.
                self.last_diagnostics.append(dict(
                    itr=itr,
                    resampled=not is_terminal,
                    ess=float((1.0 / (weights ** 2).sum()).item()),
                    std_mean=float(std.mean().item()),
                    std_min=float(std.min().item()),
                    std_max=float(std.max().item()),
                    return_spread=float((returns.max() - returns.min()).item()),
                    best_return=float(returns.max().item()),
                    mean_abs_mean=float(mean.abs().mean().item()),
                ))

            if is_terminal:
                break

            actions = torch.distributions.Normal(mean, std).sample((self.K,))            # (K, T, nu)
            actions = torch.clamp(actions, -self.max_action_value, self.max_action_value)
            returns = self._evaluate(current_state, actions)

        best_idx = int(torch.argmax(returns))
        command_action = actions[best_idx]

        self.last_actions = actions
        self.last_returns = returns

        # Warm-start from the chunk actually executed.
        self.prev_best_action = command_action.detach().clone()

        return command_action


# ---------------------------------------------------------------------------
# Sanity-check demo — trivial 2D point-mass, no dependencies beyond torch.
# Mirrors cem_core.py's demo so the two can be run side by side.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cpu"

    goal = torch.tensor([1.0, 1.0], device=device)

    # state_{t+1} = state_t + sum(action chunk); mirrors this project's "one chunk -> one
    # terminal state" dynamics model, which never exposes intermediate sub-steps.
    def dynamics_fn(state, action):
        # state: (K, 2), action: (K, T, 2) -> (K, 2)
        return state + action.sum(dim=1)

    def cost_fn(terminal_state):
        # terminal_state: (K, 2) -> (K,)
        return ((terminal_state - goal) ** 2).sum(dim=-1)

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
        collect_diagnostics=True,
    )
    print(f"MPPI: K={mppi.K}  mppi_temperature={mppi.mppi_temperature}  "
          f"n_planning_itrs={mppi.n_planning_itrs}  warm_start_std={mppi.warm_start_std}")

    state = torch.zeros(2, device=device)
    print(f"\n{'step':>4}  {'state':>26}  {'dist':>8}")
    print("-" * 46)
    for i in range(20):
        action_chunk = mppi.command(state)    # (pred_horizon, action_dim)
        action = action_chunk[0]              # execute only the first real sub-step
        state = state + action
        dist = (state - goal).norm().item()
        print(f"{i:>4}  {str([round(x, 4) for x in state.tolist()]):>26}  {dist:>8.4f}")

    print("\nTerminal-refit diagnostics from the last command() call:")
    print("   itr:  " + " ".join(f"{d['itr']:>8d}" for d in mppi.last_diagnostics))
    print("   std:  " + " ".join(f"{d['std_mean']:>8.4f}" for d in mppi.last_diagnostics))
    print("   ess:  " + " ".join(f"{d['ess']:>8.2f}" for d in mppi.last_diagnostics))
    print("  best:  " + " ".join(f"{d['best_return']:>8.4f}" for d in mppi.last_diagnostics))

    # -- Assertion: the drop-in guarantee ------------------------------------
    best_idx = int(torch.argmax(mppi.last_returns))
    assert torch.equal(mppi.last_actions[best_idx], action_chunk), \
        "argmax(last_returns) does not index the chunk command() returned"
    print("\n[1/1] OK  argmax(last_returns) indexes the returned chunk")
