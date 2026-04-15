"""Hydra: iterative complementary configuration (Xu et al., AAAI 2010).

Extension card (color 2) for portfolio axis.
Canvas ID: ext_hydra_portfolio
Canvas group: sg_portfolio / e2834f88676d1882

Iteratively adds configs tuned on the residual unsolved instance set.
Each round's inner configurator (SMAC or ParamILS) selects the best
complement from the discrete config space.
"""
from __future__ import annotations

import time
from typing import Callable, List

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from pipeline.types import PortfolioBuilder, PortfolioConfig, PortfolioResult


# ---------------------------------------------------------------------------
# Shared helpers (kept in-file per plan — no separate module)
# ---------------------------------------------------------------------------

def compute_unsolved_mask(
    cost_matrix: np.ndarray,
    portfolio_indices: List[int],
    timeout_s: float,
) -> np.ndarray:
    """Boolean mask: True for instances not solved by any portfolio config."""
    if not portfolio_indices:
        return np.ones(cost_matrix.shape[0], dtype=bool)
    portfolio_costs = cost_matrix[:, portfolio_indices]
    best_costs = portfolio_costs.min(axis=1)
    return best_costs >= timeout_s


def evaluate_portfolio(
    config_indices,
    cost_matrix: np.ndarray,
    timeout_s: float,
) -> float:
    """Negative mean PAR-2 for portfolio (higher = better for maximisation)."""
    if not config_indices:
        return -np.inf
    subset = cost_matrix[:, list(config_indices)]
    vbs = subset.min(axis=1)
    return -vbs.mean()


def indices_to_configs(
    indices: List[int],
    config_names: List[str],
    config_args: dict | None = None,
) -> List[PortfolioConfig]:
    """Convert config indices to PortfolioConfig objects.

    ``config_names`` uses ``"solver::config_name"`` format from DB.
    If ``config_args`` dict is provided, looks up actual solver args;
    otherwise uses the config_name as a placeholder arg.
    """
    configs: list[PortfolioConfig] = []
    for idx in indices:
        name = config_names[idx]
        solver, config_name = name.split("::", 1)
        if config_args and name in config_args:
            args = config_args[name]
            if isinstance(args, str):
                args = args.split()
        else:
            args = [config_name]
        configs.append(PortfolioConfig(name=name, solver=solver, args=args))
    return configs


# ---------------------------------------------------------------------------
# HydraPortfolio
# ---------------------------------------------------------------------------

class HydraPortfolio:
    """Hydra: iterative complementary configuration (Xu et al., AAAI 2010).

    Implements the :class:`PortfolioBuilder` protocol.

    Sub-ablation knobs:
        configurator : "smac" | "paramils"
        max_configs  : portfolio budget (e.g. 4, 8, 16)
        warm_start   : seed from existing portfolio indices
        targeting    : "complement" (residual unsolved) | "random" (control)
    """

    def __init__(
        self,
        max_configs: int = 8,
        configurator: str = "smac",
        warm_start: bool = False,
        targeting: str = "complement",
        seed: int = 42,
    ) -> None:
        self.max_configs = max_configs
        self.configurator = configurator
        self.warm_start = warm_start
        self.targeting = targeting
        self.seed = seed
        self.history_: list[dict] = []  # populated after build()

    # -- protocol method ----------------------------------------------------

    def build(
        self,
        instances: List[str],
        solver_space: dict,
        cost_fn: Callable,
    ) -> PortfolioResult:
        t0 = time.monotonic()

        cost_matrix: np.ndarray = solver_space["cost_matrix"]
        config_names: List[str] = solver_space["config_names"]
        timeout_s: float = solver_space["timeout_s"]
        config_args: dict | None = solver_space.get("config_args")

        n_instances, _n_configs = cost_matrix.shape
        rng = np.random.default_rng(self.seed)

        # Warm-start: seed portfolio from provided indices
        if self.warm_start and "warm_start_indices" in solver_space:
            portfolio_indices: list[int] = list(solver_space["warm_start_indices"])
        else:
            portfolio_indices = []

        self.history_ = []

        for round_idx in range(self.max_configs - len(portfolio_indices)):
            # 1. Compute residual unsolved set
            if self.targeting == "complement":
                unsolved_mask = compute_unsolved_mask(
                    cost_matrix, portfolio_indices, timeout_s
                )
            else:  # random targeting (ablation control)
                unsolved_mask = rng.random(n_instances) < 0.5
                if unsolved_mask.sum() == 0:
                    unsolved_mask[0] = True

            if unsolved_mask.sum() == 0:
                break

            # 2. Inner configurator selects best config for residual
            residual_costs = cost_matrix[unsolved_mask]
            if self.configurator == "smac":
                best_idx = self._smac_select(
                    residual_costs, exclude=portfolio_indices, rng=rng
                )
            else:
                best_idx = self._paramils_select(
                    residual_costs, exclude=portfolio_indices, rng=rng
                )

            portfolio_indices.append(best_idx)
            self.history_.append(
                {
                    "round": round_idx,
                    "unsolved": int(unsolved_mask.sum()),
                    "selected": config_names[best_idx],
                }
            )

        elapsed = time.monotonic() - t0
        configs = indices_to_configs(portfolio_indices, config_names, config_args)
        return PortfolioResult(
            configs=configs,
            construction_method="hydra",
            construction_time_s=elapsed,
        )

    # -- inner configurators ------------------------------------------------

    def _smac_select(
        self,
        residual_costs: np.ndarray,
        exclude: List[int],
        rng: np.random.Generator,
    ) -> int:
        """Bayesian optimisation over discrete config space.

        Fits a RandomForestRegressor on (config_index -> mean PAR-2 on residual).
        Uses Expected Improvement acquisition over non-excluded configs.
        """
        n_configs = residual_costs.shape[1]
        mean_costs = residual_costs.mean(axis=0)

        candidates = [i for i in range(n_configs) if i not in exclude]
        if not candidates:
            return int(rng.integers(n_configs))

        # Train RF surrogate on all configs
        X_train = np.arange(n_configs).reshape(-1, 1)
        y_train = mean_costs

        rf = RandomForestRegressor(
            n_estimators=10, random_state=int(rng.integers(2**31))
        )
        rf.fit(X_train, y_train)

        # Predict mean and variance for candidates
        X_cand = np.array(candidates).reshape(-1, 1)
        preds = np.array([t.predict(X_cand) for t in rf.estimators_])
        mu = preds.mean(axis=0)
        sigma = preds.std(axis=0) + 1e-9

        # Expected Improvement (minimising cost)
        best_so_far = mean_costs[candidates].min()
        improvement = best_so_far - mu
        ei = improvement / sigma  # simplified EI — sufficient for ranking

        best_cand_pos = int(np.argmax(ei))
        return candidates[best_cand_pos]

    def _paramils_select(
        self,
        residual_costs: np.ndarray,
        exclude: List[int],
        rng: np.random.Generator,
    ) -> int:
        """Iterative local search over discrete config space.

        Starts from random config. Evaluates neighbours (nearby indices).
        Accepts improving moves. 3 random restarts. Returns best found.
        """
        n_configs = residual_costs.shape[1]
        mean_costs = residual_costs.mean(axis=0)

        candidates = [i for i in range(n_configs) if i not in exclude]
        if not candidates:
            return int(rng.integers(n_configs))

        n_restarts = 3
        best_idx = candidates[0]
        best_cost = mean_costs[best_idx]

        for _ in range(n_restarts):
            current = candidates[int(rng.integers(len(candidates)))]
            current_cost = mean_costs[current]

            # Local search: check neighbours (+-1, +-2 in candidate list)
            improved = True
            while improved:
                improved = False
                pos = candidates.index(current)
                for delta in [-2, -1, 1, 2]:
                    nbr_pos = pos + delta
                    if 0 <= nbr_pos < len(candidates):
                        nbr = candidates[nbr_pos]
                        nbr_cost = mean_costs[nbr]
                        if nbr_cost < current_cost:
                            current = nbr
                            current_cost = nbr_cost
                            improved = True
                            break

            if current_cost < best_cost:
                best_idx = current
                best_cost = current_cost

        return best_idx
