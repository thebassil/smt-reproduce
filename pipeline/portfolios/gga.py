"""GGA: Gender-based Genetic Algorithm (Ansotegui et al., CP 2009).

Extension card (color 2) for portfolio axis.
Canvas ID: ext_gga_portfolio
Canvas group: sg_portfolio / e2834f88676d1882

Evolves portfolios (sets of config indices) via crossover, mutation,
and gender-based selection. Each individual is a frozenset of config
indices of size top_k.
"""
from __future__ import annotations

import time
from typing import Callable, List

import numpy as np

from pipeline.types import PortfolioBuilder, PortfolioConfig, PortfolioResult


# ---------------------------------------------------------------------------
# Shared helpers (kept in-file per plan — no separate module)
# ---------------------------------------------------------------------------

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
    """Convert config indices to PortfolioConfig objects."""
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
# GGAPortfolio
# ---------------------------------------------------------------------------

class GGAPortfolio:
    """GGA: Gender-based Genetic Algorithm (Ansotegui et al., CP 2009).

    Implements the :class:`PortfolioBuilder` protocol.

    Each individual is a ``frozenset`` of config indices (a portfolio of
    size ``top_k``).  Fitness = negative portfolio PAR-2 (higher = better).

    Sub-ablation knobs:
        selection       : "gender" | "tournament"
        population_size : 30, 50, 100
        generations     : 50, 100, 200
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.1,
        selection: str = "gender",
        top_k: int = 8,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection = selection
        self.top_k = top_k
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

        n_configs = len(config_names)
        rng = np.random.default_rng(self.seed)

        # Initialise: random subsets of size top_k
        population: list[frozenset[int]] = [
            frozenset(rng.choice(n_configs, self.top_k, replace=False))
            for _ in range(self.population_size)
        ]
        self.history_ = []

        for gen in range(self.generations):
            fitness = np.array(
                [evaluate_portfolio(ind, cost_matrix, timeout_s) for ind in population]
            )

            if self.selection == "gender":
                parents = self._gender_select(population, fitness, rng)
            else:
                parents = self._tournament_select(population, fitness, rng)

            offspring: list[frozenset[int]] = []
            for p1, p2 in parents:
                if rng.random() < self.crossover_rate:
                    child = self._crossover(p1, p2, n_configs, rng)
                else:
                    child = p1 if rng.random() < 0.5 else p2
                child = self._mutate(child, n_configs, rng)
                offspring.append(child)

            # Elitist survival: keep best from combined pool
            combined = population + offspring
            combined_fitness = np.array(
                [evaluate_portfolio(ind, cost_matrix, timeout_s) for ind in combined]
            )
            population = self._elitist_survive(combined, combined_fitness)

            best_fit = float(combined_fitness.max())
            self.history_.append({"generation": gen, "best_fitness": best_fit})

        # Select best individual from final population
        final_fitness = [
            evaluate_portfolio(ind, cost_matrix, timeout_s) for ind in population
        ]
        best = population[int(np.argmax(final_fitness))]

        elapsed = time.monotonic() - t0
        configs = indices_to_configs(sorted(best), config_names, config_args)
        return PortfolioResult(
            configs=configs,
            construction_method="gga",
            construction_time_s=elapsed,
        )

    # -- selection ----------------------------------------------------------

    @staticmethod
    def _gender_select(
        population: list[frozenset[int]],
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> list[tuple[frozenset[int], frozenset[int]]]:
        """Gender-based selection (Ansotegui et al.).

        Split by fitness median into competitive (top half) and
        non-competitive (bottom half). Each competitive individual pairs
        with a non-competitive one — balances exploitation and exploration.
        """
        median = float(np.median(fitness))
        competitive = [i for i, f in enumerate(fitness) if f >= median]
        non_competitive = [i for i, f in enumerate(fitness) if f < median]

        if not non_competitive:
            non_competitive = competitive

        rng.shuffle(competitive)  # type: ignore[arg-type]
        parents: list[tuple[frozenset[int], frozenset[int]]] = []
        for i in range(len(competitive)):
            nc_idx = non_competitive[int(rng.integers(len(non_competitive)))]
            parents.append((population[competitive[i]], population[nc_idx]))
        return parents

    @staticmethod
    def _tournament_select(
        population: list[frozenset[int]],
        fitness: np.ndarray,
        rng: np.random.Generator,
        tournament_size: int = 3,
    ) -> list[tuple[frozenset[int], frozenset[int]]]:
        """Standard tournament selection."""
        n = len(population)

        def _pick() -> frozenset[int]:
            contestants = rng.integers(n, size=tournament_size)
            winner = contestants[int(np.argmax(fitness[contestants]))]
            return population[winner]

        return [(_pick(), _pick()) for _ in range(n)]

    # -- genetic operators --------------------------------------------------

    def _crossover(
        self,
        p1: frozenset[int],
        p2: frozenset[int],
        n_configs: int,
        rng: np.random.Generator,
    ) -> frozenset[int]:
        """Union of parents, pruned to top_k (biased toward shared configs)."""
        union = list(p1 | p2)
        shared = p1 & p2

        if len(union) <= self.top_k:
            remaining = [i for i in range(n_configs) if i not in union]
            if remaining:
                extra = rng.choice(
                    remaining,
                    size=min(self.top_k - len(union), len(remaining)),
                    replace=False,
                )
                return frozenset(union) | frozenset(extra.tolist())
            return frozenset(union)

        # Keep shared configs, fill rest from non-shared
        non_shared = [c for c in union if c not in shared]
        rng.shuffle(non_shared)  # type: ignore[arg-type]
        keep = list(shared)
        for c in non_shared:
            if len(keep) >= self.top_k:
                break
            keep.append(c)
        return frozenset(keep[: self.top_k])

    def _mutate(
        self,
        individual: frozenset[int],
        n_configs: int,
        rng: np.random.Generator,
    ) -> frozenset[int]:
        """With probability mutation_rate per position, swap for random unused."""
        result = list(individual)
        for i in range(len(result)):
            if rng.random() < self.mutation_rate:
                unused = [c for c in range(n_configs) if c not in result]
                if unused:
                    result[i] = unused[int(rng.integers(len(unused)))]
        return frozenset(result)

    def _elitist_survive(
        self,
        combined: list[frozenset[int]],
        combined_fitness: np.ndarray,
    ) -> list[frozenset[int]]:
        """Sort by fitness, keep top population_size."""
        order = np.argsort(combined_fitness)[::-1]
        return [combined[i] for i in order[: self.population_size]]
