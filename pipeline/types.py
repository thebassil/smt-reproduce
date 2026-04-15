"""Shared types for the cross-system ablation pipeline.

Three data types flow through the pipeline:
  Featuriser -> FeatureResult -> Model -> Predictions -> Policy -> Decision

All models use Pydantic v2 (model_config, not class Config).
All protocols use typing.Protocol with @runtime_checkable.
"""
from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FeatureResult(BaseModel):
    """Output of a featuriser card for a single instance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    features: Any  # np.ndarray (VECTOR) or torch_geometric.data.Data (GRAPH)
    feature_type: Literal["VECTOR", "GRAPH"]
    wall_time_ms: float = Field(ge=0, description="Measured extraction time in milliseconds")
    n_features: int = Field(ge=0, description="Vector dim (VECTOR) or node feature dim (GRAPH)")
    instance_id: str = Field(description="Instance identifier (file path or DB id)")
    logic: Optional[str] = Field(
        default=None,
        description="SMT-LIB logic (e.g. QF_LIA). Needed by SolverLogicPWC and per-logic models.",
    )


class Predictions(BaseModel):
    """Output of a model card for a batch of instances.

    values shape: (n_instances, n_configs)
      - "scores": raw predicted runtimes. Lower = better.
      - "ranking": ordinal ranks. Lower = better.
      - "distribution": probabilities summing to 1 per instance. Higher = more likely to win.
        Required by exponential timer and probability-proportional policies.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: Any  # np.ndarray shape (n_instances, n_configs)
    output_type: Literal["scores", "ranking", "distribution"]
    config_names: List[str]  # length = n_configs
    instance_ids: List[str]  # length = n_instances


class Decision(BaseModel):
    """Output of a policy card for a single instance.

    decision_type semantics:
      - "select": single solver pick. selected_config is set.
      - "schedule": ordered solver list with time budgets. schedule is set.
      - "rank": full solver ordering without time allocation. ranking is set.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instance_id: str
    decision_type: Literal["select", "schedule", "rank"]
    selected_config: Optional[str] = Field(
        default=None,
        description="The single pick (for 'select'), or None for schedule/rank.",
    )
    schedule: Optional[List[Tuple[str, float]]] = Field(
        default=None,
        description="Ordered (config, time_budget_s) pairs. Budgets must sum <= total budget.",
    )
    ranking: Optional[List[str]] = Field(
        default=None,
        description="Full ordering best-to-worst. None for select/schedule.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Model confidence in this decision. Used by Confidence-Gated Fallback.",
    )


class PortfolioConfig(BaseModel):
    """A single solver-config in a portfolio."""

    name: str
    solver: str  # solver binary name, e.g. "z3", "cvc5"
    args: List[str]  # command-line arguments


class PortfolioResult(BaseModel):
    """Output of a portfolio construction card."""

    configs: List[PortfolioConfig]
    construction_method: str  # e.g. "hydra", "gga", "fixed"
    construction_time_s: float = Field(ge=0)


class FoldResult(BaseModel):
    """Aggregated results from one fold of cross-validation."""

    fold_id: int
    decisions: List[Decision]
    metrics: Dict[str, float]  # e.g. {"par2": ..., "solved": ..., "solved_pct": ...}


# ---------------------------------------------------------------------------
# Protocol ABCs
# ---------------------------------------------------------------------------

@runtime_checkable
class Featuriser(Protocol):
    """Protocol for featuriser cards.

    input_type declares what kind of features this featuriser produces
    (VECTOR or GRAPH). Declared as ClassVar on the implementing class.
    """

    input_type: ClassVar[Literal["VECTOR", "GRAPH"]]

    def extract(self, instance_path: Union[str, Path]) -> FeatureResult: ...

    def extract_batch(self, instance_paths: List[Union[str, Path]]) -> List[FeatureResult]:
        """Default: loop over extract(). Implementations may override for efficiency."""
        ...


@runtime_checkable
class Model(Protocol):
    """Protocol for model cards.

    input_type: what kind of features this model consumes (VECTOR or GRAPH).
    output_type: what kind of predictions it produces (scores, ranking, distribution).
    """

    input_type: ClassVar[Literal["VECTOR", "GRAPH"]]
    output_type: ClassVar[Literal["scores", "ranking", "distribution"]]

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> dict: ...

    def predict(self, features: List[FeatureResult]) -> Predictions: ...

    def save(self, path: Union[str, Path]) -> None: ...

    def load(self, path: Union[str, Path]) -> None: ...


@runtime_checkable
class Policy(Protocol):
    """Protocol for policy cards.

    decide() returns one Decision per instance in predictions.
    """

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]: ...


@runtime_checkable
class FittablePolicy(Protocol):
    """Protocol for policy cards that learn during training (e.g. ClusterDispatch).

    Extends Policy with a fit() method called by the runner during training.
    """

    def fit(
        self,
        features: List[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: List[str],
    ) -> None: ...

    def decide(self, predictions: Predictions, budget_s: float) -> List[Decision]: ...


@runtime_checkable
class PortfolioBuilder(Protocol):
    """Protocol for portfolio construction cards."""

    def build(
        self,
        instances: List[str],
        solver_space: dict,
        cost_fn: Callable,
    ) -> PortfolioResult: ...
