"""Cross-system ablation pipeline: types, protocols, validation, and composition.

Re-exports all public types for clean imports::

    from pipeline import FeatureResult, Predictions, Decision
"""
from pipeline.types import (
    Decision,
    FeatureResult,
    Featuriser,
    FittablePolicy,
    FoldResult,
    Model,
    Policy,
    PortfolioBuilder,
    PortfolioConfig,
    PortfolioResult,
    Predictions,
)
from pipeline.compose import SystemPipeline
from pipeline.validate import (
    validate_decisions,
    validate_feature_result,
    validate_pipeline,
    validate_predictions,
)

__all__ = [
    # Data types
    "Decision",
    "FeatureResult",
    "FoldResult",
    "PortfolioConfig",
    "PortfolioResult",
    "Predictions",
    # Protocols
    "Featuriser",
    "FittablePolicy",
    "Model",
    "Policy",
    "PortfolioBuilder",
    # Composition
    "SystemPipeline",
    # Validation
    "validate_decisions",
    "validate_feature_result",
    "validate_pipeline",
    "validate_predictions",
]
