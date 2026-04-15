"""AdaBoost EHM model card (from scratch).

Reimplements MachSMT's EHM model architecture:
  RobustScaler → MultiOutputRegressor(AdaBoostRegressor)

Fits on log-transformed PAR-2 runtimes, predicts per-solver costs.

Study reference: artifacts/machsmt/MachSMT/machsmt/ml/mach_scikit.py
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from ..types import FeatureResult, Predictions


class AdaBoostEHM:
    """AdaBoost regressor with EHM (Exponential loss, Hydra-configurable).

    Architecture:
        RobustScaler → MultiOutputRegressor(AdaBoostRegressor)

    Hydra config: conf/model/adaboost_ehm.yaml
    """

    input_type: str = "VECTOR"
    output_type: str = "scores"

    def __init__(
        self,
        n_estimators: int = 200,
        loss: str = "exponential",
        learning_rate: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.loss = loss
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._pipeline = None
        self._config_names: list[str] = []

    def _build_pipeline(self):
        return make_pipeline(
            RobustScaler(),
            MultiOutputRegressor(
                AdaBoostRegressor(
                    n_estimators=self.n_estimators,
                    loss=self.loss,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
            ),
        )

    def fit(
        self,
        features: list[FeatureResult],
        cost_matrix: np.ndarray,
        config_names: list[str],
    ) -> dict:
        """Train on feature vectors and PAR-2 cost matrix.

        Args:
            features: list of FeatureResult from featuriser
            cost_matrix: (n_instances, n_configs) PAR-2 scores in seconds
            config_names: solver config labels (column order)

        Returns:
            Training metadata dict.
        """
        X = np.vstack([f.features for f in features])
        # Log-transform costs (matches MachSMT: np.log(score + 1))
        Y = np.log1p(cost_matrix)

        self._config_names = list(config_names)
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X, Y)

        return {"n_train": X.shape[0], "n_features": X.shape[1]}

    def predict(self, features: list[FeatureResult]) -> Predictions:
        """Predict per-solver costs for a batch of instances.

        Returns predicted costs in original scale (inverse log-transform).
        """
        if self._pipeline is None:
            raise RuntimeError("Model not trained — call fit() first")

        X = np.vstack([f.features for f in features])
        Y_log = self._pipeline.predict(X)
        # Inverse log-transform
        Y = np.expm1(Y_log)

        return Predictions(
            values=Y,
            output_type="scores",
            config_names=self._config_names,
            instance_ids=[f.instance_id for f in features],
        )

    def save(self, path: Union[str, Path]) -> None:
        """Serialize model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "pipeline": self._pipeline,
                    "config_names": self._config_names,
                    "params": {
                        "n_estimators": self.n_estimators,
                        "loss": self.loss,
                        "learning_rate": self.learning_rate,
                        "random_state": self.random_state,
                    },
                },
                f,
            )

    def load(self, path: Union[str, Path]) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._pipeline = data["pipeline"]
        self._config_names = data["config_names"]
        params = data["params"]
        self.n_estimators = params["n_estimators"]
        self.loss = params["loss"]
        self.learning_rate = params["learning_rate"]
        self.random_state = params["random_state"]
