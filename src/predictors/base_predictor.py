"""Abstract base class for all predictors."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.models.system import RefrigerationSystem


class BasePredictor(ABC):
    """
    All predictors share this interface.

    Workflow
    --------
    1. predictor.fit(system)          — train on the system's historical data
    2. predictor.predict(X_window)    — given a look-back window, return n_steps forecasts
    3. predictor.save(path)           — persist the trained model
    4. predictor.load(path)           — restore a saved model
    """

    def __init__(self, seq_length: int, n_steps: int):
        self.seq_length = seq_length
        self.n_steps = n_steps
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, system: RefrigerationSystem) -> None:
        """Train the predictor using data from the system object."""
        ...

    @abstractmethod
    def predict(self, X_window: np.ndarray) -> np.ndarray:
        """
        Forecast n_steps ahead.

        Parameters
        ----------
        X_window : np.ndarray, shape (seq_length, n_input_features)
            The look-back window of scaled input features.

        Returns
        -------
        np.ndarray, shape (n_steps, n_output_features)
            Predicted outputs in original (unscaled) units.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the trained model to disk."""
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore a previously saved model from disk."""
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Predictor is not fitted. Call fit() first.")
