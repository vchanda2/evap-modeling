"""
Decision Tree predictor for N-step evaporator/compressor forecasting.

Strategy: recursive (autoregressive) multi-step prediction.
  - One MultiOutputDecisionTreeRegressor is trained to predict
    all outputs at t+1 given a flattened seq_length look-back window.
  - For steps 2..N the predicted outputs are fed back as the next
    window's inputs (the evap_temp columns), while exogenous inputs
    (setpoints, dry_bulb, wet_bulb, system_on) hold their last known value.
"""

import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

from src.predictors.base_predictor import BasePredictor
from src.models.system import RefrigerationSystem


class DecisionTreePredictor(BasePredictor):
    """
    Parameters
    ----------
    seq_length : int
        Number of past time steps used as input (look-back window).
    n_steps : int
        Number of future time steps to predict (forecast horizon).
    max_depth : int | None
        Max depth of the underlying decision tree (None = unconstrained).
    test_size : float
        Fraction of data held out for evaluation during fit().
    """

    def __init__(
        self,
        seq_length: int = 6,
        n_steps: int = 12,
        max_depth: int | None = 10,
        test_size: float = 0.2,
    ):
        super().__init__(seq_length=seq_length, n_steps=n_steps)
        self.max_depth = max_depth
        self.test_size = test_size

        self._model: MultiOutputRegressor | None = None
        self._x_scaler: StandardScaler = StandardScaler()
        self._y_scaler: StandardScaler = StandardScaler()
        self._input_cols: list[str] = []
        self._output_cols: list[str] = []

        # Indices into input_cols that correspond to output_cols
        # (the features that get overwritten during recursive rollout)
        self._output_in_input_indices: list[int] = []

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, system: RefrigerationSystem) -> None:
        """Train the decision tree on the system's historical data."""
        self._input_cols = system.input_feature_cols
        self._output_cols = system.output_feature_cols

        # Build sequences: X shape (N, seq_len, n_in), y shape (N, n_steps, n_out)
        X_seq, y_seq, _ = system.build_sequences(self.seq_length, self.n_steps)

        # Flatten input window: (N, seq_len * n_in)
        n_samples = X_seq.shape[0]
        X_flat = X_seq.reshape(n_samples, -1)

        # For the one-step model we only use y at step 0: (N, n_out)
        y_one_step = y_seq[:, 0, :]

        # Train/test split (no shuffle — time series)
        X_train, X_test, y_train, y_test = train_test_split(
            X_flat, y_one_step, test_size=self.test_size, shuffle=False
        )

        # Scale
        X_train_s = self._x_scaler.fit_transform(X_train)
        X_test_s = self._x_scaler.transform(X_test)
        y_train_s = self._y_scaler.fit_transform(y_train)

        # Fit multi-output decision tree
        base = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
        self._model = MultiOutputRegressor(base)
        self._model.fit(X_train_s, y_train_s)

        # Evaluate on test set
        y_test_pred_s = self._model.predict(X_test_s)
        y_test_pred = self._y_scaler.inverse_transform(y_test_pred_s)
        mse = float(np.mean((y_test_pred - y_test) ** 2))
        print(f"[DecisionTreePredictor] Test MSE (1-step): {mse:.4f}")

        # Pre-compute which input-feature indices map to output features
        # (needed during recursive rollout to overwrite evap temps)
        self._output_in_input_indices = [
            self._input_cols.index(col)
            for col in self._output_cols
            if col in self._input_cols
        ]
        self._output_col_in_input_map = {
            out_idx: self._input_cols.index(col)
            for out_idx, col in enumerate(self._output_cols)
            if col in self._input_cols
        }

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X_window: np.ndarray) -> np.ndarray:
        """
        Recursively forecast n_steps ahead.

        Parameters
        ----------
        X_window : np.ndarray, shape (seq_length, n_input_features)
            Most recent look-back window in *original* (unscaled) units.

        Returns
        -------
        np.ndarray, shape (n_steps, n_output_features)
            Predicted values in original units.
        """
        self._check_fitted()

        window = X_window.copy().astype(np.float32)  # (seq_len, n_in)
        results = []

        for _ in range(self.n_steps):
            flat = window.reshape(1, -1)
            flat_s = self._x_scaler.transform(flat)
            pred_s = self._model.predict(flat_s)                  # (1, n_out)
            pred = self._y_scaler.inverse_transform(pred_s)[0]    # (n_out,)
            results.append(pred)

            # Roll window forward: drop oldest step, append a new step
            new_step = window[-1].copy()
            # Overwrite the output-that-are-also-inputs with this step's predictions
            for out_idx, in_idx in self._output_col_in_input_map.items():
                new_step[in_idx] = pred[out_idx]

            window = np.vstack([window[1:], new_step])

        return np.array(results)  # (n_steps, n_out)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "x_scaler": self._x_scaler,
            "y_scaler": self._y_scaler,
            "input_cols": self._input_cols,
            "output_cols": self._output_cols,
            "output_col_in_input_map": self._output_col_in_input_map,
            "seq_length": self.seq_length,
            "n_steps": self.n_steps,
            "max_depth": self.max_depth,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[DecisionTreePredictor] Saved to {path}")

    def load(self, path: Path) -> None:
        path = Path(path)
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._x_scaler = payload["x_scaler"]
        self._y_scaler = payload["y_scaler"]
        self._input_cols = payload["input_cols"]
        self._output_cols = payload["output_cols"]
        self._output_col_in_input_map = payload["output_col_in_input_map"]
        self.seq_length = payload["seq_length"]
        self.n_steps = payload["n_steps"]
        self.max_depth = payload["max_depth"]
        self._is_fitted = True
        print(f"[DecisionTreePredictor] Loaded from {path}")
