"""
LSTM predictor — placeholder.

TODO: port the PyTorch LSTM from the notebook:
  notebooks/refrigeration_prediction_and_optimization_using_RNN_and_RL.ipynb
  (cells 9, 11, 14)

The fit/predict/save/load signatures below are the integration contract;
replace the NotImplementedError bodies with the real implementation.
"""

from pathlib import Path

import numpy as np

from src.predictors.base_predictor import BasePredictor
from src.models.system import RefrigerationSystem


class LSTMPredictor(BasePredictor):
    """
    LSTM-based N-step predictor (not yet implemented).

    Parameters
    ----------
    seq_length : int
        Look-back window length.
    n_steps : int
        Forecast horizon.
    hidden_size : int
        LSTM hidden layer size.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability between LSTM layers.
    learning_rate : float
        Adam optimizer learning rate.
    num_epochs : int
        Training epochs.
    use_optuna : bool
        If True, run Optuna hyperparameter search before final training.
    """

    def __init__(
        self,
        seq_length: int = 6,
        n_steps: int = 12,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        use_optuna: bool = False,
    ):
        super().__init__(seq_length=seq_length, n_steps=n_steps)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_optuna = use_optuna

    def fit(self, system: RefrigerationSystem) -> None:
        raise NotImplementedError("LSTMPredictor.fit() is not yet implemented.")

    def predict(self, X_window: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X_window : np.ndarray, shape (seq_length, n_input_features)

        Returns
        -------
        np.ndarray, shape (n_steps, n_output_features)
        """
        raise NotImplementedError("LSTMPredictor.predict() is not yet implemented.")

    def save(self, path: Path) -> None:
        raise NotImplementedError("LSTMPredictor.save() is not yet implemented.")

    def load(self, path: Path) -> None:
        raise NotImplementedError("LSTMPredictor.load() is not yet implemented.")
