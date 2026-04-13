"""
RefrigerationSystem: aggregates Evaporator and Compressor objects
and exposes the feature/target matrices used by predictors.
"""

import numpy as np
import pandas as pd

from src.models.evaporator import Evaporator
from src.models.compressor import Compressor
from src.data.loader import load_modeling_dataframe


class RefrigerationSystem:
    """
    Top-level system object that owns all evaporators and the compressor,
    and is responsible for building the feature and target matrices that
    predictors consume.

    Input features (per time step):
        <evap>_temp, <evap>_temp_setpoint  for every evaporator
        dry_bulb_temp, wet_bulb_temp, system_on

    Target outputs (per time step):
        <evap>_temp  for every evaporator
        total_compressor_power
    """

    def __init__(self):
        self.evaporators: list[Evaporator] = []
        self.compressor: Compressor = Compressor()
        self._df: pd.DataFrame | None = None
        self.evap_names: list[str] = []

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load data from CSV files and populate all domain objects."""
        df, evap_names = load_modeling_dataframe()
        self._df = df
        self.evap_names = evap_names

        self.evaporators = []
        for name in evap_names:
            evap = Evaporator(id=name)
            evap.load_from_dataframe(df)
            self.evaporators.append(evap)

        self.compressor.load_from_dataframe(df)

    # ------------------------------------------------------------------
    # Feature / target column names
    # ------------------------------------------------------------------

    @property
    def input_feature_cols(self) -> list[str]:
        cols = []
        for evap in self.evaporators:
            cols.append(f"{evap.id}_temp")
            if not self._df[f"{evap.id}_temp_setpoint"].empty:
                cols.append(f"{evap.id}_temp_setpoint")
        for extra in ("dry_bulb_temp", "wet_bulb_temp", "system_on"):
            if self._df is not None and extra in self._df.columns:
                cols.append(extra)
        return cols

    @property
    def output_feature_cols(self) -> list[str]:
        cols = [f"{evap.id}_temp" for evap in self.evaporators]
        cols.append("total_compressor_power")
        return cols

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call load() before accessing data.")
        return self._df

    def build_sequences(
        self, seq_length: int, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray, list]:
        """
        Build windowed (X, y) arrays for N-step prediction.

        X shape: (n_samples, seq_length, n_input_features)
            — the look-back window of input features

        y shape: (n_samples, n_steps, n_output_features)
            — the n_steps ahead targets

        Returns (X, y, target_timestamps) where target_timestamps[i] is
        the timestamp of the *last* predicted step for sample i.
        """
        df = self.get_dataframe()
        input_cols = self.input_feature_cols
        output_cols = self.output_feature_cols

        X_data = df[input_cols].to_numpy(dtype=np.float32)
        y_data = df[output_cols].to_numpy(dtype=np.float32)
        timestamps = df.index.tolist()

        X_seqs, y_seqs, target_ts = [], [], []
        total = len(df)
        for i in range(total - seq_length - n_steps + 1):
            X_seqs.append(X_data[i : i + seq_length])
            y_seqs.append(y_data[i + seq_length : i + seq_length + n_steps])
            target_ts.append(timestamps[i + seq_length + n_steps - 1])

        return np.array(X_seqs), np.array(y_seqs), target_ts

    def __repr__(self) -> str:
        return (
            f"RefrigerationSystem("
            f"evaporators={[e.id for e in self.evaporators]}, "
            f"compressor={self.compressor.id!r})"
        )
