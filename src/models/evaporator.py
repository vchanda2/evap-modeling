"""Evaporator domain model."""

import pandas as pd


class Evaporator:
    """
    Represents a single evaporator unit.

    Attributes
    ----------
    id : str
        Unique identifier, e.g. "H01".
    size : float | None
        Physical size / capacity (optional metadata).
    location : str | None
        Physical location description (optional metadata).
    metadata : dict
        Any additional key-value metadata.

    State variables (time-indexed Series, set by RefrigerationSystem)
    ----------
    temp : pd.Series            actual evaporator temperature
    temp_setpoint : pd.Series   temperature setpoint
    """

    def __init__(
        self,
        id: str,
        size: float | None = None,
        location: str | None = None,
        **metadata,
    ):
        self.id = id
        self.size = size
        self.location = location
        self.metadata = metadata

        self.temp: pd.Series = pd.Series(dtype=float)
        self.temp_setpoint: pd.Series = pd.Series(dtype=float)

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Populate state variables from a modeling DataFrame."""
        self.temp = df[f"{self.id}_temp"].copy()
        if f"{self.id}_temp_setpoint" in df.columns:
            self.temp_setpoint = df[f"{self.id}_temp_setpoint"].copy()

    def get_state(self, t) -> dict:
        """Return state at a given timestamp (or integer index)."""
        return {
            "temp": self.temp.loc[t] if t in self.temp.index else None,
            "temp_setpoint": (
                self.temp_setpoint.loc[t]
                if t in self.temp_setpoint.index
                else None
            ),
        }

    def __repr__(self) -> str:
        return f"Evaporator(id={self.id!r}, size={self.size}, location={self.location!r})"
