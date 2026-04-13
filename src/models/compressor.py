"""Compressor domain model."""

import pandas as pd


class Compressor:
    """
    Represents the compressor subsystem.

    Attributes
    ----------
    id : str
        Identifier, e.g. "COMP-1".
    metadata : dict
        Any additional key-value metadata.

    State variables (time-indexed Series, set by RefrigerationSystem)
    ----------
    power : pd.Series       total compressor power (kW)
    system_on : pd.Series   binary flag: 1 = running, 0 = off
    """

    def __init__(self, id: str = "COMP-1", **metadata):
        self.id = id
        self.metadata = metadata

        self.power: pd.Series = pd.Series(dtype=float) 
        self.system_on: pd.Series = pd.Series(dtype=int)

    def load_from_dataframe(self, df: pd.DataFrame) -> None:
        """Populate state variables from a modeling DataFrame."""
        self.power = df["total_compressor_power"].copy()
        self.system_on = df["system_on"].copy()

    def get_state(self, t) -> dict:
        """Return state at a given timestamp (or integer index)."""
        return {
            "power": self.power.loc[t] if t in self.power.index else None,
            "system_on": self.system_on.loc[t] if t in self.system_on.index else None,
        }

    def __repr__(self) -> str: # this function is used when you print the object or inspect it in a debugger
        return f"Compressor(id={self.id!r})" # the !r means to use the repr of the id, which will include quotes if it's a string
