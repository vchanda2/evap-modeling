"""
Load and preprocess the raw evaporator and compressor CSV files
into a clean, time-indexed DataFrame ready for model consumption.
"""

import pandas as pd

from src.config import (
    EVAP_RAW_DATA_PATH,
    COMP_RAW_DATA_PATH,
    EVAP_TEMP_PROXY_PAIRS,
    SUPPLY_VOLTAGE,
    POWER_FACTOR,
    SQRT3,
    KW_DIVISOR,
    SYSTEM_ON_POWER_THRESHOLD_KW,
)


def load_raw() -> pd.DataFrame:
    """Merge the two raw CSVs on DateTime and return a time-indexed DataFrame."""
    evap_df = pd.read_csv(EVAP_RAW_DATA_PATH)
    comp_df = pd.read_csv(COMP_RAW_DATA_PATH)
    combined = pd.merge(evap_df, comp_df, on="DateTime", how="inner")
    combined["DateTime"] = pd.to_datetime(combined["DateTime"])
    combined.set_index("DateTime", inplace=True)
    return combined


def _extract_evap_names(df: pd.DataFrame) -> list[str]:
    names = set()
    for col in df.columns:
        first_part = col.split(" - ")[0]
        if "CG " in first_part:
            names.add(first_part.split(" ")[1])
    return sorted(names)


def _build_column_map(df: pd.DataFrame, evap_names: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for col in df.columns:
        for evap in evap_names:
            if evap in col:
                if "SP" in col or "Setpoint" in col:
                    mapping[col] = f"{evap}_temp_setpoint"
                elif "Actual Temp" in col:
                    mapping[col] = f"{evap}_temp"
    return mapping


def build_modeling_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    From the raw combined DataFrame, produce a clean modeling DataFrame with:
      - <evap>_temp and <evap>_temp_setpoint for every evaporator
      - dry_bulb_temp, wet_bulb_temp, system_on
      - total_compressor_power (kW)

    Returns (modeling_df, evap_names).
    """
    evap_names = _extract_evap_names(df)
    col_map = _build_column_map(df, evap_names)

    modeling_df = df[list(col_map.keys())].rename(columns=col_map)

    # Fill missing temp sensors from proxy siblings
    for missing, source in EVAP_TEMP_PROXY_PAIRS:
        src_col = f"{source}_temp"
        dst_col = f"{missing}_temp"
        if src_col in modeling_df.columns and dst_col not in modeling_df.columns:
            modeling_df[dst_col] = modeling_df[src_col]

    # Exogenous weather inputs
    if "Condenser Sequencer - Air Temperature" in df.columns:
        modeling_df["dry_bulb_temp"] = df["Condenser Sequencer - Air Temperature"]
    if "Condenser Sequencer - Wetbulb Temperature" in df.columns:
        modeling_df["wet_bulb_temp"] = df["Condenser Sequencer - Wetbulb Temperature"]

    # Compressor power from sum of all amp readings
    amps_cols = [c for c in df.columns if "Amps" in c]
    total_amps = df[amps_cols].sum(axis=1)
    modeling_df["total_compressor_power"] = (
        total_amps * SUPPLY_VOLTAGE * SQRT3 * POWER_FACTOR / KW_DIVISOR
    )

    modeling_df["system_on"] = (
        modeling_df["total_compressor_power"] > SYSTEM_ON_POWER_THRESHOLD_KW
    ).astype(int)

    modeling_df.dropna(inplace=True)
    return modeling_df, evap_names


def load_modeling_dataframe() -> tuple[pd.DataFrame, list[str]]:
    """Convenience: load raw files and return the clean modeling DataFrame."""
    raw = load_raw()
    return build_modeling_dataframe(raw)
