from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent

# Data files — place CSVs in data/ at project root
DATA_DIR = ROOT_DIR / "data"
EVAP_RAW_DATA_PATH = DATA_DIR / "evap_raw_data.csv"
COMP_RAW_DATA_PATH = DATA_DIR / "comp_and_other_raw_data.csv"

# Electrical constants for power calculation (3-phase)
SUPPLY_VOLTAGE = 480          # Volts
POWER_FACTOR = 0.93
SQRT3 = 3 ** 0.5
KW_DIVISOR = 1000

# System threshold to determine compressor on/off
SYSTEM_ON_POWER_THRESHOLD_KW = 25

# Evaporator pairs: evaps missing temp sensors → borrow from sibling
EVAP_TEMP_PROXY_PAIRS = [
    ("H06", "H05"),
    ("G06", "G05"),
    ("G04", "G03"),
    ("H08", "H07"),
    ("H04", "H03"),
    ("H10", "H09"),
    ("G02", "G01"),
    ("G08", "G07"),
    ("H02", "H01"),
]

# Default predictor settings
DEFAULT_SEQ_LENGTH = 6      # look-back window (time steps)
DEFAULT_N_STEPS = 12        # how many steps ahead to predict
