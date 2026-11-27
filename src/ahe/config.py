"""Paths and study parameters.

Paths are taken from environment variables so that no local path is hard-coded:

    AHE_DATA_DIR    folder with the input tables (default: data/synthetic, created by `python run_pipeline.py --synthetic`)
    AHE_OUTPUT_DIR  folder where every output is written (default: output/)
    AHE_SPLIT_FILE  optional JSON file {"train": [...], "val": [...], "test": [...]} with a predefined patient split

All study parameters are defined here, in one place.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get('AHE_DATA_DIR', ROOT / 'data' / 'synthetic')).resolve()
OUTPUT_DIR = Path(os.environ.get('AHE_OUTPUT_DIR', ROOT / 'output')).resolve()
SPLIT_FILE = os.environ.get('AHE_SPLIT_FILE')

# ----------------------------------------------------------------------------------------------- time grid
PERIOD_MINUTES = 4                       # every signal is aggregated on a 4-minute grid
PERIOD_SECONDS = PERIOD_MINUTES * 60

# ----------------------------------------------------------------------------------------------- plausible ranges
# Records outside these ranges are discarded on input.
RANGES = {
    'map': (40, 150), 'sap': (50, 250), 'dap': (30, 150),
    'hr': (20, 200), 'spo2': (50, 100), 'fio2': (21, 100), 'temp': (30, 42),
}

# ----------------------------------------------------------------------------------------------- patient inclusion
MIN_PRESSURE_RECORDS = 60                # arterial pressure: at least 60 records (≥ 2 h of monitoring)
MAX_MEDIAN_INTERVAL_MIN = 4              # median interval between consecutive records < 4 min (MAP, SAP, DAP, HR, SpO2)
MAX_OUTLIER_FRACTION = 0.10              # exclude patients with > 10 % of outlying arterial-pressure periods
MAX_ZERO_MAP_PCT = 40                    # exclude patients with > 40 % of periods without MAP ...
MIN_NONZERO_MAP_PERIODS = 1000           # ... and fewer than 1,000 periods with MAP
MAX_MISSING_HR_SPO2_PCT = 40             # exclude patients with > 40 % of periods without HR (or SpO2) ...
MIN_HR_SPO2_PERIODS = 500                # ... and fewer than 500 periods with HR (or SpO2)

# ----------------------------------------------------------------------------------------------- arterial pressure
PULSE_PRESSURE_RANGE = (10, 110)         # outlier if pulse pressure < 10 or > 110 mmHg (or MAP outside DAP–SAP)
MAX_GAP_PERIODS = 2                      # forward filling of arterial pressure for gaps of up to 2 periods (8 min)
MAP_DISCREPANCY_MMHG = 8                 # measured MAP replaced by DAP + (SAP − DAP)/3 if they differ by > 8 mmHg

# ----------------------------------------------------------------------------------------------- other variables
FIO2_AMBIENT = 21.0                      # FiO2 (%) assumed when no earlier value exists (ambient air)

# ----------------------------------------------------------------------------------------------- vasoactive drugs
# Infusion rate per 4-minute period = maximum recorded rate; the last rate is carried forward unless no rate was
# recorded in the previous DRUG_GAP_PERIODS periods (then the infusion is considered stopped). Rates are capped.
DRUGS = ['norepinephrine', 'epinephrine', 'vasopressin', 'dobutamine']
DRUG_CAPS = {'norepinephrine': 5.0, 'dobutamine': 25.0, 'vasopressin': 0.2, 'epinephrine': None}
DRUG_GAP_PERIODS = 4
CARRY_FORWARD_DRUGS = ['norepinephrine', 'dobutamine', 'vasopressin']   # epinephrine: only the recorded periods

# ----------------------------------------------------------------------------------------------- event definition
MAP_THRESHOLD = 65                       # mmHg
SUSTAINED_PERIODS = 2                    # sustained hypotension: 2 consecutive periods (≥ 8 min)
VASOACTIVE_WINDOW_PERIODS = 8            # clinically significant: vasoactive infusion within ± 8 periods (± 32 min)

# ----------------------------------------------------------------------------------------------- prediction windows
OBSERVATION_PERIODS = 10                 # 40-minute observation window (model input)
GAP_PERIODS = 3                          # 12-minute gap
TARGET_PERIODS = 1                       # 4-minute target period → prediction 16 minutes ahead
WINDOW_PERIODS = OBSERVATION_PERIODS + GAP_PERIODS + TARGET_PERIODS

# ----------------------------------------------------------------------------------------------- data split and model
SEED = 42
TEST_FRACTION = 0.30                     # 70 % training; the remaining 30 % is split 50/50 into validation and test
TRAIN_NEGATIVES_KEPT = 42_872            # negative (non-AHE) training windows kept after random downsampling
LSTM_UNITS = (128, 128, 64)
LSTM_DROPOUT = (0.3, 0.2, 0.2)
DENSE_UNITS = (32, 16)
DENSE_DROPOUT = (0.3, 0.2)
LEARNING_RATE = 0.001
EPOCHS = 30
PATIENCE = 5                             # early stopping on validation loss
BATCH_SIZE = 16
TRAINING_SEED = 1

# ----------------------------------------------------------------------------------------------- evaluation
N_BOOTSTRAP = 1000                       # patient-level bootstrap resamples for confidence intervals
FOLLOW_UP_PERIODS = (7, 15)              # false-positive follow-up after the predicted period (28 and 60 min)


def output_path(*parts):
    """Path inside OUTPUT_DIR (created if needed). Nothing is ever written outside OUTPUT_DIR."""
    p = OUTPUT_DIR.joinpath(*parts).resolve()
    if p != OUTPUT_DIR and OUTPUT_DIR not in p.parents:
        raise PermissionError(f'Outputs must be written inside {OUTPUT_DIR}: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
