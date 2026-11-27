"""Input tables and intermediate files.

Input tables (in AHE_DATA_DIR, as .csv or .pkl; see data/README.md):

    patients.*      patient_id, sex ('M'/'F'), age, charlson_index, ltst, exclude
    map.*  sap.*  dap.*  hr.*  spo2.*  fio2.*  temp.*
                    patient_id, datetime, value          (one row per recorded value)
    vasoactives.*   patient_id, datetime, drug, rate     (drug: norepinephrine, epinephrine, vasopressin, dobutamine)
"""
import pandas as pd

from . import config as C

SIGNALS = ['map', 'sap', 'dap', 'hr', 'spo2', 'fio2', 'temp']


def _find(name):
    for ext in ('.pkl', '.csv'):
        p = C.DATA_DIR / f'{name}{ext}'
        if p.exists():
            return p
    raise FileNotFoundError(f'Input table "{name}" not found in {C.DATA_DIR} (.pkl or .csv)')


def read_input(name):
    """Read one input table with the expected column types."""
    p = _find(name)
    df = pd.read_pickle(p) if p.suffix == '.pkl' else pd.read_csv(p)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    if name == 'patients':
        for col in ('ltst', 'exclude'):
            df[col] = df[col].fillna(False).astype(bool) if col in df.columns else False
    return df


def read_signal(name):
    """One signal (patient_id, datetime, value), restricted to its plausible range."""
    df = read_input(name)[['patient_id', 'datetime', 'value']]
    lo, hi = C.RANGES[name]
    return df[(df.value >= lo) & (df.value <= hi)]


def save(df, name):
    df.to_pickle(C.output_path('intermediate', f'{name}.pkl'))


def load(name):
    return pd.read_pickle(C.output_path('intermediate', f'{name}.pkl'))
