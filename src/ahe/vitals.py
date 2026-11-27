"""Heart rate, SpO2, FiO2 and temperature on the 4-minute grid.

Steps
  1. Among the patients with arterial pressure: median interval between HR records and between SpO2 records < 4 min,
     and temperature recorded.
  2. Per-period means on the same grid as arterial pressure.
  3. Patients with > 40 % of periods without HR (or SpO2) and fewer than 500 periods with HR (or SpO2) are excluded.

"""
import numpy as np
import pandas as pd

from . import config as C
from . import io
from .utils import grid, median_interval_minutes, period_means

VARIABLES = ['hr', 'spo2', 'fio2', 'temp']


def cohort_grid(pressure_patients, reference, log=print):
    """Patients meeting the HR/SpO2/temperature criteria and their observed per-period values (before imputation)."""
    rec = {v: io.read_signal(v) for v in VARIABLES}
    rec = {v: r[r.patient_id.isin(pressure_patients)] for v, r in rec.items()}
    ok = None
    for v in ('hr', 'spo2'):
        med = median_interval_minutes(rec[v])
        s = set(med[med < C.MAX_MEDIAN_INTERVAL_MIN].index)
        ok = s if ok is None else ok & s
    med_t = median_interval_minutes(rec['temp'])
    ok &= set(med_t[med_t.notna()].index)                      # temperature recorded (at least two records)
    ref = reference[reference.patient_id.isin(ok)]
    g = grid(ref)
    for v in VARIABLES:
        g = g.merge(period_means(rec[v], ref[['patient_id', 'begin']], v), on=['patient_id', 'period'], how='left')
    # insufficient HR or SpO2
    drop = set()
    for v in ('hr', 'spo2'):
        miss = g[v].isna().groupby(g.patient_id).mean() * 100
        n = g[v].notna().groupby(g.patient_id).sum()
        drop |= set(miss[(miss > C.MAX_MISSING_HR_SPO2_PCT) & (n < C.MIN_HR_SPO2_PERIODS)].index)
    g = g[~g.patient_id.isin(drop)].reset_index(drop=True)
    log(f'  HR/SpO2/temperature criteria: {len(ok):,} patients; excluded for insufficient HR/SpO2: {len(drop):,}; '
        f'cohort: {g.patient_id.nunique():,}')
    return g


def impute(observed, train_patients, log=print):
    """Forward filling within each patient"""
    out = observed[['patient_id', 'period']].copy()
    tr = observed[observed.patient_id.isin(set(train_patients))]
    fallback = {v: float(tr[v].median()) for v in ('hr', 'spo2', 'temp')}
    fallback['fio2'] = C.FIO2_AMBIENT
    g = observed.groupby('patient_id', sort=False)
    for v in VARIABLES:
        out[v] = g[v].ffill().fillna(fallback[v]).astype('float32')
    log(f'  fallback values (training median; FiO2 ambient air): {fallback}')
    return out
