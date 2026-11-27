"""Cohort selection on arterial pressure and arterial-pressure processing (MAP, SAP, DAP).

Steps
  1. Records within plausible ranges, duplicates removed; patients with ltst/exclude flags removed.
  2. MAP: ≥ 60 records and median interval between records < 4 min. The first and last MAP record define each
     patient's 4-minute grid.
  3. DAP and then SAP: same criteria, among the patients retained in the previous step.
  4. Per-period means; a missing pressure is derived with MAP = DAP + (SAP − DAP)/3 when the other two are present;
     gaps of up to 2 periods (8 min) are forward-filled; longer gaps = no monitoring (value 0).
  5. Outliers (pulse pressure < 10 or > 110 mmHg, or MAP outside DAP–SAP): patients with > 10 % outlying periods are
     excluded; in the rest, outlying periods are set to missing and filled again as in step 4.
  6. Each patient's series ends at the last period with MAP; patients with > 40 % of periods without MAP and fewer than
     1,000 periods with MAP are excluded.
  7. Measured MAP is replaced by DAP + (SAP − DAP)/3 when they differ by more than 8 mmHg.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import io
from .utils import grid, median_interval_minutes, period_means


def _eligible_records(name, patients):
    r = io.read_signal(name)
    r = r[r.patient_id.isin(patients)].drop_duplicates()
    n = r.groupby('patient_id')['value'].transform('size')
    r = r[n >= C.MIN_PRESSURE_RECORDS]
    med = median_interval_minutes(r)
    keep = med[med < C.MAX_MEDIAN_INTERVAL_MIN].index
    return r[r.patient_id.isin(keep)]


def _reference(map_records, patients):
    """Reference start and duration of each patient (first and last MAP record)."""
    m = map_records[map_records.patient_id.isin(patients)]
    ref = m.groupby('patient_id')['datetime'].agg(begin='min', end='max').reset_index()
    ref['minutes'] = (ref['end'] - ref['begin']).dt.total_seconds() / 60
    return ref


def _signal_grid(records, ref, name):
    g = grid(ref)
    m = period_means(records, ref[['patient_id', 'begin']], name)
    g = g.merge(m, on=['patient_id', 'period'], how='left')
    g[name] = g[name].fillna(0).astype('float32')
    return g


def _fill_short_gaps(df, col):
    """Forward fill for gaps of up to 2 periods within each patient; longer gaps → 0 (no monitoring)."""
    v = df[col]
    g = v.groupby(df['patient_id'])
    stop = g.shift(1).isna() & g.shift(2).isna()
    filled = pd.Series(np.where(~v.isna(), v, np.where(stop, 0, np.nan)), index=df.index)
    return filled.groupby(df['patient_id']).ffill().fillna(0)


def _fill(df):
    for col in ('map', 'dap', 'sap'):
        df[col] = df[col].replace(0, np.nan)
    for col in ('map', 'dap', 'sap'):
        df[f'{col}_filled'] = _fill_short_gaps(df, col)
    df = df.drop(columns=['map', 'sap', 'dap'])
    return df.rename(columns={'map_filled': 'map', 'sap_filled': 'sap', 'dap_filled': 'dap'})


def process(patients_table, log=print):
    """Returns (pressure, reference, raw_grids):
    pressure   patient_id, period, map, dap, sap (final 4-minute series of the cohort)
    reference  patient_id, begin, end, minutes (4-minute grid of each patient, defined by MAP)
    raw_grids  per-period means before any derivation or filling (used to report the origin of MAP values)
    """
    p = patients_table[~patients_table.ltst & ~patients_table.exclude]
    patients = set(p.patient_id)
    log(f'  eligible patients: {len(patients):,}')

    map_rec = _eligible_records('map', patients)
    ref = _reference(map_rec, set(map_rec.patient_id))
    map_g = _signal_grid(map_rec, ref, 'map')
    pts = set(map_g.patient_id)
    log(f'  MAP (≥ {C.MIN_PRESSURE_RECORDS} records, median interval < {C.MAX_MEDIAN_INTERVAL_MIN} min): {len(pts):,} patients')

    dap_rec = _eligible_records('dap', pts)
    dap_g = _signal_grid(dap_rec, _reference(map_rec, set(dap_rec.patient_id)), 'dap')
    pts = set(dap_g.patient_id)
    sap_rec = _eligible_records('sap', pts)
    sap_g = _signal_grid(sap_rec, _reference(map_rec, set(sap_rec.patient_id)), 'sap')
    pts = set(sap_g.patient_id)
    log(f'  DAP and SAP (same criteria): {len(pts):,} patients')

    df = map_g[map_g.patient_id.isin(pts)]
    df = df.merge(dap_g, how='left', on=['patient_id', 'period']).merge(sap_g, how='left', on=['patient_id', 'period'])
    df = df.reset_index(drop=True)
    raw = df[['patient_id', 'period', 'map', 'sap', 'dap']].copy()

    # derive a missing pressure from the other two (if two or more are missing, the period has no pressure)
    for col in ('map', 'sap', 'dap'):
        df[col] = df[col].replace(0, np.nan)
    two_missing = ((df.map.isna() & df.dap.isna()) | (df.map.isna() & df.sap.isna()) | (df.dap.isna() & df.sap.isna()))
    df.loc[two_missing, ['map', 'sap', 'dap']] = 0
    df['map'] = df['map'].fillna(df['dap'] + (df['sap'] - df['dap']) / 3)
    df['sap'] = df['sap'].fillna(3 * df['map'] - 2 * df['dap'])
    df['dap'] = df['dap'].fillna((3 * df['map'] - df['sap']) / 2)
    df = _fill(df)

    # outliers
    pp = df['sap'] - df['dap']
    lo, hi = C.PULSE_PRESSURE_RANGE
    df['outlier'] = ((pp < lo) & (pp != 0)) | (pp > hi) | ((df['map'] < df['dap']) | (df['map'] > df['sap']))
    frac = df.groupby('patient_id')['outlier'].sum() / df.groupby('patient_id')['outlier'].count()
    drop = set(frac[frac > C.MAX_OUTLIER_FRACTION].index)
    df = df[~df.patient_id.isin(drop)]
    log(f'  excluded for > {C.MAX_OUTLIER_FRACTION:.0%} outlying periods: {len(drop):,} patients')
    df.loc[df['outlier'], ['map', 'sap', 'dap']] = 0
    df = _fill(df)

    # series end at the last period with MAP
    last = df[df['map'] != 0].groupby('patient_id').apply(lambda g: g.index.max())
    df = df[df.index.values <= df['patient_id'].map(last).fillna(-1).values]

    # insufficient MAP
    zero_pct = (df['map'] == 0).groupby(df['patient_id']).mean() * 100
    nonzero = (df['map'] != 0).groupby(df['patient_id']).sum()
    drop = set(zero_pct[(zero_pct > C.MAX_ZERO_MAP_PCT) & (nonzero < C.MIN_NONZERO_MAP_PERIODS)].index)
    df = df[~df.patient_id.isin(drop)]
    log(f'  excluded for insufficient MAP: {len(drop):,} patients')

    # measured MAP inconsistent with SAP and DAP
    theoretical = ((df['sap'] - df['dap']) / 3) + df['dap']
    replace = (df['map'] - theoretical).abs() > C.MAP_DISCREPANCY_MMHG
    df.loc[replace, 'map'] = ((df['sap'] - df['dap']) / 3) + df['dap']

    out = df[['patient_id', 'period', 'map', 'dap', 'sap']].copy()
    for col in ('map', 'dap', 'sap'):
        out[col] = out[col].astype('float32')
    out = out.reset_index(drop=True)
    log(f'  arterial pressure: {out.patient_id.nunique():,} patients, {len(out):,} periods')
    return out, ref, raw


def map_origin(pressure, raw, tol=0.01):
    """Origin of every final MAP value: measured, measured but replaced by the formula, calculated from SAP and DAP,
    carried forward, or no monitoring (MAP = 0)."""
    r = pressure.merge(raw.rename(columns={'map': 'map_raw', 'sap': 'sap_raw', 'dap': 'dap_raw'}),
                       on=['patient_id', 'period'], how='left')
    v = {k: r[k].fillna(0).to_numpy(np.float64) for k in ('map', 'sap', 'dap', 'map_raw')}
    formula = v['dap'] + (v['sap'] - v['dap']) / 3
    close = lambda a, b: np.abs(a - b) < tol
    none = v['map'] == 0
    measured = ~none & (v['map_raw'] > 0) & close(v['map'], v['map_raw'])
    replaced = ~none & ~measured & (v['map_raw'] > 0) & close(v['map'], formula) & (np.abs(v['map_raw'] - formula) > C.MAP_DISCREPANCY_MMHG)
    calculated = ~none & ~measured & (v['map_raw'] == 0) & close(v['map'], formula)
    origin = np.select([none, measured, replaced, calculated],
                       ['no monitoring', 'measured', 'replaced by formula', 'calculated'], 'carried forward')
    return pd.DataFrame({'patient_id': r.patient_id.values, 'period': r.period.values, 'map_origin': origin})
