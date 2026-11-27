"""Derived temporal features and the per-period model table.

Rolling windows end at the current period: 3 periods (12 min), 6 periods (24 min) and, for the MAP median,
7 periods (28 min). Periods without MAP (value 0) are excluded from the minimum, SD, slope and median; the slope is the
least-squares slope over the available values.
"""
import warnings
import numpy as np
import pandas as pd

from .utils import lag_matrix

FEATURES = [
    'map', 'sap', 'dap', 'hr', 'spo2', 'fio2', 'temp',
    'map_max_12min', 'map_min_12min', 'map_sd_12min', 'map_slope_12min',
    'map_max_24min', 'map_min_24min', 'map_sd_24min', 'map_slope_24min', 'map_median_28min',
    'hr_max_12min', 'hr_min_12min', 'hr_sd_12min', 'hr_slope_12min', 'spo2_sd_12min',
    'hr_max_24min', 'hr_min_24min', 'hr_sd_24min', 'hr_slope_24min', 'spo2_sd_24min',
    'sex', 'age', 'charlson_index',
]
MAP_FEATURES = [f for f in FEATURES if f.startswith('map_')]
HR_FEATURES = [f for f in FEATURES if f.startswith(('hr_', 'spo2_'))]
WINDOWS = {'12min': 3, '24min': 6}


def _stats_nonzero(M, min_n_obs=2):
    """Minimum, SD, slope and median over the non-zero values of each row of M (NaN = before the patient's start)."""
    obs = ~np.isnan(M)
    Z = np.where(obs & (M != 0), M, np.nan)
    nz = (~np.isnan(Z)).sum(1)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mn = np.nanmin(Z, 1)
        mu = np.nanmean(Z, 1)
        sd = np.where(nz > 1, np.sqrt(np.nansum((Z - mu[:, None]) ** 2, 1) / (nz - 1)), np.nan)
        x = np.where(np.isnan(Z), np.nan, np.cumsum(~np.isnan(Z), 1) - 1.0)     # positions of the non-zero values
        xm = np.nanmean(x, 1)
        sxy = np.nansum((x - xm[:, None]) * (Z - mu[:, None]), 1)
        sxx = np.nansum((x - xm[:, None]) ** 2, 1)
        slope = np.where((obs.sum(1) >= min_n_obs) & (nz >= 2), sxy / sxx, np.nan)
        med = np.nanmedian(Z, 1)
    return mn, sd, slope, med


def map_features(df):
    """MAP features per period (computed on the final MAP series, before the model table is assembled)."""
    pid, v = df.patient_id.values, df['map'].values.astype(float)
    out = pd.DataFrame({'patient_id': pid, 'period': df.period.values})
    for key, w in WINDOWS.items():
        M = lag_matrix(v, pid, w)
        mn, sd, slope, _ = _stats_nonzero(M)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out[f'map_max_{key}'] = np.nanmax(M, 1)
        out[f'map_min_{key}'] = mn
        out[f'map_sd_{key}'] = sd
        out[f'map_slope_{key}'] = slope
    out['map_median_28min'] = _stats_nonzero(lag_matrix(v, pid, 7))[3]
    for c in MAP_FEATURES:
        out[c] = out[c].astype('float32')
    return out


def hr_features(hr, spo2, pid):
    """HR maximum, minimum, SD and slope, and SpO2 SD, over the last 12 and 24 minutes."""
    out = {}
    for key, w in WINDOWS.items():
        M = lag_matrix(hr, pid, w)
        mn, sd, slope, _ = _stats_nonzero(M)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            out[f'hr_max_{key}'] = np.nanmax(M, 1)
        out[f'hr_min_{key}'] = mn
        out[f'hr_sd_{key}'] = sd
        out[f'hr_slope_{key}'] = slope
        S = lag_matrix(spo2, pid, w)
        ns = (~np.isnan(S)).sum(1)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            sm = np.nanmean(S, 1)
            out[f'spo2_sd_{key}'] = np.where(ns > 1, np.sqrt(np.nansum((S - sm[:, None]) ** 2, 1) / (ns - 1)), np.nan)
    return out


def model_table(labelled, vitals, log=print):
    """One row per patient and period: pressures, vitals, derived features and the AHE label.
    Periods without MAP have every value set to 0 (they are not part of any prediction window)."""
    df = labelled[['patient_id', 'period', 'map', 'sap', 'dap', 'ahe']].copy()
    df = df.merge(vitals, how='left', on=['patient_id', 'period'])
    mf = map_features(labelled[['patient_id', 'period', 'map']])
    df = df.merge(mf, how='left', on=['patient_id', 'period'])
    df = df.sort_values(['patient_id', 'period']).reset_index(drop=True)
    cols = [c for c in df.columns if c not in ('patient_id', 'period')]
    zero = df['map'].values == 0
    df.loc[zero, cols] = 0
    df[MAP_FEATURES] = df[MAP_FEATURES].fillna(0)
    for k, v in hr_features(df['hr'].values, df['spo2'].values, df['patient_id'].values).items():
        df[k] = v
    cols = [c for c in df.columns if c not in ('patient_id', 'period')]
    df.loc[zero, cols] = 0
    df[HR_FEATURES] = df[HR_FEATURES].fillna(0).astype('float32')
    log(f'  model table: {len(df):,} periods × {df.shape[1]} columns')
    return df
