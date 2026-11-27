"""Vectorised helpers for per-patient time series stored as consecutive rows (sorted by patient and period)."""
import numpy as np
import pandas as pd

from . import config as C


def shift_within(values, patient, k):
    """Values shifted k rows down (k > 0: previous rows) without crossing patients; NaN where it would cross."""
    v = np.asarray(values, float)
    pid = np.asarray(patient)
    out = np.full(len(v), np.nan)
    if k == 0:
        return v.copy()
    same = np.zeros(len(v), bool)
    if k > 0:
        out[k:] = v[:-k]
        same[k:] = pid[k:] == pid[:-k]
    else:
        k = -k
        out[:-k] = v[k:]
        same[:-k] = pid[:-k] == pid[k:]
    out[~same] = np.nan
    return out


def lag_matrix(values, patient, width):
    """(N, width) matrix with the last `width` values of each row (last column = current); NaN before the patient's start."""
    return np.column_stack([shift_within(values, patient, k) for k in range(width - 1, -1, -1)])


def period_of(times, begin):
    """4-minute period index of each timestamp relative to the patient's reference start."""
    return np.floor((pd.to_datetime(times) - pd.to_datetime(begin)).dt.total_seconds() / C.PERIOD_SECONDS).astype(int)


def median_interval_minutes(records):
    """Median interval (minutes) between consecutive records of each patient (records sorted by time)."""
    r = records.sort_values(['patient_id', 'datetime'], kind='mergesort')
    d = r.groupby('patient_id')['datetime'].diff().dt.total_seconds() / 60
    return d.groupby(r['patient_id']).median()


def grid(reference):
    """Regular 4-minute grid from each patient's first to last reference record: ceil(duration / 4 min) periods."""
    n = np.ceil(reference['minutes'] / C.PERIOD_MINUTES).astype(int)
    return pd.DataFrame({'patient_id': reference['patient_id'].repeat(n).values,
                         'period': np.concatenate([np.arange(k) for k in n]) if len(n) else np.array([], int)})


def period_means(records, begin, name):
    """Mean value of each signal per patient and 4-minute period."""
    r = records.merge(begin, on='patient_id', how='inner')
    r['period'] = period_of(r['datetime'], r['begin'])
    return r.groupby(['patient_id', 'period'])['value'].mean().rename(name).reset_index()
