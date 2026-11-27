"""Prediction windows: 10 observation periods (40 min, model input), 3 gap periods (12 min) and 1 target period.

Each patient's series is divided into segments that end at a period without MAP or at an AHE period. Segments
ending in an AHE give one positive window (its last 14 periods, the AHE being the target period); the other segments
are divided into consecutive, non-overlapping blocks of 14 periods (negative windows). Segments shorter than
14 periods are discarded.
"""
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES
from .utils import shift_within

KEYS = ['patient_id', 'period']
COLUMNS = KEYS + FEATURES + ['ahe']


def add_demographics(table, patients):
    demo = patients[['patient_id', 'sex', 'age', 'charlson_index']].drop_duplicates('patient_id').copy()
    demo['sex'] = demo['sex'].map({'M': 1, 'F': 0})
    return table.merge(demo, on='patient_id', how='left')


def build(table, patients, log=print):
    """Windows as consecutive rows with a 'window' identifier (WINDOW_PERIODS rows each)."""
    w = C.WINDOW_PERIODS
    df = add_demographics(table, patients)[COLUMNS]
    df = df.sort_values(KEYS).reset_index(drop=True)
    pid = df.patient_id.values
    # a new segment starts in the period after a period without MAP or an AHE period
    cut = ((df['map'].values == 0) | (df['ahe'].values == 1)).astype(int)
    seg = shift_within(pd.Series(cut).groupby(pid).cumsum().values, pid, 1)
    df['segment'] = np.where(np.isnan(seg), 0, seg).astype(int)
    # drop the last period of a segment when it has no MAP
    nxt = shift_within(df['segment'].values.astype(float), pid, -1)
    last = np.isnan(nxt) | (nxt != df['segment'].values)
    df = df[~(last & (df['map'].values == 0))].reset_index(drop=True)
    # segments of at least one window
    n = df.groupby(['patient_id', 'segment'])['map'].transform('size')
    df = df[n >= w].reset_index(drop=True)
    # segments ending in an AHE → their last w periods; the others → consecutive blocks of w periods
    g = df.groupby(['patient_id', 'segment'])
    ends_in_ahe = g['ahe'].transform('max').values == 1
    from_end = g.cumcount(ascending=False).values
    df = df[~ends_in_ahe | (from_end < w)].reset_index(drop=True)
    df['block'] = df.groupby(['patient_id', 'segment']).cumcount().values // w
    size = df.groupby(['patient_id', 'segment', 'block'])['map'].transform('size')
    df = df[size == w].reset_index(drop=True)
    df['window'] = df.groupby(['patient_id', 'segment', 'block']).ngroup()
    df = df.drop(columns=['segment', 'block']).reset_index(drop=True)
    last_rows = df.groupby('window').tail(1)
    log(f'  windows: {df.window.nunique():,} | positive {int(last_rows.ahe.sum()):,} '
        f'({last_rows.ahe.mean() * 100:.1f} %) | patients {df.patient_id.nunique():,}')
    return df


def window_info(windows):
    """One row per window: patient, prediction period (last observation period), target period and label."""
    g = windows.sort_values(['window', 'period']).groupby('window', sort=True)
    info = pd.DataFrame({'patient_id': g.patient_id.first(),
                         'prediction_period': g['period'].nth(C.OBSERVATION_PERIODS - 1).values,
                         'target_period': g['period'].last().values,
                         'label': g['ahe'].last().values.astype(int)})
    return info.reset_index()
