"""Acute hypotensive event (AHE) labelling of every 4-minute period.

An AHE is MAP < 65 mmHg that is either
  * sustained: MAP < 65 mmHg in at least two consecutive periods (≥ 8 min); once started, every period with
    MAP < 65 mmHg belongs to the episode, which ends when MAP is ≥ 65 mmHg in two consecutive periods; or
  * clinically significant: any period with MAP < 65 mmHg with a vasoactive infusion (norepinephrine, epinephrine,
    vasopressin or dobutamine) in the same period or within the preceding or following 8 periods (32 min),
    regardless of its duration.
Vasoactive infusions are used only for labelling; they are never model inputs.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import io
from .utils import period_of, shift_within


def drug_rates(cohort, reference):
    """Infusion rate of each vasoactive drug per period (carried forward, capped)."""
    rec = io.read_input('vasoactives')
    rec = rec[rec.patient_id.isin(set(cohort.patient_id))].merge(reference[['patient_id', 'begin']], on='patient_id', how='left')
    rec['period'] = period_of(rec['datetime'], rec['begin'])
    pid = cohort.patient_id.values
    rates = {}
    for drug in C.DRUGS:
        mx = rec[rec.drug == drug].groupby(['patient_id', 'period'])['rate'].max().rename('mx').reset_index()
        m = cohort[['patient_id', 'period']].merge(mx, on=['patient_id', 'period'], how='left')['mx'].values.astype(float)
        if drug in C.CARRY_FORWARD_DRUGS:
            none_before = np.ones(len(m), bool)
            for s in range(1, C.DRUG_GAP_PERIODS + 1):
                none_before &= np.isnan(shift_within(m, pid, s))
            v = np.where(~np.isnan(m), m, np.where(none_before, 0.0, np.nan))
            v = pd.Series(v).groupby(pid).ffill().values
            cap = C.DRUG_CAPS[drug]
            v = np.where(v >= cap, cap, v)
        else:
            v = np.nan_to_num(m, nan=0.0)
        rates[drug] = v
    return rates


def _any_within(v, pid, w):
    """True if any value != 0 within w periods before or after (including the current one), within each patient."""
    df = pd.DataFrame({'pid': pid, 'v': np.asarray(v, float)})
    back = df.groupby('pid', sort=False)['v'].rolling(window=w, min_periods=1, closed='both').max() \
        .reset_index(level=0, drop=True).sort_index().values
    r = df.iloc[::-1].reset_index(drop=True)
    fwd = r.groupby('pid', sort=False)['v'].rolling(window=w, min_periods=1, closed='both').max() \
        .reset_index(level=0, drop=True).sort_index().values[::-1]
    return (back != 0) | (fwd != 0)


def label(pressure, cohort_patients, reference, log=print):
    """Pressure rows of the cohort with drug rates, the vasoactive flag and the AHE label (column 'ahe')."""
    df = pressure[pressure.patient_id.isin(set(cohort_patients))].sort_values(['patient_id', 'period']).reset_index(drop=True)
    df = assign_labels(df, drug_rates(df, reference))
    log(f'  vasoactive within ± {C.VASOACTIVE_WINDOW_PERIODS} periods: {df.vasoactive.mean() * 100:.1f} % of periods; '
        f'AHE periods: {int(df.ahe.sum()):,}')
    return df


def assign_labels(df, rates):
    """AHE label of each period, given the MAP series and the drug rates."""
    df = df.copy()
    pid = df.patient_id.values
    vaso = np.zeros(len(df), bool)
    for drug, v in rates.items():
        df[f'rate_{drug}'] = v
        vaso |= _any_within(v, pid, C.VASOACTIVE_WINDOW_PERIODS)
    m = df['map'].values
    low = (m < C.MAP_THRESHOLD) & (m != 0)
    nxt = shift_within(low.astype(float), pid, -1)
    both = np.where(np.isnan(nxt), low.astype(float), (low + np.nan_to_num(nxt)) / 2)
    mark = np.where(both == 1, 1.0, np.where(both == 0, 0.0, np.nan))      # two consecutive low (1) or normal (0) periods
    episode = pd.Series(mark).groupby(pid).ffill().fillna(0).values.astype(int)
    sustained = episode & low.astype(int)
    significant = vaso & low
    df['vasoactive'] = vaso
    df['sustained'] = sustained.astype(bool)
    df['ahe'] = np.where(vaso, significant, sustained).astype(int)
    return df
