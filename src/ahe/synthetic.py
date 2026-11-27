"""Synthetic example dataset (entirely artificial; results obtained on it have no clinical meaning).

Creates data/synthetic/ with the input tables described in data/README.md, so that the whole pipeline can be run
without real data. MAP fluctuates around a patient-specific baseline with occasional falls below 65 mmHg; some
patients receive norepinephrine during those falls. A few patients are generated so that they are excluded
(withdrawal of life-sustaining treatment, too few records).
"""
import numpy as np
import pandas as pd

from . import config as C

N_PATIENTS = 60
HOURS = 36
SEED = 2024


def _times(start, hours, every_min, rng):
    n = int(hours * 60 / every_min)
    t = start + pd.to_timedelta(np.arange(n) * every_min * 60 + rng.uniform(-20, 20, n), unit='s')
    return pd.DatetimeIndex(np.sort(t.values))


def _map_series(t, rng):
    """MAP: AR(1) fluctuation around a baseline, with falls of 15–35 mmHg lasting about 20–60 minutes."""
    minutes = (t - t[0]).total_seconds().values / 60
    base = rng.uniform(72, 95)
    noise = np.zeros(len(t))
    for i in range(1, len(t)):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 1.2)
    dip = np.zeros(len(t))
    falls = []
    start = rng.uniform(60, 240)
    while start < minutes[-1] - 120:
        depth, down, hold, up = rng.uniform(15, 35), rng.uniform(15, 40), rng.uniform(8, 30), rng.uniform(15, 30)
        x = minutes - start
        dip += np.where((x >= 0) & (x < down), depth * x / down, 0)
        dip += np.where((x >= down) & (x < down + hold), depth, 0)
        dip += np.where((x >= down + hold) & (x < down + hold + up), depth * (1 - (x - down - hold) / up), 0)
        falls.append((start, depth))
        start += rng.uniform(150, 420)
    return base + noise - dip, falls


def write(n_patients=N_PATIENTS, hours=HOURS, out=None):
    rng = np.random.default_rng(SEED)
    out = out or (C.ROOT / 'data' / 'synthetic')
    out.mkdir(parents=True, exist_ok=True)
    sig = {k: [] for k in ('map', 'sap', 'dap', 'hr', 'spo2', 'fio2', 'temp')}
    drugs, pts = [], []
    t0 = pd.Timestamp('2030-01-01')
    for pid in range(1, n_patients + 1):
        start = t0 + pd.Timedelta(days=int(pid * 3))
        h = hours if pid % 20 else 1.5                     # every 20th patient: too few records → excluded
        t = _times(start, h, 2.0, rng)
        mp, falls = _map_series(t, rng)
        pp = rng.uniform(40, 60) + rng.normal(0, 3, len(t))
        sap, dap = mp + 2 * pp / 3, mp - pp / 3
        hr = 80 + rng.normal(0, 3, len(t)) + 0.3 * (np.nanmean(mp) - mp)
        spo2 = np.clip(97 + rng.normal(0, 1.2, len(t)), 85, 100)
        for k, v in (('map', mp), ('sap', sap), ('dap', dap), ('hr', hr), ('spo2', spo2)):
            sig[k].append(pd.DataFrame({'patient_id': pid, 'datetime': t, 'value': np.round(v, 1)}))
        tt = _times(start, h, 60, rng)
        sig['temp'].append(pd.DataFrame({'patient_id': pid, 'datetime': tt, 'value': np.round(36.8 + rng.normal(0, 0.4, len(tt)), 1)}))
        if pid % 3 == 0:                                   # some patients with supplemental oxygen
            tf = _times(start, h, 30, rng)
            sig['fio2'].append(pd.DataFrame({'patient_id': pid, 'datetime': tf, 'value': rng.choice([30, 35, 40, 50], len(tf))}))
        if pid % 2 == 0:                                   # norepinephrine during the deeper falls
            for f_start, depth in falls:
                if depth > 22:
                    ti = start + pd.to_timedelta(f_start + np.arange(0, 120, 15), unit='m')
                    drugs.append(pd.DataFrame({'patient_id': pid, 'datetime': ti, 'drug': 'norepinephrine',
                                               'rate': np.round(rng.uniform(0.05, 0.3), 3)}))
        pts.append({'patient_id': pid, 'sex': rng.choice(['M', 'F']), 'age': int(rng.integers(30, 90)),
                    'charlson_index': int(rng.integers(0, 7)), 'ltst': pid % 25 == 0, 'exclude': False})
    pd.DataFrame(pts).to_csv(out / 'patients.csv', index=False)
    for k, v in sig.items():
        pd.concat(v, ignore_index=True).to_csv(out / f'{k}.csv', index=False)
    pd.concat(drugs, ignore_index=True).to_csv(out / 'vasoactives.csv', index=False)
    print(f'Synthetic dataset ({n_patients} artificial patients) written to {out}')
