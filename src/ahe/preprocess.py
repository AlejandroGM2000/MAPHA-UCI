"""Preprocessing: from the input tables to the prediction windows.

    patients + signals ──► arterial pressure (cohort) ──► HR/SpO2/FiO2/temperature (cohort) ──► patient split
        ──► imputation ──► AHE labels ──► derived features ──► prediction windows
"""
import json

from sklearn.model_selection import train_test_split

from . import config as C
from . import features as F
from . import io
from . import labelling as L
from . import pressure as P
from . import vitals as V
from . import windows as W


def patient_split(patient_ids, log=print):
    """Random patient-level split (70 % training, 15 % validation, 15 % test), or the one in AHE_SPLIT_FILE."""
    if C.SPLIT_FILE:
        with open(C.SPLIT_FILE, encoding='utf-8') as f:
            split = {k: [int(x) for x in v] for k, v in json.load(f).items()}
        log(f'  patient split read from AHE_SPLIT_FILE')
    else:
        ids = sorted(int(x) for x in patient_ids)
        train, rest = train_test_split(ids, test_size=C.TEST_FRACTION, random_state=C.SEED)
        val, test = train_test_split(rest, test_size=0.5, random_state=C.SEED)
        split = {'train': sorted(train), 'val': sorted(val), 'test': sorted(test)}
    with open(C.output_path('split.json'), 'w', encoding='utf-8') as f:
        json.dump(split, f)
    log('  ' + ' · '.join(f'{k}: {len(v):,} patients' for k, v in split.items()))
    return split


def run(log=print):
    log('[1/6] Arterial pressure and cohort')
    patients = io.read_input('patients')
    pressure, reference, raw = P.process(patients, log)
    io.save(pressure, 'pressure')
    io.save(reference, 'reference')
    io.save(P.map_origin(pressure, raw), 'map_origin')
    del raw

    log('[2/6] Heart rate, SpO2, FiO2 and temperature')
    observed = V.cohort_grid(set(pressure.patient_id), reference, log)
    io.save(observed, 'vitals_observed')
    cohort = sorted(observed.patient_id.unique())

    log('[3/6] Patient split')
    split = patient_split(cohort, log)
    vitals = V.impute(observed, split['train'], log)
    del observed

    log('[4/6] AHE labels')
    labelled = L.label(pressure, cohort, reference, log)
    io.save(labelled, 'labels')
    del pressure

    log('[5/6] Derived features')
    table = F.model_table(labelled, vitals, log)
    del labelled, vitals

    log('[6/6] Prediction windows')
    windows = W.build(table, patients, log)
    io.save(windows, 'windows')
    return windows, split
