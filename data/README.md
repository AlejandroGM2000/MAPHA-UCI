# Input data

No patient data are included in this repository. The pipeline starts from tables already extracted from an ICU
clinical information system. Each site must export its own data to the format below (`.csv` or pandas `.pkl`) and set
`AHE_DATA_DIR` to the folder that contains them.

`python run_pipeline.py --synthetic` writes an entirely artificial example of every table to `data/synthetic/`.

## `patients`

One row per ICU admission eligible for the study (adults admitted to the ICU).

| Column | Type | Description |
|---|---|---|
| `patient_id` | integer | Admission identifier, used in every table |
| `sex` | `M` / `F` | Sex recorded in the electronic health record |
| `age` | number | Age at admission (years) |
| `charlson_index` | number | Charlson comorbidity index |
| `ltst` | boolean | Withdrawal or limitation of life-sustaining treatment (these patients are excluded) |
| `exclude` | boolean | Any other site-specific exclusion (optional; `false` if absent) |

## Physiological signals

One table per signal, one row per recorded value: `map`, `sap`, `dap` (mean, systolic and diastolic invasive
arterial pressure, mmHg), `hr` (heart rate, bpm), `spo2` (%), `fio2` (%), `temp` (°C).

| Column | Type | Description |
|---|---|---|
| `patient_id` | integer | Admission identifier |
| `datetime` | timestamp | Time of the recorded value |
| `value` | number | Recorded value |

Values outside these plausible ranges are discarded: MAP 40–150, SAP 50–250, DAP 30–150, HR 20–200, SpO2 50–100,
FiO2 21–100, temperature 30–42.

## `vasoactives`

One row per recorded infusion rate.

| Column | Type | Description |
|---|---|---|
| `patient_id` | integer | Admission identifier |
| `datetime` | timestamp | Time of the record |
| `drug` | text | `norepinephrine`, `epinephrine`, `vasopressin` or `dobutamine` |
| `rate` | number | Infusion rate (norepinephrine and dobutamine in µg/kg/min, vasopressin in U/min; epinephrine: recorded dose) |

Vasoactive infusions are used only to label events; they are never model inputs.
