# Early prediction of acute hypotensive events in the ICU (LSTM)

Code implementing the data preprocessing, acute hypotensive event (AHE) labelling, prediction-window generation, model
development and evaluation, calibration and explainability analyses described in the accompanying article.

The model predicts, every 4 minutes, whether an AHE will occur 16 minutes later, using 40 minutes of routinely recorded
ICU variables.

**No patient data and no trained model are included.** The pipeline runs on tables exported from an ICU clinical
information system in the format described in [`data/README.md`](data/README.md). A synthetic example dataset can be
generated to run the whole pipeline.

## Method in brief

| Step | Module | Description |
|---|---|---|
| Cohort and arterial pressure | `pressure.py` | ≥ 60 invasive arterial pressure records and median interval between records < 4 min (MAP, DAP, SAP); 4-minute grid; missing pressure derived with MAP = DAP + (SAP − DAP)/3; gaps ≤ 8 min forward-filled, longer gaps = no monitoring; outliers (pulse pressure < 10 or > 110 mmHg, or MAP outside DAP–SAP) removed and patients with > 10 % outlying periods excluded; MAP replaced by the formula value when they differ by > 8 mmHg |
| Other variables | `vitals.py` | HR, SpO2, FiO2 and temperature on the same grid; median interval between HR and SpO2 records < 4 min; forward filling within each patient; FiO2 21 % (ambient air) and the training median for the other variables when no earlier value exists |
| AHE labelling | `labelling.py` | MAP < 65 mmHg either sustained (two consecutive 4-minute periods, ≥ 8 min) or with a vasoactive infusion in the same period or within the preceding or following 32 minutes |
| Features | `features.py` | 29 inputs: MAP, SAP, DAP, HR, SpO2, FiO2, temperature; maximum, minimum, SD and slope of MAP and HR over 12 and 24 min; SpO2 SD; MAP median over 28 min; sex, age and Charlson index |
| Prediction windows | `windows.py` | 40-minute observation window, 12-minute gap and 4-minute target period |
| Model | `model.py` | Patient-level split (70/15/15 %); random downsampling of negative training windows; min–max scaling fitted on training data; LSTM 128-128-64 → dense 32-16 → sigmoid; Adam; early stopping on validation loss |
| Evaluation | `evaluation.py` | Threshold = point of the ROC curve closest to (0, 1), selected on the validation set; AUROC, AUPRC, sensitivity, specificity, PPV, F1, accuracy and false positives per true positive with patient-level bootstrap 95 % CIs; calibration (intercept, slope, Brier score) and Platt recalibration fitted on the validation set; comparison with a logistic regression on current MAP and MAP slope; performance by sex and by vasoactive support at the time of prediction; sensitivity analysis restricted to windows with unmodified MAP; comparison of observation windows preceding and not preceding an AHE; follow-up of false-positive predictions |
| Explainability | `explainability.py` | SHAP (GradientExplainer) with importance per time step and aggregated over the observation window |

All parameters are defined in [`src/ahe/config.py`](src/ahe/config.py).

## Requirements

Python 3.10 and the packages in [`requirements.txt`](requirements.txt):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the whole pipeline on the synthetic example dataset (a few minutes on a laptop):

```bash
python run_pipeline.py --synthetic
```

Run it on your own data:

```bash
export AHE_DATA_DIR=/path/to/input/tables      # Windows: set AHE_DATA_DIR=C:\path\to\input\tables
export AHE_OUTPUT_DIR=/path/to/outputs
python run_pipeline.py
```

Individual steps can be run with `--steps preprocess train evaluate explain`. A predefined patient split can be provided
with `AHE_SPLIT_FILE` (JSON with `train`, `val` and `test` lists of patient identifiers).

## Outputs (`AHE_OUTPUT_DIR`)

| File | Content |
|---|---|
| `intermediate/*.pkl` | Arterial pressure, labels, MAP origin and prediction windows |
| `split.json` | Patient-level split |
| `model.weights.h5`, `scaler.pkl`, `training_history.json` | Trained model |
| `predictions.pkl` | Predicted probabilities for the validation and test windows |
| `results/*.csv` | Evaluation and explainability results (see the docstring of `evaluation.py`) |

Nothing is written outside `AHE_OUTPUT_DIR`.

## Tests

```bash
python -m unittest discover -s tests
```

The tests check the AHE definition on hand-made cases and run the complete pipeline on a small synthetic dataset.

## Data availability

Patient-level data cannot be shared because of data-protection regulations. The synthetic dataset is artificial and
results obtained on it have no clinical meaning.
