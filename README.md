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

Python 3.10 and the packages in [`requirements.txt`](requirements.txt).

First check that `python` is version 3.10 (otherwise, call the Python 3.10 executable by its full path when creating the
environment):

```bash
python --version
```

Create and activate a virtual environment, then install the packages. Run these commands inside the repository folder.

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows, Command Prompt (cmd):

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Windows, PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell reports that running scripts is disabled, allow it for the current window only with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again.

Once activated, the prompt starts with `(.venv)`. In a new terminal, only the activation command is needed again.

## Usage

### Synthetic example

Run the whole pipeline on the synthetic example dataset (a few minutes on a laptop):

```bash
python run_pipeline.py --synthetic
```

The synthetic tables are written to `data/synthetic/` and the results to `output/`.

### Your own data

Prepare the input tables in the format described in [`data/README.md`](data/README.md), then set the input and output
folders and run the pipeline. `AHE_SPLIT_FILE` is optional: a JSON file with `train`, `val` and `test` lists of patient
identifiers; without it, patients are split at random (70/15/15 %).

Linux / macOS:

```bash
export AHE_DATA_DIR=/path/to/input/tables
export AHE_OUTPUT_DIR=/path/to/outputs
export AHE_SPLIT_FILE=/path/to/split.json        # optional
python run_pipeline.py
```

Windows, Command Prompt (cmd) — no spaces around `=`:

```bat
set AHE_DATA_DIR=C:\path\to\input\tables
set AHE_OUTPUT_DIR=C:\path\to\outputs
set AHE_SPLIT_FILE=C:\path\to\split.json
python run_pipeline.py
```

Windows, PowerShell:

```powershell
$env:AHE_DATA_DIR = "C:\path\to\input\tables"
$env:AHE_OUTPUT_DIR = "C:\path\to\outputs"
$env:AHE_SPLIT_FILE = "C:\path\to\split.json"    # optional
python run_pipeline.py
```

In cmd, omit the `AHE_SPLIT_FILE` line if no predefined split is used. These variables only last while the terminal
window is open.

With patient data, choose an output folder **outside** the repository folder, so that results derived from patient data
are never committed by mistake.

As a reference, the complete pipeline on about 4,300 patients took about 20–25 minutes on a laptop CPU (preprocessing
≈ 8 min, training ≈ 8–15 min, evaluation and SHAP a few minutes).

Individual steps can be run with `--steps preprocess train evaluate explain`.

TensorFlow prints several `WARNING` and information messages (deprecations, CPU instructions, oneDNN). They are normal
and can be ignored.

## Outputs (`AHE_OUTPUT_DIR`)

| File | Content |
|---|---|
| `intermediate/*.pkl` | Arterial pressure, labels, MAP origin and prediction windows |
| `split.json` | Patient-level split |
| `model.weights.h5`, `scaler.pkl`, `training_history.json` | Trained model |
| `predictions.pkl` | Predicted probabilities for the validation and test windows |
| `results/*.csv` | Evaluation and explainability results (see the docstring of `evaluation.py`) |

Apart from the synthetic example dataset (`data/synthetic/`), nothing is written outside `AHE_OUTPUT_DIR`.

## Tests

```bash
python -m unittest discover -s tests
```

The tests check the AHE definition on hand-made cases and run the complete pipeline on a small synthetic dataset.

## Data availability

Patient-level data cannot be shared because of data-protection regulations. The synthetic dataset is artificial and
results obtained on it have no clinical meaning.
