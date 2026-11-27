"""SHAP explanations of the LSTM (GradientExplainer, i.e. expected gradients).

Each window yields one SHAP value per variable and 4-minute time step (29 variables × 10 time steps). Feature importance
is summarised in two ways:
  * per time step: each time step of each window is treated as a separate observation (mean |SHAP|);
  * per window: SHAP values are summed over the 10 time steps before taking the absolute value.

Results: output/results/shap_importance.csv and output/results/shap_values.npz.
"""
import numpy as np
import pandas as pd

from . import config as C
from . import model as M
from .features import FEATURES

N_BACKGROUND = 500        # training windows used as background
N_EXPLAINED = 1000        # test windows explained
N_SAMPLES = 200           # samples per explanation


def run(log=print):
    import shap
    import tensorflow as tf
    d = M.prepare(log=lambda *a: None)
    model = M.load_model()
    rng = np.random.default_rng(C.SEED)
    background = d['X_train'][rng.choice(len(d['X_train']), min(N_BACKGROUND, len(d['X_train'])), replace=False)]
    idx = rng.choice(len(d['X_test']), min(N_EXPLAINED, len(d['X_test'])), replace=False)
    X = d['X_test'][idx]
    log(f'[explain] SHAP (GradientExplainer): {len(X):,} test windows, background of {len(background):,} training windows')
    tf.keras.utils.set_random_seed(C.SEED)
    explainer = shap.GradientExplainer(model, background)
    sv = np.asarray(explainer.shap_values(X, nsamples=N_SAMPLES, rseed=C.SEED)).reshape(X.shape)
    np.savez(C.output_path('results', 'shap_values.npz'), shap_values=sv, test_index=idx)
    imp = pd.DataFrame({'feature': FEATURES,
                        'mean |SHAP| per time step': np.abs(sv.reshape(-1, sv.shape[2])).mean(0),
                        'mean |SHAP summed over the window|': np.abs(sv.sum(1)).mean(0)}).set_index('feature')
    imp['rank per time step'] = imp.iloc[:, 0].rank(ascending=False).astype(int)
    imp['rank per window'] = imp.iloc[:, 1].rank(ascending=False).astype(int)
    imp.sort_values('rank per time step').to_csv(C.output_path('results', 'shap_importance.csv'), encoding='utf-8-sig')
    log(f'  most important variables: {", ".join(imp.sort_values("rank per time step").index[:5])}')
