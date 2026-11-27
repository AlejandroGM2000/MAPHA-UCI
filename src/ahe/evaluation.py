"""Evaluation on the test set. Every threshold or recalibration is fitted on the validation set.

Results (output/results/):
  performance.csv            AUROC, AUPRC, sensitivity, specificity, PPV, F1, accuracy, FP/TP with patient-level 95 % CIs
  calibration.csv            mean predicted risk, calibration intercept and slope, Brier score (raw and Platt-recalibrated)
  logistic_regression.csv    comparison with a logistic regression on current MAP and MAP slope (paired bootstrap)
  subgroups.csv              by sex and by vasoactive infusion at the time of prediction
  sensitivity_raw_map.csv    windows in which every MAP value was recorded by the monitor (no derivation or filling)
  map_origin.csv             origin of the MAP values of the cohort
  windows_preceding_ahe.csv  observation windows preceding vs not preceding an AHE (one median per patient; paired Wilcoxon)
  false_positive_follow_up.csv   events and interventions after the predicted period, by prediction class
  false_positive_nadir.csv       lowest MAP in the 32 minutes after the prediction, by prediction class
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import io
from . import model as M
from . import windows as W
from .features import FEATURES


def save(df, name):
    df.to_csv(C.output_path('results', f'{name}.csv'), encoding='utf-8-sig')
    return df


# ----------------------------------------------------------------------------------------------- metrics
def threshold_closest_to_corner(y, p):
    """Point of the ROC curve closest to (0, 1)."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmin(np.sqrt(fpr ** 2 + (1 - tpr) ** 2))])


def metrics(y, p, threshold):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    yp = (p >= threshold).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum()); fn = int(((yp == 0) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum()); tn = int(((yp == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    return dict(AUROC=roc_auc_score(y, p), AUPRC=average_precision_score(y, p), sensitivity=sens,
                specificity=tn / (tn + fp) if tn + fp else np.nan, PPV=ppv,
                F1=2 * ppv * sens / (ppv + sens) if ppv and (ppv + sens) else np.nan,
                accuracy=(tp + tn) / len(y), FP_per_TP=fp / tp if tp else np.nan, TP=tp, FP=fp, FN=fn, TN=tn)


CI_METRICS = ['AUROC', 'AUPRC', 'sensitivity', 'specificity', 'PPV', 'F1', 'accuracy', 'FP_per_TP']


def patient_bootstrap(y, p, patients, threshold, n_boot=C.N_BOOTSTRAP, seed=C.SEED):
    """95 % CIs by resampling patients (all windows of a patient together)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'y': np.asarray(y).astype(int), 'p': p, 'pid': np.asarray(patients)})
    groups = {k: v.index.values for k, v in df.groupby('pid')}
    keys = np.array(list(groups))
    res = []
    for _ in range(n_boot):
        s = df.iloc[np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])]
        if s.y.nunique() < 2:
            continue
        m = metrics(s.y.values, s.p.values, threshold)
        res.append([m[k] for k in CI_METRICS])
    res = np.array(res, float)
    return {k: (np.nanpercentile(res[:, i], 2.5), np.nanpercentile(res[:, i], 97.5)) for i, k in enumerate(CI_METRICS)}


def paired_bootstrap(y, p_ref, p_new, patients, n_boot=C.N_BOOTSTRAP, seed=C.SEED):
    """Difference new − reference in AUROC and AUPRC on the same windows, with 95 % CIs by patient bootstrap."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({'y': np.asarray(y).astype(int), 'a': p_ref, 'b': p_new, 'pid': np.asarray(patients)})
    groups = {k: v.index.values for k, v in df.groupby('pid')}
    keys = np.array(list(groups))
    f = {'AUROC': roc_auc_score, 'AUPRC': average_precision_score}
    obs = {k: fn(df.y, df.b) - fn(df.y, df.a) for k, fn in f.items()}
    boot = {k: [] for k in f}
    for _ in range(n_boot):
        s = df.iloc[np.concatenate([groups[k] for k in rng.choice(keys, len(keys), replace=True)])]
        if s.y.nunique() < 2:
            continue
        for k, fn in f.items():
            boot[k].append(fn(s.y, s.b) - fn(s.y, s.a))
    return {k: (obs[k], np.percentile(boot[k], 2.5), np.percentile(boot[k], 97.5)) for k in f}


# ----------------------------------------------------------------------------------------------- calibration
def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def platt(y_val, p_val):
    """Logistic (Platt) recalibration fitted on the validation set: returns a function p → recalibrated p."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p_val).reshape(-1, 1), np.asarray(y_val).astype(int))
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return lambda p: 1 / (1 + np.exp(-(a * _logit(p) + b)))


def calibration(y, p):
    import warnings
    import statsmodels.api as sm
    lg = _logit(p)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')        # small subgroups (e.g. synthetic data) may trigger separation warnings
        slope = sm.GLM(y, sm.add_constant(lg), family=sm.families.Binomial()).fit().params[1]
        intercept = sm.GLM(y, np.ones_like(lg), family=sm.families.Binomial(), offset=lg).fit().params[0]
    return {'mean predicted risk': float(np.mean(p)), 'observed prevalence': float(np.mean(y)),
            'calibration intercept': float(intercept), 'calibration slope': float(slope), 'Brier score': brier_score_loss(y, p)}


# ----------------------------------------------------------------------------------------------- helpers
def _row_index(labels):
    """Row of each patient's first period in the (contiguous) labels table, and number of periods."""
    g = labels.groupby('patient_id', sort=False)['period']
    first, n = g.apply(lambda s: s.index[0]), g.size()
    assert (g.max() + 1 == n).all() and (g.min() == 0).all(), 'each patient must have a contiguous grid starting at 0'
    return first, n


def _map_and_slope(X, scaler):
    """Current MAP (last observation period) and least-squares MAP slope over the window, in mmHg."""
    j = FEATURES.index('map')
    m = X[:, :, j] * scaler.data_range_[j] + scaler.data_min_[j]
    x = np.arange(m.shape[1]) - (m.shape[1] - 1) / 2
    return pd.DataFrame({'map': m[:, -1], 'map_slope': (m - m.mean(1, keepdims=True)) @ x / (x ** 2).sum()})


def _summary(m, ci=None):
    return {k: (f'{m[k]:.3f} ({ci[k][0]:.3f}–{ci[k][1]:.3f})' if ci and k in ci else m[k]) for k in m}


# ----------------------------------------------------------------------------------------------- run
def run(log=print):
    import pickle
    pred = pd.read_pickle(C.output_path('predictions.pkl'))
    val, test = pred[pred.set == 'val'], pred[pred.set == 'test'].reset_index(drop=True)
    yv, pv, yt, pt, pid = val.label.values, val.p.values, test.label.values, test.p.values, test.patient_id.values
    u = threshold_closest_to_corner(yv, pv)
    log(f'[evaluate] threshold (validation set): {u:.4f}')

    m = metrics(yt, pt, u)
    ci = patient_bootstrap(yt, pt, pid, u)
    perf = pd.Series(_summary(m, ci), name='test set').to_frame()
    perf.loc['threshold'] = u
    save(perf, 'performance')
    log(f'  AUROC {m["AUROC"]:.3f} ({ci["AUROC"][0]:.3f}–{ci["AUROC"][1]:.3f}) · AUPRC {m["AUPRC"]:.3f} · '
        f'sensitivity {m["sensitivity"]:.3f} · specificity {m["specificity"]:.3f} · PPV {m["PPV"]:.3f}')

    recal = platt(yv, pv)
    pcal = recal(pt)
    save(pd.DataFrame({'raw model output': calibration(yt, pt), 'Platt recalibration (validation set)': calibration(yt, pcal)}).T,
         'calibration')

    # comparator: logistic regression on current MAP and MAP slope, same training windows
    d = M.prepare(log=lambda *a: None)
    with open(C.output_path('scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=5000))
    lr.fit(_map_and_slope(d['X_train'], scaler), d['y_train'])
    lv = lr.predict_proba(_map_and_slope(d['X_val'], scaler))[:, 1]
    lt_ = pd.Series(lr.predict_proba(_map_and_slope(d['X_test'], scaler))[:, 1], index=d['window_test'])
    lt = lt_.loc[test.window].values
    rows = {}
    for name, a_v, a_t in (('LSTM', pv, pt), ('Logistic regression (current MAP + MAP slope)', lv, lt)):
        rows[name] = metrics(yt, a_t, threshold_closest_to_corner(yv, a_v))
    cmp_ = pd.DataFrame(rows).T
    diff = paired_bootstrap(yt, lt, pt, pid)
    for k, (dv, lo, hi) in diff.items():
        cmp_.loc['Difference LSTM − logistic regression (95% CI)', k] = f'{dv:+.3f} ({lo:+.3f} to {hi:+.3f})'
    save(cmp_, 'logistic_regression')

    # tables per period and per window
    labels = io.load('labels')
    windows = io.load('windows')
    info = W.window_info(windows).set_index('window').loc[test.window].reset_index()
    first, n = _row_index(labels)
    base = first.loc[info.patient_id].values
    t_pred = base + info.prediction_period.values
    t_target = base + info.target_period.values
    rates = labels[[f'rate_{d}' for d in C.DRUGS]].fillna(0).to_numpy() > 0
    on_drug = rates.any(1)

    # subgroups
    sex = windows.set_index('window').groupby(level=0)['sex'].first().loc[test.window].values
    groups = {'Female': sex == 0, 'Male': sex == 1,
              'Without vasoactive support at prediction': ~on_drug[t_pred], 'With vasoactive support at prediction': on_drug[t_pred]}
    rows = {}
    for g, mask in groups.items():
        if mask.sum() == 0 or len(np.unique(yt[mask])) < 2:
            continue
        mm = metrics(yt[mask], pt[mask], u)
        cc = patient_bootstrap(yt[mask], pt[mask], pid[mask], u)
        r = {'patients': len(np.unique(pid[mask])), 'windows': int(mask.sum()), 'events': int(yt[mask].sum())}
        r.update(_summary({k: mm[k] for k in ('AUROC', 'AUPRC', 'sensitivity', 'specificity', 'PPV', 'FP_per_TP')}, {'AUROC': cc['AUROC']}))
        r.update({f'{k} (Platt)': v for k, v in calibration(yt[mask], pcal[mask]).items()})
        rows[g] = r
    save(pd.DataFrame(rows).T, 'subgroups')

    # MAP origin and sensitivity analysis restricted to windows with recorded MAP only
    origin = io.load('map_origin')
    o = labels[['patient_id', 'period']].merge(origin, on=['patient_id', 'period'], how='left')
    save(o.map_origin.value_counts().rename('periods').to_frame().assign(
        pct_of_monitored=lambda x: (x.periods / x.periods.drop('no monitoring', errors='ignore').sum() * 100).round(2)), 'map_origin')
    modified = (~o.map_origin.isin(['measured', 'no monitoring'])).to_numpy().astype(int)
    cs = np.concatenate([[0], np.cumsum(modified)])
    raw_only = (cs[t_target + 1] - cs[t_pred - (C.OBSERVATION_PERIODS - 1)]) == 0
    rows = {}
    for g, mask in (('all test windows', np.ones(len(yt), bool)), ('no MAP value modified', raw_only),
                    ('at least one MAP value modified', ~raw_only)):
        if len(np.unique(yt[mask])) < 2:
            continue
        mm = metrics(yt[mask], pt[mask], u)
        cc = patient_bootstrap(yt[mask], pt[mask], pid[mask], u)
        r = {'windows': int(mask.sum()), 'patients': len(np.unique(pid[mask])), 'events': int(yt[mask].sum())}
        r.update(_summary({k: mm[k] for k in ('AUROC', 'AUPRC', 'sensitivity', 'specificity', 'PPV', 'FP_per_TP')}, {'AUROC': cc['AUROC']}))
        rows[g] = r
    save(pd.DataFrame(rows).T, 'sensitivity_raw_map')

    save(windows_preceding_ahe(windows), 'windows_preceding_ahe')

    # follow-up of the predictions
    cls = np.where(yt == 1, np.where(pt >= u, 'TP', 'FN'), np.where(pt >= u, 'FP', 'TN'))
    last = base + n.loc[info.patient_id].values - 1
    ahe = labels.ahe.to_numpy()
    mp = labels['map'].to_numpy(np.float64)
    low = ((mp < C.MAP_THRESHOLD) & (mp != 0)).astype(int)
    nor = labels['rate_norepinephrine'].fillna(0).to_numpy()
    cum = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in (('ahe', ahe), ('low', low), ('drug', on_drug.astype(int)))}
    rows = {}
    for h in C.FOLLOW_UP_PERIODS:
        end = np.minimum(t_target + h, last)
        within = lambda k: (cum[k][end + 1] - cum[k][t_target + 1]) > 0
        rise = np.array([(nor[a + 1:b + 1].max() if b > a else 0) > nor[a] * 1.2 + 1e-9 and nor[a] > 0 for a, b in zip(t_target, end)])
        rows[f'AHE within {h * 4} min'] = within('ahe')
        rows[f'MAP < 65 mmHg within {h * 4} min'] = within('low')
        rows[f'vasoactive started within {h * 4} min'] = ~on_drug[t_target] & within('drug')
        rows[f'norepinephrine increased ≥ 20% within {h * 4} min'] = rise
    fu = pd.DataFrame(rows)
    fu['class'] = cls
    tab = (fu.groupby('class').mean() * 100).T.round(1)
    tab.loc['windows'] = fu['class'].value_counts()
    save(tab[[c for c in ('TP', 'FP', 'FN', 'TN') if c in tab.columns]], 'false_positive_follow_up')

    mp_nan = np.where(mp == 0, np.nan, mp)
    nadir = np.array([np.nanmin(mp_nan[a:b + 1]) if b >= a and np.isfinite(mp_nan[a:b + 1]).any() else np.nan
                      for a, b in zip(t_pred + 1, np.minimum(t_pred + 8, last))])
    nd = pd.DataFrame({'class': cls, 'map_at_prediction': mp_nan[t_pred], 'nadir_32min': nadir})
    save(nd.groupby('class').agg(windows=('nadir_32min', 'size'), map_at_prediction_median=('map_at_prediction', 'median'),
                                 nadir_median=('nadir_32min', 'median'),
                                 nadir_below_70_pct=('nadir_32min', lambda s: (s < 70).mean() * 100),
                                 nadir_below_75_pct=('nadir_32min', lambda s: (s < 75).mean() * 100)), 'false_positive_nadir')
    log(f'  results written to {C.output_path("results")}')


def windows_preceding_ahe(windows):
    """Observation windows preceding vs not preceding an AHE: value of each variable at the time of prediction, one median
    per patient and window type, paired Wilcoxon signed-rank test (normal approximation) in patients with both types;
    effect size r = |Z| / sqrt(n)."""
    variables = [f for f in FEATURES if f not in ('sex', 'age', 'charlson_index')]
    w = windows.sort_values(['window', 'period'])
    at_pred = w.groupby('window').nth(C.OBSERVATION_PERIODS - 1)
    label = w.groupby('window')['ahe'].last()
    if 'window' in at_pred.columns:
        at_pred = at_pred.set_index('window')
    d = at_pred[['patient_id'] + variables].copy()
    d['group'] = np.where(label.loc[d.index].values == 1, 'preceding', 'not preceding')
    med = d.groupby(['patient_id', 'group'])[variables].median()
    q = lambda x: f'{x.median():.2f} ({x.quantile(.25):.2f}–{x.quantile(.75):.2f})'
    rows = []
    for v in variables:
        a = med.xs('preceding', level='group')[v]
        b = med.xs('not preceding', level='group')[v]
        common = a.index.intersection(b.index)
        diff = (a.loc[common] - b.loc[common]).to_numpy()
        dd = diff[diff != 0]
        k = len(dd)
        if k == 0:
            continue
        rk = stats.rankdata(np.abs(dd))
        z = (rk[dd > 0].sum() - k * (k + 1) / 4) / np.sqrt(k * (k + 1) * (2 * k + 1) / 24)
        rows.append({'variable': v, 'patients': len(common), 'not preceding, median (IQR)': q(b.loc[common]),
                     'preceding, median (IQR)': q(a.loc[common]), 'median difference': float(np.median(diff)),
                     'Z': float(z), 'p': 2 * stats.norm.sf(abs(z)), 'r': abs(float(z)) / np.sqrt(len(common))})
    return pd.DataFrame(rows).set_index('variable')
