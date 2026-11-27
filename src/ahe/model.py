"""LSTM model: data preparation, architecture and training.

  * Windows are assigned to the training, validation or test set by patient (output/split.json).
  * Negative (non-AHE) training windows are randomly downsampled; validation and test keep the original distribution.
  * Min–max scaling fitted on the training windows only.
  * Input: the 10 observation periods × 29 variables of each window; label: AHE in the target period.
  * Architecture: LSTM 128-128-64 (dropout 0.3/0.2/0.2) → Dense 32-16 ReLU (dropout 0.3/0.2) → sigmoid.
    Adam (learning rate 0.001), binary cross-entropy, up to 30 epochs, early stopping on validation loss (patience 5),
    keeping the best weights.
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config as C
from . import io
from .features import FEATURES


def load_split():
    with open(C.output_path('split.json'), encoding='utf-8') as f:
        return json.load(f)


def downsample(train, log=print):
    """Random downsampling of the negative training windows to TRAIN_NEGATIVES_KEPT (or all, if fewer)."""
    last = train.groupby('window').tail(1)
    neg = last.loc[last.ahe == 0, 'window']
    n_keep = min(C.TRAIN_NEGATIVES_KEPT, len(neg))
    drop = set(neg.sample(n=len(neg) - n_keep, random_state=C.SEED))
    log(f'  training negatives: {len(neg):,} → {n_keep:,} (positives: {int(last.ahe.sum()):,})')
    return train[~train.window.isin(drop)]


def to_arrays(df, scaler):
    """X (n_windows, 10, 29), y (n_windows,) and window identifiers, in window order."""
    df = df.sort_values(['window', 'period'])
    n = df.groupby('window').size()
    assert (n == C.WINDOW_PERIODS).all(), 'every window must have WINDOW_PERIODS rows'
    V = scaler.transform(df[FEATURES]).astype(np.float32).reshape(-1, C.WINDOW_PERIODS, len(FEATURES))
    y = df['ahe'].to_numpy(np.float32).reshape(-1, C.WINDOW_PERIODS)[:, -1]
    ids = df[['window', 'patient_id']].iloc[::C.WINDOW_PERIODS].reset_index(drop=True)
    return V[:, :C.OBSERVATION_PERIODS, :], y, ids


def prepare(log=print):
    """Scaled arrays for the training, validation and test sets (cached in output/arrays.npz)."""
    cache = C.output_path('arrays.npz')
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return {k: z[k] for k in z.files}
    w = io.load('windows')
    split = load_split()
    parts = {k: w[w.patient_id.isin(set(v))] for k, v in split.items()}
    parts['train'] = downsample(parts['train'], log)
    scaler = MinMaxScaler().fit(parts['train'][FEATURES])
    with open(C.output_path('scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    out = {}
    for k, df in parts.items():
        X, y, ids = to_arrays(df, scaler)
        out[f'X_{k}'], out[f'y_{k}'] = X, y
        out[f'window_{k}'], out[f'patient_{k}'] = ids.window.values, ids.patient_id.values
        log(f'  {k}: {len(y):,} windows, {int(y.sum()):,} positive, {len(np.unique(ids.patient_id)):,} patients')
    np.savez(cache, **out)
    return out


def build(input_shape=(C.OBSERVATION_PERIODS, len(FEATURES))):
    import tensorflow as tf
    layers = []
    for i, (n, d) in enumerate(zip(C.LSTM_UNITS, C.LSTM_DROPOUT)):
        kw = {'input_shape': input_shape} if i == 0 else {}
        layers += [tf.keras.layers.LSTM(n, return_sequences=i < len(C.LSTM_UNITS) - 1, **kw), tf.keras.layers.Dropout(d)]
    for n, d in zip(C.DENSE_UNITS, C.DENSE_DROPOUT):
        layers += [tf.keras.layers.Dense(n, activation='relu'), tf.keras.layers.Dropout(d)]
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    model = tf.keras.Sequential(layers)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=C.LEARNING_RATE), loss='binary_crossentropy',
                  metrics=[tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(curve='PR', name='auprc')])
    return model


def load_model():
    """The trained model (architecture rebuilt and weights loaded from output/model.weights.h5)."""
    model = build()
    model.load_weights(str(C.output_path('model.weights.h5')))
    return model


def train(d, log=print):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(C.TRAINING_SEED)
    tf.config.experimental.enable_op_determinism()      # repeated runs on the same machine give the same model
    model = build()
    cb = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=C.PATIENCE, restore_best_weights=True)]
    h = model.fit(d['X_train'], d['y_train'], validation_data=(d['X_val'], d['y_val']), epochs=C.EPOCHS,
                  batch_size=C.BATCH_SIZE, shuffle=True, callbacks=cb, verbose=2)
    model.save_weights(str(C.output_path('model.weights.h5')))
    with open(C.output_path('training_history.json'), 'w', encoding='utf-8') as f:
        json.dump({k: [float(x) for x in v] for k, v in h.history.items()}, f, indent=1)
    log(f'  trained for {len(h.history["loss"])} epochs; best validation loss {min(h.history["val_loss"]):.4f}')
    return model


def predict(model, d):
    rows = []
    for k in ('val', 'test'):
        p = model.predict(d[f'X_{k}'], batch_size=4096, verbose=0).ravel()
        rows.append(pd.DataFrame({'window': d[f'window_{k}'], 'patient_id': d[f'patient_{k}'], 'set': k,
                                  'label': d[f'y_{k}'].astype(int), 'p': p}))
    return pd.concat(rows, ignore_index=True)


def run(log=print):
    log('[train] Preparing windows')
    d = prepare(log)
    log('[train] Training the LSTM')
    model = train(d, log)
    predict(model, d).to_pickle(C.output_path('predictions.pkl'))
    log('  predictions saved (validation and test)')
