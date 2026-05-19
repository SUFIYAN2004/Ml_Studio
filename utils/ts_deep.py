"""
ts_deep.py
Deep-learning time-series forecasting using Keras/TensorFlow.
Supports LSTM, Bidirectional LSTM, GRU, and stacked variants.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
import datetime


def make_sequences(values: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a 1-D array into (X, y) sequences for LSTM."""
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i: i + seq_len])
        y.append(values[i + seq_len])
    return np.array(X), np.array(y)


def build_model(arch: str, seq_len: int, units: int, dropout: float,
                n_layers: int, learning_rate: float, bidirectional: bool = False):
    """
    Build a Keras model.
    arch: "LSTM" | "GRU" | "BiLSTM"
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    tf.get_logger().setLevel("ERROR")

    inp = keras.Input(shape=(seq_len, 1))
    x = inp

    RNNCell = layers.LSTM if arch in ("LSTM", "BiLSTM") else layers.GRU
    use_bi = (arch == "BiLSTM") or bidirectional

    for i in range(n_layers):
        return_seq = (i < n_layers - 1)
        rnn = RNNCell(units, return_sequences=return_seq, dropout=dropout)
        x = layers.Bidirectional(rnn)(x) if use_bi else rnn(x)
        if return_seq:
            x = layers.LayerNormalization()(x)

    x = layers.Dense(max(units // 2, 16), activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_dl_timeseries(
    df_ts: pd.DataFrame,
    date_col: str,
    target_col: str,
    arch: str = "LSTM",
    seq_len: int = 10,
    units: int = 64,
    n_layers: int = 2,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 32,
    test_size: float = 0.2,
    bidirectional: bool = False,
    early_stopping_patience: int = 10,
    progress_callback=None,
):
    """
    Full pipeline:
      1. Sort by date, extract target
      2. MinMax scale
      3. Create sequences
      4. Train/test split (temporal, no shuffle)
      5. Train model
      6. Inverse-scale predictions
      7. Compute metrics
    Returns results dict compatible with page_5 Results page.
    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow import keras

    tf.get_logger().setLevel("ERROR")

    # ── Prep data ──────────────────────────────────────────────────────────────
    df = df_ts[[date_col, target_col]].copy().sort_values(date_col).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df.dropna(subset=[target_col], inplace=True)

    values = df[target_col].values.astype(float).reshape(-1, 1)
    dates = df[date_col].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values).flatten()

    # ── Sequences ──────────────────────────────────────────────────────────────
    X_seq, y_seq = make_sequences(scaled, seq_len)
    X_seq = X_seq.reshape(-1, seq_len, 1)

    split = int(len(X_seq) * (1 - test_size))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test_sc = y_seq[:split], y_seq[split:]
    dates_test = dates[seq_len + split:]

    # ── Build model ────────────────────────────────────────────────────────────
    model = build_model(arch, seq_len, units, dropout, n_layers, learning_rate, bidirectional)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    # ── Predict & inverse scale ────────────────────────────────────────────────
    y_pred_sc = model.predict(X_test, verbose=0).flatten()
    y_test_orig = scaler.inverse_transform(y_test_sc.reshape(-1, 1)).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae  = mean_absolute_error(y_test_orig, y_pred_orig)
    mse  = mean_squared_error(y_test_orig, y_pred_orig)
    r2   = r2_score(y_test_orig, y_pred_orig)

    result = {
        "model": model,
        "scaler": scaler,
        "history": history.history,
        "y_test": y_test_orig,
        "y_pred": y_pred_orig,
        "dates_test": dates_test,
        "task_type": "Time Series",
        "dl_arch": arch,
        "seq_len": seq_len,
        "epochs_ran": len(history.history["loss"]),
        "df_full": df,
        "date_col": date_col,
        "target_col": target_col,
        "metrics": {
            "R² Score": round(r2, 4),
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
        },
    }
    return result


def forecast_future(model, last_sequence: np.ndarray, scaler, n_steps: int, seq_len: int):
    """
    Autoregressive future forecast.
    last_sequence: last seq_len values (original scale)
    Returns: array of n_steps future predictions (original scale)
    """
    from sklearn.preprocessing import MinMaxScaler

    sc_seq = scaler.transform(last_sequence.reshape(-1, 1)).flatten()
    buf = list(sc_seq)
    preds = []
    for _ in range(n_steps):
        inp = np.array(buf[-seq_len:]).reshape(1, seq_len, 1)
        p = model.predict(inp, verbose=0)[0, 0]
        preds.append(p)
        buf.append(p)

    preds_orig = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    return preds_orig
