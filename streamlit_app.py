import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau

st.set_page_config(page_title="Predicción precio de cierre", layout="wide")

# -------------------------
# Función principal LSTM
# -------------------------
def train_and_predict(ticker: str, start_date: str = "2020-03-01", window_size: int = 60):
    today = date.today()
    end_date = today.strftime("%Y-%m-%d")  # yfinance: end exclusivo → llega hasta AYER

    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No se pudieron descargar datos. Revisa el ticker o las fechas.")
        return None

    data = data[["Close"]].copy()
    close_prices = data["Close"].values.reshape(-1,1)

    # Escalado
    scaler = MinMaxScaler(feature_range=(0,1))
    close_scaled = scaler.fit_transform(close_prices)

    # Crear secuencias
    def create_sequences(series, window):
        X, y = [], []
        for i in range(window, len(series)):
            X.append(series[i-window:i, 0])
            y.append(series[i, 0])
        X = np.array(X)
        y = np.array(y)
        return X, y

    X_all, y_all = create_sequences(close_scaled, window_size)
    X_all = np.expand_dims(X_all, axis=-1)

    train_size = int(len(X_all) * 0.8)
    X_train = X_all[:train_size]
    y_train = y_all[:train_size]
    X_test  = X_all[train_size:]
    y_test  = y_all[train_size:]

    # Modelo
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(window_size, 1)),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        loss="mean_squared_error",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        epochs=20,            # algo más corto para que la app no tarde tanto
        batch_size=32,
        validation_split=0.1,
        callbacks=[reduce_lr],
        verbose=0
    )

    # Predicciones en test
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    mae = mean_absolute_error(y_test_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred))

    # Predicción de HOY (usando los últimos 60 cierres hasta AYER)
    last_60_closes = close_prices[-window_size:]
    arr = np.array(last_60_closes).reshape(-1,1)
    arr_scaled = scaler.transform(arr)
    X_input = arr_scaled.reshape(1, window_size, 1)
    pred_scaled = model.predict(X_input, verbose=0)[0,0]
    pred_price = scaler.inverse_transform([[pred_scaled]])[0,0]

    # Fechas para el test
    fechas_test = data.index[window_size + train_size : window_size + train_size + len(y_test_real)]

    return {
        "data": data,
        "fechas_test": fechas_test,
        "y_test_real": y_test_real,
        "y_pred": y_pred,
        "last_close": float(close_prices[-1,0]),
        "pred_price": float(pred_price),
        "mae": float(mae),
        "rmse": float(rmse),
        "history": history.history,
    }

# -------------------------
# Interfaz Streamlit
# -------------------------
st.title("📈 Predicción de precio de cierre con LSTM")
st.markdown(
    """
Esta app entrena un modelo **LSTM** con los datos históricos de la acción seleccionada 
(desde 2020, post-pandemia) y predice el **precio de cierre de HOY** usando los últimos 60 cierres.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input("Ticker", value="AAPL")
with col2:
    start_date = st.date_input("Fecha inicio datos", value=date(2020,3,1))
with col3:
    window_size = st.slider("Ventana (días)", min_value=30, max_value=120, value=60, step=5)

if st.button("Entrenar modelo y predecir"):
    with st.spinner("Entrenando modelo y generando predicción..."):
        result = train_and_predict(ticker, start_date.strftime("%Y-%m-%d"), window_size)

    if result is not None:
        st.success("¡Modelo entrenado y predicción generada!")

        colA, colB = st.columns(2)
        with colA:
            st.subheader("Predicción de cierre de HOY")
            st.metric(
                label=f"{ticker} – cierre HOY (predicho)",
                value=f"{result['pred_price']:.2f} USD",
                delta=f"{result['pred_price'] - result['last_close']:.2f} vs cierre AYER"
            )

        with colB:
            st.subheader("Rendimiento en conjunto de test")
            st.write(f"**MAE:** {result['mae']:.4f}")
            st.write(f"**RMSE:** {result['rmse']:.4f}")

        st.subheader("Histórico de precios (últimos 200 días)")
        st.line_chart(result["data"]["Close"].tail(200))

        st.subheader("Predicción vs Real en test")
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(result["fechas_test"], result["y_test_real"], label="Real")
        ax.plot(result["fechas_test"], result["y_pred"], label="Predicción LSTM")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Precio de cierre")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Curvas de pérdida (entrenamiento)")
        loss = result["history"]["loss"]
        val_loss = result["history"]["val_loss"]
        fig2, ax2 = plt.subplots(figsize=(8,3))
        ax2.plot(loss, label="Train")
        ax2.plot(val_loss, label="Val")
        ax2.set_xlabel("Época")
        ax2.set_ylabel("MSE")
        ax2.legend()
        st.pyplot(fig2)
