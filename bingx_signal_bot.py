import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
INTERVAL = "15m"   # 15m, 1h, 4h
LIMIT = 300
SLEEP_SECONDS = 20

USE_TELEGRAM = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# =========================
# LOG
# =========================
def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}", flush=True)

# =========================
# TELEGRAM
# =========================
def send_telegram(message):
    if not USE_TELEGRAM or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram no configurado.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        r = requests.post(url, data=payload, timeout=15)
        r.raise_for_status()
        log("Telegram enviado correctamente.")
    except Exception as e:
        log(f"Error enviando Telegram: {e}")

# =========================
# INTERVAL MAP
# =========================
def bybit_interval(interval):
    mapping = {
        "1m": "1",
        "3m": "3",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "1h": "60",
        "2h": "120",
        "4h": "240",
        "6h": "360",
        "12h": "720",
        "1d": "D"
    }
    return mapping.get(interval, "15")

# =========================
# GET DATA FROM BYBIT
# =========================
def get_klines(symbol, interval="15m", limit=300):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": bybit_interval(interval),
        "limit": limit
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    if data.get("retCode") != 0:
        raise Exception(f"Bybit error: {data}")

    rows = data["result"]["list"]

    if not rows:
        raise Exception(f"No data for {symbol}")

    # Bybit devuelve orden descendente, lo damos vuelta
    rows = rows[::-1]

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "turnover"
    ])

    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    return df

# =========================
# INDICATORS
# =========================
def compute_indicators(df):
    df = df.copy()

    # EMA 200
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI 14
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR 14
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()

    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    return df

# =========================
# STRATEGY
# =========================
def check_signal(df):
    if len(df) < 220:
        return {"signal": "NO_DATA"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["close"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"])
    prev_rsi = float(prev["rsi"])
    atr = float(last["atr"])
    atr_pct = float(last["atr_pct"])

    trend = "LONG_BIAS" if close > ema200 else "SHORT_BIAS"
    vol_ok = 0.25 <= atr_pct <= 3.0

    signal = "NO_TRADE"

    # LONG:
    # - precio arriba de EMA200
    # - RSI cruza arriba de 50
    # - volatilidad OK
    if close > ema200 and prev_rsi < 50 and rsi > 50 and vol_ok:
        signal = "LONG"

    # SHORT:
    # - precio abajo de EMA200
    # - RSI cruza abajo de 50
    # - volatilidad OK
    elif close < ema200 and prev_rsi > 50 and rsi < 50 and vol_ok:
        signal = "SHORT"

    return {
        "signal": signal,
        "close": round(close, 4),
        "ema200": round(ema200, 4),
        "rsi": round(rsi, 2),
        "atr": round(atr, 4),
        "atr_pct": round(atr_pct, 3),
        "trend": trend,
        "vol_ok": vol_ok
    }

# =========================
# FORMAT SIGNAL
# =========================
def format_signal(symbol, state):
    signal = state["signal"]
    close = state["close"]
    ema200 = state["ema200"]
    rsi = state["rsi"]
    atr = state["atr"]
    atr_pct = state["atr_pct"]

    if signal == "LONG":
        sl = round(close - (atr * 1.5), 4)
        tp1 = round(close + (atr * 1.5), 4)
        tp2 = round(close + (atr * 3.0), 4)

        return (
            f"🟢 SEÑAL LONG - {symbol} ({INTERVAL})\n"
            f"Entrada aprox: {close}\n"
            f"EMA200: {ema200}\n"
            f"RSI: {rsi}\n"
            f"ATR: {atr} ({atr_pct}%)\n"
            f"SL: {sl}\n"
            f"TP1: {tp1}\n"
            f"TP2: {tp2}\n"
            f"Riesgo sugerido: 0.5% - 1% por trade"
        )

    elif signal == "SHORT":
        sl = round(close + (atr * 1.5), 4)
        tp1 = round(close - (atr * 1.5), 4)
        tp2 = round(close - (atr * 3.0), 4)

        return (
            f"🔴 SEÑAL SHORT - {symbol} ({INTERVAL})\n"
            f"Entrada aprox: {close}\n"
            f"EMA200: {ema200}\n"
            f"RSI: {rsi}\n"
            f"ATR: {atr} ({atr_pct}%)\n"
            f"SL: {sl}\n"
            f"TP1: {tp1}\n"
            f"TP2: {tp2}\n"
            f"Riesgo sugerido: 0.5% - 1% por trade"
        )

    return None

# =========================
# MAIN LOOP
# =========================
def main():
    log(f"Bot iniciado para {', '.join(SYMBOLS)} en {INTERVAL} (Bybit data)")

    last_candle_times = {symbol: None for symbol in SYMBOLS}
    last_signal_sent = {symbol: None for symbol in SYMBOLS}

    send_telegram(f"🤖 Bot ONLINE en Railway\nActivos: {', '.join(SYMBOLS)}\nTimeframe: {INTERVAL}\nFuente: Bybit")

    while True:
        for symbol in SYMBOLS:
            try:
                df = get_klines(symbol, INTERVAL, LIMIT)
                df = compute_indicators(df)

                current_candle_time = df.iloc[-1]["open_time"]

                if last_candle_times[symbol] is None:
                    last_candle_times[symbol] = current_candle_time
                    log(f"{symbol} - Inicializado en vela: {current_candle_time}")
                    continue

                if current_candle_time != last_candle_times[symbol]:
                    last_candle_times[symbol] = current_candle_time
                    log(f"{symbol} - Nueva vela cerrada: {current_candle_time}")

                    state = check_signal(df)

                    if state["signal"] in ["LONG", "SHORT"]:
                        signal_key = f"{symbol}_{state['signal']}_{current_candle_time}"

                        if last_signal_sent[symbol] != signal_key:
                            msg = format_signal(symbol, state)
                            if msg:
                                log(f"{symbol} - Señal detectada: {state}")
                                send_telegram(msg)
                                last_signal_sent[symbol] = signal_key
                        else:
                            log(f"{symbol} - Señal repetida evitada.")
                    else:
                        log(f"{symbol} - Sin señal. Estado: {state}")

            except Exception as e:
                log(f"{symbol} - ERROR: {e}")

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
