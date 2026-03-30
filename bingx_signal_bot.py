import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]   # BingX futures format
INTERVAL = "15m"   # 15m, 1h, 4h
LIMIT = 300
SLEEP_SECONDS = 20

USE_TELEGRAM = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Base API BingX
BINGX_BASE_URL = "https://open-api.bingx.com"

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
# BINGX INTERVAL MAP
# =========================
def bingx_interval(interval):
    mapping = {
        "1m": "1m",
        "3m": "3m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "8h": "8h",
        "12h": "12h",
        "1d": "1d"
    }
    return mapping.get(interval, "15m")

# =========================
# GET DATA FROM BINGX FUTURES
# =========================
def get_klines(symbol, interval="15m", limit=300):
    """
    BingX Perpetual Futures public klines
    Endpoint doc family:
    /openApi/swap/v3/quote/klines
    """
    url = f"{BINGX_BASE_URL}/openApi/swap/v3/quote/klines"
    params = {
        "symbol": symbol,
        "interval": bingx_interval(interval),
        "limit": limit
    }

    headers = {
        "accept": "application/json",
        "user-agent": "Mozilla/5.0"
    }

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Algunos endpoints BingX devuelven data directo, otros dentro de "data"
    rows = None
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            rows = data["data"]
        elif "data" in data and isinstance(data["data"], dict):
            # a veces puede venir dentro de data -> klines/list
            if "klines" in data["data"]:
                rows = data["data"]["klines"]
            elif "list" in data["data"]:
                rows = data["data"]["list"]
        elif "result" in data and isinstance(data["result"], list):
            rows = data["result"]

    if not rows:
        raise Exception(f"Respuesta inesperada de BingX: {data}")

    # Normalizamos distintos formatos posibles
    parsed = []
    for row in rows:
        # Caso lista estilo [time, open, high, low, close, volume, ...]
        if isinstance(row, list):
            if len(row) >= 6:
                parsed.append({
                    "open_time": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5]
                })
        # Caso dict
        elif isinstance(row, dict):
            parsed.append({
                "open_time": row.get("time") or row.get("openTime") or row.get("timestamp"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume", 0)
            })

    if not parsed:
        raise Exception(f"No se pudieron parsear klines de BingX: {data}")

    df = pd.DataFrame(parsed)

    # tiempo ms
    df["open_time"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)

    df = df.sort_values("open_time").reset_index(drop=True)

    return df

# =========================
# INDICATORS
# =========================
def compute_indicators(df):
    df = df.copy()

    # EMA 200
    df["ema200"] = df["close"].ewm(span=200, adjust=False).mean()

    # RSI 14 (simple rolling)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, 1e-9)

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

    # LONG: tendencia + cruce RSI 50 + volatilidad aceptable
    if close > ema200 and prev_rsi < 50 and rsi > 50 and vol_ok:
        signal = "LONG"

    # SHORT: tendencia + cruce RSI 50 + volatilidad aceptable
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

    symbol_show = symbol.replace("-", "")

    if signal == "LONG":
        sl = round(close - (atr * 1.5), 4)
        tp1 = round(close + (atr * 1.5), 4)
        tp2 = round(close + (atr * 3.0), 4)

        return (
            f"🟢 SEÑAL LONG - {symbol_show} ({INTERVAL})\n"
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
            f"🔴 SEÑAL SHORT - {symbol_show} ({INTERVAL})\n"
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
    log(f"Bot iniciado para {', '.join(SYMBOLS)} en {INTERVAL} (BingX data)")
    send_telegram(f"🤖 Bot ONLINE en Railway\nActivos: {', '.join([s.replace('-', '') for s in SYMBOLS])}\nTimeframe: {INTERVAL}\nFuente: BingX Futures")

    last_candle_times = {symbol: None for symbol in SYMBOLS}
    last_signal_sent = {symbol: None for symbol in SYMBOLS}

    while True:
        for symbol in SYMBOLS:
            try:
                df = get_klines(symbol, INTERVAL, LIMIT)

                if len(df) < 50:
                    log(f"{symbol} - Pocas velas recibidas: {len(df)}")

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
