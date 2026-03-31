import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# =========================
# CONFIG
# =========================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "PON_TU_TOKEN_AQUI")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "PON_TU_CHAT_ID_AQUI")

SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
INTERVAL = "15m"
CHECK_EVERY_SECONDS = 60

# V2 PARAMS
EMA_FAST = 20
EMA_MID = 50
EMA_TREND = 200
RSI_PERIOD = 14
ATR_PERIOD = 14
VOL_MA_PERIOD = 20

# Filtros V2
MIN_ATR_PCT = 0.25   # volatilidad mínima
LONG_RSI_MIN = 55
LONG_RSI_MAX = 72
SHORT_RSI_MIN = 28
SHORT_RSI_MAX = 45

# Anti-duplicados
last_candle_time = {}
last_signal_sent = {}  # guarda "SYMBOL-candleTime-signal"

# Telegram update offset
telegram_offset = None


# =========================
# UTILS
# =========================
def log(msg):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{now}] {msg}", flush=True)


def send_telegram_message(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram no configurado. Saltando envío.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text
    }

    try:
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code == 200:
            log("Telegram enviado correctamente.")
            return True
        else:
            log(f"Error enviando Telegram: {r.status_code} - {r.text}")
            return False
    except Exception as e:
        log(f"Excepción enviando Telegram: {e}")
        return False


def get_telegram_updates():
    global telegram_offset
    if not TELEGRAM_BOT_TOKEN:
        return []

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": 1}
    if telegram_offset is not None:
        params["offset"] = telegram_offset

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        if not data.get("ok"):
            return []

        results = data.get("result", [])
        if results:
            telegram_offset = results[-1]["update_id"] + 1
        return results
    except Exception as e:
        log(f"Error getUpdates Telegram: {e}")
        return []


# =========================
# BINGX MARKET DATA
# =========================
def fetch_bingx_klines(symbol="BTC-USDT", interval="15m", limit=300):
    """
    BingX endpoint público (swap/linear)
    """
    url = "https://open-api.bingx.com/openApi/swap/v3/quote/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Esperado: {"code":0,"msg":"","data":[...]}
    if data.get("code") != 0:
        raise Exception(f"BingX API error: {data}")

    klines = data.get("data", [])
    if not klines or len(klines) < 50:
        raise Exception(f"No hay suficientes velas para {symbol}")

    rows = []
    for k in klines:
        # Campos comunes:
        # time, open, high, low, close, volume
        rows.append({
            "time": pd.to_datetime(int(k["time"]), unit="ms", utc=True),
            "open": float(k["open"]),
            "high": float(k["high"]),
            "low": float(k["low"]),
            "close": float(k["close"]),
            "volume": float(k["volume"]),
        })

    df = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    return df


# =========================
# INDICATORS
# =========================
def calculate_rsi(series, period=14):
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr


def add_indicators(df):
    df = df.copy()

    df["ema20"] = df["close"].ewm(span=EMA_FAST, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=EMA_MID, adjust=False).mean()
    df["ema200"] = df["close"].ewm(span=EMA_TREND, adjust=False).mean()

    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["atr"] = calculate_atr(df, ATR_PERIOD)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()

    return df


# =========================
# STRATEGY V2
# =========================
def generate_signal_v2(df):
    """
    Usa la última vela CERRADA.
    """
    if len(df) < 220:
        return {"signal": "NO_TRADE", "reason": "not_enough_data"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["close"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])
    ema200 = float(last["ema200"])
    rsi = float(last["rsi"])
    atr = float(last["atr"])
    atr_pct = float(last["atr_pct"])
    volume = float(last["volume"])
    vol_ma = float(last["vol_ma"]) if not pd.isna(last["vol_ma"]) else 0.0

    prev_high = float(prev["high"])
    prev_low = float(prev["low"])

    vol_ok = volume >= vol_ma if vol_ma > 0 else True

    # Tendencia
    long_trend = close > ema200 and ema20 > ema50
    short_trend = close < ema200 and ema20 < ema50

    # Momentum
    long_momentum = LONG_RSI_MIN <= rsi <= LONG_RSI_MAX
    short_momentum = SHORT_RSI_MIN <= rsi <= SHORT_RSI_MAX

    # Confirmación
    long_break = close > prev_high
    short_break = close < prev_low

    # Volatilidad
    volat_ok = atr_pct >= MIN_ATR_PCT

    # LONG
    if long_trend and long_momentum and long_break and volat_ok and vol_ok:
        entry = close
        sl = entry - (atr * 1.5)
        tp1 = entry + (atr * 1.5)
        tp2 = entry + (atr * 3.0)

        return {
            "signal": "LONG",
            "close": round(close, 4),
            "ema20": round(ema20, 4),
            "ema50": round(ema50, 4),
            "ema200": round(ema200, 4),
            "rsi": round(rsi, 2),
            "atr": round(atr, 4),
            "atr_pct": round(atr_pct, 3),
            "vol_ok": bool(vol_ok),
            "prev_high": round(prev_high, 4),
            "entry": round(entry, 4),
            "sl": round(sl, 4),
            "tp1": round(tp1, 4),
            "tp2": round(tp2, 4),
        }

    # SHORT
    if short_trend and short_momentum and short_break and volat_ok and vol_ok:
        entry = close
        sl = entry + (atr * 1.5)
        tp1 = entry - (atr * 1.5)
        tp2 = entry - (atr * 3.0)

        return {
            "signal": "SHORT",
            "close": round(close, 4),
            "ema20": round(ema20, 4),
            "ema50": round(ema50, 4),
            "ema200": round(ema200, 4),
            "rsi": round(rsi, 2),
            "atr": round(atr, 4),
            "atr_pct": round(atr_pct, 3),
            "vol_ok": bool(vol_ok),
            "prev_low": round(prev_low, 4),
            "entry": round(entry, 4),
            "sl": round(sl, 4),
            "tp1": round(tp1, 4),
            "tp2": round(tp2, 4),
        }

    # Estado informativo
    trend = "LONG_BIAS" if close > ema200 else "SHORT_BIAS"

    return {
        "signal": "NO_TRADE",
        "close": round(close, 4),
        "ema20": round(ema20, 4),
        "ema50": round(ema50, 4),
        "ema200": round(ema200, 4),
        "rsi": round(rsi, 2),
        "atr": round(atr, 4),
        "atr_pct": round(atr_pct, 3),
        "trend": trend,
        "vol_ok": bool(vol_ok),
        "long_break": bool(long_break),
        "short_break": bool(short_break),
    }


# =========================
# MESSAGE FORMAT
# =========================
def format_signal_message(symbol, interval, candle_time, signal_data):
    signal = signal_data["signal"]

    if signal == "LONG":
        return (
            f"🟢 ALERTA LONG - {symbol.replace('-', '')} ({interval})\n\n"
            f"📍 Hora vela: {candle_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Entrada aprox: {signal_data['entry']}\n\n"
            f"📊 Confirmaciones:\n"
            f"- Precio > EMA200\n"
            f"- EMA20 > EMA50\n"
            f"- RSI: {signal_data['rsi']}\n"
            f"- Rompió máximo vela anterior\n"
            f"- ATR%: {signal_data['atr_pct']}\n"
            f"- Volumen OK: {signal_data['vol_ok']}\n\n"
            f"🛡 SL: {signal_data['sl']}\n"
            f"🎯 TP1: {signal_data['tp1']}\n"
            f"🎯 TP2: {signal_data['tp2']}\n\n"
            f"⚠️ Revisar gráfico antes de entrar\n"
            f"💰 Riesgo sugerido: 0.5% - 1% por trade"
        )

    elif signal == "SHORT":
        return (
            f"🔴 ALERTA SHORT - {symbol.replace('-', '')} ({interval})\n\n"
            f"📍 Hora vela: {candle_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Entrada aprox: {signal_data['entry']}\n\n"
            f"📊 Confirmaciones:\n"
            f"- Precio < EMA200\n"
            f"- EMA20 < EMA50\n"
            f"- RSI: {signal_data['rsi']}\n"
            f"- Rompió mínimo vela anterior\n"
            f"- ATR%: {signal_data['atr_pct']}\n"
            f"- Volumen OK: {signal_data['vol_ok']}\n\n"
            f"🛡 SL: {signal_data['sl']}\n"
            f"🎯 TP1: {signal_data['tp1']}\n"
            f"🎯 TP2: {signal_data['tp2']}\n\n"
            f"⚠️ Revisar gráfico antes de entrar\n"
            f"💰 Riesgo sugerido: 0.5% - 1% por trade"
        )

    return (
        f"🔎 {symbol.replace('-', '')} sin señal ahora.\n"
        f"Hora vela: {candle_time.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Estado: {signal_data}"
    )


# =========================
# CHECK LOGIC
# =========================
def process_symbol(symbol, force_check=False, send_no_trade=False):
    global last_candle_time, last_signal_sent

    try:
        df = fetch_bingx_klines(symbol=symbol, interval=INTERVAL, limit=300)
        df = add_indicators(df)

        # Última vela cerrada = última del dataframe (BingX ya suele traer cerradas en este endpoint)
        candle = df.iloc[-1]
        candle_time = candle["time"]

        # Inicialización: no mandar señal vieja al arrancar
        if symbol not in last_candle_time:
            last_candle_time[symbol] = candle_time
            log(f"{symbol} - Inicializado en vela: {candle_time}")
            if force_check:
                signal_data = generate_signal_v2(df)
                msg = format_signal_message(symbol, INTERVAL, candle_time, signal_data)
                send_telegram_message(msg)
            return

        # Si no hay vela nueva y no es force_check, salir
        if (not force_check) and candle_time == last_candle_time[symbol]:
            return

        # Actualizar vela vista
        last_candle_time[symbol] = candle_time
        log(f"{symbol} - Nueva vela cerrada: {candle_time}")

        signal_data = generate_signal_v2(df)

        if signal_data["signal"] in ["LONG", "SHORT"]:
            signal_key = f"{symbol}-{candle_time}-{signal_data['signal']}"

            if last_signal_sent.get(symbol) != signal_key or force_check:
                msg = format_signal_message(symbol, INTERVAL, candle_time, signal_data)
                send_telegram_message(msg)
                last_signal_sent[symbol] = signal_key
                log(f"{symbol} - Señal enviada: {signal_data['signal']}")
            else:
                log(f"{symbol} - Señal duplicada evitada.")
        else:
            log(f"{symbol} - Sin señal. Estado: {signal_data}")
            if force_check and send_no_trade:
                msg = format_signal_message(symbol, INTERVAL, candle_time, signal_data)
                send_telegram_message(msg)

    except Exception as e:
        log(f"{symbol} - ERROR: {e}")


# =========================
# TELEGRAM COMMANDS
# =========================
def handle_telegram_commands():
    updates = get_telegram_updates()
    if not updates:
        return

    for upd in updates:
        msg = upd.get("message", {})
        text = msg.get("text", "")
        chat_id = str(msg.get("chat", {}).get("id", ""))

        # Seguridad: responder solo a tu chat configurado
        if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
            continue

        if text == "/start":
            send_telegram_message(
                "🤖 Bot activo en Railway.\n\n"
                "Comandos:\n"
                "/check → revisar ahora\n"
                "/status → estado del bot"
            )

        elif text == "/status":
            send_telegram_message(
                f"✅ Bot activo.\n"
                f"Símbolos: {', '.join([s.replace('-', '') for s in SYMBOLS])}\n"
                f"Timeframe: {INTERVAL}\n"
                f"Estrategia: V2 (EMA200 + EMA20/50 + RSI + Breakout + ATR + Volumen)"
            )

        elif text == "/check":
            send_telegram_message("🔎 Revisando mercado ahora...")
            for symbol in SYMBOLS:
                process_symbol(symbol, force_check=True, send_no_trade=True)


# =========================
# MAIN LOOP
# =========================
def main():
    log(f"Bot iniciado para {', '.join(SYMBOLS)} en {INTERVAL} (BingX data, V2)")

    # Mensaje de inicio
    send_telegram_message(
        f"🚀 Bot iniciado en Railway\n"
        f"Símbolos: {', '.join([s.replace('-', '') for s in SYMBOLS])}\n"
        f"Timeframe: {INTERVAL}\n"
        f"Estrategia: V2"
    )

    # Inicializar sin disparar señales viejas
    for symbol in SYMBOLS:
        process_symbol(symbol, force_check=False)

    while True:
        try:
            handle_telegram_commands()

            for symbol in SYMBOLS:
                process_symbol(symbol, force_check=False)

        except Exception as e:
            log(f"Error en loop principal: {e}")

        time.sleep(CHECK_EVERY_SECONDS)


if __name__ == "__main__":
    main()
