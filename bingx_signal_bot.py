import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# =========================
# CONFIG
# =========================
SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]
INTERVAL = "15m"         # 15m, 1h, 4h
LIMIT = 300
SLEEP_SECONDS = 20

USE_TELEGRAM = True
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = str(os.getenv("TELEGRAM_CHAT_ID", "")).strip()

BINGX_BASE_URL = "https://open-api.bingx.com"

# Heartbeat cada X horas
HEARTBEAT_HOURS = 4

# Resumen diario a esta hora UTC
DAILY_SUMMARY_HOUR_UTC = 0

# =========================
# ESTADO GLOBAL
# =========================
last_update_id = None
last_heartbeat_time = None
last_daily_summary_date = None

# para evitar señales repetidas
last_candle_times = {}
last_signal_sent = {}

# stats del día
daily_stats = {
    "BTC-USDT": {"LONG": 0, "SHORT": 0},
    "ETH-USDT": {"LONG": 0, "SHORT": 0},
    "SOL-USDT": {"LONG": 0, "SHORT": 0},
}

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
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        r = requests.post(url, data=payload, timeout=15)
        r.raise_for_status()
        log("Telegram enviado correctamente.")
        return True
    except Exception as e:
        log(f"Error enviando Telegram: {e}")
        return False

# =========================
# TELEGRAM COMMANDS (polling)
# =========================
def get_telegram_updates():
    global last_update_id

    if not TELEGRAM_BOT_TOKEN:
        return []

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {
        "timeout": 1
    }

    if last_update_id is not None:
        params["offset"] = last_update_id + 1

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        if not data.get("ok"):
            return []

        updates = data.get("result", [])
        if updates:
            last_update_id = updates[-1]["update_id"]

        return updates
    except Exception as e:
        log(f"Error leyendo comandos Telegram: {e}")
        return []

def process_telegram_commands():
    updates = get_telegram_updates()

    for upd in updates:
        try:
            message = upd.get("message", {})
            text = str(message.get("text", "")).strip()
            chat_id = str(message.get("chat", {}).get("id", "")).strip()

            # Seguridad: solo responde a tu chat
            if chat_id != TELEGRAM_CHAT_ID:
                log(f"Ignorando comando de chat no autorizado: {chat_id}")
                continue

            if text == "/start":
                send_telegram(
                    "🤖 Bot activo.\n\n"
                    "Comandos disponibles:\n"
                    "/status - estado del bot\n"
                    "/check - análisis inmediato\n"
                    "/help - ayuda"
                )

            elif text == "/help":
                send_telegram(
                    "📘 Comandos:\n"
                    "/status - estado del bot\n"
                    "/check - fuerza análisis ahora\n"
                    "/help - ver comandos"
                )

            elif text == "/status":
                send_telegram(build_status_message())

            elif text == "/check":
                send_telegram("🔎 Ejecutando análisis manual...")
                run_analysis(force_manual=True)

        except Exception as e:
            log(f"Error procesando comando Telegram: {e}")

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

    rows = None
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            rows = data["data"]
        elif "data" in data and isinstance(data["data"], dict):
            if "klines" in data["data"]:
                rows = data["data"]["klines"]
            elif "list" in data["data"]:
                rows = data["data"]["list"]
        elif "result" in data and isinstance(data["result"], list):
            rows = data["result"]

    if not rows:
        raise Exception(f"Respuesta inesperada de BingX: {data}")

    parsed = []
    for row in rows:
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

    # RSI 14
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

    if close > ema200 and prev_rsi < 50 and rsi > 50 and vol_ok:
        signal = "LONG"
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
# STATUS MESSAGE
# =========================
def build_status_message():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    msg = [
        "📡 ESTADO DEL BOT",
        f"Hora: {now}",
        f"Activos: {', '.join([s.replace('-', '') for s in SYMBOLS])}",
        f"Timeframe: {INTERVAL}",
        "",
        "📊 Señales del día:"
    ]

    for s in SYMBOLS:
        longs = daily_stats[s]["LONG"]
        shorts = daily_stats[s]["SHORT"]
        msg.append(f"{s.replace('-', '')}: LONG={longs} | SHORT={shorts}")

    return "\n".join(msg)

# =========================
# DAILY SUMMARY
# =========================
def maybe_send_daily_summary():
    global last_daily_summary_date

    now = datetime.now(timezone.utc)
    today = now.date()

    if now.hour == DAILY_SUMMARY_HOUR_UTC:
        if last_daily_summary_date != today:
            msg = ["📊 RESUMEN DIARIO"]
            for s in SYMBOLS:
                longs = daily_stats[s]["LONG"]
                shorts = daily_stats[s]["SHORT"]
                msg.append(f"{s.replace('-', '')}: LONG={longs} | SHORT={shorts}")

            send_telegram("\n".join(msg))
            last_daily_summary_date = today

            # reset stats después de enviar
            for s in SYMBOLS:
                daily_stats[s]["LONG"] = 0
                daily_stats[s]["SHORT"] = 0

# =========================
# HEARTBEAT
# =========================
def maybe_send_heartbeat():
    global last_heartbeat_time

    now = datetime.now(timezone.utc)

    if last_heartbeat_time is None:
        last_heartbeat_time = now
        return

    if now - last_heartbeat_time >= timedelta(hours=HEARTBEAT_HOURS):
        send_telegram(
            f"💓 Bot sigue activo\n"
            f"Hora UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Activos: {', '.join([s.replace('-', '') for s in SYMBOLS])}\n"
            f"Timeframe: {INTERVAL}"
        )
        last_heartbeat_time = now

# =========================
# ANALYSIS LOOP
# =========================
def run_analysis(force_manual=False):
    global last_candle_times, last_signal_sent

    for symbol in SYMBOLS:
        try:
            df = get_klines(symbol, INTERVAL, LIMIT)

            if len(df) < 50:
                log(f"{symbol} - Pocas velas recibidas: {len(df)}")

            df = compute_indicators(df)

            current_candle_time = df.iloc[-1]["open_time"]

            if symbol not in last_candle_times:
                last_candle_times[symbol] = current_candle_time
                last_signal_sent[symbol] = None
                log(f"{symbol} - Inicializado en vela: {current_candle_time}")

                if force_manual:
                    state = check_signal(df)
                    send_telegram(f"🔎 {symbol.replace('-', '')} análisis manual:\n{state}")

                continue

            # Si es manual, analiza aunque no haya vela nueva
            if force_manual:
                state = check_signal(df)

                if state["signal"] in ["LONG", "SHORT"]:
                    msg = format_signal(symbol, state)
                    if msg:
                        send_telegram(f"🔎 CHECK MANUAL\n{msg}")
                else:
                    send_telegram(f"🔎 {symbol.replace('-', '')} sin señal ahora.\nEstado: {state}")

                continue

            # Modo normal: solo cuando cierra nueva vela
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

                            # stats
                            daily_stats[symbol][state["signal"]] += 1
                    else:
                        log(f"{symbol} - Señal repetida evitada.")
                else:
                    log(f"{symbol} - Sin señal. Estado: {state}")

        except Exception as e:
            log(f"{symbol} - ERROR: {e}")

# =========================
# MAIN
# =========================
def main():
    log(f"Bot iniciado para {', '.join(SYMBOLS)} en {INTERVAL} (BingX data)")
    send_telegram(
        f"🤖 Bot ONLINE en Railway\n"
        f"Activos: {', '.join([s.replace('-', '') for s in SYMBOLS])}\n"
        f"Timeframe: {INTERVAL}\n"
        f"Fuente: BingX Futures\n"
        f"Comandos: /status /check /help"
    )

    while True:
        try:
            process_telegram_commands()
            run_analysis(force_manual=False)
            maybe_send_heartbeat()
            maybe_send_daily_summary()
        except Exception as e:
            log(f"ERROR GENERAL LOOP: {e}")

        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
