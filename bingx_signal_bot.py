import os
import time
import csv
from datetime import datetime, timezone

import requests
import pandas as pd

# ==========================================
# CONFIG DESDE VARIABLES DE ENTORNO (Railway)
# ==========================================
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
INTERVAL = os.getenv("INTERVAL", "15m")
LIMIT = int(os.getenv("LIMIT", "300"))

EMA_LEN = int(os.getenv("EMA_LEN", "200"))
RSI_LEN = int(os.getenv("RSI_LEN", "14"))
ATR_LEN = int(os.getenv("ATR_LEN", "14"))

RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "45"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "55"))
RSI_MID = float(os.getenv("RSI_MID", "50"))
PULLBACK_BARS = int(os.getenv("PULLBACK_BARS", "12"))

SL_ATR_MULT = float(os.getenv("SL_ATR_MULT", "1.5"))
TP_ATR_MULT = float(os.getenv("TP_ATR_MULT", "3.0"))

USE_MIN_ATR_FILTER = os.getenv("USE_MIN_ATR_FILTER", "true").lower() == "true"
MIN_ATR_PERCENT = float(os.getenv("MIN_ATR_PERCENT", "0.25"))

CHECK_EVERY_SECONDS = int(os.getenv("CHECK_EVERY_SECONDS", "30"))

COMMISSION_PER_SIDE = float(os.getenv("COMMISSION_PER_SIDE", "0.0004"))
ESTIMATED_SLIPPAGE_PCT = float(os.getenv("ESTIMATED_SLIPPAGE_PCT", "0.0003"))
ESTIMATED_SPREAD_PCT = float(os.getenv("ESTIMATED_SPREAD_PCT", "0.0002"))

MIN_SIGNAL_GAP_BARS = int(os.getenv("MIN_SIGNAL_GAP_BARS", "4"))

# ==========================================
# TELEGRAM
# ==========================================
USE_TELEGRAM = os.getenv("USE_TELEGRAM", "true").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SEND_STARTUP_MESSAGE = os.getenv("SEND_STARTUP_MESSAGE", "true").lower() == "true"
SEND_HOURLY_SUMMARY = os.getenv("SEND_HOURLY_SUMMARY", "false").lower() == "true"

# ==========================================
# ARCHIVOS
# ==========================================
CSV_FILE = os.getenv("CSV_FILE", "signals_log_pro.csv")

# ==========================================
# ESTADO INTERNO
# ==========================================
last_candle_time_by_symbol = {}
last_signal_sent_by_symbol = {}
last_signal_bar_time_by_symbol = {}
last_state_by_symbol = {}
last_summary_hour = None

# ==========================================
# UTILIDADES
# ==========================================
def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] {msg}", flush=True)

def send_telegram_message(message):
    if not USE_TELEGRAM:
        return

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram desactivado: falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID.")
        return

    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message
        }

        response = requests.post(url, data=payload, timeout=10)

        if response.status_code == 200:
            log("Telegram enviado correctamente.")
        else:
            log(f"Error Telegram [{response.status_code}]: {response.text}")

    except Exception as e:
        log(f"Error enviando Telegram: {e}")

def ensure_csv_exists():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp_utc",
                "symbol",
                "interval",
                "signal",
                "entry",
                "sl",
                "tp",
                "ema200",
                "rsi",
                "atr",
                "atr_pct",
                "rr_net",
                "cost_pct_estimated",
                "reason"
            ])

def save_signal_to_csv(symbol, interval, result):
    ensure_csv_exists()

    with open(CSV_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            interval,
            result.get("signal", ""),
            result.get("entry", ""),
            result.get("sl", ""),
            result.get("tp", ""),
            result.get("ema200", ""),
            result.get("rsi", ""),
            result.get("atr", ""),
            result.get("atr_pct", ""),
            result.get("rr_net", ""),
            result.get("cost_pct_estimated", ""),
            result.get("reason", "")
        ])

def format_num(x, decimals=2):
    try:
        return round(float(x), decimals)
    except Exception:
        return x

def estimate_total_cost_pct():
    return (COMMISSION_PER_SIDE * 2) + (ESTIMATED_SLIPPAGE_PCT * 2) + ESTIMATED_SPREAD_PCT

# ==========================================
# DATOS DE MERCADO
# ==========================================
def get_binance_klines(symbol="BTCUSDT", interval="15m", limit=300):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

    return df

# ==========================================
# INDICADORES
# ==========================================
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, 1e-10)
    return 100 - (100 / (1 + rs))

def atr(df, length=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def crossover(prev_val, curr_val, level):
    return prev_val <= level and curr_val > level

def crossunder(prev_val, curr_val, level):
    return prev_val >= level and curr_val < level

# ==========================================
# ESTRATEGIA PRO
# ==========================================
def generate_signal(df):
    df = df.copy()

    df["ema200"] = ema(df["close"], EMA_LEN)
    df["rsi"] = rsi(df["close"], RSI_LEN)
    df["atr"] = atr(df, ATR_LEN)
    df["atr_pct"] = (df["atr"] / df["close"]) * 100

    if len(df) < EMA_LEN + 5:
        return {"signal": "NO_DATA"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["close"])
    open_ = float(last["open"])
    ema200_val = float(last["ema200"])
    rsi_now = float(last["rsi"])
    rsi_prev = float(prev["rsi"])
    atr_now = float(last["atr"])
    atr_pct = float(last["atr_pct"])

    trend_long = close > ema200_val
    trend_short = close < ema200_val

    recent = df.iloc[-PULLBACK_BARS:]
    recent_oversold = float(recent["rsi"].min()) < RSI_OVERSOLD
    recent_overbought = float(recent["rsi"].max()) > RSI_OVERBOUGHT

    bullish_bar = close > open_
    bearish_bar = close < open_

    rsi_cross_up = crossover(rsi_prev, rsi_now, RSI_MID)
    rsi_cross_down = crossunder(rsi_prev, rsi_now, RSI_MID)

    vol_ok = (not USE_MIN_ATR_FILTER) or (atr_pct >= MIN_ATR_PERCENT)

    long_signal = (
        trend_long and
        recent_oversold and
        rsi_cross_up and
        bullish_bar and
        vol_ok
    )

    short_signal = (
        trend_short and
        recent_overbought and
        rsi_cross_down and
        bearish_bar and
        vol_ok
    )

    total_cost_pct = estimate_total_cost_pct()

    if long_signal:
        entry = close
        sl = entry - (atr_now * SL_ATR_MULT)
        tp = entry + (atr_now * TP_ATR_MULT)

        gross_risk_pct = (entry - sl) / entry
        gross_reward_pct = (tp - entry) / entry

        net_risk_pct = gross_risk_pct + total_cost_pct
        net_reward_pct = max(gross_reward_pct - total_cost_pct, 0)

        rr_net = net_reward_pct / net_risk_pct if net_risk_pct > 0 else 0

        return {
            "signal": "LONG",
            "entry": format_num(entry, 2),
            "sl": format_num(sl, 2),
            "tp": format_num(tp, 2),
            "ema200": format_num(ema200_val, 2),
            "rsi": format_num(rsi_now, 2),
            "atr": format_num(atr_now, 2),
            "atr_pct": format_num(atr_pct, 3),
            "rr_net": format_num(rr_net, 2),
            "cost_pct_estimated": format_num(total_cost_pct * 100, 3),
            "reason": "Tendencia alcista + pullback RSI + cruce RSI 50 arriba + vela alcista"
        }

    if short_signal:
        entry = close
        sl = entry + (atr_now * SL_ATR_MULT)
        tp = entry - (atr_now * TP_ATR_MULT)

        gross_risk_pct = (sl - entry) / entry
        gross_reward_pct = (entry - tp) / entry

        net_risk_pct = gross_risk_pct + total_cost_pct
        net_reward_pct = max(gross_reward_pct - total_cost_pct, 0)

        rr_net = net_reward_pct / net_risk_pct if net_risk_pct > 0 else 0

        return {
            "signal": "SHORT",
            "entry": format_num(entry, 2),
            "sl": format_num(sl, 2),
            "tp": format_num(tp, 2),
            "ema200": format_num(ema200_val, 2),
            "rsi": format_num(rsi_now, 2),
            "atr": format_num(atr_now, 2),
            "atr_pct": format_num(atr_pct, 3),
            "rr_net": format_num(rr_net, 2),
            "cost_pct_estimated": format_num(total_cost_pct * 100, 3),
            "reason": "Tendencia bajista + pullback RSI + cruce RSI 50 abajo + vela bajista"
        }

    return {
        "signal": "NO_TRADE",
        "close": format_num(close, 2),
        "ema200": format_num(ema200_val, 2),
        "rsi": format_num(rsi_now, 2),
        "atr": format_num(atr_now, 2),
        "atr_pct": format_num(atr_pct, 3),
        "trend": "LONG_BIAS" if trend_long else "SHORT_BIAS",
        "vol_ok": vol_ok,
        "recent_oversold": recent_oversold,
        "recent_overbought": recent_overbought,
        "rsi_cross_up": rsi_cross_up,
        "rsi_cross_down": rsi_cross_down,
        "bullish_bar": bullish_bar,
        "bearish_bar": bearish_bar
    }

# ==========================================
# MENSAJES TELEGRAM
# ==========================================
def build_signal_message(symbol, interval, result):
    emoji = "🟢" if result["signal"] == "LONG" else "🔴"

    return (
        f"{emoji} {symbol} - {interval}\n"
        f"SEÑAL: {result['signal']}\n\n"
        f"Motivo:\n{result['reason']}\n\n"
        f"Entrada: {result['entry']}\n"
        f"SL: {result['sl']}\n"
        f"TP: {result['tp']}\n"
        f"EMA200: {result['ema200']}\n"
        f"RSI: {result['rsi']}\n"
        f"ATR: {result['atr']} ({result['atr_pct']}%)\n"
        f"R:R Neto Est.: {result['rr_net']}\n"
        f"Coste Total Est.: {result['cost_pct_estimated']}%"
    )

def build_hourly_summary():
    lines = ["📊 Resumen horario del bot\n"]

    for symbol in SYMBOLS:
        state = last_state_by_symbol.get(symbol)

        if not state:
            lines.append(f"{symbol}: sin datos")
            continue

        if state["signal"] == "NO_TRADE":
            lines.append(
                f"{symbol}: {state['trend']} | RSI {state['rsi']} | ATR% {state['atr_pct']} | sin setup"
            )
        else:
            lines.append(
                f"{symbol}: ÚLTIMA SEÑAL {state['signal']} | Entrada {state.get('entry')}"
            )

    return "\n".join(lines)

# ==========================================
# COOL DOWN ENTRE SEÑALES
# ==========================================
def can_send_new_signal(symbol, candle_time):
    if symbol not in last_signal_bar_time_by_symbol:
        return True

    prev_time = last_signal_bar_time_by_symbol[symbol]
    if prev_time is None:
        return True

    interval_minutes = {
        "15m": 15,
        "1h": 60,
        "4h": 240
    }.get(INTERVAL, 15)

    min_gap_seconds = MIN_SIGNAL_GAP_BARS * interval_minutes * 60
    diff_seconds = (candle_time - prev_time).total_seconds()

    return diff_seconds >= min_gap_seconds

# ==========================================
# PROCESO POR SÍMBOLO
# ==========================================
def process_symbol(symbol):
    global last_candle_time_by_symbol, last_signal_sent_by_symbol, last_signal_bar_time_by_symbol, last_state_by_symbol

    df = get_binance_klines(symbol, INTERVAL, LIMIT)

    # Solo velas cerradas
    df_closed = df.iloc[:-1].copy()

    if len(df_closed) < LIMIT - 1:
        log(f"{symbol} - No hay suficientes velas cerradas todavía.")
        return

    candle_time = df_closed.iloc[-1]["close_time"]

    if symbol not in last_candle_time_by_symbol:
        last_candle_time_by_symbol[symbol] = None

    if symbol not in last_signal_sent_by_symbol:
        last_signal_sent_by_symbol[symbol] = None

    if symbol not in last_signal_bar_time_by_symbol:
        last_signal_bar_time_by_symbol[symbol] = None

    if last_candle_time_by_symbol[symbol] is None or candle_time != last_candle_time_by_symbol[symbol]:
        last_candle_time_by_symbol[symbol] = candle_time

        result = generate_signal(df_closed)
        last_state_by_symbol[symbol] = result

        log(f"{symbol} - Nueva vela cerrada: {candle_time}")

        if result["signal"] in ["LONG", "SHORT"]:
            signal_key = f"{candle_time}_{result['signal']}"

            if signal_key != last_signal_sent_by_symbol[symbol]:
                if can_send_new_signal(symbol, candle_time):
                    last_signal_sent_by_symbol[symbol] = signal_key
                    last_signal_bar_time_by_symbol[symbol] = candle_time

                    log("===================================")
                    log(f"{symbol} - SEÑAL: {result['signal']}")
                    log(f"Motivo: {result['reason']}")
                    log(f"Entrada: {result['entry']}")
                    log(f"SL: {result['sl']}")
                    log(f"TP: {result['tp']}")
                    log(f"EMA200: {result['ema200']}")
                    log(f"RSI: {result['rsi']}")
                    log(f"ATR: {result['atr']} ({result['atr_pct']}%)")
                    log(f"R:R neto estimado: {result['rr_net']}")
                    log(f"Coste total estimado ida/vuelta: {result['cost_pct_estimated']}%")
                    log("===================================")

                    save_signal_to_csv(symbol, INTERVAL, result)
                    send_telegram_message(build_signal_message(symbol, INTERVAL, result))
                else:
                    log(f"{symbol} - Señal válida detectada, pero en cooldown anti-spam.")
            else:
                log(f"{symbol} - Señal duplicada ignorada.")
        else:
            log(
                f"{symbol} - NO TRADE | "
                f"Trend={result['trend']} | "
                f"RSI={result['rsi']} | "
                f"OversoldRec={result['recent_oversold']} | "
                f"OverboughtRec={result['recent_overbought']} | "
                f"CrossUp={result['rsi_cross_up']} | "
                f"CrossDown={result['rsi_cross_down']} | "
                f"BullBar={result['bullish_bar']} | "
                f"BearBar={result['bearish_bar']} | "
                f"ATR%={result['atr_pct']} | "
                f"VolOK={result['vol_ok']}"
            )

# ==========================================
# RESUMEN HORARIO
# ==========================================
def maybe_send_hourly_summary():
    global last_summary_hour

    if not SEND_HOURLY_SUMMARY or not USE_TELEGRAM:
        return

    now = datetime.now(timezone.utc)
    current_hour = now.strftime("%Y-%m-%d %H")

    if last_summary_hour is None:
        last_summary_hour = current_hour
        return

    if current_hour != last_summary_hour:
        last_summary_hour = current_hour
        send_telegram_message(build_hourly_summary())

# ==========================================
# MAIN
# ==========================================
def main():
    log(f"Bot PRO Railway iniciado para {', '.join(SYMBOLS)} en {INTERVAL}")
    ensure_csv_exists()

    if SEND_STARTUP_MESSAGE:
        send_telegram_message(
            f"🚀 Bot PRO Railway iniciado | Pares: {', '.join(SYMBOLS)} | TF: {INTERVAL}"
        )

    while True:
        try:
            for symbol in SYMBOLS:
                try:
                    process_symbol(symbol)
                except Exception as e:
                    log(f"{symbol} - ERROR: {e}")

            maybe_send_hourly_summary()
            time.sleep(CHECK_EVERY_SECONDS)

        except KeyboardInterrupt:
            log("Bot detenido manualmente.")
            if USE_TELEGRAM:
                send_telegram_message("⛔ Bot PRO Railway detenido manualmente.")
            break

        except Exception as e:
            error_msg = f"ERROR general: {e}"
            log(error_msg)

            if USE_TELEGRAM:
                send_telegram_message(f"⚠️ {error_msg}")

            time.sleep(15)

if __name__ == "__main__":
    main()