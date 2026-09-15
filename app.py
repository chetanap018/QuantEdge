"""
app.py
-------
Backtest Studio - Beautiful Web UI for the backtesting framework.

Replaces the need to edit backtest_config.py for every run:

    python3 app.py
    # then open http://127.0.0.1:5001

The UI reads its defaults from backtest_config.py (so it stays the single
source of truth for default values), but every parameter can be changed
live in the browser instead of editing the file.

The engine, data fetcher, strategies, and broker charges modules are untouched.
"""

from __future__ import annotations

import inspect
import logging
import os
import re
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import backtest_config
import config
import strategies.base as strategies_base
from ai_strategy_writer import (
    GENERATED_DIR,
    MAX_GENERATION_ATTEMPTS,
    REPAIR_DELAY_SECONDS,
    _class_name_from_slug,
    delete_strategy,
    generate_and_validate_strategy,
    generated_slugs,
    infer_params_schema,
    load_generated_strategies,
    persist_strategy,
)
from backtest_engine import BacktestEngine
from data_fetcher import DataFetcher
from strategies import (
    GENERATED_STRATEGIES,
    HeroOrbStrategy,
    MACDCrossoverStrategy,
    OpeningCandleStrategy,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
    SuzlonStrategy,
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================
# Cache-busting for static assets -- appends file mtime as ?v=
# so browsers always fetch fresh CSS/JS after code changes.
# ============================================================
def _asset_version() -> str:
    static_dir = os.path.join(app.root_path, "static")
    stamps = []
    for name in ("style.css", "echarts.min.js"):
        path = os.path.join(static_dir, name)
        try:
            stamps.append(int(os.path.getmtime(path)))
        except OSError:
            stamps.append(0)
    return "v" + ".".join(str(s) for s in stamps)

@app.context_processor
def _inject_cache_version():
    return {"cache_version": _asset_version}


# ============================================================
# Strategy registry + parameter schemas (drives the UI forms)
# ============================================================

# Hard cap on how long an AI strategy description may be. Exposed to the
# template so the description box enforces the exact same limit client-side
# (and warns before the user hits a 400 from the API).
MAX_STRATEGY_DESCRIPTION_CHARS = 4000

BUILTIN_STRATEGY_KEYS = frozenset({
    "sma_crossover",
    "rsi_mean_reversion",
    "macd_crossover",
    "opening_candle",
    "suzlon",
    "hero_orb",
})

STRATEGY_REGISTRY: dict[str, type] = {
    "sma_crossover": SMACrossoverStrategy,
    "rsi_mean_reversion": RSIMeanReversionStrategy,
    "macd_crossover": MACDCrossoverStrategy,
    "opening_candle": OpeningCandleStrategy,
    "suzlon": SuzlonStrategy,
    "hero_orb": HeroOrbStrategy,
}

STRATEGY_SCHEMAS = {
    "sma_crossover": {
        "fast_period": {"label": "Fast SMA period", "type": "int", "default": 20, "min": 2,"max": 200},
        "slow_period": {"label": "Slow SMA period", "type": "int", "default": 50,"min": 5,"max":  500},
    },
    "rsi_mean_reversion": {
        "rsi_period": {"label": "RSI period", "type": "int", "default": 14,"min": 2,"max": 100},
        "oversold": {"label": "Oversold level", "type": "float", "default": 30.0,"min":  1.0,"max": 49.0},
        "overbought": {"label": "Overbought level", "type": "float", "default": 70.0,"min": 51.0,"max": 99.0},
    },
    "macd_crossover": {
        "fast_period": {"label": "Fast EMA period", "type": "int", "default": 12,"min": 2,"max": 100},
        "slow_period": {"label": "Slow EMA period", "type": "int", "default": 26,"min": 5,"max": 200},
        "signal_period": {"label": "Signal EMA period", "type": "int", "default": 9,"min": 2,"max": 100},
    },
    "opening_candle": {
        "target_points": {"label": "Target (points)", "type": "float", "default": 30.0,"min": 1.0,"max": 1000.0},
        "stop_loss_points": {"label": "Stop loss (points)", "type": "float", "default": 30.0,"min": 1.0,"max": 1000.0},
    },
    "hero_orb": {
        "orb_bars": {"label": "Opening range bars", "type": "int", "default": 4,"min": 1,"max": 8},
        "trend_ema": {"label": "Trend EMA period", "type": "int", "default": 30,"min": 5,"max": 200},
        "atr_period": {"label": "ATR period", "type": "int", "default": 14,"min": 2,"max": 100},
        "stop_atr": {"label": "Disaster stop (x ATR)", "type": "float", "default": 7.0,"min": 2.0,"max": 20.0},
        "min_hold_bars": {"label": "Min hold (bars)", "type": "int", "default": 8,"min": 1,"max": 40},
        "max_entry_time": {"label": "Last entry time", "type": "text", "default": "14:30"},
    },
    "suzlon": {
        "ema_fast": {"label": "Fast EMA", "type": "int", "default": 9,"min": 2,"max": 50},
        "ema_slow": {"label": "Slow EMA", "type": "int", "default": 20,"min": 5,"max": 100},
        "rsi_period": {"label": "RSI period", "type": "int", "default": 14,"min": 2,"max": 100},
        "volume_period": {"label": "Volume MA period", "type": "int", "default": 20,"min": 2,"max": 100},
        "volume_threshold": {"label": "Volume threshold", "type": "float", "default": 1.0,"min": 0.1,"max": 10.0},
        "rsi_long_min": {"label": "RSI long min", "type": "float", "default": 45.0,"min": 1.0,"max": 99.0},
        "rsi_long_max": {"label": "RSI long max", "type": "float", "default": 70.0,"min": 1.0,"max": 99.0},
        "rsi_short_min": {"label": "RSI short min", "type": "float", "default": 30.0,"min": 1.0,"max": 99.0},
        "rsi_short_max": {"label": "RSI short max", "type": "float", "default": 55.0,"min": 1.0,"max": 99.0},
        "pullback_tolerance": {"label": "Pullback tolerance", "type": "float", "default": 0.0025,"min": 0.0001,"max": 0.05},
        "stop_loss_pct": {"label": "Stop loss %", "type": "float", "default": 0.003,"min": 0.0001,"max": 0.1},
        "target_pct": {"label": "Target %", "type": "float", "default": 0.005,"min": 0.0001,"max": 0.2},
        "use_vwap": {"label": "Use VWAP filter", "type": "bool", "default": True},
        "use_rsi": {"label": "Use RSI filter", "type": "bool", "default": True},
        "use_volume": {"label": "Use volume filter", "type": "bool", "default": True},
        "use_time_filter": {"label": "Use time filter", "type": "bool", "default": True},
        "session_start": {"label": "Session start", "type": "text", "default": "09:20"},
        "session_end": {"label": "Session end", "type": "text", "default": "15:00"},
    },
}


# Reload AI-generated strategies accepted in previous sessions
for _gen_key, _gen_cls, _gen_schema in load_generated_strategies():
    STRATEGY_REGISTRY[_gen_key] = _gen_cls
    STRATEGY_SCHEMAS[_gen_key] = _gen_schema
    logger.info("Loaded AI-generated strategy from disk: %s", _gen_key)



TIMEFRAMES: list[str] = [
    "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
    "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY", "ONE_WEEK", "ONE_MONTH",
]

EXCHANGES: list[str] = ["NSE", "BSE", "NFO", "MCX"]
SEGMENTS: list[str] = ["intraday_equity", "delivery_equity", "futures", "options"]
SIZING_METHODS: list[str] = ["fixed_quantity", "fixed_capital_pct", "risk_based"]

CHARGES_COMPONENTS: list[str] = [
    "brokerage", "stt", "exchange_charges", "gst", "sebi_charges", "stamp_duty",
]


# ============================================================
# Helpers
# ============================================================
def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas/timestamp values to JSON-safe ones."""
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return obj.total_seconds()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _build_strategy(strat_key: str, params: dict | None) -> Any:
    """Instantiate a strategy with validated params (falls back to schema defaults)."""
    schema = STRATEGY_SCHEMAS.get(strat_key, {})
    clean: dict[str, Any] = {}
    for pname, pspec in schema.items():
        raw = params.get(pname) if params else None
        if raw is None:
            clean[pname] = pspec.get("default")
        elif pspec["type"] == "bool":
            clean[pname] = bool(raw)
        elif pspec["type"] == "int":
            clean[pname] = int(raw)
        elif pspec["type"] == "float":
            clean[pname] = float(raw)
        else:
            clean[pname] = str(raw)
    return STRATEGY_REGISTRY[strat_key](**clean)


def _pad_or_truncate(dates: list, values: list) -> tuple[list, list]:
    """Align equity-curve dates/values (they always match in practice)."""
    n = min(len(dates), len(values))
    return list(dates[:n]), list(values[:n])


def _compute_drawdown(values: list[float], dates: list[str]) -> dict:
    """Compute the drawdown series (in % of peak equity)."""
    eq = pd.Series(values, dtype="float64")
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max.replace(0, np.nan) * 100
    dd = dd.fillna(0.0)
    max_dd = 0.0
    max_dd_date = None
    if len(dd) > 0:
        idx = int(dd.idxmin())
        max_dd = round(float(abs(dd.iloc[idx])), 2)
        max_dd_date = str(dates[idx]) if idx < len(dates) else None
    return {
        "dates": [_jsonable(d) for d in dates],
        "values": [round(float(v), 2) for v in dd.tolist()],
        "max_dd": max_dd,
        "max_dd_date": max_dd_date,
    }


def _compute_monthly_returns(values: list[float], dates: list[str]) -> dict:
    """Build a years x months heatmap matrix of monthly returns in %."""
    if len(values) == 0 or len(dates) == 0:
        return {"years": [], "months": list(range(1, 13)), "matrix": []}
    idx = pd.to_datetime(dates)
    eq = pd.Series(values, dtype="float64", index=idx)
    ms = eq.resample("ME").last()
    ret = ms.pct_change() * 100
    ret = ret.dropna()
    years = sorted({int(ts.year) for ts in ret.index})
    matrix = []
    seen = set()
    for ts, val in ret.items():
        y, m = int(ts.year), int(ts.month)
        if (y, m) in seen:
            continue
        seen.add((y, m))
        matrix.append([y, m, round(float(val), 2)])
    return {"years": years, "months": list(range(1, 13)), "matrix": matrix}


def _trade_stats(trades: list[dict]) -> dict:
    """Compute advanced per-trade statistics."""
    empty_side = {"count": 0, "pnl": 0.0, "win_rate": 0.0}
    if not trades:
        return {
            "avg_win": 0.0, "avg_loss": 0.0, "max_win": 0.0,
            "max_loss": 0.0, "avg_holding_h": 0.0, "best_trade": 0.0,
            "worst_trade": 0.0, "max_consec_wins": 0, "max_consec_losses": 0,
            "long": dict(empty_side), "short": dict(empty_side),
        }

    net_pnls = [t["net_pnl"] for t in trades]
    wins = [p for p in net_pnls if p > 0]
    losses = [p for p in net_pnls if p <= 0]
    holding_h = [
        (pd.Timestamp(t["exit_time"]) - pd.Timestamp(t["entry_time"])).total_seconds() / 3600
        for t in trades
    ]

    max_w_streak = max_l_streak = cur_w = cur_l = 0
    for p in net_pnls:
        if p > 0:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w_streak = max(max_w_streak, cur_w)
        max_l_streak = max(max_l_streak, cur_l)

    long_trades = [t for t in trades if t["direction"] == "long"]
    short_trades = [t for t in trades if t["direction"] == "short"]

    def _side_stats(ts: list[dict]) -> dict:
        pnls = [t["net_pnl"] for t in ts]
        cnt = len(ts)
        pnl_sum = sum(pnls)
        pnl_cnt = len(pnls)
        win_val = 0.0
        if pnl_cnt:
            wins_cnt = 0
            for pp in pnls:
                if pp > 0:
                    wins_cnt = wins_cnt + 1
            win_val = round(wins_cnt * 100 / pnl_cnt, 2)
        pnl_val = round(pnl_sum, 2)
        return {"count": cnt, "pnl": pnl_val, "win_rate": win_val}
    win_sum = sum(wins)
    loss_sum = sum(losses)
    win_cnt = len(wins)
    loss_cnt = len(losses)
    hold_sum = sum(holding_h)
    hold_cnt = len(holding_h)
    avg_win =  0.0
    avg_loss =  0.0
    hold_avg =  0.0
    if win_cnt:
        avg_win = round(win_sum / win_cnt,  2)
    if loss_cnt:
        avg_loss = round(loss_sum / loss_cnt,  2)
    if hold_cnt:
        hold_avg = round(hold_sum / hold_cnt,  2)
    max_win = max(wins, default= 0.0)
    max_loss = min(losses, default= 0.0)
    best = max(net_pnls, default=  0.0)
    worst = min(net_pnls, default=  0.0)
    long_stats = _side_stats(long_trades)
    short_stats = _side_stats(short_trades)
    return {
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win": max_win,
        "max_loss": max_loss,
        "avg_holding_h": hold_avg,
        "best_trade": best,
        "worst_trade": worst,
        "max_consec_wins": max_w_streak,
        "max_consec_losses": max_l_streak,
        "long": long_stats,
        "short": short_stats,
    }
def _charges_totals(trades: list[dict]) -> dict:
    """Sum each charge component across all trades."""
    comps = ["brokerage", "stt", "exchange_charges", "gst", "sebi_charges", "stamp_duty"]
    totals = {c: 0.0 for c in comps}
    for t in trades:
        bd = t.get("charges_breakdown", {})
        for c in comps:
            val = bd.get(c)
            if val is None:
                val = 0.0
            totals[c] = totals[c] + float(val)
    grand = sum(totals.values())
    totals["total_charges"] = grand
    out = {}
    for c in comps:
        out[c] = round(totals[c], 2)
    out["total_charges"] = round(grand, 2)
    return out


def _serialize_trades(trades: list) -> list[dict]:
    """Convert Trade dataclasses to JSON-safe plain dicts."""
    out = []
    for t in trades:
        d = asdict(t)
        d["entry_time"] = _jsonable(d["entry_time"])
        d["exit_time"] = _jsonable(d["exit_time"])
        bd = d.get("charges_breakdown", {})
        clean_bd = {}
        for k, v in bd.items():
            if v is None:
                v = 0.0
            clean_bd[k] = float(v)
        d["charges_breakdown"] = clean_bd
        d["entry_price"] = float(d["entry_price"])
        d["exit_price"] = float(d["exit_price"])
        d["gross_pnl"] = float(d["gross_pnl"])
        d["total_charges"] = float(d["total_charges"])
        d["net_pnl"] = float(d["net_pnl"])
        d["quantity"] = int(d["quantity"])
        out.append(d)
    return out
def _run_one(
    strat_key: str,
    params: dict,
    data: pd.DataFrame,
    capital: float,
    sizing: str,
    segment: str,
    sizing_params: dict,
) -> dict:
    """Run a single strategy and enrich with advanced analytics."""
    strategy = _build_strategy(strat_key, params)

    # The UI's sizing dropdown sends sizing_params; pass them straight into
    # the engine (key names normalized) instead of mutating global config.
    sp = dict(sizing_params or {})
    engine_sizing = {
        "quantity": sp.get("quantity", backtest_config.FIXED_QUANTITY),
        "capital_pct": sp.get("pct", sp.get("capital_pct", backtest_config.CAPITAL_PCT)),
        "risk_pct": sp.get("risk_pct", backtest_config.RISK_PCT),
        "risk_per_unit": sp.get("risk_per_unit", backtest_config.RISK_PER_UNIT),
    }

    engine = BacktestEngine(
        strategy=strategy,
        data=data,
        initial_capital=capital,
        position_sizing=sizing,
        segment=segment,
        sizing_params=engine_sizing,
    )
    results = engine.run()

    # Regenerate signals so equity-curve dates align with the engine's rows
    df_copy = data.copy()
    signals_df = strategy.generate_signals(df_copy)
    equity_dates = []
    if "datetime" in signals_df.columns:
        equity_dates = [_jsonable(d) for d in signals_df["datetime"].tolist()]
    equity_values = [float(v) for v in results["equity_curve"]]
    equity_dates, equity_values = _pad_or_truncate(equity_dates, equity_values)

    trades = _serialize_trades(results["trades"])
    metrics = {}
    skip = {"trades", "equity_curve", "trade_log"}
    for k, v in results.items():
        if k in skip:
            continue
        metrics[k] = v
    metrics = _jsonable(metrics)

    markers = []
    for t in trades:
        markers.append(
            {"time": t["entry_time"], "price": t["entry_price"], "type": "entry", "direction": t["direction"], "pnl": t["net_pnl"]}
        )
        markers.append(
            {"time": t["exit_time"], "price": t["exit_price"], "type": "exit", "direction": t["direction"], "pnl": t["net_pnl"]}
        )

    pnl_series = [{"x": i +  1, "y": float(t["net_pnl"])} for i, t in enumerate(trades)]

    dd = _compute_drawdown(equity_values, equity_dates)
    mr = _compute_monthly_returns(equity_values, equity_dates)
    tstats = _trade_stats(trades)
    ctotals = _charges_totals(trades)
    pg = strategy.get_params()
    params_json = _jsonable(pg)

    return {
        "key": strat_key,
        "name": strategy.name,
        "params": params_json,
        "metrics": metrics,
        "equity": {"dates": equity_dates, "values": equity_values},
        "drawdown": dd,
        "monthly_returns": mr,
        "trades": trades,
        "trade_stats": tstats,
        "charges_totals": ctotals,
        "pnl_series": pnl_series,
        "markers": markers,
        "trade_count": len(trades),
    }
@app.route("/")
def index():
    return render_template(
        "index.html",
        project_name=backtest_config.PROJECT_NAME,
        ai_desc_max_chars=MAX_STRATEGY_DESCRIPTION_CHARS,
        ai_max_attempts=MAX_GENERATION_ATTEMPTS,
        ai_repair_delay=REPAIR_DELAY_SECONDS,
    )


def _generated_keys() -> set:
    """Keys of AI-generated strategies that are deletable (manifest-tracked).

    We do NOT restrict to keys already present in STRATEGY_REGISTRY here,
    because on a fresh restart a generated strategy may live on disk in the
    manifest but hasn't been imported into the in-process registry yet.  The
    delete endpoint should still be allowed to remove it.
    """
    try:
        slugs = set(generated_slugs())
    except Exception as _e:  # noqa: BLE001
        slugs = set()
    return {s for s in slugs if s not in BUILTIN_STRATEGY_KEYS}


@app.route("/api/strategies")
def api_strategies():
    import backtest_config
    fixed_qty = getattr(backtest_config, "FIXED_QUANTITY", 10)
    cap_pct = getattr(backtest_config, "CAPITAL_PCT", 1.0)
    risk_pct = getattr(backtest_config, "RISK_PCT", 0.02)
    risk_unit = getattr(backtest_config, "RISK_PER_UNIT", 10.0)
    defaults = {
        "symbol": backtest_config.SYMBOL,
        "exchange": backtest_config.EXCHANGE,
        "timeframe": backtest_config.TIMEFRAME,
        "start_date": backtest_config.START_DATE,
        "end_date": backtest_config.END_DATE,
        "capital": backtest_config.INITIAL_CAPITAL,
        "sizing": backtest_config.POSITION_SIZING,
        "fixed_quantity": fixed_qty,
        "capital_pct": cap_pct,
        "risk_pct": risk_pct,
        "risk_per_unit": risk_unit,
        "segment": backtest_config.SEGMENT,
        "strategies": backtest_config.STRATEGIES,
    }
    payload = {
        "strategies": {},
        "defaults": defaults,
        "timeframes": TIMEFRAMES,
        "exchanges": EXCHANGES,
        "segments": SEGMENTS,
        "sizing_methods": SIZING_METHODS,
    }
    for key, schema in STRATEGY_SCHEMAS.items():
        payload["strategies"][key] = {
            "name": STRATEGY_REGISTRY[key]().name,
            "params": schema,
            "generated": key in _generated_keys(),
        }
    return jsonify(payload)


@app.route("/api/run", methods=["POST"])
def api_run():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"ok": False, "error": "Invalid JSON body"}), 400
    symbol_raw = body.get("symbol", "RELIANCE")
    symbol = str(symbol_raw).strip()
    exchange_raw = body.get("exchange", "NSE")
    exchange = str(exchange_raw).upper()
    timeframe_raw = body.get("timeframe", "ONE_DAY")
    timeframe = str(timeframe_raw)
    start_date = body.get("start_date")
    end_date = body.get("end_date")
    capital_raw = body.get("capital", 100000.0)
    capital = float(capital_raw)
    sizing_raw = body.get("sizing", "fixed_quantity")
    sizing = str(sizing_raw)
    segment_raw = body.get("segment", "intraday_equity")
    segment = str(segment_raw)
    sizing_params = body.get("sizing_params") or {}
    strategies_sel = body.get("strategies") or []
    allow_synthetic = body.get("allow_synthetic")
    simulated = body.get("simulated")
    if simulated is None:
        simulated = False
    else:
        simulated = bool(simulated)

    fetcher = DataFetcher()
    try:
        data = fetcher.fetch_historical_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            allow_synthetic=allow_synthetic,
        )
    except RuntimeError as exc:
        fetcher.logout()
        return jsonify({"ok": False, "error": str(exc)}), 422
    except Exception as _exc:  # noqa: BLE001
        fetcher.logout()
        return jsonify({"ok": False, "error": f"Data fetch failed: {_exc}"}), 502
    finally:
        fetcher.logout()

    if data.empty:
        return jsonify({"ok": False, "error": "No data returned"}), 400

    data = data.sort_values("datetime").reset_index(drop=True).copy()
    bars_n = len(data)
    _src_raw = str(getattr(data, "attrs", {}).get("data_source", "unknown") or "unknown").lower()
    _SOURCE_LABELS = {
        "angelone": "Angel One SmartAPI",
        "angel": "Angel One SmartAPI",
        "yahoo": "Yahoo Finance",
        "nse": "NSE India",
        "csv": "Local CSV",
        "synthetic": "Synthetic (simulated)",
        "live": "Angel One SmartAPI",
        "cached": "Cached",
    }
    data_source = _src_raw if _src_raw != "live" else "angelone"
    data_source_label = _SOURCE_LABELS.get(data_source, data_source.replace("_", " ").title())
    data_live = data_source in ("angelone", "angel", "yahoo", "nse", "csv", "live")
    data_first = _jsonable(data["datetime"].iloc[0])
    data_last = _jsonable(data["datetime"].iloc[-1])
    logger.info("Backtest data for %s: source=%s bars=%d (%s to %s)",
                symbol, data_source, bars_n, data_first, data_last)
    ohlc = []
    i_ohlc = 0
    while i_ohlc < bars_n:
        row = data.iloc[i_ohlc]
        ohlc.append({
            "t": _jsonable(row["datetime"]),
            "o": float(row["open"]),
            "h": float(row["high"]),
            "l": float(row["low"]),
            "c": float(row["close"]),
            "v": int(row["volume"]),
        })
        i_ohlc = i_ohlc + 1

    results_list = []
    for item in strategies_sel:
        key = str(item.get("name", ""))
        if key not in STRATEGY_REGISTRY:
            continue
        params = item.get("params") or {}
        try:
            one = _run_one(key, params, data, capital, sizing, segment, sizing_params)
            results_list.append(one)
        except Exception as exc:
            logger.exception("Strategy %s failed", key)
            results_list.append({"key": key, "error": str(exc)})

    if not results_list:
        return jsonify({"ok": False, "error": "No valid strategies selected"}), 400

    run_record = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "capital": capital,
        "bars": bars_n,
        "data_source": data_source,
        "data_source_label": data_source_label,
        "data_live": data_live,
        "simulated": simulated,
        "strategies": [
            {"key": r.get("key"), "name": r.get("name"), "error": r.get("error")}
            for r in results_list
        ],
        "metrics": [
            {
                "key": r.get("key"),
                "net_profit": r.get("metrics", {}).get("net_profit"),
                "total_return_pct": r.get("metrics", {}).get("total_return_pct"),
                "max_drawdown_pct": r.get("metrics", {}).get("max_drawdown_pct"),
            }
            for r in results_list
            if "metrics" in r
        ],
    }
    log_run(run_record)

    return jsonify({
        "ok": True,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "capital": capital,
        "bars": bars_n,
        "data_source": data_source,
        "data_source_label": data_source_label,
        "data_live": data_live,
        "data_first": data_first,
        "data_last": data_last,
        "simulated": simulated,
        "ohlc": ohlc,
        "results": results_list,
    })


@app.route("/api/strategies/generate", methods=["POST"])
def api_generate_strategy():
    """
    Body: { "description": "...", "name": "..." }
    On success: registers the strategy immediately and writes to disk.
    On failure: returns the validation/dry-run error so the UI can show it.

    A rejected attempt is automatically sent back to the model with its exact
    failure reason and fixed, up to MAX_GENERATION_ATTEMPTS calls; both
    responses carry `attempts` so the UI can show what happened each round.
    """
    body = request.get_json(silent=True) or {}
    description = str(body.get("description", "")).strip()
    display_name = str(body.get("name", "")).strip()

    if not description:
        return jsonify({"ok": False, "error": "Please describe the strategy's entry/exit logic."}), 400
    if not display_name:
        return jsonify({"ok": False, "error": "Please give the strategy a short name."}), 400
    if len(description) > MAX_STRATEGY_DESCRIPTION_CHARS:
        return jsonify({
            "ok": False,
            "error": (
                "Description is too long "
                f"(max {MAX_STRATEGY_DESCRIPTION_CHARS} characters, "
                f"got {len(description)})."
            ),
        }), 400

    try:
        result = generate_and_validate_strategy(description, display_name)
    except Exception as exc:
        logger.exception("AI strategy generation failed")
        return jsonify({"ok": False, "error": f"Generation failed: {exc}"}), 500

    if not result["ok"]:
        return jsonify(result), 422

    key = result["key"]
    STRATEGY_REGISTRY[key] = result["strategy_class"]
    STRATEGY_SCHEMAS[key] = result["params_schema"]
    logger.info("Registered new AI-generated strategy: %s", key)

    return jsonify({
        "ok": True,
        "key": key,
        "name": result["name"],
        "params": result["params_schema"],
        "code": result["code"],
        "attempts": result.get("attempts", []),
        "attempts_used": result.get("attempts_used", 1),
        "attempts_allowed": result.get("attempts_allowed", MAX_GENERATION_ATTEMPTS),
    })


@app.route("/api/strategies/generated", methods=["GET"])
def api_list_generated_strategies():
    """List AI-generated strategies currently registered."""
    import ai_strategy_writer as _aiw
    out = []
    for entry in _aiw._load_manifest():
        out.append({
            "key": entry["slug"],
            "name": entry["display_name"],
            "description": entry["description"],
            "created_at": entry["created_at"],
        })
    return jsonify({"ok": True, "strategies": out})


@app.route("/api/strategies/<key>", methods=["DELETE"])
def api_delete_strategy(key: str):
    """Delete an AI-generated strategy completely (registry + disk).

    Builtin strategies are protected and can never be deleted this way.
    Deletion is allowed whenever the manifest knows about the slug, even if
    the module hasn't been imported into this process yet.
    """
    slug = str(key or "").strip().lower()
    if not slug:
        return jsonify({"ok": False, "error": "Missing strategy key."}), 400
    if slug in BUILTIN_STRATEGY_KEYS:
        return jsonify({
            "ok": False,
            "error": f"'{slug}' cannot be deleted (builtin strategy).",
        }), 403
    if slug not in _generated_keys():
        return jsonify({
            "ok": False,
            "error": f"'{slug}' cannot be deleted (unknown or not yet generated).",
        }), 403
    try:
        removed = delete_strategy(slug)
    except Exception as exc:
        logger.exception("Strategy deletion failed for %s", slug)
        return jsonify({"ok": False, "error": f"Deletion failed: {exc}"}), 500
    if not removed:
        return jsonify({"ok": False, "error": f"'{slug}' was not found on disk."}), 404
    STRATEGY_REGISTRY.pop(slug, None)
    STRATEGY_SCHEMAS.pop(slug, None)
    GENERATED_STRATEGIES.pop(slug, None)

    # Purge any previously-loaded generated module so a future restart or
    # import doesn't accidentally resurrect a deleted strategy file.
    bad = [name for name in _sys.modules if name.startswith("strategies.generated." + slug)]
    for name in bad:
        _sys.modules.pop(name, None)

    # If this was the currently focused strategy, clear the selection.
    _st_last = None
    _state = globals().get('STATE')
    if _state is not None:
        _st_last = getattr(_state, 'last', None)
    if _st_last:
        _st_last.results = [r for r in _st_last.results if r.get("key") != slug]
        if _st_last.key == slug:
            _st_last.key = None
            _st_last.results = []

    logger.info("Deleted AI-generated strategy: %s", slug)
    return jsonify({"ok": True, "key": slug})


# ----------------------------------------------------------------------
# Run audit log (JSON-lines).  No external deps; survives restarts.
# ----------------------------------------------------------------------
import json as _json
import sys as _sys
from datetime import datetime as _datetime

_RUN_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
_RUN_LOG_PATH = os.path.join(_RUN_LOG_DIR, "audit.jsonl")


def _ensure_run_log_dir() -> None:
    try:
        os.makedirs(_RUN_LOG_DIR, exist_ok=True)
    except OSError:
        pass


def log_run(record: dict[str, Any]) -> None:
    """Append one run record to the audit log as a JSON line.

    Called by POST /api/run on success so you can later review what was run,
    with which data source and simulated flag, without re-running anything.
    """
    _ensure_run_log_dir()
    record.setdefault("logged_at", _iso_now())
    try:
        with open(_RUN_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(_jsonable(record), default=str) + "\n")
    except OSError as exc:
        logger.warning("Could not write run audit log: %s", exc)


def list_runs(limit: int = 50) -> list[dict[str, Any]]:
    """Return the most recent run records from the audit log."""
    if not os.path.exists(_RUN_LOG_PATH):
        return []
    try:
        with open(_RUN_LOG_PATH, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return []
    lines.reverse()
    out: list[dict[str, Any]] = []
    for line in lines:
        if len(out) >= limit:
            break
        try:
            out.append(_json.loads(line))
        except _json.JSONDecodeError:
            continue
    out.reverse()
    return out


def _iso_now() -> str:
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.route("/api/runs", methods=["GET"])
def api_list_runs():
    """Return recent run records (data source + simulated flag included)."""
    try:
        runs = list_runs()
    except Exception as exc:
        logger.exception("Run audit list failed")
        return jsonify({"ok": False, "error": f"Cowardly refusing to list runs: {exc}"}), 500
    return jsonify({"ok": True, "runs": runs, "count": len(runs)})


def _import_module(name: str):
    """Import a module by name with module-level error logging."""
    import importlib
    return importlib.import_module(name)


def _next_copy_slug(original: str, index: int = 1) -> str:
    """Generate a unique duplicate slug for `original` (e.g. `foo_copy_1`)."""
    base = re.sub(r"[^a-z0-9]+", "_", original.strip().lower()).strip("_") or "strategy"
    slug = f"{base}_copy_{index}"
    while slug in generated_slugs():
        index += 1
        slug = f"{base}_copy_{index}"
    return slug


@app.route("/api/strategies/<key>/duplicate", methods=["POST"])
def api_duplicate_strategy(key: str):
    """Duplicate an existing AI-generated strategy under a new generated slug.

    The new copy is validated and dry-run exactly like a brand-new generation
    (reuses `load_generated_strategies`-style loading), then registered in
    memory so it appears in the UI immediately.
    """
    original = str(key or "").strip().lower()
    if not original:
        return jsonify({"ok": False, "error": "Missing strategy key."}), 400
    if original not in _generated_keys():
        return jsonify({
            "ok": False,
            "error": f"'{original}' cannot be duplicated (not a generated strategy).",
        }), 403

    try:
        mod = _import_module(f"strategies.generated.{original}")
    except Exception as exc:
        logger.exception("Could not import generated module %s", original)
        return jsonify({"ok": False, "error": f"Could not load strategy '{original}': {exc}"}), 500

    cls = getattr(mod, original, None)
    if cls is None or not inspect.isclass(cls) or not issubclass(cls, strategies_base.Strategy):
        return jsonify({
            "ok": False,
            "error": f"'{original}' is not a valid Strategy subclass.",
        }), 400

    display_name = getattr(cls, "display_name", original.replace("_", " ").title() + " (copy)")
    cls_name = getattr(cls, "__name__", _class_name_from_slug(_next_copy_slug(original)))
    new_slug = _next_copy_slug(original)
    new_class_name = cls_name

    code = getattr(mod, "__file__", None)
    if code and os.path.exists(code):
        with open(code, "r", encoding="utf-8") as f:
            code_text = f.read()
    else:
        code_text = ""

    try:
        persist_strategy(
            slug=new_slug,
            class_name=new_class_name,
            display_name=display_name,
            description=f"Duplicate of '{original}'.",
            code=code_text or "",
        )
    except Exception as exc:
        logger.exception("Failed to persist duplicate %s", new_slug)
        return jsonify({"ok": False, "error": f"Failed to persist duplicate: {exc}"}), 500

    fresh_mod = _import_module(f"strategies.generated.{new_slug}")
    fresh_cls = getattr(fresh_mod, new_class_name, None)
    if fresh_cls is None or not inspect.isclass(fresh_cls) or not issubclass(fresh_cls, strategies_base.Strategy):
        return jsonify({
            "ok": False,
            "error": f"Duplicate persisted but could not load class '{new_class_name}'.",
        }), 500

    STRATEGY_REGISTRY[new_slug] = fresh_cls
    STRATEGY_SCHEMAS[new_slug] = infer_params_schema(fresh_cls)
    GENERATED_STRATEGIES[new_slug] = {
        "name": display_name,
        "params": STRATEGY_SCHEMAS[new_slug],
        "generated": True,
    }
    logger.info("Duplicated strategy %s -> %s", original, new_slug)
    return jsonify({
        "ok": True,
        "original": original,
        "key": new_slug,
        "name": display_name,
        "params": STRATEGY_SCHEMAS[new_slug],
    })


@app.route("/api/strategies/<key>/code", methods=["GET"])
def api_export_strategy_code(key: str):
    """Export the source code of a generated strategy as JSON.

    Only AI-generated strategies can be exported this way.  Returns the raw
    Python source so the UI can either show it or trigger a file download.
    """
    slug = str(key or "").strip().lower()
    if not slug:
        return jsonify({"ok": False, "error": "Missing strategy key."}), 400
    if slug not in _generated_keys():
        return jsonify({
            "ok": False,
            "error": f"'{slug}' cannot be exported (not a generated strategy).",
        }), 403
    code_path = os.path.join(GENERATED_DIR, f"{slug}.py")
    if not os.path.exists(code_path):
        return jsonify({"ok": False, "error": f"Source file for '{slug}' not found."}), 404
    try:
        with open(code_path, "r", encoding="utf-8") as f:
            code_text = f.read()
    except OSError as exc:
        logger.exception("Failed to read code for %s", slug)
        return jsonify({"ok": False, "error": f"Failed to read source: {exc}"}), 500
    display_name = GENERATED_STRATEGIES.get(slug, {}).get("name", slug)
    return jsonify({
        "ok": True,
        "key": slug,
        "name": display_name,
        "code": code_text,
    })


if __name__ == "__main__":
    _port_default = os.environ.get("PORT", "5001")
    port = int(_port_default)
    print("=" * 60)
    print("  Backtest Studio - Web UI")
    print("  Open: http://127.0.0.1:%d", port)
    print("=" * 60)
    app.run(host="127.0.0.1", port=port, debug=False)