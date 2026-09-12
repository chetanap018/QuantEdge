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

import os
import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

import config
import backtest_config
from data_fetcher import DataFetcher
from backtest_engine import BacktestEngine
from strategies import (
    SMACrossoverStrategy,
    RSIMeanReversionStrategy,
    MACDCrossoverStrategy,
    OpeningCandleStrategy,
    SuzlonStrategy,
    HeroOrbStrategy,
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ============================================================
# Strategy registry + parameter schemas (drives the UI forms)
# ============================================================
STRATEGY_REGISTRY: Dict[str, type] = {
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

TIMEFRAMES: List[str] = [
    "ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE",
    "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY", "ONE_WEEK", "ONEEEEMONTH",
]

EXCHANGES: List[str] = ["NSE", "BSE", "NFO", "MCX"]
SEGMENTS: List[str] = ["intraday_equity", "delivery_equity", "futures", "options"]
SIZING_METHODS: List[str] = ["fixed_quantity", "fixed_capital_pct", "risk_based"]

CHARGES_COMPONENTS: List[str] = [
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


def _build_strategy(strat_key: str, params: Optional[Dict]) -> Any:
    """Instantiate a strategy with validated params (falls back to schema defaults)."""
    schema = STRATEGY_SCHEMAS.get(strat_key, {})
    clean: Dict[str, Any] = {}
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


def _pad_or_truncate(dates: List, values: List) -> Tuple[List, List]:
    """Align equity-curve dates/values (they always match in practice)."""
    n = min(len(dates), len(values))
    return list(dates[:n]), list(values[:n])


def _compute_drawdown(values: List[float], dates: List[str]) -> Dict:
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


def _compute_monthly_returns(values: List[float], dates: List[str]) -> Dict:
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


def _trade_stats(trades: List[Dict]) -> Dict:
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

    def _side_stats(ts: List[Dict]) -> Dict:
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
def _charges_totals(trades: List[Dict]) -> Dict:
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


def _serialize_trades(trades: List) -> List[Dict]:
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
    params: Dict,
    data: pd.DataFrame,
    capital: float,
    sizing: str,
    segment: str,
    sizing_params: Dict,
) -> Dict:
    """Run a single strategy and enrich with advanced analytics."""
    strategy = _build_strategy(strat_key, params)

    # Make the UI's sizing parameters actually drive the engine
    config.DEFAULT_FIXED_QUANTITY = int(sizing_params.get("quantity", backtest_config.FIXED_QUANTITY))
    config.DEFAULT_CAPITAL_PCT = float(sizing_params.get("pct", backtest_config.CAPITAL_PCT))
    config.DEFAULT_RISK_PCT = float(sizing_params.get("risk_pct", backtest_config.RISK_PCT))
    config.DEFAULT_RISK_PER_UNIT = float(sizing_params.get("risk_per_unit", backtest_config.RISK_PER_UNIT))

    engine = BacktestEngine(
        strategy=strategy,
        data=data,
        initial_capital=capital,
        position_sizing=sizing,
        segment=segment,
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
    return render_template("index.html", project_name=backtest_config.PROJECT_NAME)


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
    for key in STRATEGY_SCHEMAS:
        payload["strategies"][key] = {
            "name": STRATEGY_REGISTRY[key]().name,
            "params": STRATEGY_SCHEMAS[key],
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

    fetcher = DataFetcher()
    try:
        data = fetcher.fetch_historical_data(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
        )
    finally:
        fetcher.logout()

    if data.empty:
        return jsonify({"ok": False, "error": "No data returned"}), 400

    data = data.sort_values("datetime").reset_index(drop=True).copy()
    bars_n = len(data)
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
            logger.exception("Strategy %s failed" % key)
            results_list.append({"key": key, "error": str(exc)})

    if not results_list:
        return jsonify({"ok": False, "error": "No valid strategies selected"}), 400

    return jsonify({
        "ok": True,
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "capital": capital,
        "bars": bars_n,
        "ohlc": ohlc,
        "results": results_list,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 60)
    print("  Backtest Studio - Web UI")
    print("  Open: http://127.0.0.1:%d" % port)
    print("=" * 60)
    app.run(host="127.0.0.1", port=port, debug=False)