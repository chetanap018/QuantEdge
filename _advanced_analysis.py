"""
advanced_analysis.py - Advanced strategy diagnostics & report generation.
Run:  python3 _advanced_analysis.py
Produces:  _report_data.json  (consumed by advanced_report.html)
"""
import hashlib, os, json
from datetime import datetime
import numpy as np, pandas as pd
from backtest_engine import BacktestEngine
from strategies.hero_orb import HeroOrbStrategy

SYMBOL, EXCHANGE, TIMEFRAME = "HEROMOTOCO", "NSE", "FIFTEEN_MINUTE"
START, END = "2026-04-01", "2026-09-08"
CAPITAL   = 100_000.0
SIZING    = "fixed_capital_pct"
SEGMENT   = "intraday_equity"

RAW = f"{EXCHANGE}:{SYMBOL}_{TIMEFRAME}_{START}_{END}"
KEY = hashlib.md5(RAW.encode()).hexdigest()
DATA = pd.read_csv(os.path.join("cache", KEY + ".csv"), parse_dates=["datetime"])
DATA = DATA.sort_values("datetime").reset_index(drop=True)


def run_strategy(params):
    s = HeroOrbStrategy(**params)
    e = BacktestEngine(strategy=s, data=DATA, initial_capital=CAPITAL,
                       position_sizing=SIZING, segment=SEGMENT)
    return e.run()


def max_consecutive(pnls, positive=True):
    best = cur = 0
    for p in pnls:
        if (p > 0) == positive:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def rolling_metric(values, window, func):
        return [func(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


# ── 1. Default run ──
DEFAULTS = {"orb_bars": 4, "trend_ema": 30, "stop_atr": 7.0, "min_hold_bars": 8}
default_res = run_strategy(DEFAULTS)
dp     = [t.net_pnl for t in default_res["trades"]]
eq     = default_res["equity_curve"]
eq_arr = np.array(eq)
dd     = (eq_arr - np.maximum.accumulate(eq_arr)) / np.maximum.accumulate(eq_arr) * 100
print(f"Default: trades={len(dp)} win={default_res['win_rate']}% net={default_res['net_profit']:.2f}")

# ── 2. Parameter sweep ──
SWEEP = []
for ob in [2, 3, 4, 6]:
    for te in [20, 30, 50]:
        for mh in [4, 8, 12]:
            for sa in [5.0, 7.0, 10.0]:
                r = run_strategy({"orb_bars": ob, "trend_ema": te, "stop_atr": sa, "min_hold_bars": mh})
                SWEEP.append({
                    "orb_bars": ob, "trend_ema": te, "min_hold_bars": mh, "stop_atr": sa,
                    "net_profit": r["net_profit"], "win_rate": r["win_rate"],
                    "pf": r["profit_factor"], "trades": r["total_trades"],
                    "max_dd": r["max_drawdown_pct"], "sharpe": r["sharpe_ratio"],
                })
SWEEP_DF = pd.DataFrame(SWEEP)
best_row = SWEEP_DF.loc[SWEEP_DF["net_profit"].idxmax()]
print(f"Best sweep: {best_row.to_dict()}")

# ── 3. Per-trade diagnostics ──
trades = default_res["trades"]
trade_diag = []
cum = 0
for i, t in enumerate(trades):
    cum += t.net_pnl
    held = (t.exit_time - t.entry_time).total_seconds() / 3600
    trade_diag.append({
        "n": i + 1, "dir": t.direction,
        "entry_t": t.entry_time.strftime("%m-%d %H:%M"),
        "exit_t":  t.exit_time.strftime("%m-%d %H:%M"),
        "entry_p": t.entry_price, "exit_p": t.exit_price,
                "gross": float(t.gross_pnl), "charges": float(t.total_charges),
        "net": float(t.net_pnl), "cum": round(float(cum), 2), "held_h": round(float(held), 1),
        "win": bool(t.net_pnl > 0),
    })

win_series = [1 if t.net_pnl > 0 else 0 for t in trades]
rolling_wr = rolling_metric(win_series, 5, lambda s: round(sum(s) / len(s) * 100, 1))

rolling_pf = []
for i in range(len(trades)):
    sub = trades[max(0, i - 9):i + 1]
    wins = sum(t.gross_pnl for t in sub if t.net_pnl > 0)
    losses = abs(sum(t.gross_pnl for t in sub if t.net_pnl <= 0))
    rolling_pf.append(round(wins / losses, 2) if losses > 0 else None)

win_streak  = max_consecutive(dp, True)
loss_streak = max_consecutive(dp, False)

sorted_trades = sorted(trades, key=lambda t: t.net_pnl)
worst_5 = sorted_trades[:5]
best_5  = sorted_trades[-5:][::-1]

long_trades  = [t for t in trades if t.direction == "long"]
short_trades = [t for t in trades if t.direction == "short"]
long_pnl  = sum(t.net_pnl for t in long_trades)
short_pnl = sum(t.net_pnl for t in short_trades)
print(f"Diag done: long={len(long_trades)}({long_pnl:.0f}) short={len(short_trades)}({short_pnl:.0f})")

# ── 4. Parameter heatmap ──
te_fixed = int(best_row["trend_ema"])
mh_fixed = int(best_row["min_hold_bars"])
heatmap_data = [
    [int(row["orb_bars"]), row["stop_atr"], round(row["net_profit"], 2)]
    for _, row in SWEEP_DF[(SWEEP_DF["trend_ema"] == te_fixed) & (SWEEP_DF["min_hold_bars"] == mh_fixed)].iterrows()
]

# ── 5. Time-of-day ──
tod_buckets = {}
for t in trades:
    bucket = f"{t.entry_time.hour:02d}:00"
    tod_buckets.setdefault(bucket, []).append(t.net_pnl)
tod_summary = [
    {"hour": k, "trades": len(v), "avg_pnl": round(sum(v) / len(v), 2), "total": round(sum(v), 2)}
    for k, v in sorted(tod_buckets.items())
]

n_days = DATA["datetime"].dt.date.nunique()
trades_per_day = len(trades) / n_days if n_days else 0

# ── 6. Recommendations ──
total_charges = default_res["total_charges"]
net_profit    = default_res["net_profit"]
max_dd        = default_res["max_drawdown_pct"]
win_rate      = default_res["win_rate"]
profit_factor = default_res["profit_factor"]

recommendations = []
if long_pnl < 0 and short_pnl > 0:
    recommendations.append({"issue": "Long trades losing, shorts profitable",
        "detail": f"Long net: Rs {long_pnl:.2f} ({len(long_trades)} trades). Short net: Rs {short_pnl:.2f} ({len(short_trades)} trades).",
        "fix": "Tighten trend gate: use shorter EMA (20) or require close > EMA + 0.3%. In downtrends, add ADX>20 filter or disable longs when price < 200-EMA."})
elif short_pnl < 0 and long_pnl > 0:
    recommendations.append({"issue": "Short trades losing, longs profitable",
        "detail": f"Short net: Rs {short_pnl:.2f}. Long net: Rs {long_pnl:.2f}.",
        "fix": "ORB breakdown edge unreliable. Widen stop_atr to let runners run. Verify ORB period captures true opening volatility."})

if loss_streak >= 4:
    recommendations.append({"issue": f"Losing streak of {loss_streak} consecutive trades",
        "detail": "Consecutive losses indicate choppy regime over-trading.",
        "fix": "Increase min_hold_bars to 12+. Add volatility filter: skip when ATR < average. Add ADX>20 trend-strength filter."})

if trades_per_day > 2:
    recommendations.append({"issue": f"High frequency: {trades_per_day:.1f} trades/day",
        "detail": "Over-trading increases charges and whipsaw risk.",
        "fix": "Widen opening range (orb_bars=4-6). Reduce max_entry_time. Increase min_hold_bars to 12+."})
elif trades_per_day < 0.3:
    recommendations.append({"issue": f"Low frequency: {trades_per_day:.1f} trades/day",
        "detail": "May miss opportunities.",
        "fix": "Shorten orb_bars. Extend max_entry_time. Verify win rate doesn't degrade."})

if total_charges > abs(net_profit) * 0.5 and net_profit < 0:
    recommendations.append({"issue": "Charges are a large drag",
        "detail": f"Charges: Rs {total_charges:.2f}. Net: Rs {net_profit:.2f}.",
        "fix": "Reduce trade frequency. Use delivery segment for longer holds. Widen stops to reduce stop-outs."})

if max_dd > 5:
    recommendations.append({"issue": f"Max drawdown {max_dd:.1f}% is elevated",
        "detail": "Large drawdowns erode capital.",
        "fix": "Reduce position size to 50% capital. Add daily loss limit (-2%). Tighten disaster stop to stop_atr=5."})

if best_row["net_profit"] > net_profit + 100:
    recommendations.append({"issue": "Default params suboptimal vs sweep best",
        "detail": f"Default: Rs {net_profit:.2f}. Sweep best: Rs {best_row['net_profit']:.2f} (orb={int(best_row['orb_bars'])}, ema={int(best_row['trend_ema'])}, hold={int(best_row['min_hold_bars'])}, stop={best_row['stop_atr']}).",
        "fix": "Update defaults to sweep-best. Run sweep on out-of-sample data to avoid overfitting."})

if not recommendations:
    recommendations.append({"issue": "No critical issues",
        "detail": f"Net: Rs {net_profit:.2f}, win: {win_rate}%, PF: {profit_factor}.",
        "fix": "Strategy performing within expectations. Test on multiple symbols/timeframes."})

# ── 7. Assemble & write report ──
report = {
    "meta": {"symbol": SYMBOL, "exchange": EXCHANGE, "timeframe": TIMEFRAME,
             "start": str(DATA["datetime"].iloc[0]), "end": str(DATA["datetime"].iloc[-1]),
             "bars": len(DATA), "capital": CAPITAL,
             "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
    "metrics": {
        "net_profit": default_res["net_profit"], "total_return_pct": default_res["total_return_pct"],
        "win_rate": default_res["win_rate"], "profit_factor": default_res["profit_factor"],
        "total_trades": default_res["total_trades"], "winning_trades": default_res["winning_trades"],
        "losing_trades": default_res["losing_trades"], "max_drawdown_pct": default_res["max_drawdown_pct"],
        "sharpe_ratio": default_res["sharpe_ratio"], "cagr": default_res["cagr"],
        "total_charges": default_res["total_charges"], "final_capital": default_res["final_capital"],
    },
    "params_used": DEFAULTS,
    "equity": [round(v, 2) for v in eq],
    "drawdown": [round(v, 3) for v in dd],
    "trade_diag": trade_diag,
    "rolling_wr": rolling_wr,
    "rolling_pf": rolling_pf,
    "win_streak": win_streak, "loss_streak": loss_streak,
    "long_pnl": round(long_pnl, 2), "short_pnl": round(short_pnl, 2),
    "long_count": len(long_trades), "short_count": len(short_trades),
    "heatmap": heatmap_data,
    "heatmap_fixed": {"trend_ema": te_fixed, "min_hold_bars": mh_fixed},
    "tod_summary": tod_summary,
    "trades_per_day": round(trades_per_day, 2),
    "best_params": {k: (round(v, 2) if isinstance(v, float) else int(v)) for k, v in best_row.items()},
    "sweep_top12": [{k: (round(v, 2) if isinstance(v, float) else int(v)) for k, v in r.items()}
                    for _, r in SWEEP_DF.nlargest(12, "net_profit").iterrows()],
    "sweep_count": len(SWEEP),
    "recommendations": recommendations,
    "worst_5": [{"dir": t.direction, "entry": t.entry_time.strftime("%m-%d %H:%M"),
                 "pnl": round(t.net_pnl, 2)} for t in worst_5],
    "best_5": [{"dir": t.direction, "entry": t.entry_time.strftime("%m-%d %H:%M"),
                "pnl": round(t.net_pnl, 2)} for t in best_5],
}

with open("_report_data.json", "w") as f:
    json.dump(report, f, default=str)

with open("report_data.js", "w") as f:
    f.write("var REPORT_DATA = " + json.dumps(report, indent=2) + ";\n")

print(f"Report written. Recommendations: {len(recommendations)}")
for r in recommendations:
    print(f"  - {r['issue']}")
print("DONE")