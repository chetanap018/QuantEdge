"""runs_db.py
============

Multi-run comparison database for backtest experiments.

Stores every backtest run config + results in a small SQLite file so you
can query historical experiments instead of re-running everything.

Usage:
    from runs_db import RunStore
    store = RunStore("runs.sqlite")
    store.record_run(config, results, extra=...)
    runs = store.query_runs(symbol="RELIANCE")
    comparison = store.compare_runs([run_id1, run_id2])
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional


class RunStore:
    DEFAULT_DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "runs.sqlite"))

    def __init__(self, db_path: Optional[str] = None):
        path = db_path or self.DEFAULT_DB
        self.db_path = os.path.abspath(path)
        self._ensure_db()

    def _ensure_db(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                symbol TEXT,
                exchange TEXT,
                timeframe TEXT,
                strategy TEXT,
                capital REAL,
                sizing TEXT,
                segment TEXT,
                status TEXT,
                config_json TEXT,
                results_json TEXT,
                extra_json TEXT
            )""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_runs_symbol ON runs(symbol)""")
            conn.execute("""CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at)""")
            conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_run(self, config: Dict[str, Any], results: Dict[str, Any], extra: Optional[Dict[str, Any]] = None, *, status: str = "ok", id: Optional[str] = None) -> str:
        run_id = id or str(uuid.uuid4())
        row = {
            "id": run_id,
            "created_at": datetime.now().isoformat(),
            "symbol": config.get("symbol"),
            "exchange": config.get("exchange"),
            "timeframe": config.get("timeframe"),
            "strategy": results.get("strategy"),
            "capital": results.get("initial_capital"),
            "sizing": config.get("sizing"),
            "segment": config.get("segment"),
            "status": status,
            "config_json": json.dumps(config, default=str),
            "results_json": json.dumps(results, default=str),
            "extra_json": json.dumps(extra, default=str) if extra else None,
        }
        conn = self._connect()
        try:
            conn.execute("""INSERT INTO runs(id, created_at, symbol, exchange, timeframe,
                strategy, capital, sizing, segment, status,
                config_json, results_json, extra_json)
                VALUES(:id, :created_at, :symbol, :exchange, :timeframe,
                :strategy, :capital, :sizing, :segment, :status,
                :config_json, :results_json, :extra_json)""", row)
            conn.commit()
        finally:
            conn.close()
        return run_id

    def query_runs(self, *, symbol: Optional[str] = None, strategy: Optional[str] = None,
        status: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None,
        limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        conditions = []
        params: Dict[str, Any] = {}
        if symbol:
            conditions.append("symbol = :symbol")
            params["symbol"] = symbol
        if strategy:
            conditions.append("strategy = :strategy")
            params["strategy"] = strategy
        if status:
            conditions.append("status = :status")
            params["status"] = status
        if since:
            conditions.append("created_at >= :since")
            params["since"] = since
        if until:
            conditions.append("created_at <= :until")
            params["until"] = until
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        conn = self._connect()
        try:
            rows = conn.execute(f"SELECT * FROM runs {where} ORDER BY created_at DESC LIMIT :limit OFFSET :offset",
                {**params, "limit": limit, "offset": offset}).fetchall()
            return [{"id": r["id"], "created_at": r["created_at"], "symbol": r["symbol"],
                "exchange": r["exchange"], "timeframe": r["timeframe"], "strategy": r["strategy"],
                "capital": r["capital"], "sizing": r["sizing"], "segment": r["segment"],
                "status": r["status"],
                "config": json.loads(r["config_json"] or "{}"),
                "results": json.loads(r["results_json"] or "{}"),
                "extra": json.loads(r["extra_json"] or "{}")} for r in rows]
        finally:
            conn.close()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        conn = self._connect()
        try:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            return {"id": row["id"], "created_at": row["created_at"], "symbol": row["symbol"],
                "exchange": row["exchange"], "timeframe": row["timeframe"], "strategy": row["strategy"],
                "capital": row["capital"], "sizing": row["sizing"], "segment": row["segment"],
                "status": row["status"],
                "config": json.loads(row["config_json"] or "{}"),
                "results": json.loads(row["results_json"] or "{}"),
                "extra": json.loads(row["extra_json"] or "{}")}
        finally:
            conn.close()

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        runs = []
        for rid in run_ids:
            r = self.get_run(rid)
            if r:
                runs.append(r)
        if not runs:
            return {"runs": [], "comparison": {}}
        keys = ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "total_trades", "win_rate", "profit_factor", "net_profit"]
        comparison: Dict[str, Any] = {}
        for k in keys:
            series = []
            for r in runs:
                val = r["results"].get(k)
                if val is not None:
                    try:
                        series.append((r["id"], float(val)))
                    except (TypeError, ValueError):
                        pass
            if series:
                sorted_series = sorted(series, key=lambda x: x[1], reverse=True)
                comparison[k] = {"by_run": {rid: v for rid, v in series}, "best": sorted_series[0], "worst": sorted_series[-1],
                    "mean": sum(v for _, v in series) / len(series)}
        return {"runs": runs, "comparison": comparison}

    def get_last_runs(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.query_runs(limit=n)

    def delete_run(self, run_id: str) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def db_status(self) -> Dict[str, Any]:
        conn = self._connect()
        try:
            count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            last = conn.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT 1").fetchone()
            return {"db_path": self.db_path, "run_count": count,
                "last_run": {"id": last["id"], "created_at": last["created_at"], "symbol": last["symbol"], "strategy": last["strategy"]} if last else None}
        finally:
            conn.close()


if __name__ == "__main__":
    store = RunStore(":memory:")
    cfg: Dict[str, Any] = {"symbol": "RELIANCE", "exchange": "NSE", "timeframe": "ONE_DAY", "sizing": "fixed_quantity", "segment": "intraday_equity"}
    res: Dict[str, Any] = {"strategy": "SMA Crossover", "total_trades": 10, "total_return_pct": 5.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 3.0, "win_rate": 55.0, "profit_factor": 1.2, "net_profit": 5000.0}
    rid = store.record_run(cfg, res)
    print("recorded:", rid)
    print("status:", store.db_status())
    print("last:", store.get_last_runs(1))
    print("compare:", store.compare_runs([rid]))
    print("deleted:", store.delete_run(rid))
    print("after delete status:", store.db_status())
