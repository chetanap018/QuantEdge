"""
ai_strategy_writer.py
----------------------
Lets a user describe a trading strategy in plain English and turns it into a
registered Strategy class, the same way strategies/sma_crossover.py etc. work.

Pipeline: generate -> static-check -> load -> dry-run on synthetic data -> persist.
Each step must pass before moving to the next.

This is defense-in-depth for a personal/local tool, not a security boundary
for a multi-user or internet-facing deployment.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
import requests

from strategies.base import Strategy

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_MAX_OUTPUT_TOKENS = 2000
DRY_RUN_TIMEOUT_SECONDS = 8

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies", "generated")
MANIFEST_PATH = os.path.join(GENERATED_DIR, "manifest.json")

ALLOWED_IMPORTS = {"pandas", "numpy", "strategies.base", "strategies"}

FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "exit", "quit", "breakpoint", "help",
}

SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "sum": sum,
    "abs": abs, "round": round, "enumerate": enumerate, "zip": zip,
    "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "sorted": sorted, "reversed": reversed, "map": map, "filter": filter,
    "isinstance": isinstance, "print": print,
    "True": True, "False": False, "None": None,
    "Exception": Exception, "ValueError": ValueError, "TypeError": TypeError,
    "StopIteration": StopIteration, "KeyError": KeyError, "IndexError": IndexError,
    "super": super, "property": property, "staticmethod": staticmethod,
}


class StrategyGenerationError(Exception):
    pass


_SYSTEM_CONTRACT = """\
You write exactly one Python class for a backtesting framework. Follow this contract exactly.

RULES:
- Output ONLY a single Python code block. No prose before or after it.
- The ONLY imports allowed are: `import pandas as pd`, `import numpy as np`, and `from strategies.base import Strategy`. No other imports of any kind.
- Define exactly one class, named `{class_name}`, subclassing `Strategy`.
- `__init__(self, **kwargs)` must accept simple numeric/bool/str parameters with sensible defaults and must call `super().__init__(name="{display_name}", <same kwargs>)`, then store each as `self.<param> = <param>`.
- Implement `generate_signals(self, data: pd.DataFrame) -> pd.DataFrame`. `data` has columns: datetime, open, high, low, close, volume, sorted ascending by time. Return a copy of `data` with an added integer column `signal`: 1 = enter/flip long, -1 = enter/flip short, 0 = no action.
- CRITICAL: no lookahead bias. When computing the signal for row i, only use data from rows <= i. Never use `.shift(-n)` or any future row.
- No file I/O, no network calls, no `os`/`sys`/`subprocess`, no `eval`/`exec`, no `input()`. Pure pandas/numpy computation only.

EXAMPLE (for style reference only):

```python
from strategies.base import Strategy
import pandas as pd
import numpy as np

class MACDCrossover(Strategy):
    def __init__(self, fast=12, slow=26, signal=9):
        super().__init__(name="MACD Crossover", fast=fast, slow=slow, signal=signal)
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        df["signal"] = 0
        df.loc[macd > signal_line, "signal"] = 1
        df.loc[macd < signal_line, "signal"] = -1
        return df
```
"""


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or f"strategy_{uuid.uuid4().hex[:8]}"


def _class_name_from_slug(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_")) + "Strategy"


def _extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    code = match.group(1) if match else text
    return code.strip()


def call_gemini(description: str, class_name: str, display_name: str,
                 api_key: Optional[str] = None, model: str = GEMINI_MODEL) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise StrategyGenerationError(
            "GEMINI_API_KEY is not set. Export it in your environment "
            "before using AI strategy generation."
        )

    system = _SYSTEM_CONTRACT.format(class_name=class_name, display_name=display_name)
    user_prompt = (
        f"Strategy description from the user:\n\"\"\"\n{description.strip()}\n\"\"\"\n\n"
        f"Write the `{class_name}` class implementing this."
    )

    url = GEMINI_API_URL.format(model=model)
    resp = requests.post(
        url,
        headers={"content-type": "application/json"},
        params={"key": key},
        json={
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": GEMINI_MAX_OUTPUT_TOKENS, "temperature": 0.2},
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        block_reason = data.get("promptFeedback", {}).get("blockReason")
        if block_reason:
            raise StrategyGenerationError(f"Model declined to respond (reason: {block_reason}).")
        raise StrategyGenerationError("Model returned no candidates.")

    finish_reason = candidates[0].get("finishReason")
    parts = candidates[0].get("content", {}).get("parts", [])
    text_parts = [p["text"] for p in parts if "text" in p]
    if not text_parts:
        if finish_reason == "MAX_TOKENS":
            raise StrategyGenerationError(
                "Model response was cut off (hit max output tokens). Try a shorter description."
            )
        raise StrategyGenerationError(f"Model returned no text content (finishReason={finish_reason}).")
    return "\n".join(text_parts)


def validate_source(code: str, class_name: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise StrategyGenerationError(f"Generated code has a syntax error: {e}")

    class_defs = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if len(class_defs) != 1:
        raise StrategyGenerationError(
            f"Expected exactly one class definition, found {len(class_defs)}."
        )
    if class_defs[0].name != class_name:
        raise StrategyGenerationError(
            f"Expected class named '{class_name}', found '{class_defs[0].name}'."
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    raise StrategyGenerationError(f"Disallowed import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.split(".")[0] not in ALLOWED_IMPORTS:
                raise StrategyGenerationError(f"Disallowed import: {mod}")
        elif isinstance(node, ast.Call):
            fname = None
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname in FORBIDDEN_CALLS:
                raise StrategyGenerationError(f"Disallowed call: {fname}(...)")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in (
                "__init__", "__name__",
            ):
                raise StrategyGenerationError(f"Disallowed dunder attribute access: {node.attr}")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            raise StrategyGenerationError("`with` statements are not allowed (no file/context I/O).")


def load_strategy_class(code: str, class_name: str) -> Type[Strategy]:
    module_ns: Dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        "Strategy": Strategy,
    }
    try:
        exec(compile(code, "<ai_generated_strategy>", "exec"), module_ns)
    except Exception as e:
        raise StrategyGenerationError(f"Generated code failed to execute: {e}")

    cls = module_ns.get(class_name)
    if cls is None or not (inspect.isclass(cls) and issubclass(cls, Strategy)):
        raise StrategyGenerationError(f"Class '{class_name}' not found or is not a Strategy subclass.")
    return cls


def _synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    returns = rng.normal(0.0003, 0.015, n)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    noise = rng.uniform(0.002, 0.01, n)
    high = np.maximum(open_, close) * (1 + noise)
    low = np.minimum(open_, close) * (1 - noise)
    volume = rng.integers(50_000, 500_000, n)
    return pd.DataFrame({
        "datetime": dates, "open": open_, "high": high,
        "low": low, "close": close, "volume": volume,
    })


def infer_params_schema(cls: Type[Strategy]) -> Dict[str, Dict[str, Any]]:
    sig = inspect.signature(cls.__init__)
    schema: Dict[str, Dict[str, Any]] = {}
    for pname, p in sig.parameters.items():
        if pname == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        default = p.default if p.default is not inspect.Parameter.empty else 0
        if isinstance(default, bool):
            ptype, lo, hi = "bool", None, None
        elif isinstance(default, int):
            ptype, lo, hi = "int", max(0, default - default * 5 - 10), default * 10 + 50
        elif isinstance(default, float):
            ptype, lo, hi = "float", 0.0, max(default * 10, 10.0)
        else:
            ptype, lo, hi = "text", None, None
        entry: Dict[str, Any] = {
            "label": pname.replace("_", " ").capitalize(),
            "type": ptype,
            "default": default,
        }
        if lo is not None:
            entry["min"] = lo
        if hi is not None:
            entry["max"] = hi
        schema[pname] = entry
    return schema


def _dry_run(cls: Type[Strategy]) -> None:
    params = {k: v["default"] for k, v in infer_params_schema(cls).items()}
    instance = cls(**params)

    data = _synthetic_ohlcv()
    out = instance.generate_signals(data.copy())

    if not isinstance(out, pd.DataFrame):
        raise StrategyGenerationError("generate_signals() must return a DataFrame.")
    if "signal" not in out.columns:
        raise StrategyGenerationError("generate_signals() output is missing a 'signal' column.")
    if len(out) != len(data):
        raise StrategyGenerationError("generate_signals() must not change the number of rows.")
    bad_values = set(out["signal"].dropna().unique()) - {-1, 0, 1}
    if bad_values:
        raise StrategyGenerationError(f"'signal' column has invalid values: {bad_values}.")

    from backtest_engine import BacktestEngine
    engine = BacktestEngine(strategy=instance, data=data, initial_capital=100_000.0,
                             position_sizing="fixed_quantity", segment="intraday_equity")
    engine.run()


def dry_run_with_timeout(cls: Type[Strategy], timeout: int = DRY_RUN_TIMEOUT_SECONDS) -> None:
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_dry_run, cls)
        try:
            fut.result(timeout=timeout)
        except FutTimeout:
            raise StrategyGenerationError(
                f"Strategy timed out after {timeout}s during the test run "
                "(likely an infinite loop or pathological computation)."
            )
        except StrategyGenerationError:
            raise
        except Exception as e:
            raise StrategyGenerationError(f"Strategy raised an error during the test run: {e}")



def _load_manifest() -> List[Dict[str, str]]:
    if not os.path.exists(MANIFEST_PATH):
        return []
    with open(MANIFEST_PATH, "r") as f:
        return json.load(f)


def _save_manifest(entries: List[Dict[str, str]]) -> None:
    os.makedirs(GENERATED_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def persist_strategy(slug: str, class_name: str, display_name: str,
                      description: str, code: str) -> str:
    os.makedirs(GENERATED_DIR, exist_ok=True)
    init_path = os.path.join(GENERATED_DIR, "__init__.py")
    if not os.path.exists(init_path):
        open(init_path, "a").close()

    filepath = os.path.join(GENERATED_DIR, f"{slug}.py")
    with open(filepath, "w") as f:
        f.write(code + "\n")

    entries = _load_manifest()
    entries = [e for e in entries if e["slug"] != slug]
    entries.append({
        "slug": slug,
        "class_name": class_name,
        "display_name": display_name,
        "description": description,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_manifest(entries)
    return filepath


def generate_and_validate_strategy(description: str, display_name: str,
                                    api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Full pipeline. Returns a dict:
      ok=True  -> {ok, key, name, code, params_schema, strategy_class}
      ok=False -> {ok, error, code (if any was generated)}
    Does NOT register into any live registry -- caller does that on ok=True.
    """
    slug = _slugify(display_name)
    class_name = _class_name_from_slug(slug)

    raw = call_gemini(description, class_name, display_name, api_key=api_key)
    code = _extract_code_block(raw)

    try:
        validate_source(code, class_name)
        cls = load_strategy_class(code, class_name)
        dry_run_with_timeout(cls)
    except StrategyGenerationError as e:
        return {"ok": False, "error": str(e), "code": code}

    persist_strategy(slug, class_name, display_name, description, code)
    schema = infer_params_schema(cls)

    return {
        "ok": True,
        "key": slug,
        "name": display_name,
        "code": code,
        "params_schema": schema,
        "strategy_class": cls,
    }


def load_generated_strategies() -> List[Tuple[str, Type[Strategy], Dict[str, Any]]]:
    """
    Called at app startup: re-validates and re-loads every previously-accepted
    AI-generated strategy from disk. Skips any file that now fails validation.
    """
    results: List[Tuple[str, Type[Strategy], Dict[str, Any]]] = []
    for entry in _load_manifest():
        slug = entry["slug"]
        class_name = entry["class_name"]
        path = os.path.join(GENERATED_DIR, f"{slug}.py")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            code = f.read()
        try:
            validate_source(code, class_name)
            cls = load_strategy_class(code, class_name)
            dry_run_with_timeout(cls)
        except StrategyGenerationError as e:
            print(f"[ai_strategy_writer] Skipping '{slug}' on reload: {e}")
            continue
        results.append((slug, cls, infer_params_schema(cls)))
    return results

