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
import builtins as _builtins
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
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    """Read an int from the environment, clamped to [lo, hi], never raising."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    """Read a float from the environment, clamped to [lo, hi], never raising."""
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, value))


# Output budget per model call. 2000 was too small: a lengthy description makes
# the model write a long preamble plus a long class, the response gets cut off
# mid-code-block, and the truncated text then fails validation on its first
# lines. 4096 fits most strategies; the generation loop escalates towards
# GEMINI_MAX_OUTPUT_TOKEN_CEILING when a response is still cut off.
# Override with GEMINI_MAX_OUTPUT_TOKENS (clamped 256-8192; most Gemini models
# reject maxOutputTokens above 8192).
GEMINI_MAX_OUTPUT_TOKENS = _env_int("GEMINI_MAX_OUTPUT_TOKENS", 4096, 256, 8192)
GEMINI_MAX_OUTPUT_TOKEN_CEILING = 8192
DRY_RUN_TIMEOUT_SECONDS = 8

# Model calls allowed per generation: the first attempt plus this many minus
# one self-repair round-trips. A rejected attempt is sent back to the model
# together with its exact failure reason, which converges far faster - and
# keeps the user's original wording intact - than asking them to re-describe
# the strategy. Override with AI_STRATEGY_MAX_ATTEMPTS.
MAX_GENERATION_ATTEMPTS = _env_int("AI_STRATEGY_MAX_ATTEMPTS", 3, 1, 6)

# Pause between model calls, in seconds. Limited/free Gemini tiers enforce
# requests-per-minute quotas, and a repair fires a second call immediately
# after the first one was rejected - which is exactly when a 429 / quota error
# shows up and the run dies with nothing to show. Sleeping before each repair
# (and before retrying a throttled call) keeps the run inside the quota.
# 30-60s suits the usual 2-15 requests/minute limits.
# Override with AI_STRATEGY_REPAIR_DELAY_SECONDS; 0 disables the pause.
REPAIR_DELAY_SECONDS = _env_float("AI_STRATEGY_REPAIR_DELAY_SECONDS", 30.0, 0.0, 300.0)

# Extra model calls allowed purely to retry throttled responses (429 / quota /
# 503), on top of MAX_GENERATION_ATTEMPTS. Each retry waits at least
# REPAIR_DELAY_SECONDS, and longer when the API's own "Please retry in Xs" hint
# demands it. The free tier allows ~20 requests/minute, so a generation that
# needs several calls can trip this; 3 gives it room to breathe.
# Override with AI_STRATEGY_RATE_LIMIT_RETRIES.
RATE_LIMIT_RETRIES = _env_int("AI_STRATEGY_RATE_LIMIT_RETRIES", 3, 0, 5)

# Extra model calls allowed purely to retry a *truncated* response (MAX_TOKENS)
# with a larger output budget. Also capped by the token ceiling doubling, so
# this rarely binds. Override with AI_STRATEGY_TRUNCATION_RETRIES.
TRUNCATION_RETRIES = _env_int("AI_STRATEGY_TRUNCATION_RETRIES", 2, 0, 5)

# Cap on how long ONE automatic retry may wait, based on the API's own
# "Please retry in Xs" hint. Free-tier daily-quota 429s ask for minutes or
# hours - those must fail fast with a clear message instead of freezing the
# browser for that long. Override with AI_STRATEGY_MAX_RETRY_WAIT_SECONDS.
MAX_RETRY_WAIT_SECONDS = _env_float("AI_STRATEGY_MAX_RETRY_WAIT_SECONDS", 120.0, 1.0, 3600.0)

# Substrings that mark a throttling/transient failure worth waiting out.
# Deliberately excludes 400/401/403 style errors: a bad key or a rejected
# prompt will fail identically after any pause, so retrying just burns quota.
_RETRYABLE_API_MARKERS = (
    "429", "too many requests", "rate limit", "rate-limit", "ratelimit",
    "quota", "resource_exhausted", "resource exhausted",
    "503", "service unavailable", "overloaded", "timed out", "timeout",
)


def _is_retryable_api_error(message: str) -> bool:
    """True for throttling/transient API failures that a pause may clear."""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _RETRYABLE_API_MARKERS)


# Gemini tells us exactly how long to back off: the 429 body ends with
# "Please retry in 32.619841575s." (and sometimes a `retryDelay` JSON field or
# a `Retry-After` header). Honor it - a fixed 30s pause is what kept tripping
# the quota: the API asked for 32.6s, we waited 30s, and got throttled again.
_RETRY_AFTER_RE = re.compile(
    r"retry(?:[\s\-a-z_:=\"'\\]*?)([0-9]+(?:\.[0-9]+)?)\s*(?:s(?:ec(?:onds?)?)?)?\b",
    re.IGNORECASE,
)


def _extract_retry_after_seconds(message: str) -> Optional[float]:
    """Best-effort parse of the API's own back-off hint from an error message.

    Returns the LARGEST hint found (be conservative), or None when the message
    carries no usable hint.
    """
    if not message:
        return None
    hints = [float(m) for m in _RETRY_AFTER_RE.findall(message)]
    if not hints:
        return None
    return max(0.0, max(hints))


def _wait(seconds: float, sleeper, reason: str) -> None:
    """Pause between model calls so the API quota can recover."""
    if seconds <= 0:
        return
    print(f"[ai_strategy_writer] pausing {seconds:g}s {reason}")
    sleeper(seconds)

GENERATED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategies", "generated")
MANIFEST_PATH = os.path.join(GENERATED_DIR, "manifest.json")

ALLOWED_IMPORTS = {"pandas", "numpy", "strategies.base", "strategies"}

FORBIDDEN_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars",
    "exit", "quit", "breakpoint", "help",
}

SAFE_BUILTINS: Dict[str, Any] = {
    "range": range, "len": len, "min": min, "max": max, "sum": sum,
    "abs": abs, "round": round, "enumerate": enumerate, "zip": zip,
    "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "sorted": sorted, "reversed": reversed, "map": map, "filter": filter,
    "isinstance": isinstance, "print": print,
    "Exception": _builtins.Exception,
    "ValueError": _builtins.ValueError, "TypeError": _builtins.TypeError,
    "StopIteration": _builtins.StopIteration,
    "KeyError": _builtins.KeyError, "IndexError": _builtins.IndexError,
    "super": super, "property": property, "staticmethod": staticmethod,
    "__build_class__": _builtins.__build_class__,
}


def _safe_import(name: str, globals: Any = None, locals: Any = None,
                 fromlist: Any = (), level: int = 0) -> Any:
    """Restricted __import__ for the sandboxed exec namespace.

    Only modules in ALLOWED_IMPORTS (plus their submodules, e.g.
    ``strategies.base``) may be imported. Static validation in
    :func:`validate_source` already rejects anything else; this is the
    runtime backstop so ``import pandas as pd`` keeps working while
    ``import os`` fails even if static checks are bypassed.
    """
    top = (name or "").split(".")[0]
    if top not in ALLOWED_IMPORTS:
        raise StrategyGenerationError(f"Disallowed import: {name}")
    return _builtins.__import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS["__import__"] = _safe_import


class StrategyGenerationError(_builtins.Exception):
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


_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
_OPEN_FENCE_RE = re.compile(r"```[a-zA-Z]*[ \t]*\r?\n")
# A line that can only be top-level Python code (not prose like "from the
# data above..."): import/from-import/class/def/decorator.
_CODE_START_RE = re.compile(
    r"^[ \t]*(?:import\s+\w|from\s+[\w.]+\s+import\s|class\s+\w|def\s+\w|@)",
    re.MULTILINE,
)


def _extract_code_block(text: str, class_name: Optional[str] = None) -> str:
    """Pull the Python class out of a model response.

    Lengthy descriptions make models chatty, so this is deliberately defensive:
    the gates must report a REAL code problem, never "syntax error on line 1"
    caused by the model's prose leaking into the extracted code.

    - Several fenced blocks: prefer the one that names `class_name`, else the
      longest (a short intro snippet is never the deliverable).
    - Fence opened but never closed: the response was cut off by the output
      token limit - raise the truncation error so the caller can retry with a
      larger budget, instead of parsing prose + ```python as code.
    - No fences at all: salvage from the first line that can only be top-level
      Python (import/from-import/class/def/decorator) to the end.
    """
    text = (text or "").strip()
    if not text:
        raise StrategyGenerationError("Model returned an empty response.")

    blocks = _FENCED_BLOCK_RE.findall(text)
    if blocks:
        chosen = None
        if class_name:
            named = [b for b in blocks if class_name in b]
            if named:
                chosen = max(named, key=len)
        if chosen is None:
            chosen = max(blocks, key=len)
        return chosen.strip()

    if "```" in text:
        # An opened-but-unclosed fence means the output budget was hit
        # mid-block. Recovering the partial code would just move the syntax
        # error to the truncation point; failing loudly lets the caller retry
        # with a larger maxOutputTokens.
        raise StrategyGenerationError(
            "Model response was cut off before the code block closed "
            "(opening ``` present, closing ``` missing) - likely hit the "
            "output token limit on a lengthy strategy."
        )

    salvage = _CODE_START_RE.search(text)
    if not salvage:
        raise StrategyGenerationError(
            "Model response contained no Python code - no fenced block and no "
            "import/class/def line to salvage. Try rephrasing the description."
        )
    return text[salvage.start():].strip()


def _is_truncation_error(message: str) -> bool:
    """True when the model's response was clipped by the output token limit."""
    lowered = (message or "").lower()
    return ("cut off" in lowered or "max output tokens" in lowered
            or "maxoutputtokens" in lowered)


_REPAIR_PROMPT = """\
The {class_name} class you just wrote was REJECTED by the automated checks below.
Return ONLY the complete corrected class in a single Python code block.

REJECTION REASON (attempt {attempt} of {max_attempts}):
{error}

Requirements when fixing it:
- Fix the cause of that failure. Do not redesign the strategy; keep the user's
  intended entry/exit logic and parameter names.
- The class must still be named exactly `{class_name}` and subclass `Strategy`.
- Allowed imports only: `import pandas as pd`, `import numpy as np`,
  `from strategies.base import Strategy`.
- `__init__` must call `super().__init__(name=..., <same params>)` and store each
  parameter as `self.<param>`, with defaults that work on ~300 daily bars.
- `generate_signals(self, data)` must return a copy of `data` with an added
  integer `signal` column holding only 1, -1 or 0, and must not add or drop rows.
- No lookahead: never use `.shift(-n)` or any future row.
- No file/network I/O, no `os`/`sys`/`subprocess`, no `eval`/`exec`, no `input()`.
- If a parameter default caused the failure (for example a lookback window longer
  than the available data), choose a safe default instead of removing the parameter.
- Keep the class tight so the whole block fits in the output budget: short
  comments, no blank-line padding, no prose. If the previous response was cut
  off, simplify the code (helpers, fewer intermediate prints) - never drop logic.
"""


def _build_contents(user_prompt: str, class_name: str, attempt: int,
                    max_attempts: int, repair_code: Optional[str] = None,
                    repair_error: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build the Gemini `contents` list for a fresh call or a repair call.

    A repair call is a real conversation - original request, the rejected code,
    then the failure - so the model keeps the user's intent while it fixes the
    specific problem instead of regenerating the strategy from scratch.
    """
    contents: List[Dict[str, Any]] = [{"role": "user", "parts": [{"text": user_prompt}]}]
    if repair_code is None or not repair_error:
        return contents
    contents.append({
        "role": "model",
        "parts": [{"text": f"```python\n{repair_code.strip()}\n```"}],
    })
    contents.append({
        "role": "user",
        "parts": [{"text": _REPAIR_PROMPT.format(
            class_name=class_name, attempt=attempt,
            max_attempts=max_attempts, error=repair_error.strip(),
        )}],
    })
    return contents


def call_gemini(description: str, class_name: str, display_name: str,
                 api_key: Optional[str] = None, model: Optional[str] = None,
                 repair_code: Optional[str] = None, repair_error: Optional[str] = None,
                 attempt: int = 1, max_attempts: int = MAX_GENERATION_ATTEMPTS,
                 max_output_tokens: Optional[int] = None) -> str:
    """Ask the model for the strategy class.

    Pass `repair_code` + `repair_error` to ask for a *fix* of a rejected
    attempt rather than a brand-new strategy. `max_output_tokens` overrides the
    default output budget (the generation loop escalates it when a lengthy
    response gets cut off).
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise StrategyGenerationError(
            "GEMINI_API_KEY is not set. Export it in your environment "
            "before using AI strategy generation."
        )
    model = model or os.environ.get("GEMINI_MODEL", GEMINI_MODEL)
    budget = int(max_output_tokens or GEMINI_MAX_OUTPUT_TOKENS)

    system = _SYSTEM_CONTRACT.format(class_name=class_name, display_name=display_name)
    user_prompt = (
        f"Strategy description from the user:\n\"\"\"\n{description.strip()}\n\"\"\"\n\n"
        f"Write the `{class_name}` class implementing this."
    )
    contents = _build_contents(user_prompt, class_name, attempt, max_attempts,
                               repair_code=repair_code, repair_error=repair_error)

    url = GEMINI_API_URL.format(model=model)
    try:
        resp = requests.post(
            url,
            headers={"content-type": "application/json"},
            params={"key": key},
            json={
                "system_instruction": {"parts": [{"text": system}]},
                "contents": contents,
                "generationConfig": {"maxOutputTokens": budget, "temperature": 0.2},
            },
            timeout=60,
        )
        resp.raise_for_status()
    except requests.HTTPError as e:
        body = ""
        try:
            err = resp.json().get("error", {})
            body = str(err.get("message", "") or resp.text)[:600]
        except _builtins.Exception:
            body = (resp.text or "")[:600]
        # Gemini usually puts the back-off in the body ("Please retry in Xs"),
        # but if a Retry-After header is present surface it too so
        # _extract_retry_after_seconds() can honor it.
        header_hint = resp.headers.get("Retry-After") or ""
        if header_hint and "retry" not in body.lower():
            body = f"{body} (retry-after: {header_hint}s)".strip()
        raise StrategyGenerationError(
            f"Gemini API request failed (model={model}, status={resp.status_code}): {body or e}. "
            "Check GEMINI_API_KEY / GEMINI_MODEL and try again."
        )
    except requests.RequestException as e:
        raise StrategyGenerationError(f"Could not reach the Gemini API: {e}")
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
    if finish_reason == "MAX_TOKENS":
        # A response cut off by the output limit still carries partial text, so
        # this must be checked BEFORE the "no text" branch: silently returning
        # truncated code is what made lengthy prompts fail on their first lines
        # (unclosed code fence -> prose parsed as code).
        raise StrategyGenerationError(
            f"Model response was cut off at the {budget}-token output limit "
            f"(finishReason=MAX_TOKENS) before the strategy class finished."
        )
    if not text_parts:
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
        "__name__": "<ai_generated_strategy>",
        "__doc__": None,
        "__package__": None,
        "pd": pd,
        "np": np,
        "Strategy": Strategy,
    }
    try:
        exec(compile(code, "<ai_generated_strategy>", "exec"), module_ns)
    except _builtins.Exception as e:
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
        except _builtins.Exception as e:
            raise StrategyGenerationError(f"Strategy raised an error during the test run: {e}")



def _check_code(code: str, class_name: str) -> Type[Strategy]:
    """Run every acceptance gate on generated code and return the class.

    The three gates are deliberately in cheapest-first order: static AST checks,
    then sandboxed exec, then the synthetic-data dry run. Raises
    StrategyGenerationError with a message that is specific enough to hand
    straight back to the model for a repair round.
    """
    validate_source(code, class_name)
    cls = load_strategy_class(code, class_name)
    dry_run_with_timeout(cls)
    return cls


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


def delete_strategy(slug: str) -> bool:
    """Delete an AI-generated strategy completely: file + manifest entry.

    Returns True if something was deleted, False if the slug was unknown.
    Only touches files inside GENERATED_DIR whose name matches the slug,
    so builtin strategies can never be affected.
    """
    safe_slug = re.sub(r"[^a-z0-9_]", "", slug.strip().lower())
    if not safe_slug or safe_slug != slug.strip().lower():
        return False
    entries = _load_manifest()
    remaining = [e for e in entries if e.get("slug") != safe_slug]
    if len(remaining) == len(entries):
        return False
    _save_manifest(remaining)
    path = os.path.join(GENERATED_DIR, f"{safe_slug}.py")
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
    return True


def generated_slugs() -> set:
    """Slugs currently tracked in the manifest (i.e. deletable)."""
    return {e.get("slug") for e in _load_manifest() if e.get("slug")}


def generate_and_validate_strategy(description: str, display_name: str,
                                    api_key: Optional[str] = None,
                                    max_attempts: Optional[int] = None,
                                    repair_delay: Optional[float] = None,
                                    sleep=None) -> Dict[str, Any]:
    """Generate a strategy class, repairing rejected attempts automatically.

    Pipeline per attempt: generate -> static-check -> load -> dry-run. When an
    attempt fails a gate, the rejected code and the exact failure message are
    sent back to the model for a targeted fix, up to `max_attempts` (default
    MAX_GENERATION_ATTEMPTS). Only an attempt that passes every gate is
    persisted, so a half-broken file can never land on disk.

    Model calls are spaced by `repair_delay` seconds (default
    REPAIR_DELAY_SECONDS) because limited/free API tiers enforce
    requests-per-minute quotas: a repair lands right after a rejected call, and
    firing it immediately is what triggers 429 / quota errors. A throttled
    response is likewise retried after the same pause, up to
    RATE_LIMIT_RETRIES extra calls, while non-retryable errors (bad key, bad
    request) still fail immediately.

    `repair_delay`/`sleep` are injectable so tests can assert the spacing
    without actually waiting.

    Returns a dict:
      ok=True  -> {ok, key, name, code, params_schema, strategy_class,
                   attempts, attempts_used, attempts_allowed}
      ok=False -> {ok, error, code (last attempt), attempts, attempts_used,
                   attempts_allowed}
    Does NOT register into any live registry -- caller does that on ok=True.
    """
    slug = _slugify(display_name)
    class_name = _class_name_from_slug(slug)
    attempts_allowed = max(1, int(max_attempts or MAX_GENERATION_ATTEMPTS))
    delay = REPAIR_DELAY_SECONDS if repair_delay is None else float(repair_delay)
    delay = max(0.0, delay)
    sleeper = sleep or time.sleep
    calls_left = attempts_allowed + RATE_LIMIT_RETRIES

    history: List[Dict[str, Any]] = []
    code = ""
    error = ""
    attempts_made = 0
    # Output budget for model calls. A lengthy description can push the response
    # past the limit, cutting the class off mid-block; on a truncation failure
    # the budget doubles (capped) and the same attempt is retried, instead of
    # burning generation attempts on responses that can never fit.
    token_budget = GEMINI_MAX_OUTPUT_TOKENS
    # Retries are budgeted separately: a cut-off response (needs a bigger
    # output budget) and a throttled call (needs a pause) are different
    # problems, and sharing one pool is what let a single unlucky run - one
    # truncation plus a burst of 429s - exhaust everything before a single
    # complete response came back.
    throttles_left = RATE_LIMIT_RETRIES
    truncations_left = TRUNCATION_RETRIES
    calls_made = 0
    hard_cap = attempts_allowed + RATE_LIMIT_RETRIES + TRUNCATION_RETRIES

    def request(attempt: int, prev_code: Optional[str], prev_error: Optional[str]) -> str:
        """One model call, honoring the API's own back-off hints.

        A truncated response retries with a bigger output budget (own budget),
        a throttled call retries after max(REPAIR_DELAY_SECONDS, the API's
        "Please retry in Xs" hint) - and a hint larger than
        MAX_RETRY_WAIT_SECONDS (daily quota, billing problem) fails fast with a
        clear message instead of freezing the UI for that long.
        """
        nonlocal calls_made, token_budget, throttles_left, truncations_left
        while True:
            if calls_made >= hard_cap:
                raise StrategyGenerationError(
                    "API call budget exhausted "
                    f"({attempts_allowed} attempt(s) + {RATE_LIMIT_RETRIES} throttle retr(ies) "
                    f"+ {TRUNCATION_RETRIES} truncation retr(ies))."
                )
            calls_made += 1
            try:
                return call_gemini(
                    description, class_name, display_name, api_key=api_key,
                    repair_code=prev_code, repair_error=prev_error,
                    attempt=attempt, max_attempts=attempts_allowed,
                    max_output_tokens=token_budget,
                )
            except StrategyGenerationError as e:
                message = str(e)
                if _is_truncation_error(message):
                    if token_budget >= GEMINI_MAX_OUTPUT_TOKEN_CEILING or truncations_left <= 0:
                        raise
                    truncations_left -= 1
                    token_budget = min(GEMINI_MAX_OUTPUT_TOKEN_CEILING, token_budget * 2)
                    history.append({"attempt": attempt, "ok": False,
                                    "stage": "truncated", "error": message})
                    _wait(delay, sleeper,
                          f"before retrying attempt {attempt} with {token_budget} output tokens "
                          "(previous response was cut off)")
                    continue
                if not _is_retryable_api_error(message):
                    raise
                # The API may tell us exactly how long to back off; a fixed
                # 30s pause is what kept re-tripping the free-tier quota.
                hint = _extract_retry_after_seconds(message)
                wait_for = max(delay, hint + 1.0) if hint is not None else delay
                if wait_for > MAX_RETRY_WAIT_SECONDS:
                    raise StrategyGenerationError(
                        "the API quota needs a longer break than the automatic retry "
                        f"window (max {MAX_RETRY_WAIT_SECONDS:g}s per retry; the API asks for "
                        f"about {wait_for:g}s). This is a plan/billing limit, not a code "
                        "problem - wait a few minutes, then press Generate strategy again. "
                        f"Last error: {message}"
                    )
                if throttles_left <= 0:
                    raise StrategyGenerationError(
                        "the API is rate limiting this key and the throttle retry budget is "
                        f"spent ({attempts_allowed} attempt(s) + {RATE_LIMIT_RETRIES} throttle "
                        f"retr(ies) + {TRUNCATION_RETRIES} truncation retr(ies)). "
                        "The free tier allows ~20 requests/minute - wait a minute, then try again. "
                        f"Last error: {message}"
                    )
                throttles_left -= 1
                history.append({"attempt": attempt, "ok": False,
                                "stage": "rate_limit", "error": message})
                reason = (f"before retrying attempt {attempt} (API throttled; it asked to "
                          f"retry in {hint:g}s)" if hint is not None
                          else f"before retrying attempt {attempt} (API throttled)")
                _wait(wait_for, sleeper, reason)

    for attempt in range(1, attempts_allowed + 1):
        if attempt > 1:
            _wait(delay, sleeper, f"before repair attempt {attempt}")
        attempts_made = attempt
        try:
            raw = request(attempt,
                          code if attempt > 1 else None,
                          error if attempt > 1 else None)
        except StrategyGenerationError as e:
            if not history:
                raise  # first call failed outright: surface the request error as-is
            # The earlier attempt's code is still the best thing we have, so
            # report the last gate failure and note why repairing stopped.
            detail = f"attempt {attempt} could not run: {e}"
            error = f"{error} ({detail})" if error else detail
            history.append({"attempt": attempt, "ok": False,
                            "stage": "request", "error": str(e)})
            break

        try:
            # Extraction is inside the gate so a model that forgets the closing
            # ``` fence (or returns prose-only) is repaired like any other
            # failure instead of bubbling out as a server error.
            code = _extract_code_block(raw, class_name)
            cls = _check_code(code, class_name)
        except StrategyGenerationError as e:
            error = str(e)
            history.append({"attempt": attempt, "ok": False,
                            "stage": _failure_stage(error), "error": error})
            continue

        history.append({"attempt": attempt, "ok": True, "stage": "dry_run", "error": None})
        persist_strategy(slug, class_name, display_name, description, code)
        return {
            "ok": True,
            "key": slug,
            "name": display_name,
            "code": code,
            "params_schema": infer_params_schema(cls),
            "strategy_class": cls,
            "attempts": history,
            "attempts_used": attempt,
            "attempts_allowed": attempts_allowed,
        }

    return {
        "ok": False,
        "error": error or "Generation failed.",
        "code": code,
        "attempts": history,
        "attempts_used": attempts_made,
        "attempts_allowed": attempts_allowed,
    }


def _failure_stage(error: str) -> str:
    """Best-effort label for which gate rejected an attempt (for UI display)."""
    lowered = error.lower()
    if _is_truncation_error(lowered) or "code block closed" in lowered:
        return "truncated"
    if "syntax error" in lowered:
        return "syntax"
    if "class named" in lowered or "one class definition" in lowered:
        return "class"
    if "disallowed" in lowered:
        return "safety"
    if "failed to execute" in lowered or "not found or is not a strategy" in lowered:
        return "load"
    if "no python code" in lowered:
        return "empty"
    return "dry_run"


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
            cls = _check_code(code, class_name)
        except StrategyGenerationError as e:
            print(f"[ai_strategy_writer] Skipping '{slug}' on reload: {e}")
            continue
        results.append((slug, cls, infer_params_schema(cls)))
    return results

