"""Tests for the self-repair loop in AI strategy generation.

Covers the generate -> check -> repair cycle in ai_strategy_writer.py:
a rejected draft must be sent back to the model together with its exact
failure reason, and only a draft that passes every gate may be persisted.

Run: python3 ai_strategy_writer_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ai_strategy_writer as AIW  # noqa: E402

passed, failed = 0, 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        print("  PASS: " + name)
        passed += 1
    except AssertionError as e:
        print("  FAIL: " + name + " -> " + str(e))
        failed += 1


# ----------------------------------------------------------------------
# Fixtures: a fake model, so the tests never touch the network
# ----------------------------------------------------------------------

DISPLAY_NAME = "Unit Test Repair Loop"
SLUG = AIW._slugify(DISPLAY_NAME)
CLASS_NAME = AIW._class_name_from_slug(SLUG)

GOOD_CODE = '''\
import pandas as pd
import numpy as np
from strategies.base import Strategy


class {cls}(Strategy):
    def __init__(self, window: int = 20):
        super().__init__(name="{name}", window=window)
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        ema = df["close"].ewm(span=self.window, adjust=False).mean()
        df["signal"] = 0
        df.loc[df["close"] > ema, "signal"] = 1
        df.loc[df["close"] < ema, "signal"] = -1
        return df
'''

# Passes static checks and loads, but the dry run rejects it (no 'signal' column).
NO_SIGNAL_CODE = '''\
import pandas as pd
from strategies.base import Strategy


class {cls}(Strategy):
    def __init__(self, window: int = 20):
        super().__init__(name="{name}", window=window)
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy()
'''

# Fails the dry run differently: 'signal' holds a value outside {-1, 0, 1}.
BAD_SIGNAL_CODE = '''\
import pandas as pd
from strategies.base import Strategy


class {cls}(Strategy):
    def __init__(self, window: int = 20):
        super().__init__(name="{name}", window=window)
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["signal"] = 2
        return df
'''

# Fails the AST gate: wrong class name.
WRONG_CLASS_CODE = '''\
import pandas as pd
from strategies.base import Strategy


class NotTheClassName(Strategy):
    def __init__(self, window: int = 20):
        super().__init__(name="{name}", window=window)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["signal"] = 0
        return df
'''

# Fails the AST gate: unparseable.
SYNTAX_ERROR_CODE = '''\
import pandas as pd
from strategies.base import Strategy


class {cls}(Strategy):
    def __init__(self, window: int = 20):
        super().__init__(name="{name}", window=window)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["signal"] = ((1
        return df
'''


def code_for(template):
    return template.format(cls=CLASS_NAME, name=DISPLAY_NAME)


# Simulates what real Gemini returns for a LENGTHY prompt that gets clipped by
# the output token limit: prose preamble, an opened ```python fence, partial
# code, and NO closing fence. This is exactly the response shape that used to
# be parsed as code and failed ast.parse on its first lines.
TRUNCATED_RAW = (
    "Sure. Here is a detailed implementation of your multi-condition strategy,\n"
    "with explanations:\n\n"
    "```python\n"
    "import pandas as pd\n"
    "import numpy as np\n"
    "from strategies.base import Strategy\n\n\n"
    "class " + CLASS_NAME + "(Strategy):\n"
    "    def __init__(self, fast: int = 9, slow: int = 21):\n"
    '        super().__init__(name="' + DISPLAY_NAME + '", fast=fast, slow=slow)\n'
)

# A complete response for the same prompt: prose before AND after the block.
PROSE_WRAPPED_GOOD = (
    "Here is the complete class implementing your rules:\n\n"
    "```python\n" + code_for(GOOD_CODE) + "\n```\n\n"
    "Notes: signal is 1 for long entries, -1 for short entries, 0 otherwise."
)


class FakeModel:
    """Stands in for call_gemini: replays scripted responses.

    `fail_calls` holds 1-based *call* numbers that raise (so a 429 on call 1 can
    be followed by a success on call 2, within the same attempt), `always_error`
    makes every call fail, and `raw` returns the responses verbatim (used to
    simulate truncated/prose-wrapped model output like real Gemini sends).
    """

    def __init__(self, responses, fail_calls=None, error=None, always_error=False,
                 raw=False, fail_errors=None):
        self.responses = list(responses)
        self.calls = []
        self.fail_calls = set(fail_calls or [])
        self.error = error or "simulated API failure"
        self.always_error = always_error
        self.raw = raw
        # Per-call error messages, so a replay can mix truncation / 503 / 429s.
        self.fail_errors = dict(fail_errors or {})

    def __call__(self, description, class_name, display_name, **kwargs):
        self.calls.append({
            "description": description,
            "class_name": class_name,
            "repair_code": kwargs.get("repair_code"),
            "repair_error": kwargs.get("repair_error"),
            "attempt": kwargs.get("attempt"),
            "max_attempts": kwargs.get("max_attempts"),
            "max_output_tokens": kwargs.get("max_output_tokens"),
        })
        n = len(self.calls)
        if self.always_error or n in self.fail_calls or n in self.fail_errors:
            raise AIW.StrategyGenerationError(self.fail_errors.get(n, self.error))
        idx = min(n - 1, len(self.responses) - 1)
        if self.raw:
            return self.responses[idx]
        return "```python\n" + self.responses[idx] + "\n```"


def with_fake_model(responses, fn, fail_calls=None, error=None, always_error=False,
                    raw=False, fail_errors=None):
    """Swap call_gemini, run fn(model), then always restore + clean up disk.

    REPAIR_DELAY_SECONDS is forced to 0 so no test ever really sleeps; tests
    that assert the pause pass `repair_delay` + a recording `sleep` explicitly.
    """
    original_call = AIW.call_gemini
    original_delay = AIW.REPAIR_DELAY_SECONDS
    model = FakeModel(responses, fail_calls=fail_calls, error=error,
                      always_error=always_error, raw=raw, fail_errors=fail_errors)
    AIW.call_gemini = model
    AIW.REPAIR_DELAY_SECONDS = 0.0
    try:
        fn(model)
    finally:
        AIW.call_gemini = original_call
        AIW.REPAIR_DELAY_SECONDS = original_delay
        AIW.delete_strategy(SLUG)


def generated_path(slug):
    return os.path.join(AIW.GENERATED_DIR, slug + ".py")


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def t_ok_on_first_attempt():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME)
        assert out["ok"] is True, out
        assert out["attempts_used"] == 1, out["attempts_used"]
        assert len(model.calls) == 1, "must not repair a draft that passed"
        assert model.calls[0]["repair_code"] is None
        assert len(out["attempts"]) == 1 and out["attempts"][0]["ok"] is True
        assert os.path.exists(generated_path(out["key"])), "accepted code must be persisted"
    with_fake_model([code_for(GOOD_CODE)], run)


def t_repairs_dry_run_failure():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME)
        assert out["ok"] is True, out
        assert out["attempts_used"] == 2, out["attempts_used"]
        assert len(model.calls) == 2, len(model.calls)
        # The repair call must carry the rejected code and the exact error.
        repair = model.calls[1]
        assert repair["repair_code"].strip() == code_for(NO_SIGNAL_CODE).strip(), \
            "the full rejected code was not sent back"
        assert "signal" in repair["repair_error"], repair["repair_error"]
        assert repair["attempt"] == 2
        assert repair["max_attempts"] == AIW.MAX_GENERATION_ATTEMPTS
        hist = out["attempts"]
        assert hist[0]["ok"] is False and hist[0]["error"], hist
        assert hist[1]["ok"] is True and hist[1]["error"] is None, hist
        assert out["code"].strip().startswith("import pandas"), "final code must be the repair"
    with_fake_model([code_for(NO_SIGNAL_CODE), code_for(GOOD_CODE)], run)


def t_repairs_wrong_class_name():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME)
        assert out["ok"] is True, out
        assert out["attempts"][0]["stage"] == "class", out["attempts"]
        assert "class named" in model.calls[1]["repair_error"]
    with_fake_model([code_for(WRONG_CLASS_CODE), code_for(GOOD_CODE)], run)


def t_repairs_syntax_error():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME)
        assert out["ok"] is True, out
        assert out["attempts"][0]["stage"] == "syntax", out["attempts"]
    with_fake_model([code_for(SYNTAX_ERROR_CODE), code_for(GOOD_CODE)], run)


def t_repairs_invalid_signal_values():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME)
        assert out["ok"] is True, out
        assert out["attempts"][0]["stage"] == "dry_run", out["attempts"]
        assert "invalid values" in model.calls[1]["repair_error"]
    with_fake_model([code_for(BAD_SIGNAL_CODE), code_for(GOOD_CODE)], run)


def t_gives_up_after_max_attempts():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME, max_attempts=3)
        assert out["ok"] is False, out
        assert out["attempts_used"] == 3, out["attempts_used"]
        assert len(model.calls) == 3, len(model.calls)
        assert len(out["attempts"]) == 3
        assert all(a["ok"] is False for a in out["attempts"])
        assert out["error"], "failure must carry the last error"
        assert out["code"], "failure must carry the last attempt's code"
        assert not os.path.exists(generated_path(SLUG)), "rejected code must never persist"
    with_fake_model([code_for(NO_SIGNAL_CODE)], run)


def t_single_attempt_when_limited():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME, max_attempts=1)
        assert out["ok"] is False, out
        assert len(model.calls) == 1, "max_attempts=1 must not repair"
        assert model.calls[0]["repair_code"] is None
        assert out["attempts_allowed"] == 1
    with_fake_model([code_for(NO_SIGNAL_CODE)], run)


def t_model_failure_on_repair_is_reported():
    def run(model):
        out = AIW.generate_and_validate_strategy("desc", DISPLAY_NAME, max_attempts=3)
        assert out["ok"] is False, out
        assert out["attempts"][-1]["stage"] == "request", out["attempts"]
        assert "could not run" in out["error"], out["error"]
        assert len(model.calls) == 2, "must stop once the API call fails"
        assert not os.path.exists(generated_path(SLUG))
    with_fake_model([code_for(NO_SIGNAL_CODE), code_for(GOOD_CODE)], run, fail_calls={2})


def t_first_attempt_api_failure_raises():
    def run(model):
        raised = False
        try:
            AIW.generate_and_validate_strategy("desc", DISPLAY_NAME)
        except AIW.StrategyGenerationError:
            raised = True
        assert raised, "a failed first call should surface the request error"
        assert len(model.calls) == 1, "a non-retryable error must not be retried"
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1})


def t_throttling_budget_is_bounded():
    """A permanently throttled key must stop, not loop forever."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, max_attempts=1, repair_delay=30,
            sleep=lambda s: waits.append(s))
        assert out["ok"] is False, out
        expected_calls = 1 + AIW.RATE_LIMIT_RETRIES
        assert len(model.calls) == expected_calls, len(model.calls)
        assert "retry budget is spent" in out["error"], out["error"]
        assert out["attempts_used"] == 1, "must not inflate the attempt count"
        assert len(waits) == AIW.RATE_LIMIT_RETRIES, waits
    with_fake_model([code_for(GOOD_CODE)], run, always_error=True,
                    error="RESOURCE_EXHAUSTED: quota exceeded")


def t_non_retryable_error_is_not_retried():
    def run(model):
        waits = []
        raised = False
        try:
            AIW.generate_and_validate_strategy(
                "desc", DISPLAY_NAME, repair_delay=30,
                sleep=lambda s: waits.append(s))
        except AIW.StrategyGenerationError:
            raised = True
        assert raised, "a 400-style error must fail fast"
        assert len(model.calls) == 1, "must not burn quota on a permanent error"
        assert waits == [], waits
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1},
                    error="Gemini API request failed (status=400): invalid argument")


def t_retryable_error_classifier():
    retryable = ["Gemini API request failed (model=x, status=429): quota",
                 "RESOURCE_EXHAUSTED", "Too Many Requests", "rate limit hit",
                 "status=503 service unavailable", "the model is overloaded",
                 "Could not reach the Gemini API: HTTPSConnectionPool timed out"]
    for msg in retryable:
        assert AIW._is_retryable_api_error(msg), msg
    permanent = ["GEMINI_API_KEY is not set.",
                 "Gemini API request failed (status=400): bad request",
                 "Gemini API request failed (status=403): permission denied",
                 "Model declined to respond (reason: SAFETY).",
                 "Model returned no candidates."]
    for msg in permanent:
        assert not AIW._is_retryable_api_error(msg), msg


def t_pause_before_each_repair():
    """The repair call must wait REPAIR_DELAY_SECONDS - that is the whole point."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30,
            sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert out["attempts_used"] == 2
        assert waits == [30], "one repair should wait exactly once: %r" % (waits,)
    with_fake_model([code_for(SYNTAX_ERROR_CODE), code_for(GOOD_CODE)], run)

    def run_all_fail(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, max_attempts=3, repair_delay=45,
            sleep=lambda s: waits.append(s))
        assert out["ok"] is False
        assert waits == [45, 45], "two repairs should each wait: %r" % (waits,)
        assert len(model.calls) == 3
    with_fake_model([code_for(SYNTAX_ERROR_CODE)], run_all_fail)


def t_no_pause_when_first_draft_passes():
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=60,
            sleep=lambda s: waits.append(s))
        assert out["ok"] is True
        assert waits == [], "must not pause if nothing needs repairing: %r" % (waits,)
    with_fake_model([code_for(GOOD_CODE)], run)


def t_default_delay_is_within_30_60():
    assert 30 <= AIW.REPAIR_DELAY_SECONDS <= 60, AIW.REPAIR_DELAY_SECONDS
    assert AIW._env_float("AI_STRATEGY_NOPE", 30.0, 0.0, 300.0) == 30.0
    os.environ["AI_STRATEGY_TEST_FLOAT"] = "abc"
    try:
        assert AIW._env_float("AI_STRATEGY_TEST_FLOAT", 30.0, 0.0, 300.0) == 30.0
    finally:
        del os.environ["AI_STRATEGY_TEST_FLOAT"]
    os.environ["AI_STRATEGY_TEST_FLOAT"] = "9999"
    try:
        assert AIW._env_float("AI_STRATEGY_TEST_FLOAT", 30.0, 0.0, 300.0) == 300.0
    finally:
        del os.environ["AI_STRATEGY_TEST_FLOAT"]


def t_throttled_call_is_retried_after_pause():
    """A 429 on the first call must not kill the run - wait, then retry."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30,
            sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert out["attempts_used"] == 1, "a retry is not a new attempt"
        assert len(model.calls) == 2, "call 1 throttled, call 2 succeeded"
        assert waits == [30], "retry must wait first: %r" % (waits,)
        assert out["attempts"][0]["stage"] == "rate_limit", out["attempts"]
        assert "429" in out["attempts"][0]["error"]
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1},
                    error="Gemini API request failed (status=429): quota")


def t_rate_limit_then_repair_still_works():
    """Throttled first call, throttled repair, then a good repair."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30,
            sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert len(model.calls) == 4, "call1 429, call2 bad code, call3 429, call4 good"
        assert [a["stage"] for a in out["attempts"]] == \
            ["rate_limit", "syntax", "rate_limit", "dry_run"], out["attempts"]
        assert waits == [30, 30, 30], waits
    with_fake_model([code_for(SYNTAX_ERROR_CODE), code_for(SYNTAX_ERROR_CODE),
                     code_for(GOOD_CODE)], run,
                    fail_calls={1, 3}, error="Too Many Requests (429)")


def t_extract_takes_fenced_block_not_prose():
    out = AIW._extract_code_block(PROSE_WRAPPED_GOOD, CLASS_NAME)
    assert out == code_for(GOOD_CODE).strip(), out[:120]
    assert "Here is the complete" not in out
    assert "Notes: signal" not in out


def t_extract_prefers_class_block_or_longest():
    small = "```python\nimport pandas as pd\n```"
    big = "```python\n" + code_for(GOOD_CODE) + "\n```"
    # with the expected class name, the matching block wins...
    assert "class " + CLASS_NAME in AIW._extract_code_block(small + "\n\n" + big, CLASS_NAME)
    # ...and without a hint, the longest block wins (a snippet is never the deliverable)
    assert "class " + CLASS_NAME in AIW._extract_code_block(small + "\n\n" + big)


def t_extract_unclosed_fence_reports_truncation():
    try:
        AIW._extract_code_block(TRUNCATED_RAW, CLASS_NAME)
    except AIW.StrategyGenerationError as e:
        assert "cut off" in str(e), str(e)
    else:
        raise AssertionError("a truncated response must never be parsed as code")


def t_extract_salvages_unfenced_code():
    raw = "Of course. Here is the class implementing your rules:\n\n" + code_for(GOOD_CODE)
    out = AIW._extract_code_block(raw, CLASS_NAME)
    assert out.startswith(("import pandas as pd", "from strategies.base import Strategy")), out[:80]
    assert "Of course" not in out
    assert "class " + CLASS_NAME in out


def t_extract_rejects_prose_only_response():
    try:
        AIW._extract_code_block("I cannot help with that request.", CLASS_NAME)
    except AIW.StrategyGenerationError:
        pass
    else:
        raise AssertionError("a prose-only response must be rejected, not parsed")


def t_truncation_escalates_output_budget():
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30, sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert out["attempts_used"] == 1, "a truncation retry is not a new attempt"
        assert len(model.calls) == 2, "call 1 cut off, call 2 with a bigger budget"
        first, second = model.calls
        assert first["max_output_tokens"] == AIW.GEMINI_MAX_OUTPUT_TOKENS
        assert second["max_output_tokens"] == min(
            AIW.GEMINI_MAX_OUTPUT_TOKEN_CEILING, AIW.GEMINI_MAX_OUTPUT_TOKENS * 2)
        assert waits == [30], waits
        assert out["attempts"][0]["stage"] == "truncated", out["attempts"]
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1},
                    error="Model response was cut off at the 4096-token output limit "
                          "(finishReason=MAX_TOKENS) before the strategy class finished.")


def t_truncation_ceiling_is_respected():
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, max_attempts=1, repair_delay=30,
            sleep=lambda s: waits.append(s))
        assert out["ok"] is False, out
        assert len(model.calls) == 2, "4096 -> 8192 (ceiling), then give up"
        assert model.calls[0]["max_output_tokens"] == AIW.GEMINI_MAX_OUTPUT_TOKENS
        assert model.calls[1]["max_output_tokens"] == AIW.GEMINI_MAX_OUTPUT_TOKEN_CEILING
        assert waits == [30], waits
        assert "cut off" in out["error"], out["error"]
        assert out["attempts_used"] == 1, "must not inflate the attempt count"
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1, 2},
                    error="Model response was cut off (finishReason=MAX_TOKENS).")


def t_truncation_classifier():
    for msg in ["Model response was cut off at the 4096-token output limit",
                "hit max output tokens", "raise maxOutputTokens"]:
        assert AIW._is_truncation_error(msg), msg
    for msg in ["429 RESOURCE_EXHAUSTED: quota exceeded", "GEMINI_API_KEY is not set.",
                "Generated code has a syntax error: invalid syntax"]:
        assert not AIW._is_truncation_error(msg), msg


def t_lengthy_prompt_max_tokens_escalates_within_attempt():
    """The reported bug, API path: a lengthy prompt hits the output limit and
    Gemini returns finishReason=MAX_TOKENS. The SAME attempt retries with a
    doubled budget instead of burning a generation attempt."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30, sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert out["attempts_used"] == 1, "a truncation retry is not a new attempt"
        assert [a["stage"] for a in out["attempts"]] == ["truncated", "dry_run"], out["attempts"]
        assert len(model.calls) == 2
        assert model.calls[1]["max_output_tokens"] > model.calls[0]["max_output_tokens"]
        assert waits == [30], waits
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1},
                    error="Model response was cut off at the 4096-token output limit "
                          "(finishReason=MAX_TOKENS) before the strategy class finished.")


def t_lengthy_prompt_unclosed_fence_repairs():
    """The reported bug, extraction path: the model stops early with an unclosed
    ``` fence (no MAX_TOKENS). The prose must never reach ast.parse - the
    response goes to the repair loop with the real reason, and the persisted
    file is the class, not the prose."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30, sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert out["attempts_used"] == 2, out["attempts_used"]
        assert [a["stage"] for a in out["attempts"]] == ["truncated", "dry_run"], out["attempts"]
        assert len(model.calls) == 2
        # No budget escalation here: the API did not report MAX_TOKENS, so the
        # budgets stay at the default and the pause is the ordinary repair gap.
        assert all(c["max_output_tokens"] == AIW.GEMINI_MAX_OUTPUT_TOKENS
                   for c in model.calls), [c["max_output_tokens"] for c in model.calls]
        assert waits == [30], waits
        # The persisted file is the real class - never the model's prose.
        code = open(generated_path(out["key"])).read()
        assert "class " + CLASS_NAME in code
        assert "Here is the complete" not in code
        assert "```" not in code
    with_fake_model([TRUNCATED_RAW, PROSE_WRAPPED_GOOD], run, raw=True)


THROTTLE_503 = ("Gemini API request failed (model=gemini-3.6-flash, status=503): "
                "This model is currently experiencing high demand. Please try again later.")
THROTTLE_429_EARLY = ("Gemini API request failed (model=gemini-3.6-flash, status=429): "
                      "You exceeded your current quota. "
                      "Please retry in 3.086561023s.")
# The back-off Gemini itself asked for in the reported failure.
THROTTLE_429_LATE_HINT = 32.619841575
THROTTLE_429_LATE = ("Gemini API request failed (model=gemini-3.6-flash, status=429): "
                     "You exceeded your current quota. "
                     f"Please retry in {THROTTLE_429_LATE_HINT}s.")
TRUNCATION_MSG = ("Model response was cut off at the 4096-token output limit "
                  "(finishReason=MAX_TOKENS) before the strategy class finished.")


def t_retry_after_hint_is_honored():
    """The API said 'Please retry in 32.6s' - waiting 30s tripped the quota
    again. The pause must now be max(REPAIR_DELAY, hint + 1s)."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30, sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert waits == [THROTTLE_429_LATE_HINT + 1.0], waits
        assert out["attempts_used"] == 1
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1}, error=THROTTLE_429_LATE)


def t_long_retry_after_fails_fast():
    """A daily-quota 429 asks for minutes - never freeze the UI waiting for it."""
    def run(model):
        waits = []
        raised = False
        try:
            AIW.generate_and_validate_strategy(
                "desc", DISPLAY_NAME, repair_delay=30, sleep=lambda s: waits.append(s))
        except AIW.StrategyGenerationError as e:
            raised = True
            assert "longer break than the automatic retry window" in str(e), str(e)
        assert raised, "a minutes-long back-off must fail fast with a clear message"
        assert waits == [], waits
        assert len(model.calls) == 1, "no retry when the API asks for minutes"
    with_fake_model([code_for(GOOD_CODE)], run, fail_calls={1},
                    error="Gemini API request failed (status=429): quota. "
                          "Please retry in 1800s.")


def t_retry_after_parser():
    cases = {
        "Please retry in 32.619841575s.": 32.619841575,
        "Please retry in 1.957322531s... Check GEMINI_API_KEY": 1.957322531,
        'quota exceeded (retryDelay": "32s")': 32.0,
        "throttled (retry-after: 30s)": 30.0,
        "no hint here at all": None,
        "": None,
    }
    for msg, expected in cases.items():
        got = AIW._extract_retry_after_seconds(msg)
        if expected is None:
            assert got is None, (msg, got)
        else:
            assert got is not None and abs(got - expected) < 1e-9, (msg, got)


def t_truncation_and_throttle_budgets_are_separate():
    """A truncation retry must not eat the throttle retries (that shared pool is
    what killed the reported run after 5 calls)."""
    def run(model):
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=0, sleep=lambda s: None)
        assert out["ok"] is True, out
        assert out["attempts_used"] == 1, out["attempts_used"]
        assert len(model.calls) == 4, len(model.calls)  # trunc, 429, 429, success
        assert [a["stage"] for a in out["attempts"]] == \
            ["truncated", "rate_limit", "rate_limit", "dry_run"], out["attempts"]
    with_fake_model([code_for(GOOD_CODE)], run,
                    fail_errors={1: TRUNCATION_MSG, 2: THROTTLE_429_EARLY,
                                 3: THROTTLE_429_LATE})


def t_reported_throttle_storm_now_recovers():
    """Replay of the exact failing run from the bug report:
    truncation at 4096 -> 503 -> 429 (retry 3.1s) -> 429 (retry 32.6s) -> dead.
    With separated budgets and honored back-off hints it now completes."""
    def run(model):
        waits = []
        out = AIW.generate_and_validate_strategy(
            "desc", DISPLAY_NAME, repair_delay=30, sleep=lambda s: waits.append(s))
        assert out["ok"] is True, out
        assert out["attempts_used"] == 1, out["attempts_used"]
        stages = [a["stage"] for a in out["attempts"]]
        assert stages == ["truncated", "rate_limit", "rate_limit", "rate_limit",
                          "dry_run"], stages
        # escalation wait, 503 (no hint -> 30), 429 (3.1s -> floor 30),
        # 429 (32.6s -> honored)
        assert waits == [30, 30, 30, THROTTLE_429_LATE_HINT + 1.0], waits
        assert len(model.calls) == 5, len(model.calls)
        assert model.calls[-1]["max_output_tokens"] == AIW.GEMINI_MAX_OUTPUT_TOKEN_CEILING
    with_fake_model([code_for(GOOD_CODE)], run,
                    fail_errors={1: TRUNCATION_MSG, 2: THROTTLE_503,
                                 3: THROTTLE_429_EARLY, 4: THROTTLE_429_LATE})


def t_repair_prompt_carries_contract():
    prompt = AIW._REPAIR_PROMPT.format(class_name=CLASS_NAME, attempt=2,
                                       max_attempts=3, error="boom")
    assert CLASS_NAME in prompt and "boom" in prompt
    assert "shift(-n)" in prompt, "repair prompt must restate the no-lookahead rule"
    contents = AIW._build_contents("user request", CLASS_NAME, 2, 3,
                                   repair_code="X = 1", repair_error="boom")
    assert [c["role"] for c in contents] == ["user", "model", "user"], contents
    assert contents[1]["parts"][0]["text"].startswith("```python")
    assert AIW._build_contents("u", CLASS_NAME, 1, 3) == [
        {"role": "user", "parts": [{"text": "u"}]}
    ]


def t_check_code_gates():
    cls = AIW._check_code(code_for(GOOD_CODE), CLASS_NAME)
    assert cls.__name__ == CLASS_NAME
    cases = ((code_for(NO_SIGNAL_CODE), "signal"),
             (code_for(WRONG_CLASS_CODE), "class named"),
             (code_for(SYNTAX_ERROR_CODE), "syntax"))
    for bad, needle in cases:
        try:
            AIW._check_code(bad, CLASS_NAME)
        except AIW.StrategyGenerationError as e:
            assert needle in str(e), str(e)
        else:
            raise AssertionError("expected rejection for: " + needle)


def t_max_attempts_env_clamped():
    assert AIW.MAX_GENERATION_ATTEMPTS >= 1
    assert AIW._env_int("AI_STRATEGY_NOPE", 3, 1, 6) == 3
    os.environ["AI_STRATEGY_TEST_INT"] = "not-a-number"
    try:
        assert AIW._env_int("AI_STRATEGY_TEST_INT", 4, 1, 6) == 4
    finally:
        del os.environ["AI_STRATEGY_TEST_INT"]
    os.environ["AI_STRATEGY_TEST_INT"] = "99"
    try:
        assert AIW._env_int("AI_STRATEGY_TEST_INT", 4, 1, 6) == 6, "must clamp to hi"
    finally:
        del os.environ["AI_STRATEGY_TEST_INT"]


if __name__ == "__main__":
    for n, fn in [("ok on first attempt (no repair)", t_ok_on_first_attempt),
                  ("repairs a dry-run failure", t_repairs_dry_run_failure),
                  ("repairs a wrong class name", t_repairs_wrong_class_name),
                  ("repairs a syntax error", t_repairs_syntax_error),
                  ("repairs invalid signal values", t_repairs_invalid_signal_values),
                  ("gives up after max attempts", t_gives_up_after_max_attempts),
                  ("max_attempts=1 does not repair", t_single_attempt_when_limited),
                  ("API failure during repair is reported", t_model_failure_on_repair_is_reported),
                  ("API failure on first attempt raises", t_first_attempt_api_failure_raises),
                  ("throttled call is retried after pause", t_throttled_call_is_retried_after_pause),
                  ("throttling budget is bounded", t_throttling_budget_is_bounded),
                  ("non-retryable error is not retried", t_non_retryable_error_is_not_retried),
                  ("retryable-error classifier", t_retryable_error_classifier),
                  ("pauses before each repair", t_pause_before_each_repair),
                  ("no pause when the first draft passes", t_no_pause_when_first_draft_passes),
                  ("default delay sits in 30-60s", t_default_delay_is_within_30_60),
                  ("throttle then repair still works", t_rate_limit_then_repair_still_works),
                  ("repair prompt carries the contract", t_repair_prompt_carries_contract),
                  ("extractor takes the fenced block, not prose", t_extract_takes_fenced_block_not_prose),
                  ("extractor prefers the class block / longest", t_extract_prefers_class_block_or_longest),
                  ("unclosed fence reported as truncation", t_extract_unclosed_fence_reports_truncation),
                  ("unfenced code salvaged from prose", t_extract_salvages_unfenced_code),
                  ("prose-only response rejected", t_extract_rejects_prose_only_response),
                  ("truncation escalates the output budget", t_truncation_escalates_output_budget),
                  ("truncation ceiling is respected", t_truncation_ceiling_is_respected),
                  ("truncation classifier", t_truncation_classifier),
                  ("lengthy prompt: MAX_TOKENS escalates in-attempt", t_lengthy_prompt_max_tokens_escalates_within_attempt),
                  ("lengthy prompt: unclosed fence is repaired", t_lengthy_prompt_unclosed_fence_repairs),
                  ("throttle budgets are separate", t_truncation_and_throttle_budgets_are_separate),
                  ("retry-after hint is honored", t_retry_after_hint_is_honored),
                  ("long retry-after fails fast", t_long_retry_after_fails_fast),
                  ("retry-after parser", t_retry_after_parser),
                  ("reported throttle storm now recovers", t_reported_throttle_storm_now_recovers),
                  ("_check_code gates reject bad drafts", t_check_code_gates),
                  ("attempt limit env parsing", t_max_attempts_env_clamped)]:
        check(n, fn)
    print("AI STRATEGY REPAIR LOOP: %d passed, %d failed" % (passed, failed))
    sys.exit(1 if failed else 0)