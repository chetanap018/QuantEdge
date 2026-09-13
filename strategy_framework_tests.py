"""Runner for Fix 3 strategy-framework tests."""
import fw_event_tests as E
import fw_mtf_tests as M
import fw_opt_tests as O
import fw_combo_tests as C

CASES = []
for mod in (E, M, O, C):
    for name in dir(mod):
        if name.startswith("Test"):
            cls = getattr(mod, name)
            for meth in dir(cls):
                if meth.startswith("test_"):
                    CASES.append((f"{name}.{meth}", cls, meth))

passed = failed = 0
for label, cls, meth in CASES:
    try:
        getattr(cls(), meth)()
        print(f"PASS {label}")
        passed += 1
    except Exception as e:
        print(f"FAIL {label}: {type(e).__name__}: {e}")
        failed += 1
print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
raise SystemExit(1 if failed else 0)
