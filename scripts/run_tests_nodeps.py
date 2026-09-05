#!/usr/bin/env python3
"""Run the test suite without pytest installed.

``pytest`` is the supported way to run these tests (``make test``). This
runner exists so the suite is still executable in a bare environment -- a
locked-down CI box, a minimal container, a reviewer's laptop -- because a test
suite you cannot run is not evidence of anything. It implements the small
slice of the pytest API the tests actually use: ``approx``, ``raises``,
``mark.parametrize``, ``skip``, and session ``fixture``.

    python scripts/run_tests_nodeps.py             # all tests
    python scripts/run_tests_nodeps.py test_crc    # one module
"""
from __future__ import annotations

import importlib.util
import inspect
import itertools
import math
import sys
import traceback
import types
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT / "src"))


class Skipped(Exception):
    pass


class _Approx:
    # make numpy defer to us so `ndarray == approx(...)` reaches __eq__ here
    __array_priority__ = 1000
    __array_ufunc__ = None

    def __init__(self, expected: Any, rel: float | None = None, abs: float | None = None) -> None:
        self.expected, self.rel, self.abs = expected, rel, abs

    def _close(self, a: float, b: float) -> bool:
        rel = 1e-6 if self.rel is None else self.rel
        tol = 0.0 if self.abs is None else self.abs
        return math.isclose(a, b, rel_tol=rel, abs_tol=max(tol, 1e-12))

    def __eq__(self, other: Any) -> bool:
        expected = self.expected
        if hasattr(expected, "tolist"):
            expected = expected.tolist()
        if hasattr(other, "tolist"):
            other = other.tolist()
        if isinstance(expected, (list, tuple)):
            if not isinstance(other, (list, tuple)) or len(other) != len(expected):
                return False
            return all(self._close(float(x), float(y)) for x, y in zip(other, expected))
        return self._close(float(other), float(expected))

    def __repr__(self) -> str:
        return f"approx({self.expected})"


class _Raises:
    def __init__(self, exc: type[BaseException] | tuple[type[BaseException], ...]) -> None:
        self.exc = exc

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            raise AssertionError(f"expected {self.exc} to be raised")
        return issubclass(exc_type, self.exc)


def _make_pytest_stub() -> types.ModuleType:
    module = types.ModuleType("pytest")

    def fixture(*args, **kwargs):
        def wrap(fn):
            fn.__aegis_fixture__ = True
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap

    def parametrize(argnames, argvalues, **_):
        names = [n.strip() for n in argnames.split(",")] if isinstance(argnames, str) else list(argnames)

        def wrap(fn):
            cases = getattr(fn, "__aegis_params__", [])
            fn.__aegis_params__ = cases + [(names, list(argvalues))]
            return fn
        return wrap

    mark = types.SimpleNamespace(parametrize=parametrize, slow=lambda fn: fn,
                                 skipif=lambda *a, **k: (lambda fn: fn))
    module.fixture = fixture
    module.mark = mark
    module.approx = lambda expected, rel=None, abs=None: _Approx(expected, rel, abs)
    module.raises = lambda exc, **_: _Raises(exc)

    def skip(reason: str = "", allow_module_level: bool = False) -> None:
        raise Skipped(reason)

    module.skip = skip
    module.importorskip = lambda name, **_: __import__(name)
    return module


sys.modules.setdefault("pytest", _make_pytest_stub())


def _load(path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class FixtureRegistry:
    """Session-scoped fixture resolution with simple dependency injection."""

    def __init__(self, providers: dict[str, Callable[..., Any]]) -> None:
        self.providers = providers
        self.cache: dict[str, Any] = {}

    def get(self, name: str) -> Any:
        if name in self.cache:
            return self.cache[name]
        provider = self.providers.get(name)
        if provider is None:
            raise KeyError(f"no fixture named {name!r}")
        kwargs = {p: self.get(p) for p in inspect.signature(provider).parameters}
        value = provider(**kwargs)
        self.cache[name] = value
        return value


def _expand(fn: Callable[..., Any]) -> list[tuple[str, dict[str, Any]]]:
    """Expand stacked parametrize decorators into concrete keyword sets."""
    stacks = list(reversed(getattr(fn, "__aegis_params__", [])))
    if not stacks:
        return [("", {})]
    out: list[tuple[str, dict[str, Any]]] = []
    grids = [[(names, v) for v in values] for names, values in stacks]
    for combo in itertools.product(*grids):
        kwargs: dict[str, Any] = {}
        labels: list[str] = []
        for names, value in combo:
            values = value if len(names) > 1 else (value,)
            for name, item in zip(names, values):
                kwargs[name] = item
                labels.append(f"{name}={item!r}")
        out.append(("[" + ", ".join(labels) + "]", kwargs))
    return out


def main(argv: list[str]) -> int:
    selected = argv[1:]
    conftest = _load(TESTS / "conftest.py") if (TESTS / "conftest.py").exists() else None
    providers: dict[str, Callable[..., Any]] = {}
    if conftest:
        for name, obj in vars(conftest).items():
            if callable(obj) and getattr(obj, "__aegis_fixture__", False):
                providers[name] = obj

    passed = failed = skipped = 0
    failures: list[str] = []
    for path in sorted(TESTS.glob("test_*.py")):
        if selected and not any(s in path.stem for s in selected):
            continue
        print(f"\n--- {path.name}")
        try:
            module = _load(path)
        except Skipped as exc:
            print(f"  SKIP module: {exc}")
            skipped += 1
            continue
        # module-local fixtures shadow conftest ones, as pytest does
        local = dict(providers)
        for fname, fobj in vars(module).items():
            if callable(fobj) and getattr(fobj, "__aegis_fixture__", False):
                local[fname] = fobj
        registry = FixtureRegistry(local)
        for name, fn in sorted(vars(module).items()):
            if not (name.startswith("test_") and callable(fn)):
                continue
            for label, kwargs in _expand(fn):
                needed = [
                    p for p in inspect.signature(fn).parameters if p not in kwargs
                ]
                try:
                    for p in needed:
                        kwargs[p] = registry.get(p)
                    fn(**kwargs)
                except Skipped as exc:
                    skipped += 1
                    print(f"  skip {name}{label}: {exc}")
                except Exception:
                    failed += 1
                    failures.append(f"{path.name}::{name}{label}")
                    print(f"  FAIL {name}{label}")
                    traceback.print_exc(limit=6)
                else:
                    passed += 1
    print(f"\n{'=' * 62}\npassed={passed} failed={failed} skipped={skipped}")
    if failures:
        print("failures:")
        for item in failures:
            print(f"  - {item}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
