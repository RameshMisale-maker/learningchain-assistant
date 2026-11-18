"""Per-request stage timings.

A `Timings` object is bound to the current request through a context variable, so deep components
(vector store, BM25, re-ranker, LLM client) can record their own duration without passing it around:

    with request_timings() as t:          # in the service entry point
        ...
        with stage("vector_search_ms"):   # anywhere below it
            store.query(...)
    t.as_dict()  -> {"vector_search_ms": 12.3, ...}

Repeated stages (e.g. one vector search per knowledge base) are summed. Outside a request, `stage()` is a no-op.
"""
from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from typing import Iterator


class Timings:
    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self.started = time.perf_counter()

    def add(self, name: str, ms: float) -> None:
        self._values[name] = self._values.get(name, 0.0) + ms

    def set(self, name: str, ms: float) -> None:
        self._values[name] = ms

    def get(self, name: str, default: float = 0.0) -> float:
        return self._values.get(name, default)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def as_dict(self) -> dict[str, float]:
        return {k: round(v, 2) for k, v in self._values.items()}


_current: contextvars.ContextVar[Timings | None] = contextvars.ContextVar("request_timings", default=None)


@contextmanager
def request_timings() -> Iterator[Timings]:
    timings = Timings()
    token = _current.set(timings)
    try:
        yield timings
    finally:
        _current.reset(token)


def current() -> Timings | None:
    return _current.get()


@contextmanager
def stage(name: str) -> Iterator[None]:
    timings = _current.get()
    started = time.perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            timings.add(name, (time.perf_counter() - started) * 1000)
