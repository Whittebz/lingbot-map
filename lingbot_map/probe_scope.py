"""Opt-in eager profiling scopes; no CUDA work when a probe is not attached."""
from contextlib import nullcontext
from contextvars import ContextVar

_recorder = ContextVar('lingbot_probe_recorder', default=None)


def scope(name):
    recorder = _recorder.get()
    return recorder.span('L2', name) if recorder is not None else nullcontext()
