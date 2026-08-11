"""Model-invisible, run-scoped private context for AG-UI server tools.

Embedders may resolve opaque authorization or tenancy data from the trusted
HTTP request and bind it to the worker that executes one Hermes run. Server
and plugin tool handlers can read the value with
:func:`get_run_private_context`; it is never added to AG-UI input, model
messages, provider headers, shared state, or outbound events.

The value lives in a revocable context-local scope rather than a process
global. Concurrent runs receive separate scopes, while Hermes's audited
``propagate_context_to_thread`` wrapper copies the same scope into parallel
tool workers. Revoking the scope on disconnect makes subsequent reads return
``None`` even when non-cooperative work has not unwound yet. Outside an active
AG-UI run the getter also returns ``None``.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar, Token
from typing import Any, Optional


class _RunPrivateContext:
    """Thread-safe revocable holder shared by one run's copied contexts."""

    def __init__(self, value: Any) -> None:
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> Any:
        with self._lock:
            return self._value

    def clear(self) -> None:
        with self._lock:
            self._value = None


_RUN_PRIVATE_CONTEXT: ContextVar[Optional[_RunPrivateContext]] = ContextVar(
    "agui_run_private_context", default=None
)


def get_run_private_context() -> Any:
    """Return the opaque private context for the current AG-UI run, if any."""

    scope = _RUN_PRIVATE_CONTEXT.get()
    return scope.get() if scope is not None else None


def _new_run_private_context(value: Any) -> _RunPrivateContext:
    """Create the revocable holder owned by one adapter run."""

    return _RunPrivateContext(value)


def _coerce_run_private_context(value: Any) -> _RunPrivateContext:
    """Return *value* as a scope, wrapping raw embedder/test values once."""

    if isinstance(value, _RunPrivateContext):
        return value
    return _new_run_private_context(value)


def _set_run_private_context(scope: _RunPrivateContext) -> Token:
    """Bind *scope* to the current execution context and return its reset token."""

    return _RUN_PRIVATE_CONTEXT.set(scope)


def _reset_run_private_context(token: Token) -> None:
    """Restore the private context that preceded *token*."""

    _RUN_PRIVATE_CONTEXT.reset(token)
