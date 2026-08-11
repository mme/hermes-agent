"""Security contract for model-invisible AG-UI run private context."""

from __future__ import annotations

import concurrent.futures
import json
import threading

import pytest
from ag_ui.core import RunAgentInput
from fastapi.testclient import TestClient

from agui_adapter.private_context import get_run_private_context
from agui_adapter.server import AGUIEventBridge, AgentConfig, _run_turn, create_app


_PRIVATE_HEADER = "x-test-private-context"


def _body(*, thread_id: str = "t-private", run_id: str = "r-private") -> dict:
    return {
        "threadId": thread_id,
        "runId": run_id,
        "state": {},
        "messages": [{"id": "m1", "role": "user", "content": "hello"}],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def _run_input(*, thread_id: str, run_id: str, message: str) -> RunAgentInput:
    body = _body(thread_id=thread_id, run_id=run_id)
    body["messages"][0]["content"] = message
    return RunAgentInput.model_validate(body)


def _interrupt_id(sse_text: str) -> str:
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        event = json.loads(line[len("data:"):].strip())
        outcome = event.get("outcome")
        if isinstance(outcome, dict) and outcome.get("type") == "interrupt":
            return outcome["interrupts"][0]["id"]
    raise AssertionError("no interrupt outcome in SSE stream")


class _BaseFakeAgent:
    def __init__(self) -> None:
        for attr in (
            "stream_delta_callback",
            "reasoning_callback",
            "tool_progress_callback",
            "step_callback",
            "thinking_callback",
        ):
            setattr(self, attr, None)

    def interrupt(self, *_args) -> None:
        pass


def test_private_context_is_absent_outside_an_agui_run():
    assert get_run_private_context() is None


def test_private_context_resolver_binds_only_the_worker_and_never_serializes(
    monkeypatch, caplog
):
    """The trusted HTTP seam may bind a value for server-side code, but that
    value is not added to model input, provider headers, SSE, health, or logs."""
    import agui_adapter.server as server

    secret = "opaque-token-DO-NOT-LEAK"
    captured = {}

    def resolver(request):
        return {"authorization": request.headers[_PRIVATE_HEADER]}

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            from agui_adapter.session import _CURRENT_STATE

            captured["private"] = get_run_private_context()
            captured["user_message"] = user_message
            captured["history"] = conversation_history
            captured["shared_state"] = _CURRENT_STATE.get().state
            return {"final_response": "safe answer", "messages": []}

    def build_agent(*args, **kwargs):
        captured["default_headers"] = kwargs.get("default_headers") or {}
        captured["frontend_tool_schemas"] = kwargs.get("frontend_tool_schemas") or []
        return _FakeAgent()

    monkeypatch.setattr(server, "build_run_agent", build_agent)

    body = _body()
    body["state"] = {"lesson": "safe-state"}
    body["context"] = [{"description": "safe context", "value": "safe-context"}]
    body["forwardedProps"] = {"safeProp": "safe-forwarded-prop"}
    body["tools"] = [
        {
            "name": "show_safe_card",
            "description": "Render safe client data",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    with TestClient(
        create_app(private_context_resolver=resolver),
        base_url="http://127.0.0.1",
    ) as client:
        health = client.get("/health", headers={_PRIVATE_HEADER: secret})
        response = client.post("/", json=body, headers={_PRIVATE_HEADER: secret})

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert response.status_code == 200
    assert captured["private"] == {"authorization": secret}
    assert get_run_private_context() is None

    assert captured["shared_state"] == {"lesson": "safe-state"}
    assert captured["frontend_tool_schemas"][0]["function"]["name"] == "show_safe_card"
    exposed = repr(
        {
            "user_message": captured["user_message"],
            "history": captured["history"],
            "shared_state": captured["shared_state"],
            "frontend_tool_schemas": captured["frontend_tool_schemas"],
            "forwarded_props": body["forwardedProps"],
            "default_headers": captured["default_headers"],
            "response": response.text,
            "health": health.text,
            "logs": caplog.text,
        }
    )
    assert secret not in exposed


def test_private_context_resolver_failure_is_controlled_and_does_not_log_secret(caplog):
    secret = "resolver-secret-DO-NOT-LEAK"

    def resolver(_request):
        raise RuntimeError(secret)

    with TestClient(
        create_app(private_context_resolver=resolver),
        base_url="http://127.0.0.1",
    ) as client:
        response = client.post("/", json=_body(), headers={_PRIVATE_HEADER: secret})

    assert response.status_code == 401
    assert response.json() == {"detail": "Private run context unavailable."}
    assert secret not in response.text
    assert secret not in caplog.text
    assert get_run_private_context() is None


def test_private_context_resolver_rejects_before_json_body_parsing():
    """A configured resolver may be the embedder's auth boundary. Reject an
    unauthenticated caller before consuming or validating its request body."""

    def resolver(_request):
        raise PermissionError("untrusted resolver detail")

    with TestClient(
        create_app(private_context_resolver=resolver),
        base_url="http://127.0.0.1",
    ) as client:
        response = client.post(
            "/",
            content="{malformed-json",
            headers={"content-type": "application/json"},
        )

    # If JSON parsing ran first this would be 400 instead.
    assert response.status_code == 401
    assert response.json() == {"detail": "Private run context unavailable."}
    assert "untrusted resolver detail" not in response.text
    assert get_run_private_context() is None


def test_successful_resolver_revokes_context_when_body_is_invalid(monkeypatch):
    import agui_adapter.server as server

    created_scopes = []
    original_new_scope = server._new_run_private_context

    def recording_new_scope(value):
        scope = original_new_scope(value)
        created_scopes.append(scope)
        return scope

    monkeypatch.setattr(server, "_new_run_private_context", recording_new_scope)

    with TestClient(
        create_app(private_context_resolver=lambda _request: {"authorization": "opaque"}),
        base_url="http://127.0.0.1",
    ) as client:
        response = client.post(
            "/",
            content="{malformed-json",
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid request body."}
    assert len(created_scopes) == 1
    assert created_scopes[0].get() is None
    assert get_run_private_context() is None


async def _none_async():
    return None


@pytest.mark.parametrize(
    "resolver",
    [lambda _request: None, lambda _request: _none_async()],
    ids=["sync", "async"],
)
def test_configured_resolver_fails_closed_when_no_context_is_returned(resolver):
    with TestClient(
        create_app(private_context_resolver=resolver),
        base_url="http://127.0.0.1",
    ) as client:
        response = client.post("/", json=_body())

    assert response.status_code == 401
    assert response.json() == {"detail": "Private run context unavailable."}


def test_run_turn_isolates_concurrent_contexts_and_resets_each_worker(monkeypatch):
    import agui_adapter.server as server

    barrier = threading.Barrier(2)
    seen = {}
    after = {}
    lock = threading.Lock()

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            barrier.wait(timeout=5)
            with lock:
                seen[user_message] = get_run_private_context()
            return {"final_response": "ok", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())

    def execute(label: str):
        private = {"tenant": label}
        result = _run_turn(
            _run_input(thread_id=f"thread-{label}", run_id=f"run-{label}", message=label),
            AgentConfig(),
            AGUIEventBridge(lambda _event: None),
            {},
            private_context=private,
        )
        after[label] = get_run_private_context()
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, label) for label in ("alpha", "beta")]
        for future in futures:
            future.result(timeout=10)

    assert seen == {
        "alpha": {"tenant": "alpha"},
        "beta": {"tenant": "beta"},
    }
    assert after == {"alpha": None, "beta": None}


def test_http_requests_isolate_private_contexts_end_to_end(monkeypatch):
    import agui_adapter.server as server

    barrier = threading.Barrier(2)
    seen = {}
    lock = threading.Lock()

    def resolver(request):
        return {"authorization": request.headers[_PRIVATE_HEADER]}

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            barrier.wait(timeout=5)
            with lock:
                seen[user_message] = get_run_private_context()
            return {"final_response": "safe answer", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())
    app = create_app(private_context_resolver=resolver)

    def post(label: str) -> str:
        with TestClient(app, base_url="http://127.0.0.1") as client:
            response = client.post(
                "/",
                json={
                    **_body(thread_id=f"thread-{label}", run_id=f"run-{label}"),
                    "messages": [
                        {"id": "m1", "role": "user", "content": label}
                    ],
                },
                headers={_PRIVATE_HEADER: f"secret-{label}"},
            )
        assert response.status_code == 200
        return response.text

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(post, ("alpha", "beta")))

    assert seen == {
        "alpha": {"authorization": "secret-alpha"},
        "beta": {"authorization": "secret-beta"},
    }
    assert all("secret-alpha" not in response for response in responses)
    assert all("secret-beta" not in response for response in responses)
    assert get_run_private_context() is None


def test_private_context_reaches_parallel_server_tool_handler(monkeypatch):
    """Hermes's audited tool-thread wrapper must propagate the run ContextVar
    so a plugin handler can read the same value even in a parallel tool batch."""
    import agui_adapter.server as server
    from tools.registry import registry
    from tools.thread_context import propagate_context_to_thread

    tool_name = "agui_private_context_probe"
    observed = {}

    def handler(_args, **_kwargs):
        observed["handler"] = get_run_private_context()
        return "ok"

    registry.register(
        name=tool_name,
        toolset="agui-private-context-test",
        schema={
            "name": tool_name,
            "description": "test-only private context probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler,
        check_fn=lambda: True,
    )

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(
                    propagate_context_to_thread(
                        lambda: registry.dispatch(tool_name, {})
                    )
                ).result(timeout=5)
            return {"final_response": str(result), "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())
    private = {"authorization": "opaque-value"}
    try:
        _run_turn(
            _run_input(thread_id="thread-tool", run_id="run-tool", message="tool"),
            AgentConfig(),
            AGUIEventBridge(lambda _event: None),
            {},
            private_context=private,
        )
    finally:
        registry.deregister(tool_name)

    assert observed["handler"] == private
    assert get_run_private_context() is None


def test_tool_handler_exception_still_revokes_and_resets_private_context(monkeypatch):
    import agui_adapter.server as server
    from tools.registry import registry

    tool_name = "agui_private_context_failure_probe"
    private = {"authorization": "handler-failure-secret"}
    observed = []

    def handler(_args, **_kwargs):
        observed.append(get_run_private_context())
        raise RuntimeError("controlled handler failure")

    registry.register(
        name=tool_name,
        toolset="agui-private-context-test",
        schema={
            "name": tool_name,
            "description": "test-only failing private context probe",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler,
        check_fn=lambda: True,
    )

    handler_result = []

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            handler_result.append(registry.dispatch(tool_name, {}))
            return {"final_response": "handled", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())
    try:
        _run_turn(
            _run_input(
                thread_id="thread-handler-error",
                run_id="run-handler-error",
                message="tool",
            ),
            AgentConfig(),
            AGUIEventBridge(lambda _event: None),
            {},
            private_context=private,
        )
    finally:
        registry.deregister(tool_name)

    assert observed == [private]
    assert "controlled handler failure" in str(handler_result[0])
    assert get_run_private_context() is None


@pytest.mark.asyncio
async def test_approval_resume_keeps_the_original_private_context(monkeypatch):
    """A resume request reattaches to the parked worker. It must not replace
    that worker's private authority with data resolved for the resume request."""
    import agui_adapter.server as server
    from ag_ui.encoder import EventEncoder

    monkeypatch.setattr(server, "_approval_timeout", lambda: 5.0)
    observed = []

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool

            observed.append(("before-park", get_run_private_context()))
            callback = terminal_tool._get_approval_callback()
            assert callback is not None
            callback("rm -rf build", "test approval")
            observed.append(("after-resume", get_run_private_context()))
            return {"final_response": "done", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())

    run1 = _run_input(thread_id="thread-resume", run_id="run-1", message="approve")
    stream1 = server._event_stream(
        run1,
        EventEncoder(),
        AgentConfig(),
        {},
        private_context={"authorization": "original"},
    )
    first_leg = "".join([frame async for frame in stream1])
    assert '"type":"interrupt"' in first_leg.replace(" ", "")

    resume_body = _body(thread_id="thread-resume", run_id="run-2")
    resume_body["resume"] = [
        {
            "interruptId": _interrupt_id(first_leg),
            "status": "resolved",
            "payload": {"approved": True, "scope": "once"},
        }
    ]
    resumed = RunAgentInput.model_validate(resume_body)
    second_leg = "".join(
        [
            frame
            async for frame in server._event_stream(
                resumed,
                EventEncoder(),
                AgentConfig(),
                {},
                private_context={"authorization": "attacker-controlled-replacement"},
            )
        ]
    )

    assert "RUN_ERROR" not in second_leg
    assert observed == [
        ("before-park", {"authorization": "original"}),
        ("after-resume", {"authorization": "original"}),
    ]
    for worker in [t for t in threading.enumerate() if t.name == "hermes-agui-run"]:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert get_run_private_context() is None


@pytest.mark.asyncio
async def test_immediate_disconnect_after_run_started_revokes_fresh_run(monkeypatch):
    import agui_adapter.server as server
    from ag_ui.encoder import EventEncoder

    entered = threading.Event()
    inspect_after_disconnect = threading.Event()
    allow_worker_to_finish = threading.Event()
    interrupted = threading.Event()
    observed = []

    class _FakeAgent(_BaseFakeAgent):
        def interrupt(self, *_args):
            interrupted.set()

        def run_conversation(self, user_message, conversation_history=None):
            observed.append(("before", get_run_private_context()))
            entered.set()
            assert inspect_after_disconnect.wait(timeout=5)
            observed.append(("after-disconnect", get_run_private_context()))
            assert allow_worker_to_finish.wait(timeout=5)
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())
    stream = server._event_stream(
        _run_input(
            thread_id="thread-immediate-disconnect",
            run_id="run-immediate-disconnect",
            message="disconnect",
        ),
        EventEncoder(),
        AgentConfig(),
        {},
        private_context={"authorization": "revoke-immediately"},
    )

    # Do not consume any queue event after the initial lifecycle frame.
    assert "RUN_STARTED" in await stream.__anext__()
    assert entered.wait(2)
    await stream.aclose()
    assert interrupted.wait(2)

    inspect_after_disconnect.set()
    deadline = threading.Event()
    for _ in range(100):
        if len(observed) == 2:
            break
        deadline.wait(0.01)
    assert observed == [
        ("before", {"authorization": "revoke-immediately"}),
        ("after-disconnect", None),
    ]

    allow_worker_to_finish.set()
    for worker in [t for t in threading.enumerate() if t.name == "hermes-agui-run"]:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert get_run_private_context() is None


@pytest.mark.asyncio
async def test_immediate_disconnect_after_resume_started_denies_and_revokes(monkeypatch):
    import agui_adapter.server as server
    from ag_ui.encoder import EventEncoder

    monkeypatch.setattr(server, "_approval_timeout", lambda: 5.0)
    parked = threading.Event()
    allow_worker_to_finish = threading.Event()
    observed = []

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            from tools import terminal_tool

            observed.append(("before-park", get_run_private_context()))
            callback = terminal_tool._get_approval_callback()
            assert callback is not None
            parked.set()
            decision = callback("rm -rf build", "test approval")
            observed.append(("decision", decision))
            observed.append(("after-resume-disconnect", get_run_private_context()))
            assert allow_worker_to_finish.wait(timeout=5)
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())

    first_run = _run_input(
        thread_id="thread-immediate-resume-disconnect",
        run_id="run-park",
        message="approve",
    )
    first_stream = server._event_stream(
        first_run,
        EventEncoder(),
        AgentConfig(),
        {},
        private_context={"authorization": "original-authority"},
    )
    first_leg = "".join([frame async for frame in first_stream])
    assert parked.wait(2)
    assert '"type":"interrupt"' in first_leg.replace(" ", "")

    resume_body = _body(
        thread_id="thread-immediate-resume-disconnect", run_id="run-resume"
    )
    resume_body["resume"] = [
        {
            "interruptId": _interrupt_id(first_leg),
            "status": "resolved",
            "payload": {"approved": True, "scope": "once"},
        }
    ]
    resume_stream = server._event_stream(
        RunAgentInput.model_validate(resume_body),
        EventEncoder(),
        AgentConfig(),
        {},
        private_context={"authorization": "replacement-must-not-bind"},
    )

    # Closing here happens before resume_to_decision() can approve the command.
    assert "RUN_STARTED" in await resume_stream.__anext__()
    await resume_stream.aclose()

    deadline = threading.Event()
    for _ in range(100):
        if len(observed) == 3:
            break
        deadline.wait(0.01)
    assert observed == [
        ("before-park", {"authorization": "original-authority"}),
        ("decision", "deny"),
        ("after-resume-disconnect", None),
    ]

    allow_worker_to_finish.set()
    for worker in [t for t in threading.enumerate() if t.name == "hermes-agui-run"]:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert get_run_private_context() is None


@pytest.mark.asyncio
async def test_disconnect_resets_private_context_after_worker_termination(monkeypatch):
    import agui_adapter.server as server
    from ag_ui.encoder import EventEncoder

    interrupted = threading.Event()
    release = threading.Event()
    reset_observed = threading.Event()
    observed_after_reset = []
    original_reset = server._reset_run_private_context

    def recording_reset(token):
        original_reset(token)
        observed_after_reset.append(get_run_private_context())
        reset_observed.set()

    monkeypatch.setattr(server, "_reset_run_private_context", recording_reset)

    class _FakeAgent(_BaseFakeAgent):
        def interrupt(self, *_args):
            interrupted.set()
            release.set()

        def run_conversation(self, user_message, conversation_history=None):
            assert get_run_private_context() == {"authorization": "disconnect-secret"}
            self.stream_delta_callback("working")
            release.wait(timeout=5)
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())
    run_input = _run_input(
        thread_id="thread-disconnect", run_id="run-disconnect", message="disconnect"
    )
    stream = server._event_stream(
        run_input,
        EventEncoder(),
        AgentConfig(),
        {},
        private_context={"authorization": "disconnect-secret"},
    )

    assert "RUN_STARTED" in await stream.__anext__()
    while "TEXT_MESSAGE" not in await stream.__anext__():
        pass
    await stream.aclose()

    assert interrupted.wait(2)
    for worker in [t for t in threading.enumerate() if t.name == "hermes-agui-run"]:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert reset_observed.wait(2)
    assert observed_after_reset == [None]
    assert get_run_private_context() is None


@pytest.mark.asyncio
async def test_disconnect_revokes_context_before_noncooperative_worker_stops(monkeypatch):
    import agui_adapter.server as server
    from ag_ui.encoder import EventEncoder

    interrupted = threading.Event()
    allow_worker_to_finish = threading.Event()
    read_after_disconnect = threading.Event()
    observed = []

    class _FakeAgent(_BaseFakeAgent):
        def interrupt(self, *_args):
            interrupted.set()

        def run_conversation(self, user_message, conversation_history=None):
            observed.append(("before", get_run_private_context()))
            self.stream_delta_callback("working")
            assert read_after_disconnect.wait(timeout=5)
            observed.append(("after-disconnect", get_run_private_context()))
            assert allow_worker_to_finish.wait(timeout=5)
            return {"final_response": "", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())
    stream = server._event_stream(
        _run_input(
            thread_id="thread-noncooperative",
            run_id="run-noncooperative",
            message="disconnect",
        ),
        EventEncoder(),
        AgentConfig(),
        {},
        private_context={"authorization": "revoke-me"},
    )

    assert "RUN_STARTED" in await stream.__anext__()
    while "TEXT_MESSAGE" not in await stream.__anext__():
        pass
    await stream.aclose()
    assert interrupted.wait(2)

    # The fake deliberately ignores interrupt and stays alive. Its next read
    # must nevertheless observe revocation immediately.
    read_after_disconnect.set()
    deadline = threading.Event()
    for _ in range(100):
        if len(observed) == 2:
            break
        deadline.wait(0.01)
    assert observed == [
        ("before", {"authorization": "revoke-me"}),
        ("after-disconnect", None),
    ]

    allow_worker_to_finish.set()
    for worker in [t for t in threading.enumerate() if t.name == "hermes-agui-run"]:
        worker.join(timeout=2)
        assert not worker.is_alive()
    assert get_run_private_context() is None


@pytest.mark.parametrize("raises", [False, True], ids=["success", "agent-error"])
def test_run_turn_always_resets_private_context(monkeypatch, raises):
    import agui_adapter.server as server

    secret = {"authorization": "lifecycle-secret"}
    seen = []

    class _FakeAgent(_BaseFakeAgent):
        def run_conversation(self, user_message, conversation_history=None):
            seen.append(get_run_private_context())
            if raises:
                raise RuntimeError("controlled failure")
            return {"final_response": "ok", "messages": []}

    monkeypatch.setattr(server, "build_run_agent", lambda *args, **kwargs: _FakeAgent())

    call = lambda: _run_turn(
        _run_input(thread_id="thread-reset", run_id="run-reset", message="reset"),
        AgentConfig(),
        AGUIEventBridge(lambda _event: None),
        {},
        private_context=secret,
    )

    if raises:
        with pytest.raises(RuntimeError, match="controlled failure"):
            call()
    else:
        call()

    assert seen == [secret]
    assert get_run_private_context() is None
