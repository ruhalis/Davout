"""Tests for the HTTP server and CLI; no torch, no real model."""
from __future__ import annotations

import http.client
import json
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request
from typing import Any, Iterator

import pytest

from davout import cli, server
from davout.backends.base import PromptTooLongError

VALID = {
    "state": "The patient reports a mild headache.",
    "model": "jev-1",
    "questions": {
        "q": {"type": "choice", "instructions": "Pick one.", "criteria": {"a": None, "b": None}},
    },
}
CANNED = {
    "model": "fake",
    "answers": {"q": {"type": "choice", "choice": "a", "probabilities": {"a": 0.7, "b": 0.3}}},
    "usage": {"input_tokens": 3, "output_tokens": 0},
    "timing": {"total_ms": 1.0},
}
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class FakeEngine:
    def __init__(
        self, raises: bool = False, block: threading.Event | None = None, exc: Exception | None = None
    ) -> None:
        self.raises = raises
        self.exc = exc
        self.block = block
        self.entered = threading.Event()

    def answer(self, request: Any) -> dict:
        self.entered.set()
        if self.block is not None:
            self.block.wait(10)
        if self.exc is not None:
            raise self.exc
        if self.raises:
            raise RuntimeError("boom")
        return CANNED

    def info(self) -> dict:
        return {"model": "fake", "backend": "fake"}


class Running:
    def __init__(self, app: server.App) -> None:
        self.app = app
        self.httpd = server.make_server(app, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def call(self, method: str, path: str, body: Any = None, headers: dict | None = None,
             raw: bytes | None = None) -> tuple[int, dict]:
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers=headers or {}
        )
        try:
            with OPENER.open(req, timeout=15) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def make() -> Iterator[Any]:
    started: list[Running] = []

    def _make(engine: Any = None, **kw: Any) -> Running:
        app = server.App(**kw)
        if engine is not None:
            app.load(lambda: engine)
        r = Running(app)
        started.append(r)
        return r

    yield _make
    for r in started:
        r.stop()


def test_health_loading_then_ready(make) -> None:
    r = make()
    assert r.call("GET", "/health") == (503, {"status": "loading"})
    r.app.load(lambda: FakeEngine())
    status, body = r.call("GET", "/health")
    assert status == 200
    assert body["status"] == "ready" and body["model"] == "fake" and body["requests"] == 0


def test_health_error(make) -> None:
    r = make()

    def bad() -> Any:
        raise OSError("no weights")

    r.app.load(bad)
    status, body = r.call("GET", "/health")
    assert status == 500 and body["status"] == "error" and "no weights" in body["error"]


def test_models(make) -> None:
    r = make()
    assert r.call("GET", "/v1/models") == (503, {"error": "loading"})
    r.app.load(lambda: FakeEngine())
    assert r.call("GET", "/v1/models") == (200, {"data": [{"id": "fake"}]})


def test_valid_request(make) -> None:
    r = make(FakeEngine())
    assert r.call("POST", "/v1/systemone", VALID) == (200, CANNED)
    assert r.call("GET", "/health")[1]["requests"] == 1


def test_not_ready_request(make) -> None:
    r = make()
    assert r.call("POST", "/v1/systemone", VALID) == (503, {"error": "loading"})


def test_bad_json(make) -> None:
    r = make(FakeEngine())
    assert r.call("POST", "/v1/systemone", raw=b"{nope")[0] == 400
    assert r.call("POST", "/v1/systemone", raw=b"[1]")[0] == 400


def test_validation_error(make) -> None:
    r = make(FakeEngine())
    bad = json.loads(json.dumps(VALID))
    bad["questions"]["q"]["criteria"] = {"a": None}
    status, body = r.call("POST", "/v1/systemone", bad)
    assert status == 422
    assert body["path"] == "questions.q.criteria" and body["error"]


def test_auth(make) -> None:
    r = make(FakeEngine(), api_key="secret")
    assert r.call("POST", "/v1/systemone", VALID)[0] == 401
    assert r.call("POST", "/v1/systemone", VALID, {"Authorization": "Bearer wrong"})[0] == 401
    assert r.call("POST", "/v1/systemone", VALID, {"Authorization": "Bearer secret"})[0] == 200
    assert r.call("GET", "/v1/models")[0] == 401
    assert r.call("GET", "/health")[0] == 200


def test_not_found_and_method(make) -> None:
    r = make(FakeEngine())
    assert r.call("GET", "/nope") == (404, {"error": "not found"})
    assert r.call("GET", "/v1/systemone")[0] == 405
    assert r.call("POST", "/health", {})[0] == 405


def test_too_large_and_length_required(make) -> None:
    r = make(FakeEngine())
    conn = http.client.HTTPConnection("127.0.0.1", r.port, timeout=10)
    conn.putrequest("POST", "/v1/systemone")
    conn.putheader("Content-Length", str(5 * 1024 * 1024))
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413
    conn.close()

    conn = http.client.HTTPConnection("127.0.0.1", r.port, timeout=10)
    conn.putrequest("POST", "/v1/systemone")
    conn.endheaders()
    assert conn.getresponse().status == 411
    conn.close()


def test_busy(make) -> None:
    gate = threading.Event()
    engine = FakeEngine(block=gate)
    r = make(engine, max_wait_ms=100)
    first: list[tuple[int, dict]] = []
    t = threading.Thread(target=lambda: first.append(r.call("POST", "/v1/systemone", VALID)))
    t.start()
    assert engine.entered.wait(5)
    assert r.call("POST", "/v1/systemone", VALID) == (503, {"error": "busy"})
    assert r.call("GET", "/health")[0] == 200
    gate.set()
    t.join(10)
    assert first[0][0] == 200


def test_engine_error(make) -> None:
    r = make(FakeEngine(raises=True))
    assert r.call("POST", "/v1/systemone", VALID) == (500, {"error": "internal error"})
    assert r.call("GET", "/health")[1]["errors"] == 1


def test_prompt_too_long_is_422(make) -> None:
    r = make(FakeEngine(exc=PromptTooLongError("too long")))
    assert r.call("POST", "/v1/systemone", VALID) == (422, {"error": "too long", "path": "questions"})
    assert r.call("GET", "/health")[1]["errors"] == 0


def test_plain_value_error_is_500(make) -> None:
    r = make(FakeEngine(exc=ValueError("nope")))
    assert r.call("POST", "/v1/systemone", VALID) == (500, {"error": "internal error"})
    assert r.call("GET", "/health")[1]["errors"] == 1


def test_import_is_light() -> None:
    code = (
        "import sys, davout.server, davout.cli\n"
        "bad = [m for m in ('torch', 'transformers', 'davout.engine') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_no_command_returns_2() -> None:
    assert cli.main([]) == 2


def test_cli_ask(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "req.json"
    path.write_text(json.dumps(VALID))
    stub = types.ModuleType("davout.engine")
    stub.build_engine = lambda **kw: FakeEngine()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "davout.engine", stub)
    assert cli.main(["ask", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == CANNED


def test_cli_model_flag_reaches_the_engine(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "req.json"
    path.write_text(json.dumps(VALID))
    built: list[dict] = []
    stub = types.ModuleType("davout.engine")
    stub.build_engine = lambda **kw: built.append(kw) or FakeEngine()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "davout.engine", stub)
    monkeypatch.delenv("DAVOUT_MODEL", raising=False)

    assert cli.main(["ask", str(path)]) == 0
    assert built[-1]["model"] is None  # the stock model
    assert cli.main(["ask", str(path), "--model", "checkpoints/ftA/model"]) == 0
    assert built[-1]["model"] == "checkpoints/ftA/model"

    served: list[Any] = []
    monkeypatch.setattr(server, "serve", lambda factory, **kw: served.append((factory, kw)))
    assert cli.main(["serve", "--model", "checkpoints/ftB/model", "--port", "9"]) == 0
    factory, kw = served[-1]
    assert kw["port"] == 9 and isinstance(factory(), FakeEngine)
    assert built[-1]["model"] == "checkpoints/ftB/model" and built[-1]["device"] == "auto"
    capsys.readouterr()


def test_cli_ask_prompt_too_long(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "req.json"
    path.write_text(json.dumps(VALID))
    stub = types.ModuleType("davout.engine")
    stub.build_engine = lambda **kw: FakeEngine(exc=PromptTooLongError("too long"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "davout.engine", stub)
    assert cli.main(["ask", str(path)]) == 2
    captured = capsys.readouterr()
    assert "too long" in captured.err and captured.out == ""


def test_cli_ask_invalid(tmp_path, capsys) -> None:
    path = tmp_path / "req.json"
    path.write_text(json.dumps({"state": "x"}))
    assert cli.main(["ask", str(path)]) == 2
    assert capsys.readouterr().err.startswith("model:")
