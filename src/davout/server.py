"""HTTP server exposing the typed-decision API."""
from __future__ import annotations

import hmac
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from davout.backends.base import PromptTooLongError
from davout.schema import ValidationError, parse_request

MAX_BODY_BYTES = 4 * 1024 * 1024

ROUTES: dict[str, str] = {
    "/health": "GET",
    "/v1/models": "GET",
    "/v1/systemone": "POST",
}


class App:
    """Shared server state: engine lifecycle, counters and the engine lock."""

    def __init__(self, max_wait_ms: int = 8000, api_key: str | None = None) -> None:
        self.state = "loading"
        self.engine: Any = None
        self.error = ""
        self.requests = 0
        self.errors = 0
        self.load_seconds = 0.0
        self.lock = threading.Lock()
        self.max_wait_ms = max_wait_ms
        self.api_key = api_key

    def load(self, factory: Callable[[], Any]) -> None:
        """Build the engine with `factory` and flip state to ready or error."""
        start = time.monotonic()
        try:
            engine = factory()
        except BaseException as e:
            self.load_seconds = time.monotonic() - start
            self.error = f"{type(e).__name__}: {e}"
            self.state = "error"
            return
        self.load_seconds = time.monotonic() - start
        self.engine = engine
        self.state = "ready"


class Handler(BaseHTTPRequestHandler):
    server_version = "davout"

    def log_message(self, format: str, *args: Any) -> None:
        return

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, **extra: Any) -> None:
        self._send(status, {"error": message, **extra})

    def _authorized(self) -> bool:
        key = self.app.api_key
        if not key:
            return True
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(token.strip().encode("utf-8"), key.encode("utf-8"))

    def _dispatch(self, method: str) -> None:
        path = self.path.split("?", 1)[0]
        expected = ROUTES.get(path)
        if expected is None:
            self._error(404, "not found")
            return
        if method != expected:
            self._error(405, "method not allowed")
            return
        if path != "/health" and not self._authorized():
            self._error(401, "unauthorized")
            return
        if path == "/health":
            self._health()
        elif path == "/v1/models":
            self._models()
        else:
            self._systemone()

    def _health(self) -> None:
        app = self.app
        if app.state == "loading":
            self._send(503, {"status": "loading"})
        elif app.state == "error":
            self._send(500, {"status": "error", "error": app.error})
        else:
            self._send(200, {
                "status": "ready",
                "load_seconds": app.load_seconds,
                "requests": app.requests,
                "errors": app.errors,
                **app.engine.info(),
            })

    def _models(self) -> None:
        app = self.app
        if app.state != "ready":
            self._error(503, "loading")
            return
        self._send(200, {"data": [{"id": app.engine.info()["model"]}]})

    def _systemone(self) -> None:
        app = self.app
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._error(411, "Content-Length required")
            return
        try:
            length = int(raw_length)
            if length < 0:
                raise ValueError
        except ValueError:
            self._error(400, "invalid Content-Length")
            return
        if length > MAX_BODY_BYTES:
            self._error(413, "request body too large")
            return
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            self._error(400, "invalid JSON")
            return
        if not isinstance(body, dict):
            self._error(400, "body must be a JSON object")
            return
        try:
            request = parse_request(body)
        except ValidationError as e:
            self._error(422, e.message, path=e.path)
            return
        if app.state != "ready":
            self._error(503, "loading")
            return
        if not app.lock.acquire(timeout=app.max_wait_ms / 1000):
            self._error(503, "busy")
            return
        try:
            app.requests += 1
            result = app.engine.answer(request)
            payload = json.dumps(result, allow_nan=False).encode("utf-8")
        except PromptTooLongError as e:
            self._error(422, str(e), path="questions")
            return
        except Exception as e:
            app.errors += 1
            print(f"davout: engine error: {type(e).__name__}", file=sys.stderr)
            self._error(500, "internal error")
            return
        finally:
            app.lock.release()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PUT(self) -> None:
        self._dispatch("PUT")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    app: App


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    """Bind a server for `app`; port 0 picks a free port."""
    server = Server((host, port), Handler)
    server.app = app
    return server


def serve(
    factory: Callable[[], Any],
    host: str = "127.0.0.1",
    port: int = 8766,
    api_key: str | None = None,
    max_wait_ms: int = 8000,
) -> None:
    """Bind, load the engine in the background, and serve until interrupted."""
    app = App(max_wait_ms=max_wait_ms, api_key=api_key or None)
    server = make_server(app, host, port)
    bound_host, bound_port = server.server_address[:2]

    def _load() -> None:
        app.load(factory)
        if app.state == "ready":
            print(f"Davout ready on http://{bound_host}:{bound_port}", flush=True)
        else:
            print(f"davout: engine failed to load: {app.error}", file=sys.stderr, flush=True)

    threading.Thread(target=_load, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
