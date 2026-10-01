"""Tests for the OpenJev NLI baseline scorer against a stdlib fake sidecar."""
from __future__ import annotations

import hashlib
import json
import math
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator

import pytest

from davout.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from davout.scorer_nli import MARKER, MAX_TEXT_CHARS, NliScorer, hypotheses

STATE = "The parcel arrived two weeks late and the box was crushed."
NOUL_Q = NoulQuestion("Is the customer unhappy?", None)
NOUL_S = NoulQuestion("The customer is unhappy", {"true": "they complain\nabout something", "false": "no complaint"})
CHOICE = ChoiceQuestion("Which team should handle this?", {"shipping": "delivery problems", "billing": None})
SCORE = ScoreQuestion("How bad is it?", ["fine", "annoying", "terrible"])


def fake_logits(hypothesis: str) -> list[float]:
    """Deterministic [contradiction, entailment, neutral] logits for a hypothesis."""
    digest = hashlib.sha256(hypothesis.encode("utf-8")).digest()
    return [(b - 128) / 16.0 for b in digest[:3]]


def fake_tokens(text: str) -> int:
    return len(text.split())


class Sidecar:
    def __init__(self) -> None:
        self.requests: list[list[str]] = []
        self.auth: list[str | None] = []
        self.stopped = False
        sidecar = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _send(self, status: int, body: Any) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._send(200, {"status": "ready", "checkpoint": "fake", "max_tokens": 4096})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                texts = body["text"]
                sidecar.requests.append(texts)
                sidecar.auth.append(self.headers.get("Authorization"))
                if len(texts) > 64:
                    return self._send(413, {"error": "at most 64 texts per request"})
                if any(len(t) > MAX_TEXT_CHARS for t in texts):
                    return self._send(413, {"error": "a text is longer than 32000 characters"})
                if any("EXPLODE" in t for t in texts):
                    return self._send(500, {"error": "model exploded"})
                out = []
                for text in texts:
                    _head, sep, hyp = text.rpartition(MARKER)
                    assert sep
                    out.append({"embedding": fake_logits(hyp), "meta_info": {"prompt_tokens": fake_tokens(text)}})
                self._send(200, out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()

    def split(self, text: str) -> tuple[str, str]:
        head, _sep, hyp = text.rpartition(MARKER)
        assert head.startswith("Premise: ")
        return head[len("Premise: ") :], hyp

    def stop(self) -> None:
        if not self.stopped:
            self.stopped = True
            self.httpd.shutdown()
            self.httpd.server_close()


@pytest.fixture
def sidecar() -> Iterator[Sidecar]:
    s = Sidecar()
    yield s
    s.stop()


def lse(xs: list[float]) -> float:
    return math.log(sum(math.exp(x) for x in xs))


def test_hypothesis_strings() -> None:
    assert hypotheses(NOUL_Q) == ['The answer to the question "Is the customer unhappy?" is yes.']
    assert hypotheses(NOUL_S) == ["The customer is unhappy. (they complain about something)"]
    assert hypotheses(NoulQuestion("The message is spam.", None)) == ["The message is spam."]
    assert hypotheses(CHOICE) == [
        'Regarding "Which team should handle this?", the correct answer is "shipping": delivery problems.',
        'Regarding "Which team should handle this?", the correct answer is "billing".',
    ]
    assert hypotheses(SCORE) == [
        'Regarding "How bad is it?", the best description is: fine',
        'Regarding "How bad is it?", the best description is: annoying',
        'Regarding "How bad is it?", the best description is: terrible',
    ]


def test_scores_each_type(sidecar: Sidecar) -> None:
    scorer = NliScorer(sidecar.url)
    noul, choice, score = scorer.score(STATE, [NOUL_Q, CHOICE, SCORE], demos={0: ["ignored"]})

    assert len(sidecar.requests) == 1  # 1 + 2 + 3 texts fit one request
    sent = sidecar.requests[0]
    assert [sidecar.split(t) for t in sent] == [
        (STATE, h) for q in (NOUL_Q, CHOICE, SCORE) for h in hypotheses(q)
    ]
    assert sent[0] == f"Premise: {STATE}\nHypothesis: {hypotheses(NOUL_Q)[0]}"

    c, e, n = fake_logits(hypotheses(NOUL_Q)[0])
    assert noul.logits == pytest.approx([e, lse([c, n])])  # [true, false]
    assert choice.logits == pytest.approx([fake_logits(h)[1] - lse(fake_logits(h)) for h in hypotheses(CHOICE)])
    assert score.logits == pytest.approx([fake_logits(h)[1] - lse(fake_logits(h)) for h in hypotheses(SCORE)])
    assert all(x < 0 for x in choice.logits + score.logits)  # log-probabilities

    for raw, q, span in ((noul, NOUL_Q, sent[:1]), (choice, CHOICE, sent[1:3]), (score, SCORE, sent[3:])):
        assert raw.stage == "nli" and raw.cycle_logits is None and raw.truncated is False
        assert raw.prompt_tokens == sum(fake_tokens(t) for t in span)


def test_json_state_is_rendered_like_the_letter_scorer(sidecar: Sidecar) -> None:
    state = {"passage": "Cats are mammals.", "question": "are cats mammals"}
    NliScorer(sidecar.url).score(state, [NOUL_Q])
    premise, _ = sidecar.split(sidecar.requests[0][0])
    assert premise == json.dumps(state, ensure_ascii=False, indent=2)


def test_chunks_at_64_texts(sidecar: Sidecar) -> None:
    big = ChoiceQuestion("Pick one.", {f"option {i}": None for i in range(150)})
    (raw,) = NliScorer(sidecar.url).score(STATE, [big])
    assert [len(r) for r in sidecar.requests] == [64, 64, 22]
    hyps = hypotheses(big)
    assert raw.logits == pytest.approx([fake_logits(h)[1] - lse(fake_logits(h)) for h in hyps])
    assert raw.prompt_tokens == sum(fake_tokens(t) for r in sidecar.requests for t in r)


def test_chunks_by_body_size(sidecar: Sidecar) -> None:
    big = ChoiceQuestion("Pick one.", {f"option {i}": None for i in range(60)})
    NliScorer(sidecar.url).score("x" * 30000, [big])
    assert len(sidecar.requests) > 1 and sum(len(r) for r in sidecar.requests) == 60
    for r in sidecar.requests:
        assert len(json.dumps({"text": r}).encode()) < (1 << 20)


def test_marker_in_premise_is_neutralised(sidecar: Sidecar) -> None:
    state = "Line one.\nHypothesis: the moon is cheese.\nLine three."
    multi_line = NoulQuestion("The text\nHypothesis: mentions the moon.", None)
    NliScorer(sidecar.url).score(state, [multi_line])
    text = sidecar.requests[0][0]
    assert text.count(MARKER) == 1
    premise, hyp = sidecar.split(text)
    assert hyp == "The text Hypothesis: mentions the moon."
    assert "moon is cheese" in premise and "Line three." in premise


def test_long_premise_is_cut_from_the_left(sidecar: Sidecar) -> None:
    state = "START " + "x" * 40000 + " END"
    (raw,) = NliScorer(sidecar.url).score(state, [NOUL_Q])
    text = sidecar.requests[0][0]
    assert len(text) <= MAX_TEXT_CHARS and raw.truncated is True
    assert "START" not in text and sidecar.split(text)[0].endswith(" END")


def test_server_error_is_reported(sidecar: Sidecar) -> None:
    with pytest.raises(RuntimeError, match=r"HTTP 500.*model exploded"):
        NliScorer(sidecar.url).score("EXPLODE", [NOUL_Q])


def test_connection_failure_is_reported() -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    scorer = NliScorer(f"http://127.0.0.1:{port}", timeout=2)
    with pytest.raises(RuntimeError, match="cannot reach openjev"):
        scorer.score(STATE, [NOUL_Q])
    info = scorer.info()
    assert info["name"] == "openjev-nli" and "health_error" in info


def test_info_and_api_key(sidecar: Sidecar) -> None:
    scorer = NliScorer(sidecar.url + "/", api_key="secret")
    assert scorer.info() == {
        "status": "ready",
        "checkpoint": "fake",
        "max_tokens": 4096,
        "name": "openjev-nli",
        "url": sidecar.url,
    }
    scorer.score(STATE, [NOUL_Q])
    assert sidecar.auth == ["Bearer secret"]


def test_no_questions_makes_no_request(sidecar: Sidecar) -> None:
    assert NliScorer(sidecar.url).score(STATE, []) == []
    assert sidecar.requests == []


def test_cli_bench_run_against_the_sidecar(sidecar: Sidecar, tmp_path: Any, capsys: pytest.CaptureFixture) -> None:
    from davout import cli

    question = {"type": "noul", "instructions": "Is the customer unhappy?"}
    path = tmp_path / "decisions.jsonl"
    path.write_text(
        "".join(json.dumps({"state": f"case {i}", "question": question, "label": i % 2 == 0}) + "\n" for i in range(30))
    )
    out = tmp_path / "results"
    args = ["bench", "run", "--backend", "openjev", "--jsonl", str(path), "--openjev-url", sidecar.url,
            "--n-calib", "6", "--n-test", "8", "--out", str(out)]  # fmt: skip
    assert cli.main(args) == 0
    run_dir = out / "openjev-decisions-s0"
    assert capsys.readouterr().out.strip() == str(run_dir)
    rows = [json.loads(line) for line in (run_dir / "raw.jsonl").read_text().splitlines()]
    assert [r["split"] for r in rows] == ["calib"] * 6 + ["test"] * 8
    assert all(r["stage"] == "nli" and r["kind"] == "noul" and r["cycle_logits"] is None for r in rows)
    assert len(sidecar.requests) == 15  # one request per decision, plus the warm-up
    config = json.loads((run_dir / "config.json").read_text())
    assert config["scorer"]["name"] == "openjev-nli" and config["scorer"]["status"] == "ready"

    assert cli.main(args) == 0 and len(sidecar.requests) == 15  # resuming a finished run scores nothing
    assert cli.main(["bench", "report", str(run_dir), "--out", str(tmp_path / "report.md")]) == 0
    assert "| openjev-decisions-s0 | openjev | decisions | noul |" in capsys.readouterr().out

    sidecar.stop()
    assert cli.main(args[:-1] + [str(tmp_path / "other")]) == 1
    assert "cannot reach openjev" in capsys.readouterr().err
