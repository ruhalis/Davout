"""Tests for the benchmark runner with fake scorers and synthetic tasks; no torch."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from davout.backends.base import LETTERS, Readout
from davout.bench import runner
from davout.bench.runner import LazyScorer, RunConfig, build_scorer, run, run_suite, suite_configs
from davout.bench.tasks import TASKS, Example, TaskData
from davout.prompts import Demo
from davout.schema import ChoiceQuestion, NoulQuestion
from davout.scorer import RawScore
from davout.scorer_nli import NliScorer

NOUL = NoulQuestion("Is it spam?", None)
CHOICE = ChoiceQuestion("Pick.", {"a": None, "b": None, "c": None})


def noul_task(n_calib: int = 4, n_test: int = 6, n_shots: int = 8) -> TaskData:
    def ex(prefix: str, i: int) -> Example:
        return Example(f"{prefix}{i}", f"message {prefix}{i}", NOUL, i % 2)

    return TaskData(
        "toy",
        "noul",
        [ex("s", i) for i in range(n_shots)],
        [ex("c", i) for i in range(n_calib)],
        [ex("t", i) for i in range(n_test)],
    )


class FakeScorer:
    def __init__(self, cycles: bool = True, fail_on: str | None = None) -> None:
        self.calls: list[tuple[Any, list, Any]] = []
        self.cycles = cycles
        self.fail_on = fail_on

    def score(self, state: Any, questions: Any, demos: Any = None) -> list[RawScore]:
        self.calls.append((state, list(questions), demos))
        if self.fail_on is not None and self.fail_on in str(state) and len(self.calls) > 1:
            raise RuntimeError("scorer crashed")
        n = float(len(str(state)))
        logits = [n, -n]
        return [RawScore(logits, 7, False, [[0.0, 0.0], logits] if self.cycles else None, "letter")]

    def info(self) -> dict:
        return {"name": "fake", "h_cycles": 2}


def rows_of(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "raw.jsonl").read_text().splitlines()]


def test_run_id_is_stable() -> None:
    assert RunConfig("hrm", "boolq", "task", 5).run_id == "hrm-boolq-task5-prefix-s0"
    assert RunConfig("hrm", "boolq", "task").run_id == "hrm-boolq-task5-prefix-s0"  # default k for task shots
    assert RunConfig("hrm", "boolq").run_id == "hrm-boolq-generic3-prefix-s0"
    assert RunConfig("hrm", "ag_news", "zero", 9, prefix_lm=False, seed=2).run_id == "hrm-ag_news-zero-causal-s2"
    assert RunConfig("openjev", "boolq", "task", 5, seed=1).run_id == "openjev-boolq-s1"
    assert RunConfig("hrm", "my decisions-v2", "zero").run_id == "hrm-my_decisions_v2-zero-prefix-s0"
    # settings that do not change what is measured do not change the id
    assert RunConfig("hrm", "boolq", device="cpu", batch_size=1, out_dir="x").run_id == RunConfig("hrm", "boolq").run_id


def test_model_and_tag_change_the_run_id_only_when_set() -> None:
    # without --model the ids (and so the existing results) are exactly as before
    assert RunConfig("hrm", "boolq", "zero", model=None, tag=None).run_id == "hrm-boolq-zero-prefix-s0"
    tuned = RunConfig("hrm", "boolq", "zero", model="/ckpt/ftA/model", tag="ftA")
    assert tuned.run_id == "hrm-ftA-boolq-zero-prefix-s0"
    assert RunConfig("hrm", "ag_news", "task", 5, prefix_lm=False, seed=2, model="m", tag="ft_B.1").run_id == (
        "hrm-ft_B.1-ag_news-task5-causal-s2"
    )
    with pytest.raises(ValueError, match="model and tag go together"):
        RunConfig("hrm", "boolq", model="/ckpt/ftA/model")
    with pytest.raises(ValueError, match="model and tag go together"):
        RunConfig("hrm", "boolq", tag="ftA")
    with pytest.raises(ValueError, match="tag must be letters"):
        RunConfig("hrm", "boolq", model="m", tag="ft-A")  # '-' separates the parts of a run id
    with pytest.raises(ValueError, match="only applies to the hrm backend"):
        RunConfig("openjev", "boolq", model="m", tag="x")
    assert [c.run_id for c in suite_configs(tuned, ["boolq", "yelp"])][:2] == [
        "hrm-ftA-boolq-zero-prefix-s0", "hrm-ftA-yelp-zero-prefix-s0",
    ]  # fmt: skip
    assert all(c.model == "/ckpt/ftA/model" for c in suite_configs(tuned, ["boolq"]))


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="backend"):
        RunConfig("gpt", "boolq")
    with pytest.raises(ValueError, match="shots_mode"):
        RunConfig("hrm", "boolq", "many")
    with pytest.raises(ValueError, match="k must be >= 1"):
        RunConfig("hrm", "boolq", "generic", 0)
    cfg = RunConfig("openjev", "boolq", "task", 5)
    assert (cfg.shots_mode, cfg.k, cfg.attention) == ("zero", 0, None)


def test_run_writes_one_row_per_example(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    task = noul_task()
    scorer = FakeScorer()
    cfg = RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path))
    run_dir = run(cfg, scorer=scorer, task_data=task)

    assert run_dir == tmp_path / "hrm-toy-zero-prefix-s0"
    rows = rows_of(run_dir)
    assert [r["id"] for r in rows] == ["c0", "c1", "c2", "c3", "t0", "t1", "t2", "t3", "t4", "t5"]
    assert [r["split"] for r in rows] == ["calib"] * 4 + ["test"] * 6
    first = rows[0]
    assert set(first) == {
        "id", "split", "label", "kind", "logits", "cycle_logits",
        "prompt_tokens", "truncated", "stage", "latency_ms",
    }  # fmt: skip
    n = float(len("message c0"))
    assert first["logits"] == [n, -n] and first["cycle_logits"] == [[0.0, 0.0], [n, -n]]
    assert (first["label"], first["kind"], first["prompt_tokens"], first["truncated"], first["stage"]) == (
        0, "noul", 7, False, "letter",
    )  # fmt: skip
    assert all(r["latency_ms"] >= 0 for r in rows)
    assert [r["label"] for r in rows[4:]] == [0, 1, 0, 1, 0, 1]

    # one question per call, no task demos in zero-shot mode, plus one untimed warm-up call
    assert len(scorer.calls) == 11 and scorer.calls[0][0] == scorer.calls[1][0] == "message c0"
    assert all(questions == [NOUL] and demos is None for _, questions, demos in scorer.calls)

    config = json.loads((run_dir / "config.json").read_text())
    assert config["run_id"] == cfg.run_id and config["complete"] is True
    assert config["config"]["shots_mode"] == "zero" and config["config"]["k"] == 0
    assert config["scorer"] == {"name": "fake", "h_cycles": 2}
    assert config["task"] == {"name": "toy", "kind": "noul", "n_shots": 0, "n_calib": 4, "n_test": 6}
    assert config["davout_version"] and config["created_utc"].endswith("+00:00")
    assert "10/10" in capsys.readouterr().err


def test_resume_skips_completed_ids(tmp_path: Path) -> None:
    task = noul_task()
    cfg = RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path))
    crashing = FakeScorer(fail_on="t2")
    with pytest.raises(RuntimeError, match="scorer crashed"):
        run(cfg, scorer=crashing, task_data=task)
    run_dir = tmp_path / cfg.run_id
    assert [r["id"] for r in rows_of(run_dir)] == ["c0", "c1", "c2", "c3", "t0", "t1"]
    assert json.loads((run_dir / "config.json").read_text())["complete"] is False
    with (run_dir / "raw.jsonl").open("a") as f:
        f.write('{"id": "t2", "split": "te')  # a half-written line from the crash

    scorer = FakeScorer()
    run(cfg, scorer=scorer, task_data=task)
    assert [r["id"] for r in rows_of(run_dir)] == ["c0", "c1", "c2", "c3", "t0", "t1", "t2", "t3", "t4", "t5"]
    assert [state for state, _, _ in scorer.calls] == ["message t2"] + [f"message t{i}" for i in range(2, 6)]
    assert json.loads((run_dir / "config.json").read_text())["complete"] is True

    # a finished run needs no scorer at all
    untouched = LazyScorer(lambda: pytest.fail("the scorer must not be built"))
    assert run(cfg, scorer=untouched, task_data=task) == run_dir
    assert len(rows_of(run_dir)) == 10


def test_resume_refuses_different_settings(tmp_path: Path) -> None:
    run(RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path)), scorer=FakeScorer(), task_data=noul_task())
    changed = RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path), n_test=50)
    with pytest.raises(RuntimeError, match="different settings.*n_test: 300 -> 50"):
        run(changed, scorer=FakeScorer(), task_data=noul_task())
    # latency-only settings may change between sessions
    slower = RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path), batch_size=1)
    run(slower, scorer=FakeScorer(), task_data=noul_task())


def test_resume_guard_covers_the_model(tmp_path: Path) -> None:
    task = noul_task()
    tuned = RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path), model="/ckpt/ftA/model", tag="ftA")
    run_dir = run(tuned, scorer=FakeScorer(), task_data=task)
    assert run_dir == tmp_path / "hrm-ftA-toy-zero-prefix-s0"
    saved = json.loads((run_dir / "config.json").read_text())["config"]
    assert (saved["model"], saved["tag"]) == ("/ckpt/ftA/model", "ftA")
    run(tuned, scorer=LazyScorer(lambda: pytest.fail("the scorer must not be built")), task_data=task)

    # the same tag pointed at other weights must not append to those results
    other = RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path), model="/ckpt/ftB/model", tag="ftA")
    with pytest.raises(RuntimeError, match="different settings.*model: '/ckpt/ftA/model' -> '/ckpt/ftB/model'"):
        run(other, scorer=FakeScorer(), task_data=task)

    # stock-model runs are untouched: same directory as before, model recorded as None
    plain = RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path))
    plain_dir = run(plain, scorer=FakeScorer(), task_data=task)
    assert plain_dir == tmp_path / "hrm-toy-zero-prefix-s0"
    assert json.loads((plain_dir / "config.json").read_text())["config"]["model"] is None
    # a run recorded before the `model` field existed still resumes
    config = json.loads((plain_dir / "config.json").read_text())
    del config["config"]["model"], config["config"]["tag"]
    (plain_dir / "config.json").write_text(json.dumps(config))
    assert run(plain, scorer=FakeScorer(), task_data=task) == plain_dir


def test_build_backend_passes_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import davout.backends.hrm as hrm

    seen: list[tuple[tuple, dict]] = []

    class Stub:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            seen.append((args, kwargs))

    monkeypatch.setattr(hrm, "HrmBackend", Stub)
    runner.build_backend(RunConfig("hrm", "boolq", "zero", device="cpu", batch_size=2, max_tokens=512))
    assert seen[-1] == ((), {"device": "cpu", "batch_size": 2, "max_tokens": 512, "prefix_lm": True})
    runner.build_backend(RunConfig("hrm", "boolq", "zero", prefix_lm=False, model="/ckpt/ftA/model", tag="ftA"))
    assert seen[-1][1]["model_id_or_path"] == "/ckpt/ftA/model" and seen[-1][1]["prefix_lm"] is False


def test_task_shots_are_passed_as_demos(tmp_path: Path) -> None:
    task = noul_task()
    scorer = FakeScorer(cycles=False)
    run_dir = run(RunConfig("hrm", "toy", "task", 3, out_dir=str(tmp_path)), scorer=scorer, task_data=task)
    expected = [Demo("message s0", NOUL, 1), Demo("message s1", NOUL, 0), Demo("message s2", NOUL, 1)]
    assert all(demos == {0: expected} for _, _, demos in scorer.calls)  # label 1 (true) -> answer 0 (Yes)
    assert rows_of(run_dir)[0]["cycle_logits"] is None
    assert json.loads((run_dir / "config.json").read_text())["task"]["n_shots"] == 3

    generic = FakeScorer()
    run(RunConfig("hrm", "toy", "generic", 3, out_dir=str(tmp_path)), scorer=generic, task_data=task)
    assert all(demos is None for _, _, demos in generic.calls)

    with pytest.raises(ValueError, match="fewer than k=9"):
        run(RunConfig("hrm", "toy", "task", 9, out_dir=str(tmp_path)), scorer=scorer, task_data=task)


def test_bad_scorer_output_is_rejected(tmp_path: Path) -> None:
    class Wrong(FakeScorer):
        def score(self, state: Any, questions: Any, demos: Any = None) -> list[RawScore]:
            return [RawScore([0.1, 0.2, 0.3], 1, False, None, "letter")]

    with pytest.raises(RuntimeError, match="bad logits for c0"):
        run(RunConfig("hrm", "toy", "zero", out_dir=str(tmp_path)), scorer=Wrong(), task_data=noul_task())


def test_build_scorer_openjev() -> None:
    scorer = build_scorer(RunConfig("openjev", "boolq", openjev_url="http://127.0.0.1:9"))
    assert isinstance(scorer, NliScorer) and scorer.url == "http://127.0.0.1:9"


class FakeBackend:
    """A `LabelBackend` whose answer is always the first letter."""

    name = "fake-hrm"
    max_labels = len(LETTERS)

    def __init__(self, prefix_lm: bool) -> None:
        self.prefix_lm = prefix_lm
        self.prompts: list[Any] = []

    def info(self) -> dict:
        return {"name": self.name, "prefix_lm": self.prefix_lm}

    def read(self, prompts: Any) -> list[Readout]:
        self.prompts.extend(prompts)
        out = []
        for p in prompts:
            logits = [2.0] + [0.0] * (p.n_labels - 1)
            out.append(Readout(logits, len(p.text.split()), False, [[0.0] * p.n_labels, logits]))
        return out


def test_build_scorer_hrm_with_injected_backend() -> None:
    backend = FakeBackend(True)
    generic = build_scorer(RunConfig("hrm", "boolq", "generic", 2, shortlist=4), backend=backend)
    assert (generic.backend, generic.shots, generic.shortlist) == (backend, 2, 4)
    assert build_scorer(RunConfig("hrm", "boolq", "task", 5), backend=backend).shots == 0
    assert build_scorer(RunConfig("hrm", "boolq", "zero"), backend=backend).shots == 0
    assert runner.scorer_info(generic) == {
        "name": "fake-hrm", "prefix_lm": True, "scorer": "letter", "shots": 2, "shortlist": 4,
    }  # fmt: skip


def test_suite_configs() -> None:
    base = RunConfig("hrm", "suite", n_test=20, seed=1)
    ids = [c.run_id for c in suite_configs(base, ["boolq", "yelp"])]
    assert ids == [
        "hrm-boolq-zero-prefix-s1", "hrm-yelp-zero-prefix-s1",
        "hrm-boolq-generic3-prefix-s1", "hrm-yelp-generic3-prefix-s1",
        "hrm-boolq-task5-prefix-s1", "hrm-yelp-task5-prefix-s1",
        "hrm-boolq-task5-causal-s1", "hrm-yelp-task5-causal-s1",
    ]  # fmt: skip
    assert all(c.n_test == 20 for c in suite_configs(base, ["boolq"]))
    assert len(suite_configs(base)) == 4 * len(TASKS)
    assert [c.run_id for c in suite_configs(RunConfig("openjev", "suite"))] == [f"openjev-{t}-s0" for t in TASKS]
    with pytest.raises(ValueError, match="unknown task"):
        suite_configs(base, ["boolq", "nope"])


def test_run_suite_shares_backends_and_survives_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    choice_task = TaskData(
        "ag_news",
        "choice",
        [Example(f"s{i}", f"shot {i}", CHOICE, i % 3) for i in range(8)],
        [Example(f"c{i}", f"calib {i}", CHOICE, i % 3) for i in range(3)],
        [Example(f"t{i}", f"test {i}", CHOICE, i % 3) for i in range(3)],
    )

    def fake_load(cfg: RunConfig) -> TaskData:
        if cfg.task == "yelp":
            raise RuntimeError("dataset unavailable")
        return noul_task() if cfg.task == "boolq" else choice_task

    built: list[FakeBackend] = []

    def build(cfg: RunConfig) -> FakeBackend:
        built.append(FakeBackend(cfg.prefix_lm))
        return built[-1]

    monkeypatch.setattr(runner, "load_task_data", fake_load)
    base = RunConfig("hrm", "suite", out_dir=str(tmp_path))
    done, failed = run_suite(suite_configs(base, ["boolq", "ag_news", "yelp"]), build=build)

    assert [b.prefix_lm for b in built] == [True, False]  # one backend per attention mode
    assert [d.name for d in done] == [
        "hrm-boolq-zero-prefix-s0", "hrm-ag_news-zero-prefix-s0",
        "hrm-boolq-generic3-prefix-s0", "hrm-ag_news-generic3-prefix-s0",
        "hrm-boolq-task5-prefix-s0", "hrm-ag_news-task5-prefix-s0",
        "hrm-boolq-task5-causal-s0", "hrm-ag_news-task5-causal-s0",
    ]  # fmt: skip
    assert [name for name, _ in failed] == [
        "hrm-yelp-zero-prefix-s0", "hrm-yelp-generic3-prefix-s0",
        "hrm-yelp-task5-prefix-s0", "hrm-yelp-task5-causal-s0",
    ]  # fmt: skip
    assert all("dataset unavailable" in why for _, why in failed)

    rows = rows_of(tmp_path / "hrm-boolq-task5-prefix-s0")
    assert len(rows) == 10 and rows[0]["logits"] == [2.0, 0.0] and len(rows[0]["cycle_logits"]) == 2
    config = json.loads((tmp_path / "hrm-boolq-task5-causal-s0" / "config.json").read_text())
    assert (config["scorer"]["name"], config["scorer"]["prefix_lm"]) == ("fake-hrm", False)
    # task shots reach the prompt: five solved blocks before the live one
    task_prompts = [p for p in built[1].prompts if "message t0" in p.state]
    assert task_prompts and task_prompts[0].prefix.count("Answer: ") == 5
    assert "message s0" in task_prompts[0].prefix


def test_run_suite_does_not_share_a_backend_between_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "load_task_data", lambda cfg: noul_task())
    built: list[str | None] = []

    def build(cfg: RunConfig) -> FakeBackend:
        built.append(cfg.model)
        return FakeBackend(cfg.prefix_lm)

    base = RunConfig("hrm", "boolq", "zero", out_dir=str(tmp_path))
    tuned = RunConfig("hrm", "boolq", "zero", out_dir=str(tmp_path), model="/ckpt/ftA/model", tag="ftA")
    done, failed = run_suite([base, tuned], build=build)
    assert not failed and built == [None, "/ckpt/ftA/model"]
    assert [d.name for d in done] == ["hrm-boolq-zero-prefix-s0", "hrm-ftA-boolq-zero-prefix-s0"]


def test_cli_bench_model_and_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    from davout import cli

    seen: list[RunConfig] = []
    monkeypatch.setattr(runner, "run", lambda cfg: seen.append(cfg) or Path(cfg.out_dir) / cfg.run_id)
    common = ["--backend", "hrm", "--out", str(tmp_path), "--n-calib", "5", "--n-test", "5"]
    assert cli.main(["bench", "run", "--task", "boolq", "--shots-mode", "zero", *common]) == 0
    assert (seen[-1].model, seen[-1].tag, seen[-1].run_id) == (None, None, "hrm-boolq-zero-prefix-s0")
    args = ["bench", "run", "--task", "boolq", "--shots-mode", "zero", "--model", "/ckpt/ftA/model", *common]
    assert cli.main([*args, "--tag", "ftA"]) == 0
    assert (seen[-1].model, seen[-1].tag) == ("/ckpt/ftA/model", "ftA")
    assert capsys.readouterr().out.splitlines()[-1] == str(tmp_path / "hrm-ftA-boolq-zero-prefix-s0")
    assert cli.main(args) == 1  # --model without --tag
    assert "model and tag go together" in capsys.readouterr().err

    suites: list[list[RunConfig]] = []
    monkeypatch.setattr(runner, "run_suite", lambda cfgs: suites.append(list(cfgs)) or ([], []))
    assert cli.main(["bench", "suite", "--tasks", "boolq,yelp", "--model", "/ckpt/ftB/model", "--tag", "ftB", *common]) == 0
    assert {c.model for c in suites[-1]} == {"/ckpt/ftB/model"}
    assert suites[-1][0].run_id == "hrm-ftB-boolq-zero-prefix-s0" and len(suites[-1]) == 8


def test_bench_modules_import_without_heavy_dependencies() -> None:
    code = (
        "import sys, davout.cli, davout.scorer_nli, davout.bench.tasks, davout.bench.runner, davout.bench.report\n"
        "davout.cli._build_parser()\n"
        "bad = [m for m in ('torch', 'transformers', 'datasets', 'davout.engine', 'davout.train.loop') if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
