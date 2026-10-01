"""Tests for benchmark reports on synthetic raw rows with known properties."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from davout import cli, metrics
from davout.bench import report
from davout.bench.report import evaluate, export_calibrator, fit_calibrator, load_raw, write_report
from davout.calibrate import Calibrator, softmax


def write_run(
    root: Path,
    run_id: str,
    rows: list[dict[str, Any]],
    backend: str = "hrm",
    task: str = "toy",
    shots_mode: str = "task",
    k: int = 5,
    prefix_lm: bool = True,
    complete: bool = True,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    config = {
        "run_id": run_id,
        "config": {"backend": backend, "task": task, "shots_mode": shots_mode, "k": k,
                   "prefix_lm": prefix_lm, "seed": 0},
        "task": {"name": task, "kind": rows[0]["kind"]},
        "scorer": {"name": "fake"},
        "complete": complete,
    }  # fmt: skip
    (run_dir / "config.json").write_text(json.dumps(config))
    return run_dir


def row(i: int, split: str, kind: str, label: int, logits: list[float], cycles: Any = None, **kw: Any) -> dict:
    return {
        "id": f"{split}{i}", "split": split, "label": label, "kind": kind, "logits": logits,
        "cycle_logits": cycles, "prompt_tokens": kw.get("tokens", 100), "truncated": kw.get("truncated", False),
        "stage": kw.get("stage", "letter"), "latency_ms": kw.get("latency", 10.0),
    }  # fmt: skip


def overconfident_choice(n: int = 400, k: int = 4, scale: float = 6.0, cycles: bool = False) -> list[dict]:
    """Choice rows from a model that is right about 70% of the time but always nearly certain."""
    rng = np.random.default_rng(0)
    out = []
    for split in ("calib", "test"):
        for i in range(n):
            label = int(rng.integers(k))
            z = rng.normal(size=k)
            z[label] += 1.6
            final = (z * scale).tolist()
            early = (z * scale + rng.normal(size=k) * 2 * scale).tolist()  # a noisier first cycle
            out.append(row(i, split, "choice", label, final, [early, final] if cycles else None,
                           latency=10.0 + i % 10, tokens=100 + i % 3))  # fmt: skip
    return out


def noul_rows(n: int = 300, a: float = 0.25, b: float = -1.0) -> list[dict]:
    """Noul rows whose true probability is sigmoid(a*m + b) for the raw margin m."""
    rng = np.random.default_rng(1)
    out = []
    for split in ("calib", "test"):
        for i in range(n):
            m = float(rng.normal() * 8)
            p = 1 / (1 + np.exp(-(a * m + b)))
            out.append(row(i, split, "noul", int(rng.random() < p), [m / 2, -m / 2]))
    return out


def test_load_raw_and_find_run_dirs(tmp_path: Path) -> None:
    rows = overconfident_choice(5)
    run_dir = write_run(tmp_path, "hrm-toy-zero-prefix-s0", rows)
    assert load_raw(run_dir) == rows
    with (run_dir / "raw.jsonl").open("a") as f:
        f.write(json.dumps({**rows[0], "label": 3}) + "\n")
    assert len(load_raw(run_dir)) == len(rows)  # a repeated id is not counted twice
    assert report.find_run_dirs([tmp_path]) == [run_dir] == report.find_run_dirs([run_dir])
    with pytest.raises(ValueError, match="no raw.jsonl"):
        load_raw(tmp_path)
    with pytest.raises(ValueError, match="not a run directory"):
        report.find_run_dirs([tmp_path / "missing"])
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no raw.jsonl here or one level below"):
        report.find_run_dirs([tmp_path / "empty"])


def test_calibration_helps_an_overconfident_model(tmp_path: Path) -> None:
    rows = overconfident_choice()
    summary = evaluate(write_run(tmp_path, "hrm-toy-task5-prefix-s0", rows))
    raw, cal = summary["raw"], summary["calibrated"]

    assert summary["kind"] == "choice" and summary["n_test"] == 400 == raw["n"] and summary["n_calib"] == 400
    assert (summary["backend"], summary["task"], summary["shots_mode"], summary["k"], summary["attention"]) == (
        "hrm", "toy", "task", 5, "prefix",
    )  # fmt: skip
    assert 0.55 < raw["accuracy"] < 0.9 and cal["accuracy"] == raw["accuracy"]  # temperature keeps the argmax
    assert raw["ece"] > 0.15 and cal["ece"] < raw["ece"] / 2
    assert cal["nll"] < raw["nll"] - 0.2 and cal["brier"] < raw["brier"]
    assert summary["calibrator"]["choice_T"] > 3 and summary["calibrator"]["score_T"] == 1.0
    assert summary["calibrator"]["meta"] == {
        "run_id": "hrm-toy-task5-prefix-s0", "n": 400, "split": "calib", "kind": "choice",
    }  # fmt: skip

    # the raw numbers are exactly the metrics of the plain softmax on the test rows
    test = [r for r in rows if r["split"] == "test"]
    probs = softmax([r["logits"] for r in test])
    expected = metrics.summarize(probs, [r["label"] for r in test])
    assert raw == pytest.approx(expected)
    # reliability is reported for the calibrated output
    T = summary["calibrator"]["choice_T"]
    labels = [r["label"] for r in test]
    assert summary["reliability"] == metrics.reliability_bins(softmax([r["logits"] for r in test], T), labels)

    assert summary["latency_ms"]["p50"] == pytest.approx(14.5) and summary["latency_ms"]["mean"] == pytest.approx(14.5)
    assert summary["latency_ms"]["p95"] == pytest.approx(19.0)
    assert summary["prompt_tokens_mean"] == pytest.approx(np.mean([r["prompt_tokens"] for r in test]))
    assert summary["truncated"] == 0 and summary["stages"] == {"letter": 400}
    assert "cycles" not in summary  # no cycle logits, no early-exit section
    json.dumps(summary, allow_nan=False)


def test_cycles_section(tmp_path: Path) -> None:
    rows = overconfident_choice(cycles=True)
    summary = evaluate(write_run(tmp_path, "hrm-toy-task5-prefix-s0", rows))
    cycles = summary["cycles"]
    assert [c["cycle"] for c in cycles] == [0, 1]
    assert cycles[1]["raw"] == summary["raw"] and cycles[1]["calibrated"] == summary["calibrated"]
    assert cycles[1]["agreement_with_final"] == 1.0 and cycles[0]["agreement_with_final"] < 0.9
    assert cycles[0]["raw"]["accuracy"] < cycles[1]["raw"]["accuracy"]
    assert cycles[0]["calibrator"]["choice_T"] > cycles[1]["calibrator"]["choice_T"]  # fitted per cycle
    assert cycles[0]["calibrator"]["meta"]["cycle"] == 0

    # rows without cycle logits (two-stage choice) switch the section off
    rows[-1]["cycle_logits"] = None
    assert "cycles" not in evaluate(write_run(tmp_path, "other", rows))


def test_score_mae(tmp_path: Path) -> None:
    big = 50.0  # one-hot logits: the expected level equals the predicted level
    preds = [(0, 0), (1, 2), (4, 1), (3, 3)]  # (true level, predicted level): distances 0, 1, 3, 0
    rows = []
    for split in ("calib", "test"):
        for i, (label, pred) in enumerate(preds):
            rows.append(row(i, split, "score", label, [big if j == pred else 0.0 for j in range(5)]))
    summary = evaluate(write_run(tmp_path, "hrm-yelp-zero-prefix-s0", rows, task="yelp", shots_mode="zero", k=0))
    assert summary["raw"]["mae"] == pytest.approx(1.0) and summary["raw"]["exact"] == 0.5 == summary["raw"]["accuracy"]
    assert summary["calibrated"]["mae"] > 0 and "auroc" not in summary["raw"]
    assert summary["calibrator"]["score_T"] > 1.0 and summary["calibrator"]["choice_T"] == 1.0

    uniform = [row(i, "test", "score", 4, [0.0] * 5) for i in range(3)]
    flat = evaluate(write_run(tmp_path, "flat", uniform))
    assert flat["raw"]["mae"] == pytest.approx(2.0)  # expected level 2 against true level 4
    assert flat["calibrated"] is None and flat["calibrator"] is None and flat["n_calib"] == 0


def test_noul_platt_and_auroc(tmp_path: Path) -> None:
    rows = noul_rows()
    summary = evaluate(write_run(tmp_path, "openjev-toy-s0", rows, backend="openjev", shots_mode="zero", k=0))
    assert summary["attention"] is None and summary["kind"] == "noul"
    cal = summary["calibrator"]
    assert cal["noul_a"] == pytest.approx(0.25, abs=0.08) and cal["noul_b"] == pytest.approx(-1.0, abs=0.4)
    assert summary["calibrated"]["nll"] < summary["raw"]["nll"] and summary["calibrated"]["ece"] < summary["raw"]["ece"]
    assert summary["calibrated"]["accuracy"] >= summary["raw"]["accuracy"]  # the bias moves the threshold

    test = [r for r in rows if r["split"] == "test"]
    margins = [r["logits"][0] - r["logits"][1] for r in test]
    labels = [r["label"] for r in test]
    assert summary["raw"]["auroc"] == pytest.approx(metrics.auroc(margins, labels))
    assert summary["calibrated"]["auroc"] == pytest.approx(summary["raw"]["auroc"])  # monotone rescaling
    p_true = [Calibrator().noul_prob(m) for m in margins]
    assert summary["raw"]["accuracy"] == pytest.approx(np.mean([(p > 0.5) == bool(y) for p, y in zip(p_true, labels)]))

    fitted = fit_calibrator(rows, run_id="x")
    assert (fitted.noul_a, fitted.noul_b) == (cal["noul_a"], cal["noul_b"]) and fitted.meta["n"] == 300
    with pytest.raises(ValueError, match="no calibration rows"):
        fit_calibrator(test)


def test_ragged_choice_rows(tmp_path: Path) -> None:
    rows = []
    for split in ("calib", "test"):
        for i in range(40):
            k = 2 + i % 3
            pick = i % k if i % 4 else 0  # every fourth row picks option 0 regardless of the label
            rows.append(row(i, split, "choice", i % k, [3.0 if j == pick else 0.0 for j in range(k)]))
    summary = evaluate(write_run(tmp_path, "ragged", rows))
    assert summary["raw"]["n"] == 40 and 0.5 < summary["raw"]["accuracy"] < 1.0
    assert sum(b["count"] for b in summary["reliability"]) == 40


def test_write_report(tmp_path: Path) -> None:
    runs = [
        write_run(tmp_path, "hrm-toy-task5-prefix-s0", overconfident_choice(cycles=True)),
        write_run(tmp_path, "hrm-toy-zero-causal-s0", overconfident_choice(), shots_mode="zero", k=0, prefix_lm=False,
                  complete=False),
        write_run(tmp_path, "openjev-spam-s0", noul_rows(), backend="openjev", task="spam", shots_mode="zero", k=0),
        write_run(tmp_path, "hrm-yelp-generic3-prefix-s0",
                  [row(i, s, "score", i % 5, [9.0 if j == i % 5 else 0.0 for j in range(5)], truncated=i == 0)
                   for s in ("calib", "test") for i in range(20)], task="yelp", shots_mode="generic", k=3),
        write_run(tmp_path, "hrm-empty-zero-prefix-s0", [row(0, "calib", "noul", 1, [1.0, 0.0])]),
    ]  # fmt: skip
    out = write_report(runs, tmp_path / "reports" / "report.md")
    text = out.read_text()
    lines = text.splitlines()

    header = next(l for l in lines if l.startswith("| run |"))
    for column in ("backend", "task", "shots", "attention", "n", "accuracy raw → cal", "ECE raw → cal",
                   "NLL raw → cal", "Brier", "AUROC", "MAE", "p50 ms", "p95 ms", "tokens"):  # fmt: skip
        assert f"| {column} |" in header
    runs_section = lines[: lines.index("## Early exit: metrics after each H cycle")]
    table = {l.split(" | ")[0].lstrip("| "): [c.strip() for c in l.strip("|").split("|")] for l in runs_section
             if l.startswith("| ") and not l.startswith("| run")}  # fmt: skip
    hrm = table["hrm-toy-task5-prefix-s0"]
    assert hrm[1:7] == ["hrm", "toy", "choice", "task-5", "prefix", "400"]
    assert hrm[11] == "–" and hrm[12] == "–"  # no AUROC or MAE for a choice task
    summary = json.loads((runs[0] / "summary.json").read_text())
    assert hrm[8] == f"{summary['raw']['ece']:.3f} → {summary['calibrated']['ece']:.3f}"
    assert hrm[13:] == ["14", "19", "101"]
    assert table["hrm-toy-zero-causal-s0"][4:6] == ["zero", "causal"]
    nli = table["openjev-spam-s0"]
    assert nli[1:6] == ["openjev", "spam", "noul", "–", "–"] and nli[11] != "–" and nli[12] == "–"
    yelp = table["hrm-yelp-generic3-prefix-s0"]
    assert yelp[4] == "generic-3" and yelp[11] == "–" and yelp[12] != "–"

    assert "## Early exit: metrics after each H cycle" in text
    cycle_header = next(l for l in lines if l.startswith("| run | cycle |"))
    assert "agrees with final" in cycle_header
    cycle_rows = [l for l in lines if l.startswith("| hrm-toy-task5-prefix-s0 | ") and " of 2 |" in l]
    assert len(cycle_rows) == 2 and "| 1 of 2 |" in cycle_rows[0] and cycle_rows[1].endswith("| 1.000 |")
    assert not any(l.startswith("| hrm-toy-zero-causal-s0 | 1 of") for l in lines)

    assert "hrm-toy-zero-causal-s0: the run is incomplete" in text
    assert "hrm-yelp-generic3-prefix-s0: 1 of 20 test states were truncated" in text
    assert "## Skipped" in text and "hrm-empty-zero-prefix-s0" in text and "no test rows" in text
    assert not (runs[4] / "summary.json").exists()
    for run_dir in runs[:4]:
        assert json.loads((run_dir / "summary.json").read_text())["run_id"] == run_dir.name

    # without any recurrent run there is no early-exit section
    plain = write_report(runs[1:3], tmp_path / "plain.md").read_text()
    assert "Early exit" not in plain and "## Skipped" not in plain
    with pytest.raises(ValueError, match="no run could be evaluated"):
        write_report(runs[4:], tmp_path / "none.md")


def mixed_stage_choice(n: int = 600, k: int = 5) -> list[dict]:
    """Letter rows 3x over-confident; two-stage rows (log-probabilities) well calibrated."""
    rng = np.random.default_rng(2)
    out = []
    for split in ("calib", "test"):
        for i in range(n):
            z = rng.normal(size=k) * 1.5
            label = int(rng.choice(k, p=softmax(z)))
            out.append(row(i, split, "choice", label, (z * 3).tolist()))
            out.append(row(n + i, split, "choice", label, (z - np.logaddexp.reduce(z)).tolist(), stage="two_stage"))
    return out


def test_choice_stages_fit_independently(tmp_path: Path) -> None:
    rows = mixed_stage_choice()
    cal = fit_calibrator(rows)
    assert cal.choice_T == pytest.approx(3.0, rel=0.25)
    assert cal.choice_two_stage_T == pytest.approx(1.0, rel=0.25)

    letter_only = fit_calibrator([r for r in rows if r["stage"] == "letter"])
    two_only = fit_calibrator([r for r in rows if r["stage"] == "two_stage"])
    assert letter_only.choice_two_stage_T == 1.0 and two_only.choice_T == 1.0

    summary = evaluate(write_run(tmp_path, "mixed", rows))
    assert summary["calibrator"]["choice_T"] == pytest.approx(cal.choice_T)
    assert summary["calibrator"]["choice_two_stage_T"] == pytest.approx(cal.choice_two_stage_T)
    assert summary["calibrated"]["nll"] < summary["raw"]["nll"]


def test_export_calibrator(tmp_path: Path) -> None:
    choice = write_run(tmp_path, "hrm-toy-task5-prefix-s0", overconfident_choice())
    noul = write_run(tmp_path, "hrm-spam-task5-prefix-s0", noul_rows(), task="spam")
    noul2 = write_run(tmp_path, "hrm-spam2-task5-prefix-s0", noul_rows(), task="spam2")
    out = tmp_path / "cal" / "calibration.json"
    returned = export_calibrator([choice, noul, noul2], out)

    cal = Calibrator.load(out)  # the file `davout serve --calibration` reads
    assert cal == returned
    assert 3 < cal.choice_T < 20 and cal.score_T == 1.0  # no score run: identity
    assert cal.noul_a == pytest.approx(0.25, abs=0.08) and cal.noul_b == pytest.approx(-1.0, abs=0.4)
    assert cal.choice_T == pytest.approx(evaluate(choice)["calibrator"]["choice_T"])
    assert cal.meta == {
        "source": "davout bench",
        "split": "calib",
        "runs": {"choice": ["hrm-toy-task5-prefix-s0"],
                 "noul": ["hrm-spam-task5-prefix-s0", "hrm-spam2-task5-prefix-s0"]},
        "n": {"choice": 400, "noul": 600},
    }  # fmt: skip
    assert sum(cal.choice_probs([1.0, 2.0, 3.0])) == pytest.approx(1.0) and 0 < cal.noul_prob(0.3) < 1

    only_test = write_run(tmp_path, "t", [row(0, "test", "noul", 1, [1.0, 0.0])])
    with pytest.raises(ValueError, match="none of the given runs has calibration rows"):
        export_calibrator([only_test], tmp_path / "x.json")


def test_cli_report_and_calibrate(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    results = tmp_path / "results"
    write_run(results, "hrm-toy-task5-prefix-s0", overconfident_choice(cycles=True))
    write_run(results, "openjev-toy-s0", noul_rows(), backend="openjev", shots_mode="zero", k=0)

    out = tmp_path / "report.md"
    assert cli.main(["bench", "report", str(results), "--out", str(out)]) == 0  # a results directory expands
    captured = capsys.readouterr()
    assert captured.out.strip() == out.read_text().strip() and f"wrote {out}" in captured.err
    assert "| hrm-toy-task5-prefix-s0 | hrm |" in captured.out and "| openjev-toy-s0 | openjev |" in captured.out
    assert (results / "openjev-toy-s0" / "summary.json").exists()

    assert cli.main(["bench", "report", str(results / "openjev-toy-s0"), "--out", str(out)]) == 0
    assert "hrm-toy" not in out.read_text()
    capsys.readouterr()

    cal_path = tmp_path / "calibration.json"
    assert cli.main(["calibrate", str(results), "--out", str(cal_path)]) == 0
    cal = Calibrator.load(cal_path)
    assert cal.choice_T > 3 and cal.noul_a > 0
    assert json.loads(capsys.readouterr().out)["choice_T"] == cal.choice_T

    assert cli.main(["bench", "report", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err
    assert cli.main(["bench"]) == 2


def test_export_calibrator_splits_stages(tmp_path: Path) -> None:
    letter = write_run(tmp_path, "letter", overconfident_choice())
    mixed = write_run(tmp_path, "mixed", mixed_stage_choice())
    big = write_run(tmp_path, "big", [r for r in mixed_stage_choice() if r["stage"] == "two_stage"])
    cal = export_calibrator([letter, big], tmp_path / "a.json")
    assert cal.choice_T == pytest.approx(evaluate(letter)["calibrator"]["choice_T"])  # unskewed by `big`
    assert cal.choice_two_stage_T == pytest.approx(1.0, rel=0.25)

    cal = export_calibrator([mixed], tmp_path / "b.json")
    assert cal.choice_T == pytest.approx(3.0, rel=0.25)
    assert cal.choice_two_stage_T == pytest.approx(1.0, rel=0.25)
    assert Calibrator.load(tmp_path / "b.json") == cal

    cal = export_calibrator([big], tmp_path / "c.json")
    assert cal.choice_T == 1.0 and cal.choice_two_stage_T == pytest.approx(1.0, rel=0.25)
