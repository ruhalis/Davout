"""Turn raw benchmark rows into metrics, a markdown comparison and a serving calibrator.

Calibration parameters are always fitted on a run's `calib` rows and the
metrics computed on its `test` rows, both uncalibrated ("raw") and calibrated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from davout import metrics
from davout.bench.runner import read_raw
from davout.calibrate import Calibrator, fit_platt, fit_temperature

KINDS = ("choice", "score", "noul")
N_BINS = 15


def load_raw(run_dir: str | Path) -> list[dict[str, Any]]:
    """Rows of `<run_dir>/raw.jsonl` (the last row wins if an id repeats)."""
    path = Path(run_dir) / "raw.jsonl"
    if not path.exists():
        raise ValueError(f"{run_dir}: no raw.jsonl (not a run directory)")
    by_id = {row["id"]: row for row in read_raw(path)}
    return list(by_id.values())


def load_config(run_dir: str | Path) -> dict[str, Any]:
    """`<run_dir>/config.json`, or `{}` when the file is missing."""
    path = Path(run_dir) / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def find_run_dirs(paths: Sequence[str | Path]) -> list[Path]:
    """Expand each path into run directories: itself, or its sub-directories holding a raw.jsonl."""
    out: list[Path] = []
    for p in map(Path, paths):
        if (p / "raw.jsonl").exists():
            out.append(p)
        elif p.is_dir():
            subs = sorted(d for d in p.iterdir() if (d / "raw.jsonl").exists())
            if not subs:
                raise ValueError(f"{p}: no raw.jsonl here or one level below")
            out.extend(subs)
        else:
            raise ValueError(f"{p}: not a run directory")
    return out


def _kind(rows: Sequence[dict[str, Any]], where: Any = "rows") -> str:
    kinds = {row["kind"] for row in rows}
    if len(kinds) != 1 or not kinds <= set(KINDS):
        raise ValueError(f"{where}: expected rows of one kind, found {sorted(kinds)}")
    return kinds.pop()


def _logits(row: dict[str, Any], cycle: int | None) -> list[float]:
    return row["logits"] if cycle is None else row["cycle_logits"][cycle]


def _fit(
    kind: str,
    logit_rows: list[list[float]],
    labels: list[int],
    meta: dict[str, Any],
    stages: Sequence[str | None] | None = None,
) -> Calibrator:
    """Fit one calibrator; choice rows are split by stage (two-stage log-probs get their own T)."""
    if not logit_rows:
        raise ValueError("no calibration rows to fit on")
    if kind == "noul":
        a, b = fit_platt([l[0] - l[1] for l in logit_rows], labels)
        return Calibrator(noul_a=a, noul_b=b, meta=meta)
    if kind == "score":
        return Calibrator(score_T=fit_temperature(logit_rows, labels), meta=meta)
    stages = stages if stages is not None else [None] * len(logit_rows)
    params: dict[str, float] = {}
    for name, two_stage in (("choice_T", False), ("choice_two_stage_T", True)):
        idx = [i for i, st in enumerate(stages) if (st == "two_stage") == two_stage]
        if idx:
            params[name] = fit_temperature([logit_rows[i] for i in idx], [labels[i] for i in idx])
    return Calibrator(meta=meta, **params)


def fit_calibrator(
    rows: Sequence[dict[str, Any]], run_id: str | None = None, cycle: int | None = None
) -> Calibrator:
    """Fit a `Calibrator` on the `calib` rows (temperature for choice/score, Platt for noul).

    `cycle` fits on the logits of one H cycle instead of the final output.
    """
    calib = [row for row in rows if row.get("split") == "calib"]
    if not calib:
        raise ValueError("no calibration rows to fit on")
    kind = _kind(calib)
    meta: dict[str, Any] = {"run_id": run_id, "n": len(calib), "split": "calib", "kind": kind}
    if cycle is not None:
        meta["cycle"] = cycle
    return _fit(
        kind,
        [_logits(r, cycle) for r in calib],
        [int(r["label"]) for r in calib],
        meta,
        [r.get("stage") for r in calib],
    )


def _probs(kind: str, logits: Sequence[float], cal: Calibrator, stage: str | None = None) -> list[float]:
    """Class probabilities for one row; noul becomes [P(false), P(true)]."""
    if kind == "choice":
        return cal.choice_probs(logits, stage or "letter")
    if kind == "score":
        return cal.score_probs(logits)
    p = cal.noul_prob(logits[0] - logits[1])
    return [1.0 - p, p]


def _metrics(
    kind: str,
    logit_rows: list[list[float]],
    labels: list[int],
    cal: Calibrator,
    stages: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    stages = stages if stages is not None else [None] * len(logit_rows)
    probs = [_probs(kind, l, cal, st) for l, st in zip(logit_rows, stages)]
    out = metrics.ragged_summarize(probs, labels)
    if kind == "noul":
        out["auroc"] = metrics.auroc([p[1] for p in probs], labels)
    if kind == "score":
        expected = [sum(i * p for i, p in enumerate(row)) for row in probs]
        out["mae"] = float(np.mean([abs(e - y) for e, y in zip(expected, labels)]))
        out["exact"] = out["accuracy"]
    return out


def _reliability(
    kind: str,
    logit_rows: list[list[float]],
    labels: list[int],
    cal: Calibrator,
    stages: Sequence[str | None] | None = None,
) -> list[dict]:
    """Top-label reliability bins (same shape as `metrics.reliability_bins`; rows may be ragged)."""
    stages = stages if stages is not None else [None] * len(logit_rows)
    probs = [_probs(kind, l, cal, st) for l, st in zip(logit_rows, stages)]
    conf = np.array([max(p) for p in probs])
    correct = np.array([float(int(np.argmax(p)) == y) for p, y in zip(probs, labels)])
    idx = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
    out = []
    for b in range(N_BINS):
        m = idx == b
        c = int(m.sum())
        out.append(
            {
                "lo": b / N_BINS,
                "hi": (b + 1) / N_BINS,
                "count": c,
                "confidence": float(conf[m].mean()) if c else None,
                "accuracy": float(correct[m].mean()) if c else None,
            }
        )
    return out


def _n_cycles(rows: Sequence[dict[str, Any]]) -> int:
    """Number of H cycles recorded on every row, or 0 if any row has none."""
    counts = set()
    for row in rows:
        cl = row.get("cycle_logits")
        if not cl or any(len(c) != len(row["logits"]) for c in cl):
            return 0
        counts.add(len(cl))
    return counts.pop() if len(counts) == 1 else 0


def _clean(value: Any) -> Any:
    """Replace NaN/inf by None so the summary is strict JSON."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(run_dir: str | Path) -> dict[str, Any]:
    """Metrics of one run on its test split, raw and calibrated, plus latency and per-cycle results."""
    run_dir = Path(run_dir)
    rows = load_raw(run_dir)
    cfg = load_config(run_dir)
    conf = cfg.get("config", {})
    run_id = cfg.get("run_id", run_dir.name)
    kind = _kind(rows, run_dir)
    calib = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]
    if not test:
        raise ValueError(f"{run_dir}: no test rows yet")
    labels = [int(r["label"]) for r in test]
    test_stages = [r.get("stage") for r in test]

    def both(cycle: int | None) -> tuple[dict[str, Any], dict[str, Any] | None, Calibrator | None]:
        logit_rows = [_logits(r, cycle) for r in test]
        raw = _metrics(kind, logit_rows, labels, Calibrator(), test_stages)
        if not calib or (cycle is not None and not _n_cycles(calib)):
            return raw, None, None
        cal = fit_calibrator(calib, run_id=run_id, cycle=cycle)
        return raw, _metrics(kind, logit_rows, labels, cal, test_stages), cal

    raw, calibrated, cal = both(None)
    latency = np.array([float(r["latency_ms"]) for r in test])
    stages: dict[str, int] = {}
    for r in test:
        stages[r["stage"]] = stages.get(r["stage"], 0) + 1
    backend = conf.get("backend")
    out: dict[str, Any] = {
        "run_id": run_id,
        "backend": backend,
        "task": cfg.get("task", {}).get("name") or conf.get("task"),
        "kind": kind,
        "shots_mode": conf.get("shots_mode"),
        "k": conf.get("k"),
        "attention": None if backend != "hrm" else ("prefix" if conf.get("prefix_lm", True) else "causal"),
        "seed": conf.get("seed"),
        "n_calib": len(calib),
        "n_test": len(test),
        "complete": cfg.get("complete"),
        "raw": raw,
        "calibrated": calibrated,
        "calibrator": cal.to_json() if cal else None,
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "mean": float(latency.mean()),
        },
        "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in test])),
        "truncated": int(sum(bool(r["truncated"]) for r in test)),
        "stages": stages,
        "reliability": _reliability(kind, [r["logits"] for r in test], labels, cal or Calibrator(), test_stages),
        "scorer": cfg.get("scorer"),
    }
    n_cycles = _n_cycles(test)
    if n_cycles:
        final = [int(np.argmax(r["logits"])) for r in test]
        cycles = []
        for c in range(n_cycles):
            c_raw, c_cal, c_fit = both(c)
            picks = [int(np.argmax(r["cycle_logits"][c])) for r in test]
            cycles.append(
                {
                    "cycle": c,
                    "raw": c_raw,
                    "calibrated": c_cal,
                    "calibrator": c_fit.to_json() if c_fit else None,
                    "agreement_with_final": float(np.mean([a == b for a, b in zip(picks, final)])),
                }
            )
        out["cycles"] = cycles
    return _clean(out)


# -- markdown ---------------------------------------------------------------------------


def _num(x: Any, digits: int = 3) -> str:
    return "–" if x is None else f"{x:.{digits}f}"


def _pair(summary: dict[str, Any], key: str) -> str:
    cal = summary.get("calibrated") or {}
    return f"{_num(summary['raw'].get(key))} → {_num(cal.get(key))}"


def _best(summary: dict[str, Any], key: str) -> Any:
    """The calibrated value of a metric when there is one, else the raw value."""
    return (summary.get("calibrated") or summary["raw"]).get(key)


def _shots(s: dict[str, Any]) -> str:
    if s.get("backend") != "hrm" or s.get("shots_mode") is None:
        return "–"
    return "zero" if s["shots_mode"] == "zero" else f"{s['shots_mode']}-{s['k']}"


def _table(header: list[str], body: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    return lines + ["| " + " | ".join(row) + " |" for row in body]


RUN_COLUMNS = [
    "run", "backend", "task", "kind", "shots", "attention", "n", "accuracy raw → cal",
    "ECE raw → cal", "NLL raw → cal", "Brier", "AUROC", "MAE", "p50 ms", "p95 ms", "tokens",
]  # fmt: skip
CYCLE_COLUMNS = ["run", "cycle", "accuracy raw → cal", "ECE raw → cal", "NLL raw → cal", "agrees with final"]


def render_markdown(summaries: Sequence[dict[str, Any]], skipped: Sequence[tuple[str, str]] = ()) -> str:
    """The comparison report: one row per run, then the per-cycle early-exit table."""
    lines = ["# Davout benchmark report", ""]
    body = []
    for s in summaries:
        body.append(
            [
                s["run_id"],
                str(s.get("backend") or "–"),
                str(s.get("task") or "–"),
                s["kind"],
                _shots(s),
                s.get("attention") or "–",
                str(s["n_test"]),
                _pair(s, "accuracy"),
                _pair(s, "ece"),
                _pair(s, "nll"),
                _num(_best(s, "brier")),
                _num(_best(s, "auroc")) if s["kind"] == "noul" else "–",
                _num(_best(s, "mae")) if s["kind"] == "score" else "–",
                _num(s["latency_ms"]["p50"], 0),
                _num(s["latency_ms"]["p95"], 0),
                _num(s["prompt_tokens_mean"], 0),
            ]
        )
    lines += ["## Runs", ""] + _table(RUN_COLUMNS, body) + [""]
    lines += [
        "Metrics are computed on each run's test split. \"raw\" uses the scorer's logits as they are; "
        "\"cal\" applies temperature scaling (choice, score) or Platt scaling (noul) fitted on the run's "
        "calibration split. Brier, AUROC and MAE are the calibrated values (raw when a run has no "
        "calibration rows). MAE is the mean distance between the expected level and the true level. "
        "Latency is per decision (one `score` call); tokens is the mean prompt length per decision, "
        "summed over every prompt or hypothesis the decision needed.",
        "",
    ]
    notes = []
    for s in summaries:
        if s.get("truncated"):
            notes.append(f"- {s['run_id']}: {s['truncated']} of {s['n_test']} test states were truncated.")
        if s.get("complete") is False:
            notes.append(f"- {s['run_id']}: the run is incomplete; the numbers cover the rows scored so far.")
        if not s.get("n_calib"):
            notes.append(f"- {s['run_id']}: no calibration rows, so only raw metrics are available.")
    if notes:
        lines += ["## Notes", ""] + notes + [""]

    recurrent = [s for s in summaries if s.get("cycles")]
    if recurrent:
        body = [
            [
                s["run_id"],
                f"{c['cycle'] + 1} of {len(s['cycles'])}",
                _pair(c, "accuracy"),
                _pair(c, "ece"),
                _pair(c, "nll"),
                _num(c["agreement_with_final"]),
            ]
            for s in recurrent
            for c in s["cycles"]
        ]
        lines += ["## Early exit: metrics after each H cycle", ""] + _table(CYCLE_COLUMNS, body) + [""]
        lines += [
            "Each row reads the answer after that H cycle, with its own calibrator fitted on the "
            "calibration split; the last cycle is the model's normal output. \"agrees with final\" is the "
            "share of test decisions whose top answer already matches the final cycle. Two-stage choice "
            "runs record no per-cycle logits and are not listed.",
            "",
        ]
    if skipped:
        lines += ["## Skipped", ""] + [f"- {name}: {why}" for name, why in skipped] + [""]
    return "\n".join(lines)


def write_report(run_dirs: Sequence[str | Path], out_path: str | Path) -> Path:
    """Write `summary.json` into every run directory and one markdown comparison to `out_path`."""
    summaries: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for run_dir in map(Path, run_dirs):
        try:
            summary = evaluate(run_dir)
        except ValueError as e:
            skipped.append((run_dir.name, str(e)))
            continue
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        summaries.append(summary)
    if not summaries:
        raise ValueError("no run could be evaluated: " + "; ".join(why for _, why in skipped))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(summaries, skipped))
    return out


def export_calibrator(run_dirs: Sequence[str | Path], out_path: str | Path) -> Calibrator:
    """Fit one serving `Calibrator` from the calibration rows of the given runs and save it.

    Rows are pooled per question type: choice runs give `choice_T`, score runs
    `score_T`, noul runs the Platt pair (two-stage choice rows fit `choice_two_stage_T` apart from `choice_T`); a type with no run keeps the identity.
    """
    pooled: dict[str, tuple[list[list[float]], list[int]]] = {k: ([], []) for k in KINDS}
    pooled_stages: dict[str, list[str | None]] = {k: [] for k in KINDS}
    runs: dict[str, list[str]] = {k: [] for k in KINDS}
    for run_dir in map(Path, run_dirs):
        calib = [r for r in load_raw(run_dir) if r["split"] == "calib"]
        if not calib:
            continue
        kind = _kind(calib, run_dir)
        pooled[kind][0].extend(r["logits"] for r in calib)
        pooled[kind][1].extend(int(r["label"]) for r in calib)
        pooled_stages[kind].extend(r.get("stage") for r in calib)
        runs[kind].append(load_config(run_dir).get("run_id", run_dir.name))
    params: dict[str, float] = {}
    for kind, (logit_rows, labels) in pooled.items():
        if not logit_rows:
            continue
        fitted = _fit(kind, logit_rows, labels, {}, pooled_stages[kind])
        if kind == "noul":
            params.update(noul_a=fitted.noul_a, noul_b=fitted.noul_b)
        elif kind == "choice":
            stages = pooled_stages[kind]
            if any(st != "two_stage" for st in stages):
                params["choice_T"] = fitted.choice_T
            if any(st == "two_stage" for st in stages):
                params["choice_two_stage_T"] = fitted.choice_two_stage_T
        else:
            params[f"{kind}_T"] = getattr(fitted, f"{kind}_T")
    if not params:
        raise ValueError("none of the given runs has calibration rows")
    meta = {
        "source": "davout bench",
        "split": "calib",
        "runs": {k: v for k, v in runs.items() if v},
        "n": {k: len(v[1]) for k, v in pooled.items() if v[1]},
    }
    cal = Calibrator(meta=meta, **params)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cal.save(out)
    return cal
