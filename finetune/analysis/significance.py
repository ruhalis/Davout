"""Paired significance on the six benchmark tasks: base vs ftA vs ftB vs OpenJev (offline, no model).

Usage (from the Davout checkout):  uv run python finetune/analysis/significance.py

Reads raw.jsonl of the zero-shot seed-0 runs (300 calib + 300 test, identical rows across models).
Accuracy uses calibrated predictions as `davout bench report` does (temperature for choice/score,
Platt for noul, fitted on the run's calib split and held fixed during resampling); NLL and ECE are
raw (identity calibrator). Paired bootstrap: 10,000 resamples of the 300 test rows, seed 0,
percentile 95% interval. McNemar: exact two-sided binomial on the discordant pairs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from davout.bench import report
from davout.calibrate import Calibrator

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["boolq", "sms_spam", "sst2", "ag_news", "yelp", "banking77"]
RUNS = {
    "base": lambda t: ROOT / "results" / f"hrm-{t}-zero-prefix-s0",
    "ftA": lambda t: ROOT / "results-ftA" / f"hrm-ftA-{t}-zero-prefix-s0",
    "ftB": lambda t: ROOT / "results-ftB" / f"hrm-ftB-{t}-zero-prefix-s0",
    "openjev": lambda t: ROOT / "results" / f"openjev-{t}-s0",
}
PAIRS = [("base", "ftA"), ("base", "ftB"), ("ftA", "ftB"), ("openjev", "ftA")]  # diff = second - first
B = 10_000
EPS = 1e-12


def load(model: str, task: str) -> dict:
    run_dir = RUNS[model](task)
    rows = report.load_raw(run_dir)
    kind = report._kind(rows, run_dir)
    cal = report.fit_calibrator(rows)
    test = sorted((r for r in rows if r["split"] == "test"), key=lambda r: r["id"])
    labels = np.array([int(r["label"]) for r in test])
    stages = [r.get("stage") for r in test]
    logit_rows = [r["logits"] for r in test]
    raw = [report._probs(kind, l, Calibrator(), st) for l, st in zip(logit_rows, stages)]
    calp = [report._probs(kind, l, cal, st) for l, st in zip(logit_rows, stages)]
    out = {
        "kind": kind,
        "ids": [r["id"] for r in test],
        "labels": labels,
        "correct_cal": np.array([int(np.argmax(p)) == y for p, y in zip(calp, labels)], dtype=float),
        "correct_raw": np.array([int(np.argmax(p)) == y for p, y in zip(raw, labels)], dtype=float),
        "nll_raw": np.array([-math.log(max(p[y], EPS)) for p, y in zip(raw, labels)]),
        "nll_cal": np.array([-math.log(max(p[y], EPS)) for p, y in zip(calp, labels)]),
        "m_raw": report._metrics(kind, logit_rows, list(labels), Calibrator(), stages),
        "m_cal": report._metrics(kind, logit_rows, list(labels), cal, stages),
        "p50_ms": float(np.percentile([r["latency_ms"] for r in test], 50)),
    }
    if kind == "noul":
        out["score"] = np.array([p[1] for p in calp])
    if kind == "score":
        out["abs_err"] = np.array([abs(sum(i * q for i, q in enumerate(p)) - y) for p, y in zip(calp, labels)])
    # cross-check against the summary.json davout wrote on the remote
    summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else None
    if summary:
        assert abs(summary["raw"]["nll"] - out["m_raw"]["nll"]) < 1e-9, (model, task)
        assert abs(summary["calibrated"]["accuracy"] - out["m_cal"]["accuracy"]) < 1e-9, (model, task)
    return out


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """(first right & second wrong, first wrong & second right, exact two-sided p)."""
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n, k = n10 + n01, min(n10, n01)
    if n == 0:
        return n10, n01, 1.0
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n)
    return n10, n01, p


def boot_mean(d: np.ndarray, idx: np.ndarray) -> tuple[float, float, float]:
    b = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def auroc_weights(score: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """AUROC for each row of resample counts `w` [B, n] (ties count half)."""
    pos, neg = y == 1, y == 0
    gt = (score[pos][:, None] > score[neg][None, :]).astype(float) + 0.5 * (score[pos][:, None] == score[neg][None, :])
    wp, wn = w[:, pos], w[:, neg]
    return np.einsum("bi,ij,bj->b", wp, gt, wn) / (wp.sum(1) * wn.sum(1))


def fmt_ci(t: tuple[float, float, float], digits: int = 3) -> str:
    star = "*" if (t[1] > 0 or t[2] < 0) else " "
    return f"{t[0]:+.{digits}f} [{t[1]:+.{digits}f}, {t[2]:+.{digits}f}]{star}"


def main() -> None:
    rng = np.random.default_rng(0)
    data = {t: {m: load(m, t) for m in RUNS} for t in TASKS}
    for t in TASKS:  # identical rows across models
        ref = data[t]["base"]
        for m in RUNS:
            assert data[t][m]["ids"] == ref["ids"] and (data[t][m]["labels"] == ref["labels"]).all(), (t, m)
        assert len(ref["ids"]) == 300

    print("## Benchmark table (test split, n=300 per task, zero-shot, seed 0)\n")
    print("| task | kind | model | acc cal (raw) | NLL raw (cal) | ECE raw (cal) | AUROC | MAE | p50 ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        for m in RUNS:
            d = data[t][m]
            r, c = d["m_raw"], d["m_cal"]
            au = f"{c['auroc']:.3f}" if "auroc" in c else "–"
            mae = f"{c['mae']:.3f}" if "mae" in c else "–"
            print(
                f"| {t} | {d['kind']} | {m} | {c['accuracy']:.3f} ({r['accuracy']:.3f}) | {r['nll']:.3f} ({c['nll']:.3f}) "
                f"| {r['ece']:.3f} ({c['ece']:.3f}) | {au} | {mae} | {d['p50_ms']:.0f} |"
            )
    print()
    print("Mean raw NLL over the six tasks: " + ", ".join(f"{m} {np.mean([data[t][m]['m_raw']['nll'] for t in TASKS]):.4f}" for m in RUNS))
    print("Mean calibrated NLL over the six tasks: " + ", ".join(f"{m} {np.mean([data[t][m]['m_cal']['nll'] for t in TASKS]):.4f}" for m in RUNS))
    print("Mean calibrated accuracy over the six tasks: " + ", ".join(f"{m} {np.mean([data[t][m]['m_cal']['accuracy'] for t in TASKS]):.4f}" for m in RUNS))
    print()

    print("## Paired differences (second minus first), 95% paired-bootstrap CI; * = CI excludes 0\n")
    print("| task | pair | d acc (cal) | McNemar b/c, p | d raw NLL | d cal NLL | d AUROC or d MAE |")
    print("|---|---|---|---|---|---|---|")
    res: dict = {}
    for t in TASKS:
        n = 300
        idx = rng.integers(0, n, size=(B, n))
        w = np.zeros((B, n))
        np.add.at(w, (np.arange(B)[:, None], idx), 1.0)
        for a, b in PAIRS:
            A, Bm = data[t][a], data[t][b]
            dacc = boot_mean(Bm["correct_cal"] - A["correct_cal"], idx)
            n10, n01, p = mcnemar(A["correct_cal"], Bm["correct_cal"])
            dnll = boot_mean(Bm["nll_raw"] - A["nll_raw"], idx)
            dnllc = boot_mean(Bm["nll_cal"] - A["nll_cal"], idx)
            extra = "–"
            if A["kind"] == "noul":
                y = A["labels"]
                diff = auroc_weights(Bm["score"], y, w) - auroc_weights(A["score"], y, w)
                point = Bm["m_cal"]["auroc"] - A["m_cal"]["auroc"]
                extra = "AUROC " + fmt_ci((point, float(np.nanpercentile(diff, 2.5)), float(np.nanpercentile(diff, 97.5))))
            elif A["kind"] == "score":
                extra = "MAE " + fmt_ci(boot_mean(Bm["abs_err"] - A["abs_err"], idx))
            res[(t, a, b)] = {"dacc": dacc, "dnll": dnll, "mcnemar_p": p}
            print(f"| {t} | {a} → {b} | {fmt_ci(dacc)} | {n10}/{n01}, p={p:.3g} | {fmt_ci(dnll)} | {fmt_ci(dnllc)} | {extra} |")
    print()

    print("## Spec D5, applied literally\n")
    for arm in ("ftA", "ftB"):
        base_mean = np.mean([data[t]["base"]["m_raw"]["nll"] for t in TASKS])
        arm_mean = np.mean([data[t][arm]["m_raw"]["nll"] for t in TASKS])
        improved, improved_nll_only, worse_nll, worse_acc, ece_up = [], [], [], [], []
        for t in TASKS:
            r = res[(t, "base", arm)]
            nll_better = r["dnll"][2] < 0
            acc_up = r["dacc"][0] >= 0.05 - 1e-9
            if nll_better or acc_up:
                improved.append(f"{t} ({'NLL CI<0' if nll_better else ''}{' & ' if nll_better and acc_up else ''}{'acc +%.3f' % r['dacc'][0] if acc_up else ''})")
            if nll_better:
                improved_nll_only.append(t)
            if r["dnll"][0] > 0.05:
                worse_nll.append(f"{t} (+{r['dnll'][0]:.3f})")
            if r["dacc"][0] < -0.05:
                worse_acc.append(f"{t} ({r['dacc'][0]:+.3f})")
            d_ece = data[t][arm]["m_raw"]["ece"] - data[t]["base"]["m_raw"]["ece"]
            if d_ece > 0.05:
                ece_up.append(f"{t} (+{d_ece:.3f})")
        print(f"### {arm} vs base")
        print(f"- mean raw NLL over six tasks: base {base_mean:.4f} → {arm} {arm_mean:.4f} ({'lower: PASS' if arm_mean < base_mean else 'not lower: FAIL'})")
        print(f"- tasks improved (raw-NLL CI excludes 0 in the better direction, or calibrated accuracy ≥ +0.05): {len(improved)} of 6 ({'PASS' if len(improved) >= 4 else 'FAIL'}; need ≥ 4): {', '.join(improved) or 'none'}")
        print(f"  (by the NLL-CI clause alone: {len(improved_nll_only)} of 6: {', '.join(improved_nll_only) or 'none'})")
        print(f"- tasks worse by more than 0.05 raw NLL: {', '.join(worse_nll) or 'none'}; worse by more than 0.05 calibrated accuracy: {', '.join(worse_acc) or 'none'}")
        print("  raw NLL change per task: " + ", ".join(f"{t} {res[(t, 'base', arm)]['dnll'][0]:+.3f}" for t in TASKS))
        print(f"- raw ECE up by more than 0.05: {len(ece_up)} task(s) ({', '.join(ece_up) or 'none'}); 'overfit' needs ≥ 3")
        print("  raw ECE change per task: " + ", ".join(f"{t} {data[t][arm]['m_raw']['ece'] - data[t]['base']['m_raw']['ece']:+.3f}" for t in TASKS))
        print()
    for label, tasks in (("six tasks", TASKS), ("five single-prompt tasks", TASKS[:5])):
        a = np.mean([data[t]["ftA"]["m_raw"]["nll"] for t in tasks])
        b = np.mean([data[t]["ftB"]["m_raw"]["nll"] for t in tasks])
        ac = np.mean([data[t]["ftA"]["m_cal"]["nll"] for t in tasks])
        bc = np.mean([data[t]["ftB"]["m_cal"]["nll"] for t in tasks])
        print(f"- ftB final readout vs ftA, mean NLL over {label}: raw {a:.4f} → {b:.4f} (Δ {b - a:+.4f}, {'within' if abs(b - a) <= 0.02 else 'NOT within'} 0.02); calibrated {ac:.4f} → {bc:.4f} (Δ {bc - ac:+.4f}, {'within' if abs(bc - ac) <= 0.02 else 'NOT within'} 0.02)")


if __name__ == "__main__":
    main()
