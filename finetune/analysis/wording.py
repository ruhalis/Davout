"""Wording-robustness checks: analysis of the logits scored on the GPU box (offline, no model).

Usage (from the Davout checkout):  uv run python finetune/analysis/wording.py

Inputs: finetune/analysis/wording/dv_wording_{base,ftA,ftB}.json, written on the remote by an ad hoc
script (one decision per score call, zero-shot, the same 300 calib + 300 test rows as the benchmark).
Noul accuracy "cal" uses Platt scaling fitted on the calib rows (a calib-fitted threshold); "raw" is
P(yes) >= 0.5. AUROC is on the test rows (rank statistic, unaffected by calibration when the slope is positive).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from davout import metrics
from davout.bench import report
from davout.calibrate import Calibrator, fit_platt

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent / "wording"
MODELS = ["base", "ftA", "ftB"]
BENCH = {
    "base": lambda t: ROOT / "results" / f"hrm-{t}-zero-prefix-s0",
    "ftA": lambda t: ROOT / "results-ftA" / f"hrm-ftA-{t}-zero-prefix-s0",
    "ftB": lambda t: ROOT / "results-ftB" / f"hrm-ftB-{t}-zero-prefix-s0",
}


def noul(rows: list[dict], rng: np.random.Generator) -> dict:
    calib = [r for r in rows if r["split"] == "calib"]
    test = [r for r in rows if r["split"] == "test"]
    a, b = fit_platt([r["logits"][0] - r["logits"][1] for r in calib], [r["label"] for r in calib])
    cal = Calibrator(noul_a=a, noul_b=b)
    y = np.array([r["label"] for r in test])
    m = np.array([r["logits"][0] - r["logits"][1] for r in test])
    p_cal = np.array([cal.noul_prob(x) for x in m])
    auroc = metrics.auroc(m, y)
    boots = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        v = metrics.auroc(m[i], y[i])
        if v == v:
            boots.append(v)
    p_raw = 1 / (1 + np.exp(-m))
    nll_raw = float(-np.mean(np.log(np.clip(np.where(y == 1, p_raw, 1 - p_raw), 1e-12, None))))
    return {
        "auroc": auroc,
        "ci": (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
        "acc_raw": float(((m >= 0) == (y == 1)).mean()),
        "acc_cal": float(((p_cal >= 0.5) == (y == 1)).mean()),
        "nll_raw": nll_raw,
        "yes_rate_raw": float((m >= 0).mean()),
        "pos_rate": float(y.mean()),
        "platt": (a, b),
        "margins": m,
        "y": y,
    }


def same_as_bench(rows: list[dict], model: str, task: str) -> tuple[int, int, float]:
    bench = {r["id"]: r for r in report.load_raw(BENCH[model](task))}
    same = sum(1 for r in rows if r["id"] in bench and bench[r["id"]]["label"] == r["label"] and bench[r["id"]]["split"] == r["split"])
    diff = max(abs(x - y) for r in rows for x, y in zip(r["logits"], bench[r["id"]]["logits"]))
    return same, len(rows), diff


def main() -> None:
    data = {m: json.loads((HERE / f"dv_wording_{m}.json").read_text()) for m in MODELS}
    rng = np.random.default_rng(0)

    print("## Sanity: the ad hoc script reproduces the benchmark rows and logits for the registered wording\n")
    for m in MODELS:
        for key, task in (("sms_registered", "sms_spam"), ("ag_order_registered", "ag_news"), ("sst2_registered_statement", "sst2")):
            same, n, diff = same_as_bench(data[m]["checks"][key]["rows"], m, task)
            print(f"- {m} {task}: {same}/{n} rows match the benchmark run by id, split and label; max |logit difference| {diff:.4f}")
    print()

    for title, keys in (
        ("SMS spam: old bare statement vs the registered question + criteria", (("sms_bare_statement", 'bare statement "The message is spam."'), ("sms_registered", "registered: question + both criteria"))),
        ("SST-2: question form vs the registered statement", (("sst2_question_form", 'question "Does the review express a positive sentiment?"'), ("sst2_registered_statement", 'registered statement "The review expresses a positive sentiment."'))),
    ):
        print(f"## {title} (test n=300)\n")
        print("| wording | model | AUROC [95% CI] | accuracy raw | accuracy at calib-fitted threshold | raw NLL | raw yes-rate (positives) |")
        print("|---|---|---|---|---|---|---|")
        res = {}
        for key, label in keys:
            for m in MODELS:
                r = noul(data[m]["checks"][key]["rows"], rng)
                res[(key, m)] = r
                print(f"| {label} | {m} | {r['auroc']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}] | {r['acc_raw']:.3f} | {r['acc_cal']:.3f} | {r['nll_raw']:.3f} | {r['yes_rate_raw']:.3f} ({r['pos_rate']:.3f}) |")
        print()
        (k1, _), (k2, _) = keys
        for m in MODELS:
            a, b = res[(k1, m)], res[(k2, m)]
            y = a["y"]
            d = []
            for _ in range(2000):
                i = rng.integers(0, len(y), len(y))
                va, vb = metrics.auroc(a["margins"][i], y[i]), metrics.auroc(b["margins"][i], y[i])
                if va == va and vb == vb:
                    d.append(va - vb)
            print(f"- {m}: AUROC gap (first wording minus registered) {a['auroc'] - b['auroc']:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]; calibrated accuracy gap {a['acc_cal'] - b['acc_cal']:+.3f}")
        print()

    print("## AG News under three option orders (test n=300, argmax of raw letter logits)\n")
    orders = ["ag_order_registered", "ag_order_reversed", "ag_order_perm2"]
    print("| model | " + " | ".join(f"acc {o.replace('ag_order_', '')}" for o in orders) + " | spread (max − min) | choice changes with order (any of 3) | registered vs reversed | registered vs perm2 | reversed vs perm2 | all three agree and correct | majority-vote acc |")
    print("|---|" + "---|" * (len(orders) + 7))
    for m in MODELS:
        picks, accs = {}, {}
        for o in orders:
            c = data[m]["checks"][o]
            names, order = c["registered_names"], c["order"]
            rows = c["rows"]
            y = np.array([r["label"] for r in rows])  # index in the registered order
            pick = np.array([names.index(order[int(np.argmax(r["logits"]))]) for r in rows])
            picks[o], accs[o] = pick, float((pick == y).mean())
        p = np.stack([picks[o] for o in orders])
        changed = float((p.min(0) != p.max(0)).mean())
        pair = lambda i, j: float((p[i] != p[j]).mean())  # noqa: E731
        all_ok = float(((p.min(0) == p.max(0)) & (p[0] == y)).mean())
        vote = np.array([np.bincount(col, minlength=4).argmax() if np.bincount(col, minlength=4).max() > 1 else col[0] for col in p.T])
        print(f"| {m} | " + " | ".join(f"{accs[o]:.3f}" for o in orders) + f" | {max(accs.values()) - min(accs.values()):.3f} | {changed:.3f} | {pair(0, 1):.3f} | {pair(0, 2):.3f} | {pair(1, 2):.3f} | {all_ok:.3f} | {float((vote == y).mean()):.3f} |")
    print()
    print("Orders: registered = world, sports, business, sci_tech; reversed = sci_tech, business, sports, world; perm2 = business, world, sci_tech, sports.")
    print("Letter picked (A/B/C/D share) per order, to show position bias; the gold labels are near-uniform:\n")
    for m in MODELS:
        parts = []
        for o in orders:
            rows = data[m]["checks"][o]["rows"]
            share = np.bincount([int(np.argmax(r["logits"])) for r in rows], minlength=4) / len(rows)
            parts.append(f"{o.replace('ag_order_', '')} " + "/".join(f"{s:.2f}" for s in share))
        print(f"- {m}: " + "; ".join(parts))


if __name__ == "__main__":
    main()
