"""Early-exit analysis from the per-cycle logits already in raw.jsonl (offline, no model).

Usage (from the Davout checkout):
    uv run python finetune/analysis/early_exit.py            # the five single-prompt public tasks
    uv run python finetune/analysis/early_exit.py --private-router    # a private routing evaluation (data not public; keep the output git-ignored)

Per model and task: cycle 1 vs final (accuracy raw and calibrated, NLL raw and calibrated, agreement),
then a confidence gate: answer from cycle 1 when its calibrated top probability >= tau, else use the
final cycle. Each cycle has its own calibrator fitted on the calib split, as `davout bench report` does.
tau is chosen per task on the CALIB split as the smallest tau (grid 0.00..1.00 step 0.01, else "never")
whose combined calib accuracy is within 0.01 of final-only calib accuracy; everything reported is on TEST.
Compute saving = 0.5 x exit fraction (cycle 1 is 4 of the 8 stack passes). The current implementation
always runs both cycles, so this is the potential saving, not a measured one.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from davout.bench import report
from davout.calibrate import Calibrator

ROOT = Path(__file__).resolve().parents[2]
PRIVATE_ROUTER = "--private-router" in sys.argv
if PRIVATE_ROUTER:
    # Task and directory names below are the on-disk names of the private runs (git-ignored).
    TASKS = ["forum_needs_reply", "forum_responder", "forum_primary_choice"]
    RUNS = {
        "base": lambda t: ROOT / "results-forum" / f"hrm-{t}-zero-prefix-s0",
        "ftA": lambda t: ROOT / "results-forum-ftA" / f"hrm-ftA-{t}-zero-prefix-s0",
        "ftB": lambda t: ROOT / "results-forum-ftB" / f"hrm-ftB-{t}-zero-prefix-s0",
    }
else:
    TASKS = ["boolq", "sms_spam", "sst2", "ag_news", "yelp"]
    RUNS = {
        "base": lambda t: ROOT / "results" / f"hrm-{t}-zero-prefix-s0",
        "ftA": lambda t: ROOT / "results-ftA" / f"hrm-ftA-{t}-zero-prefix-s0",
        "ftB": lambda t: ROOT / "results-ftB" / f"hrm-ftB-{t}-zero-prefix-s0",
    }
GRID = [round(i / 100, 2) for i in range(101)]
CURVE = [0.6, 0.7, 0.8, 0.9, 0.95]
EPS = 1e-12
NEVER = float("inf")


def split_arrays(rows: list[dict], kind: str, split: str, cal1: Calibrator, calf: Calibrator) -> dict:
    sub = sorted((r for r in rows if r["split"] == split), key=lambda r: r["id"])
    y = np.array([int(r["label"]) for r in sub])
    out = {"y": y, "n": len(sub)}
    for name, cyc, cal in (("c1", 0, cal1), ("fin", None, calf)):
        logit_rows = [report._logits(r, cyc) for r in sub]
        raw = [report._probs(kind, l, Calibrator()) for l in logit_rows]
        calp = [report._probs(kind, l, cal) for l in logit_rows]
        out[name] = {
            "pred_raw": np.array([int(np.argmax(p)) for p in raw]),
            "pred_cal": np.array([int(np.argmax(p)) for p in calp]),
            "conf_cal": np.array([max(p) for p in calp]),
            "nll_raw": float(np.mean([-math.log(max(p[t], EPS)) for p, t in zip(raw, y)])),
            "nll_cal": float(np.mean([-math.log(max(p[t], EPS)) for p, t in zip(calp, y)])),
            "argmax_logits": np.array([int(np.argmax(l)) for l in logit_rows]),
        }
    return out


def gate(d: dict, tau: float) -> tuple[float, float, np.ndarray]:
    """(exit fraction, combined accuracy, combined predictions) at threshold tau."""
    exit_ = d["c1"]["conf_cal"] >= tau
    pred = np.where(exit_, d["c1"]["pred_cal"], d["fin"]["pred_cal"])
    return float(exit_.mean()), float((pred == d["y"]).mean()), pred


def choose_tau(calib: dict) -> float:
    final_acc = float((calib["fin"]["pred_cal"] == calib["y"]).mean())
    for tau in GRID:
        if gate(calib, tau)[1] >= final_acc - 0.01 - 1e-9:
            return tau
    return NEVER


def main() -> None:
    summary: dict = {}
    print("## Cycle 1 vs final (test split)\n")
    print("| task | model | n | c1 acc raw | c1 acc cal | final acc raw | final acc cal | c1 − final (cal) | c1 NLL raw / cal | final NLL raw / cal | agree (raw argmax) | agree (cal pred) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    store: dict = {}
    for t in TASKS:
        for m in RUNS:
            rows = report.load_raw(RUNS[m](t))
            kind = report._kind(rows)
            assert report._n_cycles(rows) == 2, (t, m)
            assert all(r["cycle_logits"][-1] == r["logits"] for r in rows), (t, m)
            cal1 = report.fit_calibrator(rows, cycle=0)
            calf = report.fit_calibrator(rows)
            calib = split_arrays(rows, kind, "calib", cal1, calf)
            test = split_arrays(rows, kind, "test", cal1, calf)
            store[(t, m)] = (calib, test)
            y = test["y"]
            a = {k: float((test[c][p] == y).mean()) for k, c, p in (("c1r", "c1", "pred_raw"), ("c1c", "c1", "pred_cal"), ("fr", "fin", "pred_raw"), ("fc", "fin", "pred_cal"))}
            agree_raw = float((test["c1"]["argmax_logits"] == test["fin"]["argmax_logits"]).mean())
            agree_cal = float((test["c1"]["pred_cal"] == test["fin"]["pred_cal"]).mean())
            summary[(t, m)] = {"c1_cal": a["c1c"], "fin_cal": a["fc"], "fin_nll_raw": test["fin"]["nll_raw"], "fin_nll_cal": test["fin"]["nll_cal"]}
            print(
                f"| {t} | {m} | {test['n']} | {a['c1r']:.3f} | {a['c1c']:.3f} | {a['fr']:.3f} | {a['fc']:.3f} | {a['c1c'] - a['fc']:+.3f} "
                f"| {test['c1']['nll_raw']:.3f} / {test['c1']['nll_cal']:.3f} | {test['fin']['nll_raw']:.3f} / {test['fin']['nll_cal']:.3f} | {agree_raw:.3f} | {agree_cal:.3f} |"
            )
    print()

    print("## Confidence gate, tau chosen on calib (smallest tau with combined calib accuracy within 0.01 of final-only)\n")
    print("| task | model | tau | calib: exit / combined / final | TEST exit frac | TEST combined acc | TEST final-only acc | Δ acc | rows changed (worse/better) | potential compute saving | ≥50% exit & within 0.01 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in TASKS:
        for m in RUNS:
            calib, test = store[(t, m)]
            tau = choose_tau(calib)
            ce, ca, _ = gate(calib, tau)
            cf = float((calib["fin"]["pred_cal"] == calib["y"]).mean())
            te, ta, pred = gate(test, tau)
            fin_ok = test["fin"]["pred_cal"] == test["y"]
            tf = float(fin_ok.mean())
            worse = int((fin_ok & (pred != test["y"])).sum())
            better = int((~fin_ok & (pred == test["y"])).sum())
            ok = te >= 0.5 and ta >= tf - 0.01 - 1e-9
            summary[(t, m)].update({"tau": tau, "exit": te, "comb": ta, "ok": ok, "n": test["n"], "comb_correct": ta * test["n"], "fin_correct": tf * test["n"]})
            tau_s = "never" if tau == NEVER else f"{tau:.2f}"
            print(f"| {t} | {m} | {tau_s} | {ce:.3f} / {ca:.3f} / {cf:.3f} | {te:.3f} | {ta:.3f} | {tf:.3f} | {ta - tf:+.3f} | {worse}/{better} | {0.5 * te:.1%} | {'yes' if ok else 'no'} |")
    print()

    print("## Gate curve on TEST at fixed tau: exit fraction / combined accuracy (final-only accuracy in the last column)\n")
    print("| task | model | " + " | ".join(f"tau {x}" for x in CURVE) + " | always exit (c1 only) | final only |")
    print("|---|---|" + "---|" * (len(CURVE) + 2))
    for t in TASKS:
        for m in RUNS:
            _, test = store[(t, m)]
            cells = []
            for tau in CURVE:
                e, a, _ = gate(test, tau)
                cells.append(f"{e:.2f} / {a:.3f}")
            best = [tau for tau in GRID if gate(test, tau)[0] >= 0.5 and gate(test, tau)[1] >= summary[(t, m)]["fin_cal"] - 0.01 - 1e-9]
            summary[(t, m)]["oracle"] = bool(best)
            print(f"| {t} | {m} | " + " | ".join(cells) + f" | {summary[(t, m)]['c1_cal']:.3f} | {summary[(t, m)]['fin_cal']:.3f} |")
    print()

    print("## Spec D5 'early exit usable', applied literally" + (" (router files: for reference only, the criterion names the five public single-prompt tasks)" if PRIVATE_ROUTER else "") + "\n")
    for m in RUNS:
        within = [t for t in TASKS if summary[(t, m)]["c1_cal"] >= summary[(t, m)]["fin_cal"] - 0.03 - 1e-9]
        gate_ok = [t for t in TASKS if summary[(t, m)]["ok"]]
        oracle_ok = [t for t in TASKS if summary[(t, m)]["oracle"]]
        n = sum(summary[(t, m)]["n"] for t in TASKS)
        pooled_exit = sum(summary[(t, m)]["exit"] * summary[(t, m)]["n"] for t in TASKS) / n
        pooled_comb = sum(summary[(t, m)]["comb_correct"] for t in TASKS) / n
        pooled_fin = sum(summary[(t, m)]["fin_correct"] for t in TASKS) / n
        print(f"### {m}")
        print(f"- (a) cycle-1 calibrated accuracy within 0.03 of final: {len(within)} of {len(TASKS)} tasks ({', '.join(within) or 'none'})" + ("" if PRIVATE_ROUTER else f" → {'MET' if len(within) >= 4 else 'NOT met'} (needs ≥ 4 of 5)"))
        print(f"- (b) per task, calib-chosen tau, on test: ≥50% exit with combined accuracy within 0.01 of final on {len(gate_ok)} of {len(TASKS)} tasks ({', '.join(gate_ok) or 'none'})")
        print(f"  pooled over the {len(TASKS)} tasks ({n} test decisions, per-task calib-chosen tau): exit {pooled_exit:.3f}, combined accuracy {pooled_comb:.3f} vs final-only {pooled_fin:.3f} (Δ {pooled_comb - pooled_fin:+.3f}) → {'MET' if pooled_exit >= 0.5 and pooled_comb >= pooled_fin - 0.01 - 1e-9 else 'NOT met'} on the pooled reading")
        print(f"  best case (tau picked on the test rows themselves, optimistic): possible on {len(oracle_ok)} of {len(TASKS)} tasks ({', '.join(oracle_ok) or 'none'})")
        print()
    if "ftA" in RUNS and "ftB" in RUNS:
        a = np.mean([summary[(t, "ftA")]["fin_nll_raw"] for t in TASKS])
        b = np.mean([summary[(t, "ftB")]["fin_nll_raw"] for t in TASKS])
        ac = np.mean([summary[(t, "ftA")]["fin_nll_cal"] for t in TASKS])
        bc = np.mean([summary[(t, "ftB")]["fin_nll_cal"] for t in TASKS])
        print(f"- ftB final readout vs ftA, mean NLL over these {len(TASKS)} tasks: raw {a:.4f} → {b:.4f} (Δ {b - a:+.4f}, {'within' if abs(b - a) <= 0.02 else 'NOT within'} 0.02); calibrated {ac:.4f} → {bc:.4f} (Δ {bc - ac:+.4f}, {'within' if abs(bc - ac) <= 0.02 else 'NOT within'} 0.02)")


if __name__ == "__main__":
    main()
