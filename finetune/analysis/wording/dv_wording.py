"""Wording-robustness checks (ad hoc, remote /tmp). One decision per score call, zero-shot, like davout bench.

usage: python dv_wording.py <base|checkpoint path> <out.json>
"""
import json
import sys
import time

from davout.backends.hrm import MODEL_ID, HrmBackend
from davout.bench.tasks import TASKS, load_task
from davout.schema import ChoiceQuestion, NoulQuestion
from davout.scorer import LetterScorer

model, out_path = sys.argv[1], sys.argv[2]
backend = HrmBackend(MODEL_ID if model == "base" else model, device="auto", batch_size=8)
scorer = LetterScorer(backend, shots=0)
print("loaded", backend.info(), flush=True)


def run(name, examples, question, extra=None):
    t0 = time.perf_counter()
    rows = []
    for split, ex in examples:
        raw = scorer.score(ex.state, [question])[0]
        rows.append(
            {
                "id": ex.id,
                "split": split,
                "label": ex.label,
                "logits": [float(x) for x in raw.logits],
                "cycle_logits": raw.cycle_logits,
                "prompt_tokens": int(raw.prompt_tokens),
            }
        )
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(rows))
    print(f"{name}: {len(rows)} rows, {ms:.1f} ms/decision", flush=True)
    return {"question": {"type": question.type, "instructions": question.instructions, "criteria": question.criteria}, "rows": rows, **(extra or {})}


def both(task):
    return [("calib", e) for e in task.calib] + [("test", e) for e in task.test]


def test_only(task):
    return [("test", e) for e in task.test]


res = {"model": model, "info": {k: str(v) for k, v in backend.info().items()}, "checks": {}}

sms = load_task("sms_spam")
scorer.score(sms.test[0].state, [TASKS["sms_spam"].question])  # warm-up
res["checks"]["sms_registered"] = run("sms_registered", both(sms), TASKS["sms_spam"].question)
res["checks"]["sms_bare_statement"] = run("sms_bare_statement", both(sms), NoulQuestion("The message is spam.", None))

ag = load_task("ag_news")
reg = TASKS["ag_news"].question
names = list(reg.criteria)
orders = {
    "ag_order_registered": names,  # world, sports, business, sci_tech
    "ag_order_reversed": names[::-1],  # sci_tech, business, sports, world
    "ag_order_perm2": [names[2], names[0], names[3], names[1]],  # business, world, sci_tech, sports
}
for key, order in orders.items():
    q = ChoiceQuestion(reg.instructions, {n: reg.criteria[n] for n in order})
    res["checks"][key] = run(key, test_only(ag), q, {"order": order, "registered_names": names})

sst = load_task("sst2")
res["checks"]["sst2_registered_statement"] = run("sst2_registered_statement", both(sst), TASKS["sst2"].question)
res["checks"]["sst2_question_form"] = run(
    "sst2_question_form", both(sst), NoulQuestion("Does the review express a positive sentiment?", None)
)

with open(out_path, "w") as f:
    json.dump(res, f)
print("wrote", out_path, flush=True)
