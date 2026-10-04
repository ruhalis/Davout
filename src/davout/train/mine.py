"""Mine hard negatives for final-choice rows: what a model's candidate stage ranks highest.

For every utterance of a recipe's mined sources (`S.MINED_SOURCES`) each label of the source's
label set is judged with the Yes/No candidate prompt, exactly as the first stage of a large
Choice does at serving time (bare utterance, the label set's first instruction, spaced names).
`<out_dir>/<source>.jsonl` gets one line per usable train row::

    {"id": "<source>/<split>/<row>", "gold": "<raw label>", "gold_rank": int, "wrong": ["<raw label>", ...]}

`wrong` holds the `keep` best-ranked wrong labels, best first; `gold_rank` is 0 when the gold
label ranks first. `davout.train.data.build(..., negatives=out_dir)` turns them into shortlists.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from davout.prompts import build_candidate_prompt
from davout.schema import ChoiceQuestion
from davout.train import sources as S
from davout.train.data import hf_loader, normalise
from davout.train.sources import LabelSet, Source

KEEP = 15  # wrong labels recorded per row
ROWS_PER_READ = 64  # utterances scored per `backend.read` call


def rank_labels(backend: Any, texts: Sequence[str], labels: LabelSet) -> list[list[int]]:
    """For each text, the label indices as the candidate stage ranks them, best first.

    The prompt is the first stage of a large Choice at serving time: the bare text, the label
    set's first instruction and the spaced label names without descriptions.
    """
    question = ChoiceQuestion(labels.choice_instructions[0], {label.name: None for label in labels.labels})
    n = len(labels.labels)
    out: list[list[int]] = []
    for start in range(0, len(texts), ROWS_PER_READ):
        part = texts[start : start + ROWS_PER_READ]
        outs = backend.read([build_candidate_prompt(text, question, label.name) for text in part for label in labels.labels])
        for j in range(len(part)):
            scores = [r.logits[0] - r.logits[1] for r in outs[j * n : (j + 1) * n]]
            out.append(sorted(range(n), key=lambda k: (-scores[k], k)))
    return out


def mine(
    out_dir: str | Path,
    backend: Any,
    recipe: str = "intent_hn_v1",
    *,
    loader: Callable[[Source, str], Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Write the mined negatives of every mined source of `recipe`; returns per-source statistics.

    `backend` is a `LabelBackend` (the model whose candidate stage ranks the labels); `limit`
    stops after that many rows per split (timing runs).
    """
    if recipe not in S.RECIPES:
        raise ValueError(f"unknown recipe {recipe!r}; available: {', '.join(S.RECIPES)}")
    registry = S.RECIPES[recipe]
    names = [n for n in registry if n in S.MINED_SOURCES]
    if not names:
        raise ValueError(f"recipe {recipe!r} has no sources with mined negatives")
    loader = loader or hf_loader
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}
    for name in names:
        src = registry[name]
        setname, labels = next(iter(S.label_sets(name).items()))
        question = ChoiceQuestion(labels.choice_instructions[0], {label.name: None for label in labels.labels})
        index = {label.raw: i for i, label in enumerate(labels.labels)}
        n = len(labels.labels)
        done = top1 = top10 = prompts_read = 0
        t0 = time.perf_counter()
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for split in src.train_splits:
                rows = loader(src, split)
                todo = []
                for idx in range(len(rows) if limit is None else min(limit, len(rows))):
                    item = normalise(name, rows[idx], random.Random(0))
                    if not isinstance(item, str):
                        todo.append((idx, item))
                for start in range(0, len(todo), ROWS_PER_READ):
                    part = todo[start : start + ROWS_PER_READ]
                    outs = backend.read(
                        [build_candidate_prompt(item["text"], question, label.name) for _idx, item in part for label in labels.labels]
                    )
                    prompts_read += len(outs)
                    for j, (idx, item) in enumerate(part):
                        scores = [r.logits[0] - r.logits[1] for r in outs[j * n : (j + 1) * n]]
                        order = sorted(range(n), key=lambda k: (-scores[k], k))
                        gold = index[item["golds"][setname][0].raw]
                        rank = order.index(gold)
                        wrong = [labels.labels[k].raw for k in order if k != gold][:KEEP]
                        line = {"id": f"{name}/{split}/{idx}", "gold": labels.labels[gold].raw, "gold_rank": rank, "wrong": wrong}
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                        done += 1
                        top1 += rank == 0
                        top10 += rank < 10
                    if (start // ROWS_PER_READ) % 10 == 0:
                        rate = prompts_read / (time.perf_counter() - t0)
                        print(f"[mine] {name}/{split} {done}/{len(todo)} rows, {rate:.0f} prompts/s", file=sys.stderr, flush=True)
        stats[name] = {
            "rows": done, "labels": n, "prompts": prompts_read, "seconds": round(time.perf_counter() - t0, 1),
            "gold_top1": round(top1 / max(1, done), 4), "gold_top10": round(top10 / max(1, done), 4),
        }  # fmt: skip
    (out_dir / "stats.json").write_text(json.dumps({"recipe": recipe, "keep": KEEP, "sources": stats}, indent=2) + "\n")
    return stats
