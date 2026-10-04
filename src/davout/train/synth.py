"""Teacher-labelled intent rows: synthetic taxonomies, shortlists and soft targets.

Three steps, each a resumable file transformation (nothing here is part of the default mix):

1. `generate`: a teacher model writes intent taxonomies for customer-support domains other
   than banking, and a dozen utterances per intent -> `taxonomies.jsonl`, `utterances.jsonl`.
2. `decisions`: utterances are de-duplicated and filtered (held-out texts, banking-like
   intents), a student model's candidate stage ranks each domain's labels, and every utterance
   becomes a ten-option final-choice decision -> `decisions.jsonl` (rows the teacher can score).
3. `soft_rows`: teacher distributions (`davout teacher`) are merged into training rows with a
   `target` -> `<out>/train.jsonl`, `<out>/dev_in.jsonl` for `build(recipe="teacher_intent_v1", extra=out)`.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from davout.train import sources as S
from davout.train.data import TextGuard, _column, _key, hf_loader, shortlist_example, wrap_text
from davout.train.mine import KEEP, rank_labels
from davout.train.sources import Label, LabelSet, Source

SOURCE = "synth_intent"
DOMAINS: tuple[str, ...] = (
    "a mobile phone and internet provider", "an airline", "a home and car insurance claims desk",
    "an online clothing store", "an electricity and gas utility", "a hospital appointment and scheduling line",
    "a project-management software product", "a parcel delivery and logistics company", "a hotel chain",
    "a ride-hailing app", "a city council and government services office", "a university registrar and admissions office",
    "a food delivery app", "a streaming video service", "a car rental company", "a gym and fitness club chain",
    "a property rental and letting agency", "a consumer electronics repair centre", "a furniture and home goods retailer",
    "a train and bus ticketing service", "a veterinary clinic", "a pharmacy and prescription service",
    "a web hosting and domain registrar", "an online learning platform", "a job board and recruiting platform",
    "a travel agency and tour operator", "a home broadband installation and repair service", "a video game publisher's player support",
    "a smart-home device maker", "a grocery supermarket's online ordering service", "a car dealership and service garage",
    "an event ticketing platform", "a public library", "a dental practice", "a moving and storage company",
    "a home cleaning and handyman marketplace", "a social media platform's help centre", "a cloud storage and email provider",
    "a water and waste collection utility", "a cinema chain",
)  # fmt: skip
INTENTS_PER_DOMAIN = 40
UTTERANCES_PER_INTENT = 12
INSTRUCTIONS = (
    "Which intent does the customer's message express?",
    "What is the customer asking for?",
    "Classify the customer message by intent.",
    "Which of these intents matches the customer's request?",
    "What does the customer want?",
)
# Too close to the held-out banking benchmark: intents about money movement, cards or charges are dropped.
BANKING_LIKE = re.compile(
    r"\b(bank\w*|cards?|loans?|mortgage\w*|payments?|pay|paid|paying|refund\w*|charges?|charged|fees?|transfers?|transferred|"
    r"deposits?|withdraw\w*|balance|top ?ups?|currency|exchange rate|atm|pin|wallet|billing|bills?|billed|invoices?|credit|debit|"
    r"price\w*|costs?|financ\w*|money)\b",
    re.IGNORECASE,
)
_BANK_TEXT = re.compile(r"\bbank|credit card|debit card|\bloans?\b|mortgage", re.IGNORECASE)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_NAME = re.compile(r"[^a-z0-9]+")


def taxonomy_prompt(domain: str, n: int = INTENTS_PER_DOMAIN) -> list[dict[str, str]]:
    text = (
        f"Design an intent taxonomy for classifying customer messages sent to {domain}.\n"
        f"- Exactly {n} distinct, fine-grained intents, with several groups of closely related intents that are easy to "
        "confuse with each other (different stages, variants or causes of the same kind of request).\n"
        "- No intents about payments, billing, refunds, fees, prices, cards or anything financial.\n"
        "- One intent per line, in the form: intent_name_in_snake_case | one sentence saying when it applies\n"
        "Output the lines and nothing else."
    )
    return [{"role": "user", "content": text}]


def utterance_prompt(domain: str, names: Sequence[str], name: str, description: str, k: int = UTTERANCES_PER_INTENT) -> list[dict[str, str]]:
    text = (
        f"Customers write to {domain}. Their messages are sorted into these intents: {', '.join(names)}.\n\n"
        f"Target intent: {name} - {description}\n\n"
        f"Write {k} different messages a real customer might send that express the target intent and none of the others. "
        "Make them realistic rather than textbook: some terse, some long and rambling, some indirect that avoid the obvious "
        "keywords, some with typos or casual wording, some with irrelevant extra detail. Never use the intent's name.\n"
        "One message per line, no numbering, no quotation marks, nothing else."
    )
    return [{"role": "user", "content": text}]


def parse_taxonomy(text: str) -> list[dict[str, str]]:
    """`name | description` lines as intents; malformed lines and repeated names are skipped."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in text.splitlines():
        name, sep, desc = _BULLET.sub("", line).partition("|")
        name = _NAME.sub("_", name.strip().lower()).strip("_")
        desc = " ".join(desc.split()).rstrip(".")
        if not sep or not name or not desc or name in seen or len(name) > 60 or name in ("other", "none_of_the_above"):
            continue
        seen.add(name)
        out.append({"intent": name, "description": desc})
    return out


def parse_utterances(text: str) -> list[str]:
    """One message per line: bullets, numbering and wrapping quotes removed; 2 to 80 words."""
    out = []
    for line in text.splitlines():
        line = " ".join(_BULLET.sub("", line).strip().strip('"“”').split())
        if 2 <= len(line.split()) <= 80:
            out.append(line)
    return out


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate(out_dir: str | Path, teacher: Any, minutes: float = 60.0, batch_size: int = 16, domains: Sequence[str] = DOMAINS) -> dict[str, Any]:
    """Write taxonomies, then utterances (intents interleaved across domains) until `minutes` have passed.

    `teacher.generate(prompts, max_new_tokens)` returns one text per prompt. Both files are
    appended to, so a rerun continues where the last one stopped.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tax_path, utt_path = out_dir / "taxonomies.jsonl", out_dir / "utterances.jsonl"
    t0 = time.perf_counter()
    done = {t["domain"] for t in _read(tax_path)}
    todo = [d for d in domains if d not in done]
    with tax_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), batch_size):
            part = todo[start : start + batch_size]
            for domain, text in zip(part, teacher.generate([taxonomy_prompt(d) for d in part], 2200)):
                f.write(json.dumps({"domain": domain, "intents": parse_taxonomy(text)}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[synth] taxonomies {start + len(part)}/{len(todo)} after {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
    taxonomies = _read(tax_path)
    have = {(u["domain"], u["intent"]) for u in _read(utt_path)}
    jobs = []
    for j in range(max((len(t["intents"]) for t in taxonomies), default=0)):
        for t in taxonomies:
            if j < len(t["intents"]) and (t["domain"], t["intents"][j]["intent"]) not in have:
                jobs.append((t, t["intents"][j]))
    written = 0
    with utt_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(jobs), batch_size):
            if time.perf_counter() - t0 > minutes * 60:
                break
            part = jobs[start : start + batch_size]
            prompts = [utterance_prompt(t["domain"], [i["intent"] for i in t["intents"]], it["intent"], it["description"]) for t, it in part]
            for (t, it), text in zip(part, teacher.generate(prompts, 520)):
                for k, u in enumerate(parse_utterances(text)[: UTTERANCES_PER_INTENT + 3]):
                    f.write(json.dumps({"domain": t["domain"], "intent": it["intent"], "n": k, "text": u}, ensure_ascii=False) + "\n")
                    written += 1
            f.flush()
            print(f"[synth] intents {start + len(part)}/{len(jobs)}, {written} utterances after {time.perf_counter() - t0:.0f}s", file=sys.stderr, flush=True)
    return {"taxonomies": len(taxonomies), "intents": sum(len(t["intents"]) for t in taxonomies), "utterances_written": written,
            "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip


def label_set(taxonomy: dict[str, Any]) -> LabelSet:
    """A domain's usable intents (banking-like ones removed) as a label set."""
    labels = tuple(
        Label(i["intent"], i["intent"].replace("_", " "), desc=i["description"])
        for i in taxonomy["intents"]
        if not BANKING_LIKE.search(i["intent"].replace("_", " ") + " " + i["description"])
    )
    return LabelSet(labels=labels, choice_instructions=INSTRUCTIONS)


def decisions(synth_dir: str | Path, backend: Any, *, seed: int = 0, loader: Callable[[Source, str], Any] | None = None) -> dict[str, Any]:
    """Clean the utterances, mine each one's shortlist with `backend`'s candidate stage, write `decisions.jsonl`.

    Every decision is a row `{"id", "source", "family": "choice", "state", "question", "label"}` whose
    ten options are the generating intent plus the nine wrong labels the candidate stage ranks highest;
    `label` is the generating intent. Returns the drop counts (also written to `decisions.stats.json`).
    """
    synth_dir = Path(synth_dir)
    loader = loader or hf_loader
    guard = TextGuard([t for src in S.GUARD_TEXTS for split in src.train_splits for t in _column(loader(src, split), "text")])
    drops: Counter[str] = Counter()
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    kept_taxonomies = 0
    rank_seconds = 0.0
    gold_top1 = gold_top10 = 0
    utterances = defaultdict(list)
    for u in _read(synth_dir / "utterances.jsonl"):
        utterances[u["domain"]].append(u)
    for d, taxonomy in enumerate(_read(synth_dir / "taxonomies.jsonl")):
        labels = label_set(taxonomy)
        drops["banking_like_intent"] += len(taxonomy["intents"]) - len(labels.labels)
        if len(labels.labels) < 20:
            drops["taxonomy_too_small"] += 1
            continue
        kept_taxonomies += 1
        by_raw = labels.by_raw()
        usable = []
        for u in utterances[taxonomy["domain"]]:
            key = _key(u["text"])
            if u["intent"] not in by_raw:
                drops["utterance_of_dropped_intent"] += 1
            elif key in seen:
                drops["duplicate"] += 1
            elif guard.hit(u["text"]):
                drops["held_out_text_overlap"] += 1
            elif _BANK_TEXT.search(u["text"]):
                drops["banking_like_utterance"] += 1
            else:
                seen.add(key)
                usable.append(u)
        t0 = time.perf_counter()
        orders = rank_labels(backend, [u["text"] for u in usable], labels)
        rank_seconds += time.perf_counter() - t0
        for u, order in zip(usable, orders):
            gold = by_raw[u["intent"]]
            ranked = [labels.labels[k].raw for k in order]
            rank = ranked.index(gold.raw)
            gold_top1 += rank == 0
            gold_top10 += rank < 10
            rng = random.Random(f"{seed}:synth:{d}:{u['intent']}:{u['n']}")
            state, wrapper, _ = wrap_text(u["text"], S.STYLES["clinc"], rng)
            ex = shortlist_example(state, wrapper, gold, labels, rng, [r for r in ranked if r != gold.raw][:KEEP])
            rows.append({"id": f"{SOURCE}/{d}/{u['intent']}/{u['n']}", "source": SOURCE, "family": "choice", "state": ex["state"],
                         "question": ex["question"], "label": ex["label"]})  # fmt: skip
        print(f"[synth] {taxonomy['domain']}: {len(usable)} utterances, {len(labels.labels)} intents", file=sys.stderr, flush=True)
    with (synth_dir / "decisions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {"taxonomies_kept": kept_taxonomies, "decisions": len(rows), "dropped": dict(drops), "rank_seconds": round(rank_seconds, 1),
             "student_gold_top1": round(gold_top1 / max(1, len(rows)), 4), "student_gold_top10": round(gold_top10 / max(1, len(rows)), 4)}  # fmt: skip
    (synth_dir / "decisions.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _top(probs: Sequence[float]) -> int:
    return max(range(len(probs)), key=lambda i: (probs[i], -i))


def soft_rows(out_dir: str | Path, *, synth_rows: str | Path, synth_teacher: str | Path, gold_rows: Sequence[str | Path],
              gold_teacher: str | Path, gold_dev_rows: str | Path, gold_weight: float = 0.5, dev_each: int = 200, seed: int = 0) -> dict[str, Any]:  # fmt: skip
    """Merge teacher distributions into training rows; writes `<out_dir>/train.jsonl`, `dev_in.jsonl`, `stats.json`.

    Synthetic rows take the teacher's distribution as `target` and its top option as `label`; rows on
    which the teacher's two option orders disagree are dropped. Gold-labelled rows (`gold_rows`, with
    `gold_dev_rows` held out) get `gold_weight` x one-hot gold + the rest x teacher. `dev_each` rows of
    each kind are held out for dev_in.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}

    teacher = {r["id"]: r for r in _read(Path(synth_teacher))}
    synth: list[dict[str, Any]] = []
    n = agree_gen = disagree = unlabelled = kept_agree_gen = 0
    for row in _read(Path(synth_rows)):
        t = teacher.get(row["id"])
        if t is None:
            unlabelled += 1
            continue
        n += 1
        top = _top(t["probs"])
        agree_gen += top == row["label"]
        if not t["agree"]:
            disagree += 1
            continue
        kept_agree_gen += top == row["label"]
        synth.append({**row, "label": top, "target": [round(p, 6) for p in t["probs"]]})
    random.Random(f"{seed}:synth-dev").shuffle(synth)
    stats["synthetic"] = {"scored": n, "unlabelled": unlabelled, "dropped_order_disagreement": disagree, "kept": len(synth),
                          "teacher_top_is_generating_intent": round(agree_gen / max(1, n), 4),
                          "teacher_top_is_generating_intent_kept": round(kept_agree_gen / max(1, len(synth)), 4)}  # fmt: skip

    teacher = {r["id"]: r for r in _read(Path(gold_teacher))}

    def mixed(rows: Sequence[dict[str, Any]], names: set[str]) -> tuple[list[dict[str, Any]], int, int]:
        out, right, agree = [], 0, 0
        for row in rows:
            if row["source"] not in names or row["id"] not in teacher:
                continue
            t = teacher[row["id"]]
            right += _top(t["probs"]) == row["label"]
            agree += t["agree"]
            target = [(1.0 - gold_weight) * p for p in t["probs"]]
            target[row["label"]] += gold_weight
            out.append({**row, "target": [round(p, 6) for p in target]})
        return out, right, agree

    names = set(S.MINED_SOURCES)
    gold_train, right, agree = mixed([r for p in gold_rows for r in _read(Path(p))], names)
    gold_dev, dev_right, _ = mixed(_read(Path(gold_dev_rows)), names)
    per_source = max(1, dev_each // max(1, len(names)))
    picked: Counter[str] = Counter()
    gold_dev = [r for r in gold_dev if (picked.update([r["source"]]) or picked[r["source"]] <= per_source)]
    stats["gold"] = {"train": len(gold_train), "dev": len(gold_dev), "teacher_accuracy_train": round(right / max(1, len(gold_train)), 4),
                     "teacher_order_agreement_train": round(agree / max(1, len(gold_train)), 4), "gold_weight": gold_weight}  # fmt: skip

    dev = gold_dev + synth[:dev_each]
    train = gold_train + synth[dev_each:]
    for name, rows in (("train", train), ("dev_in", dev)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["rows"] = {"train": len(train), "dev_in": len(dev), "train_per_source": dict(Counter(r["source"] for r in train))}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
