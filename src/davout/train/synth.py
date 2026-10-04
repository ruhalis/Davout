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
# The second, wider batch: consumer, business, public-sector and internal help desks not listed above.
DOMAINS_V2: tuple[str, ...] = (
    "a bookshop and e-reader store", "a cosmetics and skincare retailer", "a pet supplies retailer",
    "a sporting goods and outdoor equipment store", "a toy and baby products retailer", "a home improvement and DIY store",
    "a garden centre and plant nursery", "a jewellery and watch repair shop", "an optician and eyewear retailer",
    "a musical instrument shop", "a bicycle shop and repair workshop", "a custom print and photo-book service",
    "a florist and gift delivery service", "a meal-kit subscription service", "a wine and beverage subscription club",
    "a second-hand marketplace app", "a kitchen appliance manufacturer's support line", "a mattress and bedding company",
    "a tailoring and dry-cleaning service", "a camera and photography equipment store",
    "a cable and satellite TV provider", "a business phone system provider", "a mobile phone manufacturer's device support",
    "a laptop and PC manufacturer's technical support", "a printer and scanner manufacturer's support", "a wearable fitness tracker maker",
    "a password manager and security software vendor", "a VPN and antivirus provider", "a website builder for small merchants",
    "a customer-relationship-management software vendor", "a video-conferencing software product", "a human-resources software platform",
    "an email marketing platform", "a developer cloud platform for servers and databases", "an API and developer tools company",
    "a mobile app store's developer support", "a navigation and maps app", "a dating app", "a music streaming service",
    "a podcast and audiobook app", "a photo backup and sharing service", "a note-taking and document collaboration app",
    "an online survey and forms tool", "a language-learning app", "a smart TV and home audio manufacturer",
    "an electric scooter and bike sharing service", "an electric vehicle charging network", "a satellite internet provider",
    "a company's internal IT helpdesk", "a company's internal HR helpdesk", "a company's facilities and office management desk",
    "a corporate travel booking desk", "a university IT service desk", "a hospital's internal clinical IT helpdesk",
    "a school district's staff support line", "a corporate legal and compliance intake desk", "a company's security and access badge office",
    "a procurement and supplier onboarding desk",
    "a national passport and visa office", "a driver and vehicle licensing agency", "a public transport authority's lost property and complaints office",
    "a national postal service", "a court and jury service information line", "a police non-emergency reporting line",
    "an immigration and residency permit office", "a public housing authority", "a public employment service and job centre",
    "a parks and recreation department", "a building permits and planning office", "a national census office",
    "an electoral registration office", "a public school admissions office", "a municipal parking and street permits office",
    "an environmental health and noise complaints service", "a national health service's patient helpline", "a veterans' services office",
    "a consumer protection and complaints agency",
    "a medical laboratory and test results service", "a physiotherapy clinic", "a mental health counselling service's intake line",
    "a telemedicine app", "a hearing aid and medical device supplier", "a home care and nursing agency", "a maternity clinic",
    "a blood donation service", "an eye surgery clinic", "a hospital medical records office",
    "a cruise line", "an airport information desk", "a ferry operator", "a holiday home rental platform", "a ski resort",
    "a theme park", "a campsite and caravan park network", "a long-distance coach operator", "a museum and gallery",
    "a zoo and aquarium", "a national park visitor centre",
    "a primary school office", "a driving school", "a coding bootcamp", "a tutoring and test-prep company",
    "a student accommodation provider", "a university library and research support desk", "a study-abroad programme office",
    "a music school",
    "an estate agency selling homes", "a property management company for apartment buildings", "a home security and alarm company",
    "a solar panel installer", "a plumbing and heating repair service", "a kitchen and bathroom renovation company",
    "a pest control company", "a locksmith and key service", "a landscaping and lawn care company",
    "an elevator and building maintenance contractor",
    "a car manufacturer's connected-car app support", "a roadside assistance and breakdown service", "a tyre and windscreen replacement chain",
    "a motorcycle dealer and workshop", "a car-sharing club", "a vehicle inspection and emissions testing centre",
    "a truck fleet telematics provider",
    "a law firm's new client intake line", "an employment law advice line", "a family law and mediation service",
    "a trademark and patent office", "a tenants' rights advice service",
    "an industrial equipment and forklift supplier", "a commercial cleaning contractor", "a wholesale food distributor for restaurants",
    "a freight forwarding company", "a warehouse and fulfilment provider", "an office furniture and supplies vendor",
    "a staffing and temp agency", "a conference and exhibition organiser", "a commercial printing company",
    "a laboratory equipment supplier", "an agricultural machinery dealer", "a packaging manufacturer",
    "a games console maker's online network support", "an esports tournament organiser", "a board game and hobby store",
    "a sports club membership office", "a marathon and race event organiser", "a dance and yoga studio",
    "a public swimming pool and leisure centre", "a golf club",
    "a wedding planning service", "a funeral home", "a childcare nursery", "a volunteer organisation's coordinator desk",
    "a community centre office", "a newspaper and magazine subscription desk", "a TV broadcaster's viewer services",
    "an animal shelter and pet adoption centre",
)  # fmt: skip
DOMAIN_SETS: dict[str, tuple[str, ...]] = {"v1": DOMAINS, "v2": DOMAINS_V2}
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


def label_set(taxonomy: dict[str, Any], flagged: frozenset[tuple[str, str]] = frozenset(), descriptions: bool = True) -> LabelSet:
    """A domain's usable intents as a label set: banking-like ones (keyword filter, or `flagged`
    (domain, intent) pairs from `flag_banking`) are removed."""
    labels = tuple(
        Label(i["intent"], i["intent"].replace("_", " "), desc=i["description"] if descriptions else None)
        for i in taxonomy["intents"]
        if not BANKING_LIKE.search(i["intent"].replace("_", " ") + " " + i["description"])
        and (taxonomy["domain"], i["intent"]) not in flagged
    )
    return LabelSet(labels=labels, choice_instructions=INSTRUCTIONS)


FLAG_QUESTION = "Does this customer-support intent concern banking, payments, cards, loans or personal finance?"


def flag_banking(taxonomies: str | Path, out_path: str | Path, teacher_backend: Any, chunk: int = 256) -> dict[str, Any]:
    """Ask the teacher, intent by intent, whether it is a banking or money-movement intent.

    Appends `{"domain", "intent", "p_yes", "flag"}` lines to `out_path` (resumable); `flag` is
    `p_yes > 0.5`. Returns counts.
    """
    from davout import teacher
    from davout.schema import NoulQuestion

    out_path = Path(out_path)
    done = {(r["domain"], r["intent"]) for r in _read(out_path)}
    todo = [(t["domain"], i) for t in _read(Path(taxonomies)) for i in t["intents"] if (t["domain"], i["intent"]) not in done]
    question = NoulQuestion(FLAG_QUESTION, None)
    with out_path.open("a", encoding="utf-8") as f:
        for start in range(0, len(todo), chunk):
            part = todo[start : start + chunk]
            states = [f"Support desk: {d}\nIntent: {i['intent'].replace('_', ' ')}\nDescription: {i['description']}" for d, i in part]
            results = teacher.score(teacher_backend, [(f"{d}/{i['intent']}", s, question) for (d, i), s in zip(part, states)])
            for (d, i), res in zip(part, results):
                f.write(json.dumps({"domain": d, "intent": i["intent"], "p_yes": round(res["probs"][0], 4), "flag": res["probs"][0] > 0.5}) + "\n")
            f.flush()
            print(f"[synth] banking check {start + len(part)}/{len(todo)}", file=sys.stderr, flush=True)
    rows = _read(out_path)
    return {"intents": len(rows), "flagged": sum(r["flag"] for r in rows)}


def read_flags(path: str | Path | None) -> frozenset[tuple[str, str]]:
    return frozenset((r["domain"], r["intent"]) for r in _read(Path(path)) if r["flag"]) if path else frozenset()


def decisions(synth_dir: str | Path, backend: Any, *, seed: int = 0, loader: Callable[[Source, str], Any] | None = None,
              tag: str = "", flags: str | Path | None = None, desc_rate: float = 0.5) -> dict[str, Any]:  # fmt: skip
    """Clean the utterances, mine each one's shortlist with `backend`'s candidate stage, write `decisions.jsonl`.

    Every decision is a row `{"id", "source", "family": "choice", "state", "question", "label"}` whose
    ten options are the generating intent plus the nine wrong labels the candidate stage ranks highest;
    `label` is the generating intent. `flags` is the output of `flag_banking`; `desc_rate` is the share
    of rows that show intent descriptions (at most 0.5); `tag` goes into the row ids so that several
    synthetic sets can be mixed. Finished taxonomies are recorded in `decisions.progress.jsonl`, so a
    rerun only ranks what is missing. Returns the drop counts (also in `decisions.stats.json`).
    """
    synth_dir = Path(synth_dir)
    loader = loader or hf_loader
    flagged = read_flags(flags)
    guard = TextGuard([t for src in S.GUARD_TEXTS for split in src.train_splits for t in _column(loader(src, split), "text")])
    drops: Counter[str] = Counter()
    seen: set[str] = set()
    utterances = defaultdict(list)
    for u in _read(synth_dir / "utterances.jsonl"):
        utterances[u["domain"]].append(u)
    progress_path, rows_path = synth_dir / "decisions.progress.jsonl", synth_dir / "decisions.jsonl"
    progress = {p["domain"]: p for p in _read(progress_path)}
    if not progress and rows_path.exists():
        rows_path.unlink()  # rows without a progress record are from an unfinished first taxonomy
    plain_rate = max(0.0, 1.0 - desc_rate / 0.5)
    kept_taxonomies = 0
    for d, taxonomy in enumerate(_read(synth_dir / "taxonomies.jsonl")):
        domain = taxonomy["domain"]
        labels = label_set(taxonomy, flagged)
        plain = label_set(taxonomy, flagged, descriptions=False)
        keyword = sum(bool(BANKING_LIKE.search(i["intent"].replace("_", " ") + " " + i["description"])) for i in taxonomy["intents"])
        drops["banking_like_intent_keyword"] += keyword
        drops["banking_like_intent_teacher_only"] += len(taxonomy["intents"]) - len(labels.labels) - keyword
        if len(labels.labels) < 20:
            drops["taxonomy_too_small"] += 1
            continue
        kept_taxonomies += 1
        by_raw = labels.by_raw()
        usable = []
        for u in utterances[domain]:
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
        if domain in progress:
            continue
        t0 = time.perf_counter()
        orders = rank_labels(backend, [u["text"] for u in usable], labels)
        top1 = top10 = 0
        rows = []
        for u, order in zip(usable, orders):
            gold = by_raw[u["intent"]]
            ranked = [labels.labels[k].raw for k in order]
            rank = ranked.index(gold.raw)
            top1 += rank == 0
            top10 += rank < 10
            rng = random.Random(f"{seed}:synth:{tag}{d}:{u['intent']}:{u['n']}")
            shown = plain if rng.random() < plain_rate else labels
            state, wrapper, _ = wrap_text(u["text"], S.STYLES["clinc"], rng)
            ex = shortlist_example(state, wrapper, shown.by_raw()[gold.raw], shown, rng, [r for r in ranked if r != gold.raw][:KEEP])
            rows.append({"id": f"{SOURCE}/{tag}{d}/{u['intent']}/{u['n']}", "source": SOURCE, "family": "choice", "state": ex["state"],
                         "question": ex["question"], "label": ex["label"]})  # fmt: skip
        with rows_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        record = {"domain": domain, "intents": len(labels.labels), "decisions": len(rows), "gold_top1": top1, "gold_top10": top10,
                  "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        progress[domain] = record
        print(f"[synth] {domain}: {len(usable)} utterances, {len(labels.labels)} intents", file=sys.stderr, flush=True)
    n = sum(p["decisions"] for p in progress.values())
    stats = {"taxonomies_kept": kept_taxonomies, "decisions": n, "dropped": dict(drops),
             "rank_seconds": round(sum(p["seconds"] for p in progress.values()), 1),
             "student_gold_top1": round(sum(p["gold_top1"] for p in progress.values()) / max(1, n), 4),
             "student_gold_top10": round(sum(p["gold_top10"] for p in progress.values()) / max(1, n), 4)}  # fmt: skip
    (synth_dir / "decisions.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def _top(probs: Sequence[float]) -> int:
    return max(range(len(probs)), key=lambda i: (probs[i], -i))


def soft_rows(out_dir: str | Path, *, synth_rows: str | Path, synth_teacher: str | Path, gold_rows: Sequence[str | Path] = (),
              gold_teacher: str | Path | None = None, gold_dev_rows: str | Path | None = None, gold_weight: float = 0.5,
              dev_each: int = 200, seed: int = 0, carry: Sequence[str | Path] = (), carry_dev: Sequence[str | Path] = (),
              carry_drop: tuple[str | Path, str | Path] | None = None) -> dict[str, Any]:  # fmt: skip
    """Merge teacher distributions into training rows; writes `<out_dir>/train.jsonl`, `dev_in.jsonl`, `stats.json`.

    Synthetic rows take the teacher's distribution as `target` and its top option as `label`; rows on
    which the teacher's two option orders disagree are dropped; `dev_each` of them are held out.
    Gold-labelled rows (`gold_rows`, with `gold_dev_rows` held out) get `gold_weight` x one-hot gold +
    the rest x teacher. `carry` and `carry_dev` are files of rows finished by an earlier call, added to
    train and dev_in as they are; `carry_drop = (taxonomies, flags)` of the earlier synthetic set
    removes its carried rows whose generating intent `flag_banking` flagged.
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
    dev = synth[:dev_each]
    train = synth[dev_each:]

    if gold_teacher is not None:
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
        gold_dev, _dev_right, _ = mixed(_read(Path(gold_dev_rows)) if gold_dev_rows else [], names)
        per_source = max(1, dev_each // max(1, len(names)))
        picked: Counter[str] = Counter()
        gold_dev = [r for r in gold_dev if (picked.update([r["source"]]) or picked[r["source"]] <= per_source)]
        stats["gold"] = {"train": len(gold_train), "dev": len(gold_dev), "teacher_accuracy_train": round(right / max(1, len(gold_train)), 4),
                         "teacher_order_agreement_train": round(agree / max(1, len(gold_train)), 4), "gold_weight": gold_weight}  # fmt: skip
        dev = gold_dev + dev
        train = gold_train + train

    if carry or carry_dev:
        dropped_intents: set[tuple[str, str]] = set()
        if carry_drop is not None:
            index = {str(d): t["domain"] for d, t in enumerate(_read(Path(carry_drop[0])))}
            flagged = read_flags(carry_drop[1])
            dropped_intents = {(d, intent) for d, domain in index.items() for dom, intent in flagged if dom == domain}
        ids = {r["id"] for r in train + dev}
        counts: Counter[str] = Counter()
        for role, paths, into in (("train", carry, train), ("dev_in", carry_dev, dev)):
            for row in (r for p in paths for r in _read(Path(p))):
                parts = row["id"].split("/")
                if row["source"] == SOURCE and (parts[1], parts[2]) in dropped_intents:
                    counts["dropped_flagged_intent"] += 1
                elif row["id"] in ids:
                    counts["dropped_duplicate_id"] += 1
                else:
                    ids.add(row["id"])
                    into.append(row)
                    counts[f"{role}:{row['source']}"] += 1
        stats["carried"] = dict(counts)

    for name, rows in (("train", train), ("dev_in", dev)):
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["rows"] = {"train": len(train), "dev_in": len(dev), "train_per_source": dict(Counter(r["source"] for r in train)),
                     "dev_per_source": dict(Counter(r["source"] for r in dev))}  # fmt: skip
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats
