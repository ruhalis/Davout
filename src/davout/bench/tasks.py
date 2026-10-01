"""Labelled decision tasks: a registry of public datasets and a JSONL loader.

Public tasks need the `datasets` library (`uv sync --extra eval`); it is
imported only when a task is loaded. Sampling is deterministic for a given
(task, seed): each split is permuted once, the test examples are the head of
the permutation, the calibration examples follow, and the few-shot examples
are drawn from its tail and ordered so the first k cover as many labels as
possible.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from davout.prompts import Demo
from davout.schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
    ValidationError,
    parse_question,
)

SHOT_POOL = 64  # candidates the few-shot examples are picked from
YELP_MAX_CHARS = 2000


@dataclass(frozen=True)
class Example:
    """One labelled decision.

    `label` is the option index (choice), the level index (score), or
    1 = true / 0 = false (noul).
    """

    id: str
    state: Any
    question: Question
    label: int


@dataclass
class TaskData:
    name: str
    kind: str  # "choice", "score" or "noul"
    shots: list[Example]
    calib: list[Example]
    test: list[Example]


@dataclass(frozen=True)
class TaskSpec:
    """A public task: where the rows come from and how a row becomes a decision."""

    name: str
    dataset: str  # Hugging Face dataset id
    config: str | None
    shots_split: str
    calib_split: str
    test_split: str
    question: Question
    to_example: Callable[[Mapping[str, Any]], tuple[Any, int]]  # row -> (state, label)
    columns: tuple[str, ...]  # columns `to_example` reads
    label_column: str
    label_names: tuple[str, ...] | None  # expected ClassLabel names (None: not a ClassLabel)

    @property
    def kind(self) -> str:
        return self.question.type


def to_demo(example: Example) -> Demo:
    """An example as a few-shot demonstration (noul: Demo answer 0 = Yes, 1 = No)."""
    if isinstance(example.question, NoulQuestion):
        return Demo(example.state, example.question, 0 if example.label == 1 else 1)
    return Demo(example.state, example.question, example.label)


# -- registry ---------------------------------------------------------------------------
#
# Every entry below was checked on 2026-10-01 with metadata requests only
# (https://datasets-server.huggingface.co/info?dataset=<id> for configs, splits, columns
# and label names; https://huggingface.co/api/datasets/<id> for the file list). No rows
# or files were fetched. All six repositories hold plain parquet files and no loading
# script.
#
# google/boolq            config "default"; train 9427, validation 3270;
#                         columns question: string, answer: bool, passage: string.
# ucirvine/sms_spam       config "plain_text"; train 5574 (the only split);
#                         columns sms: string, label: ClassLabel[ham, spam].
# stanfordnlp/sst2        config "default"; train 67349, validation 872, test 1821 (the
#                         test labels are hidden, so it is not used);
#                         columns idx: int32, sentence: string, label: ClassLabel[negative, positive].
# fancyzhx/ag_news        config "default"; train 120000, test 7600;
#                         columns text: string, label: ClassLabel[World, Sports, Business, Sci/Tech].
# legacy-datasets/banking77  config "default"; train 10003, test 3080;
#                         columns text: string, label: ClassLabel with the 77 names below.
#                         Substituted for PolyAI/banking77, which only ships a loading
#                         script (banking77.py) that current `datasets` refuses to run.
# Yelp/yelp_review_full   config "yelp_review_full"; train 650000, test 50000 (about
#                         320 MB to download); columns label: ClassLabel[1 star, 2 star,
#                         3 stars, 4 stars, 5 stars], text: string.

BANKING77_LABELS: tuple[str, ...] = (
    "activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire", "card_acceptance",
    "card_arrival", "card_delivery_estimate", "card_linking", "card_not_working",
    "card_payment_fee_charged", "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge", "cash_withdrawal_not_recognised", "change_pin",
    "compromised_card", "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer", "direct_debit_payment_not_recognised",
    "disposable_card_limits", "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer", "fiat_currency_support",
    "get_disposable_virtual_card", "get_physical_card", "getting_spare_card", "getting_virtual_card",
    "lost_or_stolen_card", "lost_or_stolen_phone", "order_physical_card", "passcode_forgotten",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up", "request_refund",
    "reverted_card_payment?", "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge", "top_up_by_cash_or_cheque",
    "top_up_failed", "top_up_limits", "top_up_reverted", "topping_up_by_card",
    "transaction_charged_twice", "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing", "unable_to_verify_identity",
    "verify_my_identity", "verify_source_of_funds", "verify_top_up", "virtual_card_not_working",
    "visa_or_mastercard", "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
)  # fmt: skip


def _boolq(row: Mapping[str, Any]) -> tuple[Any, int]:
    return {"passage": row["passage"], "question": row["question"]}, int(bool(row["answer"]))


def _sms_spam(row: Mapping[str, Any]) -> tuple[Any, int]:
    return row["sms"].strip(), int(row["label"] == 1)


def _sst2(row: Mapping[str, Any]) -> tuple[Any, int]:
    return row["sentence"].strip(), int(row["label"] == 1)


def _text_label(row: Mapping[str, Any]) -> tuple[Any, int]:
    return row["text"].strip(), int(row["label"])


def _yelp(row: Mapping[str, Any]) -> tuple[Any, int]:
    return row["text"].strip()[:YELP_MAX_CHARS], int(row["label"])


TASKS: dict[str, TaskSpec] = {
    spec.name: spec
    for spec in (
        TaskSpec(
            name="boolq",
            dataset="google/boolq",
            config="default",
            shots_split="train",
            calib_split="validation",
            test_split="validation",
            question=NoulQuestion("According to the passage, is the answer to `question` yes?", None),
            to_example=_boolq,
            columns=("question", "answer", "passage"),
            label_column="answer",
            label_names=None,
        ),
        TaskSpec(
            name="sms_spam",
            dataset="ucirvine/sms_spam",
            config="plain_text",
            shots_split="train",
            calib_split="train",
            test_split="train",
            question=NoulQuestion("The message is spam.", None),
            to_example=_sms_spam,
            columns=("sms", "label"),
            label_column="label",
            label_names=("ham", "spam"),
        ),
        TaskSpec(
            name="sst2",
            dataset="stanfordnlp/sst2",
            config="default",
            shots_split="train",
            calib_split="validation",
            test_split="validation",
            question=NoulQuestion("The review expresses a positive sentiment.", None),
            to_example=_sst2,
            columns=("sentence", "label"),
            label_column="label",
            label_names=("negative", "positive"),
        ),
        TaskSpec(
            name="ag_news",
            dataset="fancyzhx/ag_news",
            config="default",
            shots_split="train",
            calib_split="train",
            test_split="test",
            question=ChoiceQuestion(
                "Which topic does the news article belong to?",
                {
                    "world": "international affairs, politics, conflicts and diplomacy",
                    "sports": "games, matches, athletes, teams and tournaments",
                    "business": "companies, markets, finance and the economy",
                    "sci_tech": "science, technology, computing and the internet",
                },
            ),
            to_example=_text_label,
            columns=("text", "label"),
            label_column="label",
            label_names=("World", "Sports", "Business", "Sci/Tech"),
        ),
        TaskSpec(
            name="banking77",
            dataset="legacy-datasets/banking77",
            config="default",
            shots_split="train",
            calib_split="train",
            test_split="test",
            question=ChoiceQuestion(
                "Which intent does the customer's banking query express?",
                {name.replace("_", " "): None for name in BANKING77_LABELS},
            ),
            to_example=_text_label,
            columns=("text", "label"),
            label_column="label",
            label_names=BANKING77_LABELS,
        ),
        TaskSpec(
            name="yelp",
            dataset="Yelp/yelp_review_full",
            config="yelp_review_full",
            shots_split="train",
            calib_split="train",
            test_split="test",
            question=ScoreQuestion(
                "How good was the reviewer's experience?",
                [
                    "Terrible experience; would warn others away",
                    "Poor experience; mostly disappointed and unlikely to return",
                    "Mixed experience; some good and some bad, nothing special",
                    "Good experience; satisfied, with at most minor complaints",
                    "Excellent; enthusiastic recommendation",
                ],
            ),
            to_example=_yelp,
            columns=("text", "label"),
            label_column="label",
            label_names=("1 star", "2 star", "3 stars", "4 stars", "5 stars"),
        ),
    )
}


# -- sampling ---------------------------------------------------------------------------


def _n_labels(question: Question) -> int:
    return 2 if isinstance(question, NoulQuestion) else len(question.criteria)


def _balanced(pool: Sequence[Example], k: int) -> list[Example]:
    """First `k` of `pool` taken round-robin over labels (labels in order of first appearance)."""
    groups: dict[int, list[Example]] = {}
    for ex in pool:
        groups.setdefault(ex.label, []).append(ex)
    out: list[Example] = []
    queues = list(groups.values())
    depth = 0
    while len(out) < k and any(depth < len(q) for q in queues):
        for q in queues:
            if depth < len(q) and len(out) < k:
                out.append(q[depth])
        depth += 1
    return out


def _content(example: Example) -> str:
    """What the model is shown for an example (state and question), whatever its row id."""
    q = example.question
    return json.dumps([example.state, q.type, q.instructions, q.criteria], ensure_ascii=False)


def _allocate(name: str, split: str, n_rows: int, wants: dict[str, int]) -> dict[str, int]:
    """How many rows each role gets from one split; shrinks proportionally if the split is small."""
    total = sum(wants.values())
    if total <= n_rows:
        return dict(wants)
    got = {role: want * n_rows // total for role, want in wants.items()}
    for role in ("calib", "test"):
        if wants.get(role, 0) > 0 and got[role] == 0:
            raise ValueError(
                f"task {name!r}: split {split!r} has {n_rows} rows, too few for "
                + ", ".join(f"{w} {r}" for r, w in wants.items())
            )
    print(
        f"[davout bench] {name}: split {split!r} has only {n_rows} rows; using "
        + ", ".join(f"{got[r]} {r} (asked {wants[r]})" for r in wants),
        file=sys.stderr,
    )
    return got


def _sample(
    name: str,
    kind: str,
    splits: Mapping[str, Sequence[Any]],
    roles: Mapping[str, str],
    make: Callable[[str, int], Example],
    n_calib: int,
    n_test: int,
    n_shots: int,
    seed: int,
) -> TaskData:
    """Draw disjoint shots/calib/test examples. `roles` maps role -> split name.

    Disjoint by content as well as by row: datasets repeat rows (ucirvine/sms_spam
    holds 5574 messages but 5160 distinct ones), so a row whose state and question
    already belong to an earlier role (test, then calib) is passed over.
    """
    if min(n_calib, n_test, n_shots) < 0:
        raise ValueError("n_calib, n_test and n_shots must be >= 0")
    wants_all = {"test": n_test, "calib": n_calib, "shots": n_shots}
    picked: dict[str, list[Example]] = {"test": [], "calib": [], "shots": []}
    taken: set[str] = set()  # content of the examples given to earlier roles
    for split in dict.fromkeys(roles[r] for r in ("test", "calib", "shots")):
        n_rows = len(splits[split])
        wants = {r: wants_all[r] for r in ("test", "calib", "shots") if roles[r] == split}
        got = _allocate(name, split, n_rows, wants)
        perm = list(range(n_rows))
        random.Random(f"{seed}:{name}:{split}").shuffle(perm)
        cursor = 0
        limit = n_rows - got.get("shots", 0)  # rows kept back for the shots
        for role in ("test", "calib"):
            if role in got:
                chosen: list[Example] = []
                while len(chosen) < got[role] and cursor < limit:
                    ex = make(split, perm[cursor])
                    cursor += 1
                    if _content(ex) not in taken:
                        chosen.append(ex)
                picked[role] = chosen
                taken.update(_content(ex) for ex in chosen)
        if got.get("shots"):
            pool_size = min(n_rows - cursor, max(SHOT_POOL, 4 * got["shots"]))
            pool = [make(split, i) for i in reversed(perm[n_rows - pool_size :])]
            picked["shots"] = _balanced([ex for ex in pool if _content(ex) not in taken], got["shots"])
    return TaskData(name, kind, picked["shots"], picked["calib"], picked["test"])


# -- public datasets --------------------------------------------------------------------


def _load_hf_split(spec: TaskSpec, split: str) -> Any:
    """Load one split with `datasets` and check it still has the verified schema."""
    try:
        import datasets
    except ImportError:
        raise RuntimeError(
            "public benchmark tasks need the `datasets` library; install it with `uv sync --extra eval` "
            "(or evaluate your own decisions with --jsonl)"
        ) from None
    ds = datasets.load_dataset(spec.dataset, spec.config, split=split)
    missing = [c for c in spec.columns if c not in ds.column_names]
    if missing:
        raise RuntimeError(f"{spec.dataset} [{split}] has no column(s) {missing}; found {ds.column_names}")
    if spec.label_names is not None:
        names = getattr(ds.features[spec.label_column], "names", None)
        if names is not None and tuple(names) != spec.label_names:
            raise RuntimeError(f"{spec.dataset} label names changed: expected {spec.label_names}, got {names}")
    return ds


def _example_from_row(spec: TaskSpec, split: str, index: int, row: Mapping[str, Any]) -> Example:
    state, label = spec.to_example(row)
    if not 0 <= label < _n_labels(spec.question):
        raise ValueError(f"{spec.dataset} [{split}] row {index}: label {label} is out of range")
    return Example(f"{spec.name}/{split}/{index}", state, spec.question, label)


def sample_task(
    spec: TaskSpec,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    n_calib: int = 300,
    n_test: int = 300,
    n_shots: int = 8,
    seed: int = 0,
) -> TaskData:
    """Sample a registered task from already-loaded splits (split name -> indexable rows)."""
    roles = {"shots": spec.shots_split, "calib": spec.calib_split, "test": spec.test_split}

    def make(split: str, index: int) -> Example:
        return _example_from_row(spec, split, index, splits[split][index])

    return _sample(spec.name, spec.kind, splits, roles, make, n_calib, n_test, n_shots, seed)


def load_task(name: str, n_calib: int = 300, n_test: int = 300, n_shots: int = 8, seed: int = 0) -> TaskData:
    """Load a registered public task (downloads the dataset on first use)."""
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; available: {', '.join(TASKS)}")
    spec = TASKS[name]
    names = dict.fromkeys((spec.shots_split, spec.calib_split, spec.test_split))
    splits = {split: _load_hf_split(spec, split) for split in names}
    return sample_task(spec, splits, n_calib, n_test, n_shots, seed)


# -- the user's own decisions -----------------------------------------------------------


def _jsonl_label(question: Question, label: Any) -> int:
    if isinstance(question, ChoiceQuestion):
        names = list(question.criteria)
        if not isinstance(label, str) or label not in names:
            raise ValueError(f"label must be one of the option names {names}, got {label!r}")
        return names.index(label)
    if isinstance(question, ScoreQuestion):
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < len(question.criteria):
            raise ValueError(f"label must be a level index 0..{len(question.criteria) - 1}, got {label!r}")
        return label
    if isinstance(label, bool) or (isinstance(label, int) and label in (0, 1)):
        return int(label)
    raise ValueError(f"label must be true or false, got {label!r}")


def read_jsonl(path: str | Path, name: str | None = None) -> tuple[str, str, list[Example]]:
    """Parse a decisions file into (task name, question type, examples); errors name the line."""
    path = Path(path)
    name = name or path.stem
    examples: list[Example] = []
    seen: dict[str, int] = {}
    kind: str | None = None
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path}:{lineno}"
            try:
                obj = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{where}: invalid JSON ({e})") from None
            if not isinstance(obj, dict):
                raise ValueError(f"{where}: each line must be a JSON object")
            unknown = sorted(k for k in obj if k not in ("id", "state", "question", "label"))
            if unknown:
                raise ValueError(f"{where}: unknown key(s) {unknown}")
            for key in ("state", "question", "label"):
                if key not in obj:
                    raise ValueError(f"{where}: {key} is required")
            state = obj["state"]
            if not isinstance(state, (str, dict, list)):
                raise ValueError(f"{where}: state must be a string, object or array")
            try:
                question = parse_question("question", obj["question"])
            except ValidationError as e:
                at = e.path.replace("questions.question", "question", 1)
                raise ValueError(f"{where}: {at}: {e.message}") from None
            if kind is None:
                kind = question.type
            elif question.type != kind:
                raise ValueError(
                    f"{where}: question type {question.type!r} differs from earlier rows ({kind!r}); "
                    "all rows must share one type"
                )
            try:
                label = _jsonl_label(question, obj["label"])
            except ValueError as e:
                raise ValueError(f"{where}: {e}") from None
            ex_id = obj.get("id", f"{name}/{lineno}")
            if not isinstance(ex_id, str) or not ex_id:
                raise ValueError(f"{where}: id must be a non-empty string")
            if ex_id in seen:
                raise ValueError(f"{where}: duplicate id {ex_id!r} (first used on line {seen[ex_id]})")
            seen[ex_id] = lineno
            examples.append(Example(ex_id, state, question, label))
    if kind is None:
        raise ValueError(f"{path}: no rows")
    return name, kind, examples


def load_jsonl_task(
    path: str | Path,
    name: str | None = None,
    n_calib: int = 300,
    n_test: int = 300,
    n_shots: int = 8,
    seed: int = 0,
) -> TaskData:
    """Load labelled decisions from a JSONL file and split them into shots/calib/test.

    Each line is `{"id"?: str, "state": ..., "question": {type, instructions,
    criteria}, "label": option name | level index | true/false}`.
    """
    name, kind, examples = read_jsonl(path, name)
    roles = {"shots": "rows", "calib": "rows", "test": "rows"}
    return _sample(
        name, kind, {"rows": examples}, roles, lambda _split, i: examples[i], n_calib, n_test, n_shots, seed
    )
