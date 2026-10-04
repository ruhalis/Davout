"""The fine-tuning loop on a tiny random-weight `hrm_text` model: CPU, no network, no real weights.

The tokenizer is found as in test_hrm_backend.py; without it the model tests are skipped.
"""
from __future__ import annotations

import json
import math
import random
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_hrm_backend import _tiny_config, _tokenizer_source

from davout import cli, metrics
from davout.backends.hrm import HrmBackend
from davout.prompts import build_candidate_prompt, build_prompt
from davout.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from davout.train.format import (
    FAMILIES,
    Item,
    Row,
    chunks,
    collate,
    micro_batches,
    pack,
    read_rows,
    render,
    target,
    tokenize,
)

ANIMALS = ("cat", "dog", "bird", "fish")
CHOICE = {"type": "choice", "instructions": "Which animal is the note about?", "criteria": {a: None for a in ANIMALS}}
SCORE = {"type": "score", "instructions": "How many legs does the animal have?", "criteria": ["none", "two", "four"]}
LEGS = {"cat": 2, "dog": 2, "bird": 1, "fish": 0}


def make_row(i: int, rng: random.Random, prefix: str = "r") -> dict[str, Any]:
    family = FAMILIES[i % len(FAMILIES)]
    # skewed on purpose: even a tiny random model can learn the label priors in a few steps
    animal = "cat" if rng.random() < 0.85 else ANIMALS[rng.randrange(4)]
    other = animal if rng.random() < 0.9 else ANIMALS[rng.randrange(4)]
    filler = " ".join(f"w{rng.randrange(50)}" for _ in range(rng.randrange(1, 30)))
    row: dict[str, Any] = {
        "id": f"{prefix}{i}",
        "source": f"src_{family}",
        "family": family,
        "state": f"Note: the {animal} sat on the mat. {filler}",
    }
    if family == "nli":
        row["question"] = {"type": "noul", "instructions": f"The note is about a {other}."}
        row["label"] = int(animal == other)
    elif family == "reading":
        row["question"] = {"type": "noul", "instructions": f"Does the note mention a {other}?", "criteria": None}
        row["label"] = int(animal == other)
    elif family == "judgement":
        crit = {"true": "the animal has four legs", "false": "it has fewer"}
        row["question"] = {"type": "noul", "instructions": "The animal has four legs.", "criteria": crit}
        row["label"] = int(LEGS[animal] == 2)
    elif family == "ovr":
        row["state"] = {"note": row["state"]}  # an object state
        row["question"] = {"type": "noul", "instructions": f"Is this note about a {other}?"}
        row["label"] = int(animal == other)
    elif family == "candidate":
        row["question"] = CHOICE
        row["candidate"] = other
        row["label"] = int(animal == other)
    elif family == "choice":
        row["question"] = CHOICE
        row["label"] = ANIMALS.index(animal)
    else:
        row["question"] = SCORE
        row["label"] = LEGS[animal]
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def make_data(root: Path, n_train: int = 192, n_dev: int = 56, seed: int = 0) -> Path:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "train.jsonl", [make_row(i, rng, "t") for i in range(n_train)])
    write_jsonl(root / "dev_in.jsonl", [make_row(i, rng, "i") for i in range(n_dev)])
    write_jsonl(root / "dev_xfer.jsonl", [make_row(i, rng, "x") for i in range(n_dev)])
    return root


# -- reader, targets, packing: no model ---------------------------------------------------


def test_reader_and_target_mapping(tmp_path: Path) -> None:
    rng = random.Random(1)
    raw = [make_row(i, rng) for i in range(28)]
    rows = read_rows(write_jsonl(tmp_path / "rows.jsonl", raw))
    assert [r.id for r in rows] == [f"r{i}" for i in range(28)]  # file order
    assert {r.family for r in rows} == set(FAMILIES)
    for obj, row in zip(raw, rows):
        y, n_opt = target(row)
        prompt = render(row)
        assert prompt.n_labels == n_opt and 0 <= y < n_opt
        if row.family in ("nli", "reading", "judgement", "ovr"):
            assert isinstance(row.question, NoulQuestion)
            assert (y, n_opt) == (0 if obj["label"] == 1 else 1, 2)  # true -> A (Yes), false -> B (No)
            assert prompt == build_prompt(row.state, row.question)
            assert prompt.suffix.endswith("\nA. Yes\nB. No\nAnswer:") or "A. Yes: " in prompt.suffix
        elif row.family == "candidate":
            assert isinstance(row.question, ChoiceQuestion) and row.candidate == obj["candidate"]
            assert (y, n_opt) == (0 if obj["label"] == 1 else 1, 2)
            assert prompt == build_candidate_prompt(row.state, row.question, row.candidate)
            assert f"Proposed answer: {row.candidate}\nQuestion: Is the proposed answer correct?" in prompt.suffix
        elif row.family == "choice":
            assert (y, n_opt) == (obj["label"], 4)
            assert prompt == build_prompt(row.state, row.question) and "\nD. fish\nAnswer:" in prompt.suffix
        else:
            assert isinstance(row.question, ScoreQuestion) and (y, n_opt) == (obj["label"], 3)
    ovr = next(r for r in rows if r.family == "ovr")
    assert isinstance(ovr.state, dict) and '"note"' in render(ovr).state  # objects render as JSON, as in serving


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": 1}, "unknown key"),
        ({"family": "other"}, "family must be one of"),
        ({"family": "score"}, "needs a score question"),
        ({"label": True}, "label must be an integer"),
        ({"label": 2}, "label must be 0 or 1"),
        ({"candidate": "cat"}, "candidate only applies"),
        ({"source": ""}, "source must be a non-empty string"),
        ({"question": {"type": "noul"}}, "question.instructions: instructions is required"),
    ],
)
def test_reader_rejects_bad_rows(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    good = {"id": "a", "source": "s", "family": "nli", "state": "x", "question": {"type": "noul", "instructions": "It is."}, "label": 1}
    path = write_jsonl(tmp_path / "rows.jsonl", [good, {**good, "id": "b", **change}])
    with pytest.raises(ValueError, match=f"rows.jsonl:2: .*{message}"):
        read_rows(path)


def test_reader_rejects_bad_choice_and_candidate_rows(tmp_path: Path) -> None:
    base = {"id": "a", "source": "s", "state": "x", "question": CHOICE}
    cases = [
        ({**base, "family": "choice", "label": 4}, "label 4 out of range for 4 options"),
        ({**base, "family": "candidate", "label": 1}, "candidate must name an option"),
        ({**base, "family": "candidate", "label": 1, "candidate": "cow"}, "candidate must name an option"),
        ({**base, "family": "candidate", "label": 3, "candidate": "cat"}, "label must be 0 or 1"),
        (
            {**base, "family": "choice", "label": 0, "question": {**CHOICE, "criteria": {f"o{i}": None for i in range(27)}}},
            "27 options do not fit the 26 answer letters",
        ),
    ]
    for obj, message in cases:
        with pytest.raises(ValueError, match=message):
            read_rows(write_jsonl(tmp_path / "rows.jsonl", [obj]))
    good = {**base, "family": "choice", "label": 0}
    with pytest.raises(ValueError, match="duplicate id 'a'"):
        read_rows(write_jsonl(tmp_path / "rows.jsonl", [good, good]))
    with pytest.raises(ValueError, match="no rows"):
        read_rows(write_jsonl(tmp_path / "rows.jsonl", []))


def fake_item(length: int, n_opt: int = 2, y: int = 0, first: int = 100) -> Item:
    row = Row(f"id{length}", "s", "nli", "x", NoulQuestion("It is.", None), 1 - y)
    return Item(row, np.arange(first, first + length, dtype=np.int32), n_opt, y)


def test_pack_bounds_rows_times_longest() -> None:
    items = [fake_item(n) for n in (30, 10, 60, 10, 40, 20, 10)]
    batches = micro_batches(items, 100)
    assert [[len(it) for it in b] for b in batches] == [[10, 10, 10, 20], [30, 40], [60]]
    assert all(len(b) * max(len(it) for it in b) <= 100 for b in batches)
    assert [[len(it) for it in b] for b in pack([fake_item(500)], 100)] == [[500]]  # an oversized row still runs alone
    assert [len(c) for c in chunks(items, 3)] == [3, 3]  # the short tail is left out
    assert [it.row.id for c in chunks(items, 3) for it in c] == [it.row.id for it in items[:6]]


def test_collate_left_pads() -> None:
    items = [fake_item(3, 2, 1), fake_item(5, 4, 3), fake_item(1, 26, 25)]
    ids, mask, n_opt, y = collate(items, pad_id=5)
    assert ids.tolist() == [[5, 5, 100, 101, 102], [100, 101, 102, 103, 104], [5, 5, 5, 5, 100]]
    assert mask.tolist() == [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1]]
    assert n_opt.tolist() == [2, 4, 26] and y.tolist() == [1, 3, 25]
    assert all(str(t.dtype) == "torch.int64" for t in (ids, mask, n_opt, y))


# -- schedule, loss, stop rules: torch but no model ---------------------------------------


def test_schedule_values() -> None:
    from davout.train.loop import schedule

    lr = lambda s: schedule(s, 1e-5, 30, 1500, 0.1)  # noqa: E731
    assert lr(1) == pytest.approx(1e-5 / 30)
    assert lr(15) == pytest.approx(5e-6)
    assert lr(30) == pytest.approx(1e-5)
    assert lr(31) == pytest.approx(1e-5, rel=1e-5) and lr(31) < 1e-5
    assert lr(765) == pytest.approx(1e-5 * (0.1 + 0.9 * 0.5))  # the middle of the cosine
    assert lr(1500) == pytest.approx(1e-6)
    assert all(lr(s) > lr(s + 1) for s in range(30, 1500))
    assert schedule(1, 3e-6, 0, 10) == pytest.approx(3e-6 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.1))))


def test_restricted_ce_matches_hand_computation() -> None:
    import torch

    from davout.train.loop import rce

    torch.manual_seed(0)
    logits = torch.randn(4, 26)
    n_opt = torch.tensor([2, 4, 26, 3])
    y = torch.tensor([1, 0, 25, 2])
    got = rce(logits, n_opt, y)
    for i in range(4):
        row = logits[i, : n_opt[i]].double()
        want = math.log(row.exp().sum().item()) - row[y[i]].item()
        assert got[i].item() == pytest.approx(want, abs=1e-5)
    # letters at or beyond n_opt do not matter, however large their logits
    noisy = logits.clone()
    for i in range(4):
        noisy[i, n_opt[i] :] += 50.0
    assert torch.allclose(rce(noisy, n_opt, y), got, atol=1e-6)
    assert rce(torch.zeros(1, 26), torch.tensor([2]), torch.tensor([0])).item() == pytest.approx(math.log(2))


def test_summarize_groups_and_auroc() -> None:
    from davout.train.loop import summarize

    rng = np.random.default_rng(0)
    n_opt = [2, 2, 2, 2, 4, 4, 3, 3]
    y = [0, 1, 0, 1, 3, 0, 2, 1]
    families = ["nli", "nli", "candidate", "candidate", "choice", "choice", "score", "score"]
    sources = ["a", "a", "b", "b", "b", "c", "c", "c"]
    probs = np.zeros((8, 26))
    for i, n in enumerate(n_opt):
        p = rng.random(n)
        probs[i, :n] = p / p.sum()
    mass = np.linspace(0.1, 0.8, 8)
    out = summarize(probs, n_opt, y, families, sources, mass)
    rows = [probs[i, :n] for i, n in enumerate(n_opt)]
    want = metrics.ragged_summarize(rows, y)
    assert {k: out["all"][k] for k in want} == want and "auroc" not in out["all"]
    assert out["all"]["letter_mass"] == pytest.approx(mass.mean())
    assert set(out["family"]) == {"nli", "candidate", "choice", "score"} and set(out["source"]) == {"a", "b", "c"}
    nli = out["family"]["nli"]
    assert nli["n"] == 2 and nli["nll"] == pytest.approx(-(math.log(probs[0, 0]) + math.log(probs[1, 1])) / 2)
    assert nli["auroc"] == metrics.auroc(probs[:2, 0], [1, 0])  # positive = the Yes target
    assert "auroc" not in out["family"]["choice"] and "auroc" in out["source"]["a"] and "auroc" not in out["source"]["b"]
    assert out["family"]["score"]["letter_mass"] == pytest.approx(mass[6:].mean())


def test_stop_rules_abort_on_bad_trajectories() -> None:
    from davout.train.loop import StopRules

    # healthy: the loss falls, dev NLL falls then wobbles once
    ok = StopRules(base_nll=0.60)
    assert all(ok.on_step(s, 1.0 - 0.002 * s, True) is None for s in range(1, 301))
    assert [ok.on_eval(s, v) for s, v in ((0, 0.60), (100, 0.55), (250, 0.56), (500, 0.50))] == [(None, None)] * 4

    # two non-finite steps (not necessarily adjacent)
    nan = StopRules()
    assert nan.on_step(1, 1.0, True) is None and nan.on_step(2, None, False) is None
    assert nan.on_step(3, 1.0, True) is None
    assert "2 non-finite steps (latest at step 4)" in nan.on_step(4, None, False)

    # the 50-step mean after step 150 exceeds its step-50 value
    up = StopRules()
    assert all(up.on_step(s, 0.7, True) is None for s in range(1, 100))
    assert up.on_step(100, 5.0, True) is None  # worse, but before step 150
    assert all(up.on_step(s, 0.69, True) is None for s in range(101, 151))  # steps 101-150 are better
    assert all(up.on_step(s, 0.75, True) is None for s in range(151, 200))
    reason = up.on_step(200, 0.75, True)
    assert "steps 151-200 (0.7500) exceeds its step-50 value (0.7000)" in reason

    # dev_xfer NLL at step 250 above base + 0.02
    hot = StopRules(base_nll=0.60)
    assert hot.on_eval(100, 0.70) == (None, None)  # only the gate step is gated
    abort, early = hot.on_eval(250, 0.621)
    assert early is None and "above base 0.6000 + 0.02; restart at lr 3e-6" in abort
    assert StopRules(base_nll=0.60).on_eval(250, 0.62) == (None, None)
    assert StopRules(base_nll=None).on_eval(250, 9.0) == (None, None)
    assert "non-finite dev_xfer NLL" in StopRules().on_eval(100, float("nan"))[0]

    # early stop: two consecutive rises
    rise = StopRules(base_nll=1.0)
    assert [rise.on_eval(s, v) for s, v in ((100, 0.50), (250, 0.51), (500, 0.505))] == [(None, None)] * 3
    assert rise.on_eval(750, 0.51) == (None, None)
    abort, early = rise.on_eval(1000, 0.52)
    assert abort is None and "two consecutive evaluations (steps 750 and 1000)" in early

    # the state round-trips through plain data, as it does in a checkpoint
    from dataclasses import asdict

    again = StopRules(**json.loads(json.dumps(asdict(up))))
    assert again == up


# -- the tiny model -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    src = _tokenizer_source()
    if src is None:
        pytest.skip("HRM-Text-1B tokenizer files are not available")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(src, local_files_only=True)


def tiny_model(seed: int = 0):
    import torch
    from transformers.models.hrm_text import HrmTextForCausalLM

    torch.manual_seed(seed)
    m = HrmTextForCausalLM(_tiny_config())
    with torch.no_grad():
        m.model.z_L_init.normal_()
    return m


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory, tokenizer) -> str:
    """A tiny random model saved with its tokenizer: the `base` of every training run here."""
    path = tmp_path_factory.mktemp("base")
    tiny_model().save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> str:
    return str(make_data(tmp_path_factory.mktemp("data")))


@pytest.fixture()
def model():
    return tiny_model()


@pytest.fixture()
def items(model, tokenizer, data_dir):
    backend = HrmBackend(model=model, tokenizer=tokenizer, max_tokens=512)
    got, dropped = tokenize(read_rows(Path(data_dir) / "train.jsonl"), backend)
    assert not dropped
    return got


def test_tokenize_matches_serving_and_drops_long_rows(model, tokenizer, data_dir) -> None:
    rows = read_rows(Path(data_dir) / "train.jsonl")[:14]
    backend = HrmBackend(model=model, tokenizer=tokenizer, max_tokens=512)
    items, dropped = tokenize(rows, backend)
    assert dropped == [] and [it.row for it in items] == rows
    for it in items:
        ids, truncated = backend._encode([render(it.row)])[0]
        assert not truncated and it.ids.tolist() == ids and it.ids.dtype == np.int32
        assert ids[:2] == [6, 8] and ids[-1] == 7  # <|im_start|><|object_ref_start|> ... <|im_end|>
        assert (it.y, it.n_opt) == target(it.row)

    long = Row("long", "s", "nli", "word " * 400, NoulQuestion("It is long.", None), 1)
    hopeless = Row("hopeless", "s", "nli", "x", NoulQuestion("word " * 200, None), 1)
    short = [Row(f"short{i}", "s", "nli", "A cat.", NoulQuestion("It is a cat.", None), i) for i in (0, 1)]
    small = HrmBackend(model=model, tokenizer=tokenizer, max_tokens=64)
    kept, dropped = tokenize([short[0], long, hopeless, short[1]], small)
    assert dropped == ["long", "hopeless"] and [it.row.id for it in kept] == ["short0", "short1"]
    assert [it.y for it in kept] == [1, 0]


def test_forward_uses_prefix_attention_and_matches_serving(model, tokenizer, items) -> None:
    import torch

    from davout.train.loop import forward_cycles

    backend = HrmBackend(model=model, tokenizer=tokenizer)
    letter_ids = torch.tensor(backend.letter_ids)
    batch = items[:7]
    ids, mask, n_opt, y = collate(batch, backend._pad_id)
    seen: dict[str, Any] = {}
    hook = model.model.register_forward_pre_hook(lambda m, args, kwargs: seen.update(kwargs), with_kwargs=True)
    try:
        model.train()
        with torch.no_grad():
            logits = forward_cycles(model, ids, mask, letter_ids)
    finally:
        hook.remove()
    assert seen["use_cache"] is False and torch.equal(seen["token_type_ids"], mask)
    assert torch.equal(seen["attention_mask"], mask) and torch.equal(seen["input_ids"], ids)
    assert (ids[mask == 0] == backend._pad_id).all() and (ids[:, -1] == 7).all()  # the answer position is last

    assert len(logits) == 2 and all(tuple(lg.shape) == (7, 26) and lg.dtype == torch.float32 for lg in logits)
    served = backend._forward([it.ids.tolist() for it in batch])  # the serving path: [cycles, rows, 26]
    assert float((torch.stack(logits) - served).abs().max()) < 1e-4  # train mode changes nothing (dropout is 0)
    with torch.no_grad():
        alone = forward_cycles(model, *collate(batch[:1], backend._pad_id)[:2], letter_ids)
    assert float((alone[1][0] - logits[1][0]).abs().max()) < 1e-3  # padding does not change a row


def test_gradients_reach_the_expected_parameters(model, tokenizer, items) -> None:
    import torch

    from davout.train.loop import forward_cycles, rce

    backend = HrmBackend(model=model, tokenizer=tokenizer)
    letter_ids = torch.tensor(backend.letter_ids)
    model.gradient_checkpointing_enable()
    model.train()
    ids, mask, n_opt, y = collate(items[:6], backend._pad_id)
    assert int(n_opt.max()) == 4

    def norms(which: int) -> dict[str, float]:
        model.zero_grad(set_to_none=True)
        rce(forward_cycles(model, ids, mask, letter_ids)[which], n_opt, y).sum().backward()
        out: dict[str, float] = {}
        for name, p in model.named_parameters():
            group = name.split(".")[1] if name.startswith("model.") else name.split(".")[0]
            out[group] = out.get(group, 0.0) + (0.0 if p.grad is None else float(p.grad.abs().sum()))
        head = model.lm_head.weight.grad
        rows = torch.zeros(head.shape[0], dtype=torch.bool)
        rows[letter_ids[:4]] = True
        assert float(head[~rows].abs().sum()) == 0.0 and float(head[rows].abs().sum()) > 0.0  # used letters only
        return out

    final = norms(-1)
    assert final["z_L_init"] == 0.0 and not model.model.z_L_init.requires_grad
    assert all(final[g] > 0.0 for g in ("embed_tokens", "L_module", "H_module", "lm_head"))
    first = norms(0)  # under L_bp_cycles [0, 3] the cycle-1 loss never reaches the L module
    assert first["L_module"] == 0.0
    assert all(first[g] > 0.0 for g in ("embed_tokens", "H_module", "lm_head"))

    # checkpointing changes memory, not gradients
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.gradient_checkpointing_disable()
    norms(0)
    assert all(torch.allclose(p.grad, grads[n], atol=1e-6) for n, p in model.named_parameters() if p.grad is not None)


def step_setup(model, tokenizer, **overrides):
    import torch

    from davout.train.loop import TrainConfig

    cfg = TrainConfig(data_dir="unused", out_dir="unused", lr=1e-3, **overrides)
    backend = HrmBackend(model=model, tokenizer=tokenizer)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)
    return cfg, opt, params, torch.tensor(backend.letter_ids), backend._pad_id


def test_accumulated_micro_batches_equal_one_batch(model, tokenizer, items) -> None:
    import torch

    from davout.train.loop import train_step

    chunk = items[:16]
    grads = []
    for budget in (10**6, 120):  # one micro-batch, then several
        cfg, opt, params, letter_ids, pad_id = step_setup(model, tokenizer, max_batch_tokens=budget, aux_weight=0.5)
        got: list[Any] = []
        opt.step = lambda got=got, params=params: got.extend(p.grad.clone() for p in params)  # capture, do not update
        info = train_step(model, opt, params, chunk, cfg, letter_ids, pad_id)
        grads.append(got)
        assert info["finite"] and info["tokens"] == sum(len(it) for it in chunk)
        assert info["loss"] == pytest.approx(info["loss_final"] + 0.5 * info["loss_cycle1"], rel=1e-5)
    assert info["micro_batches"] > 1 and info["padded_tokens"] <= info["micro_batches"] * 120
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(*grads))


def test_steps_reduce_train_loss_and_aux_trains_cycle_one(tokenizer, data_dir) -> None:
    from davout.train.loop import train_step

    losses: dict[float, list[dict[str, Any]]] = {}
    for aux in (0.0, 0.5):
        model = tiny_model()
        chunk, _ = tokenize(read_rows(Path(data_dir) / "train.jsonl")[:16], HrmBackend(model=model, tokenizer=tokenizer))
        cfg, opt, params, letter_ids, pad_id = step_setup(model, tokenizer, aux_weight=aux, max_batch_tokens=400)
        losses[aux] = [train_step(model, opt, params, chunk, cfg, letter_ids, pad_id) for _ in range(8)]
    for aux, infos in losses.items():
        assert infos[-1]["loss"] < 0.6 * infos[0]["loss"]  # the same 16 rows, eight steps
        assert infos[-1]["loss_final"] < infos[0]["loss_final"]
    assert losses[0.0][0]["loss_cycle1"] == pytest.approx(losses[0.5][0]["loss_cycle1"])  # same start
    # training the cycle-1 readout directly leaves it better than getting it as a side effect
    assert losses[0.5][-1]["loss_cycle1"] < losses[0.0][-1]["loss_cycle1"] - 0.005


def test_aux_gradient_is_the_weighted_cycle_one_gradient(model, tokenizer, items) -> None:
    import torch

    from davout.train.loop import train_step

    chunk = items[:8]

    def grads(aux: float) -> dict[str, Any]:
        cfg, opt, params, letter_ids, pad_id = step_setup(model, tokenizer, aux_weight=aux, clip=1e9)
        got: dict[str, Any] = {}
        opt.step = lambda: got.update({n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        train_step(model, opt, params, chunk, cfg, letter_ids, pad_id)
        return got

    g0, g1, g2 = grads(0.0), grads(1.0), grads(2.0)
    assert set(g0) == set(g1) and "model.z_L_init" not in g0
    moved = 0
    for name in g0:
        aux = g1[name] - g0[name]  # the cycle-1 gradient
        assert torch.allclose(g2[name], g0[name] + 2.0 * aux, atol=1e-5)
        if name.startswith("model.L_module."):
            assert float(aux.abs().max()) < 1e-7  # the L module learns from the final cycle only
        else:
            moved += float(aux.abs().max()) > 1e-6
    assert moved >= 0.9 * sum(not n.startswith("model.L_module.") for n in g0)  # embeddings, H module, head


def test_non_finite_gradient_step_is_skipped(model, tokenizer, items) -> None:
    import torch

    from davout.train.loop import train_step

    cfg, opt, params, letter_ids, pad_id = step_setup(model, tokenizer)
    before = [p.detach().clone() for p in params]
    poison = params[3].register_hook(lambda g: g * float("nan"))
    info = train_step(model, opt, params, items[:8], cfg, letter_ids, pad_id)
    poison.remove()
    assert info["finite"] is False and info["grad_norm"] is None and info["loss"] is not None
    assert all(torch.equal(a, b) for a, b in zip(before, params))  # nothing moved
    assert all(p.grad is None for p in params)  # and the bad gradient is gone
    assert not opt.state  # the optimizer never saw it

    info = train_step(model, opt, params, items[:8], cfg, letter_ids, pad_id)
    assert info["finite"] is True and info["grad_norm"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, params))


def test_gradient_clipping_bounds_the_update(model, tokenizer, items) -> None:
    import torch

    from davout.train.loop import train_step

    cfg, opt, params, letter_ids, pad_id = step_setup(model, tokenizer, clip=1e-3)
    seen: list[float] = []
    opt.step = lambda: seen.append(float(torch.sqrt(sum(p.grad.pow(2).sum() for p in params if p.grad is not None))))
    info = train_step(model, opt, params, items[:8], cfg, letter_ids, pad_id)
    assert info["grad_norm"] > 1e-3  # the norm before clipping is what is logged
    assert seen[0] == pytest.approx(1e-3, rel=1e-3)


# -- whole runs ---------------------------------------------------------------------------


def run_config(base_dir: str, data_dir: str, out: Path, **overrides):
    from davout.train.loop import TrainConfig

    settings = {
        "base": base_dir, "lr": 2e-3, "warmup": 2, "batch": 8, "max_batch_tokens": 400,
        "eval_steps": (8, 16), "device": "cpu",
    }  # fmt: skip
    settings.update(overrides)
    return TrainConfig(data_dir=data_dir, out_dir=str(out), **settings)


def read_metrics(out: Path) -> tuple[list[dict], list[dict]]:
    lines = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
    return [r for r in lines if r["kind"] == "train"], [r for r in lines if r["kind"] == "eval"]


@pytest.fixture(scope="module")
def run_a(base_dir, data_dir, tmp_path_factory):
    """Arm A (final-cycle loss only): one epoch of 24 steps."""
    from davout.train.loop import train

    out = tmp_path_factory.mktemp("arm_a")
    return out, train(run_config(base_dir, data_dir, out))


@pytest.fixture(scope="module")
def run_b(base_dir, data_dir, tmp_path_factory):
    """Arm B: the same run with the cycle-1 loss added."""
    from davout.train.loop import train

    out = tmp_path_factory.mktemp("arm_b")
    return out, train(run_config(base_dir, data_dir, out, aux_weight=0.5))


def test_run_reduces_loss_and_logs_everything(run_a) -> None:
    out, result = run_a
    steps, evals = read_metrics(out)
    assert [r["step"] for r in steps] == list(range(1, 25))
    assert [r["step"] for r in evals] == [0, 8, 16, 24]  # base, the eval steps, the last step
    assert result["state"] == "complete" and result["reason"] is None
    assert (result["step"], result["total_steps"], result["stop_at"]) == (24, 24, 24)

    first = steps[0]
    assert set(first) >= {"loss", "loss_final", "loss_cycle1", "lr", "grad_norm", "tok_s", "peak_memory_gib", "seconds"}
    assert first["lr"] == pytest.approx(1e-3) and steps[1]["lr"] == pytest.approx(2e-3)
    assert steps[-1]["lr"] == pytest.approx(2e-4)  # the cosine ends at lr * 0.1
    assert first["loss"] == pytest.approx(first["loss_final"]) and first["micro_batches"] >= 2
    assert all(r["finite"] and r["tok_s"] > 0 and r["padded_tokens"] >= r["tokens"] for r in steps)
    assert result["padded_tokens"] == sum(r["padded_tokens"] for r in steps) and result["tok_s"] > 0

    base, last = evals[0], evals[-1]
    for name in ("dev_in", "dev_xfer"):
        assert set(base[name]) == {"c1", "final"}
        final = last[name]["final"]
        assert final["all"]["n"] == 56 and set(final["family"]) == set(FAMILIES)
        assert set(final["source"]) == {f"src_{f}" for f in FAMILIES}
        assert set(final["all"]) == {"accuracy", "nll", "brier", "ece", "n", "letter_mass"}
        assert all(("auroc" in m) == (f in ("nli", "reading", "judgement", "ovr", "candidate")) for f, m in final["family"].items())
        assert 0.0 < final["all"]["letter_mass"] <= 1.0
        assert final["all"]["nll"] < base[name]["final"]["all"]["nll"]  # training helped on held-out rows
    assert result["base"]["dev_xfer"]["final"] == base["dev_xfer"]["final"]["all"]["nll"]
    assert result["base_dev_xfer_nll"] == result["base"]["dev_xfer"]["final"]

    best = result["best"]
    by_step = {r["step"]: r["dev_xfer"]["final"]["all"]["nll"] for r in evals if r["step"] > 0}
    assert best["step"] == min(by_step, key=by_step.get) and best["dev_xfer_nll"] == by_step[best["step"]]
    assert json.loads((out / "status.json").read_text()) == result
    config = json.loads((out / "config.json").read_text())
    assert config["rows"] == {"train": 192, "dev_in": 56, "dev_xfer": 56} and config["total_steps"] == 24
    assert config["dropped_too_long"] == {"train": 0, "dev_in": 0, "dev_xfer": 0}


def test_checkpoints_and_exported_model(run_a, data_dir, tokenizer) -> None:
    import torch
    from transformers import AutoModelForCausalLM

    from davout.train.loop import forward_cycles

    out, result = run_a
    last = torch.load(out / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert last["trainer"]["step"] == 24 and best["step"] == result["best"]["step"]
    assert all(v.dtype == torch.float32 for v in best["model"].values() if v.is_floating_point())
    assert set(last["optimizer"]) == {"state", "param_groups"} and len(last["optimizer"]["state"]) > 0

    # the exported model: bf16 weights of the best step, with the tokenizer, loadable as serving loads it
    model_dir = Path(result["model_dir"])
    assert model_dir == out / "model" and (model_dir / "tokenizer.json").is_file()
    assert json.loads((model_dir / "config.json").read_text())["dtype"] == "bfloat16"
    backend = HrmBackend(str(model_dir), device="cpu")
    assert backend.info()["model_id"] == str(model_dir)
    rows = read_rows(Path(data_dir) / "dev_xfer.jsonl")
    dev, _ = tokenize(rows, backend)
    served = backend.read([render(r) for r in rows])

    # 1. the served logits are the letter logits of the saved weights
    saved = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, local_files_only=True).eval()
    letter_ids = torch.tensor(backend.letter_ids)
    for it, out_row in zip(dev[:10], served):
        ids, mask, _, _ = collate([it], backend._pad_id)
        with torch.no_grad():
            direct = torch.stack(forward_cycles(saved, ids, mask, letter_ids))[:, 0, : it.n_opt]
        assert float((direct - torch.tensor(out_row.cycle_logits)).abs().max()) < 1e-3
    # 2. and they are the best fp32 weights up to bf16 rounding
    master = tiny_model(1)
    master.load_state_dict(best["model"])
    master.eval()
    worst = 0.0
    for it, out_row in zip(dev, served):
        ids, mask, _, _ = collate([it], backend._pad_id)
        with torch.no_grad():
            direct = forward_cycles(master, ids, mask, letter_ids)[-1][0, : it.n_opt]
        worst = max(worst, float((direct - torch.tensor(out_row.logits)).abs().max()))
    assert 0.0 < worst < 0.25

    gate = result["sanity_gate"]
    assert gate["pass"] is True and gate["tolerance"] == 0.01 and gate["served_dtype"] == "float32"
    assert gate["in_loop_nll"] == result["best"]["nll"]
    assert gate["max_final_diff"] == max(gate["abs_diff"][s]["final"] for s in ("dev_in", "dev_xfer")) < 0.01
    assert set(gate["served_nll"]["dev_xfer"]) == {"c1", "final"}


def test_aux_weight_trains_the_first_cycle(run_a, run_b) -> None:
    (out_a, res_a), (out_b, res_b) = run_a, run_b
    steps_a, evals_a = read_metrics(out_a)
    steps_b, evals_b = read_metrics(out_b)
    assert evals_a[0] == {**evals_b[0], "seconds": evals_a[0]["seconds"]}  # the same base model
    assert steps_b[0]["loss"] == pytest.approx(steps_b[0]["loss_final"] + 0.5 * steps_b[0]["loss_cycle1"], rel=1e-5)
    assert steps_a[0]["loss_cycle1"] == pytest.approx(steps_b[0]["loss_cycle1"], rel=1e-5)  # logged in arm A too
    c1_a, c1_b = (e[-1]["dev_in"]["c1"]["all"] for e in (evals_a, evals_b))
    assert c1_a != c1_b
    assert c1_b["nll"] < c1_a["nll"]  # the aux loss is what trains the cycle-1 readout
    assert c1_b["nll"] < evals_b[0]["dev_in"]["c1"]["all"]["nll"]
    assert res_b["aux_weight"] == 0.5 and res_a["aux_weight"] == 0.0


def test_resume_continues_from_the_last_checkpoint(base_dir, data_dir, tmp_path, run_a) -> None:
    import torch

    from davout.train.loop import train

    out = tmp_path / "resumed"
    part = train(run_config(base_dir, data_dir, out, max_steps=8))
    assert (part["state"], part["step"], part["total_steps"], part["stop_at"]) == ("complete", 8, 24, 8)
    assert part["best"]["step"] == 8 and (out / "model" / "model.safetensors").is_file()
    assert torch.load(out / "last.pt", weights_only=True)["trainer"]["step"] == 8

    # an interrupted run: steps logged after the last checkpoint are repeated, not duplicated
    with (out / "metrics.jsonl").open("a") as f:
        f.write(json.dumps({"kind": "train", "step": 9, "loss": 99.0}) + "\n" + '{"kind": "train", "st')
    full = train(run_config(base_dir, data_dir, out))
    assert (full["state"], full["step"]) == ("complete", 24)
    steps, evals = read_metrics(out)
    assert [r["step"] for r in steps] == list(range(1, 25)) and steps[8]["loss"] != 99.0
    assert [r["step"] for r in evals] == [0, 8, 16, 24]

    # the resumed run is the uninterrupted run: same losses, same weights
    straight_out, straight = run_a
    straight_steps, _ = read_metrics(straight_out)
    assert [r["loss"] for r in steps] == pytest.approx([r["loss"] for r in straight_steps], rel=1e-6)
    a = torch.load(out / "last.pt", weights_only=True)["model"]
    b = torch.load(straight_out / "last.pt", weights_only=True)["model"]
    assert all(torch.allclose(a[k], b[k], atol=1e-7) for k in b)
    assert full["best"] == straight["best"]

    # running a finished run again trains nothing more
    again = train(run_config(base_dir, data_dir, out))
    assert again["step"] == 24 and len(read_metrics(out)[0]) == 24

    with pytest.raises(RuntimeError, match="different settings.*lr: 0.002 -> 0.0005"):
        train(run_config(base_dir, data_dir, out, lr=5e-4))


def test_gate_aborts_the_run(base_dir, data_dir, tmp_path) -> None:
    from davout.train.loop import train

    out = tmp_path / "hot"
    cfg = run_config(base_dir, data_dir, out, base_dev_xfer_nll=0.001, gate_step=8)
    result = train(cfg)
    assert result["state"] == "aborted" and result["step"] == 8
    assert "at step 8 is above base 0.0010 + 0.02; restart at lr 3e-6" in result["reason"]
    assert json.loads((out / "status.json").read_text()) == result
    assert not (out / "model").exists() and "sanity_gate" not in result
    assert [r["step"] for r in read_metrics(out)[1]] == [0, 8]
    assert train(cfg) == result  # the same settings would abort the same way: nothing is rerun
    assert len(read_metrics(out)[0]) == 8


def test_early_stop_exports_the_best_checkpoint(base_dir, data_dir, tmp_path, monkeypatch) -> None:
    from davout.train import loop

    # Script dev_xfer final NLL: base 0.70, then 0.20, 0.25, 0.28 -> two consecutive rises at the third eval.
    scripted = iter([0.70, 0.20, 0.25, 0.28, 0.10])
    real = loop.evaluate
    calls = {"n": 0}

    def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
        out = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] % 2 == 0:  # dev_in, dev_xfer, dev_in, dev_xfer, ...
            out["final"]["all"]["nll"] = next(scripted)
        return out

    monkeypatch.setattr(loop, "evaluate", fake)
    out = tmp_path / "rising"
    result = loop.train(run_config(base_dir, data_dir, out, eval_steps=(2, 4, 6, 8)))
    assert result["state"] == "early_stopped" and result["step"] == 6
    assert "two consecutive evaluations (steps 4 and 6)" in result["reason"]
    assert result["best"]["step"] == 2 and result["best"]["dev_xfer_nll"] == 0.20
    assert (out / "model" / "model.safetensors").is_file()
    # the exported step-2 model is real, the scripted NLL is not: the sanity gate notices
    gate = result["sanity_gate"]
    assert gate["pass"] is False and gate["abs_diff"]["dev_xfer"]["final"] > 0.1
    assert gate["abs_diff"]["dev_in"]["final"] < 0.01
    assert loop.train(run_config(base_dir, data_dir, out, eval_steps=(2, 4, 6, 8)))["step"] == 6  # stays stopped


def test_train_rejects_bad_setups(base_dir, data_dir, tmp_path) -> None:
    from davout.train.loop import TrainConfig, train

    with pytest.raises(ValueError, match="25 steps of 8 need 200 examples"):
        train(run_config(base_dir, data_dir, tmp_path / "a", steps=25))
    with pytest.raises(ValueError, match="train.jsonl not found"):
        train(run_config(base_dir, str(tmp_path / "nowhere"), tmp_path / "b"))
    with pytest.raises(RuntimeError, match="could not load"):
        train(run_config(str(tmp_path / "no-model"), data_dir, tmp_path / "c"))
    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        TrainConfig("d", "o", max_steps=0)
    cfg = TrainConfig("d", "o")
    assert (cfg.lr, cfg.warmup, cfg.steps, cfg.batch, cfg.max_batch_tokens) == (1e-5, 30, None, 64, 8192)
    assert (cfg.aux_weight, cfg.weight_decay, cfg.betas, cfg.clip) == (0.0, 0.1, (0.9, 0.95), 30.0)
    assert cfg.eval_steps == (100, 250, 500, 750, 1000, 1250, 1500) and cfg.lr_min_ratio == 0.1
    assert cfg.grad_checkpointing is True and cfg.base == "sapientinc/HRM-Text-1B" and cfg.max_tokens == 512


# -- command line -------------------------------------------------------------------------


def test_cli_train_run(monkeypatch, capsys) -> None:
    from davout.train import loop

    seen: list[Any] = []
    result = {"state": "complete", "sanity_gate": {"pass": True}}

    def fake_train(cfg):
        seen.append(cfg)
        return result

    monkeypatch.setattr(loop, "train", fake_train)
    assert cli.main(["train", "run", "--data", "d", "--out", "o", "--device", "cpu"]) == 0
    cfg = seen[-1]
    assert cfg == loop.TrainConfig("d", "o", device="cpu")
    assert json.loads(capsys.readouterr().out) == result

    args = [
        "train", "run", "--data", "d", "--out", "o", "--aux-weight", "0.5", "--lr", "3e-6", "--steps", "1500",
        "--batch", "32", "--max-batch-tokens", "4096", "--base", "/ckpt", "--base-dev-xfer-nll", "0.61",
        "--max-steps", "20", "--device", "cuda", "--eval-steps", "10,20", "--seed", "3",
    ]  # fmt: skip
    result = {"state": "complete", "sanity_gate": {"pass": False}}
    assert cli.main(args) == 1  # a failed sanity gate is a failed run
    cfg = seen[-1]
    assert (cfg.aux_weight, cfg.lr, cfg.steps, cfg.batch, cfg.max_batch_tokens) == (0.5, 3e-6, 1500, 32, 4096)
    assert (cfg.base, cfg.base_dev_xfer_nll, cfg.max_steps, cfg.device) == ("/ckpt", 0.61, 20, "cuda")
    assert (cfg.eval_steps, cfg.seed) == ((10, 20), 3)

    result = {"state": "aborted", "reason": "x"}
    assert cli.main(args) == 1
    capsys.readouterr()
    assert cli.main(["train", "run", "--data", "d", "--out", "o", "--batch", "0"]) == 1
    assert "batch and max_batch_tokens must be >= 1" in capsys.readouterr().err
    assert cli.main(["train"]) == 2
    with pytest.raises(SystemExit):
        cli.main(["train", "run", "--out", "o"])  # --data is required


def test_cli_train_build_data(monkeypatch, capsys) -> None:
    seen: list[Any] = []
    stub = types.ModuleType("davout.train.data")
    stub.build = lambda out_dir, **kw: seen.append((out_dir, kw))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "davout.train.data", stub)
    assert cli.main(["train", "build-data", "--out", "data/v1"]) == 0
    assert seen[-1] == ("data/v1", {"seed": 0, "scale": 1.0, "sources": None, "recipe": "v1"})
    assert cli.main(["train", "build-data", "--out", "x", "--seed", "2", "--scale", "0.1", "--sources", "mnli, anli"]) == 0
    assert seen[-1] == ("x", {"seed": 2, "scale": 0.1, "sources": ["mnli", "anli"], "recipe": "v1"})
    assert cli.main(["train", "build-data", "--out", "c", "--recipe", "cand_v1"]) == 0
    assert seen[-1] == ("c", {"seed": 0, "scale": 1.0, "sources": None, "recipe": "cand_v1"})
    with pytest.raises(SystemExit):
        cli.main(["train", "build-data"])  # --out is required


def test_soft_targets_match_the_label_loss_and_ignore_masked_letters(tmp_path: Path) -> None:
    import torch

    from davout.train.format import read_rows, soft_targets
    from davout.train.loop import N_LETTERS, rce, row_loss, soft_ce

    torch.manual_seed(0)
    logits = torch.randn(4, N_LETTERS)
    n_opt = torch.tensor([3, 10, 2, 5])
    y = torch.tensor([1, 7, 0, 4])
    onehot = torch.zeros(4, N_LETTERS)
    onehot[torch.arange(4), y] = 1.0
    assert torch.allclose(soft_ce(logits, n_opt, onehot), rce(logits, n_opt, y), atol=1e-6)
    # a real distribution: the loss is the target-weighted mean of the per-option losses, and is finite
    target = torch.zeros(4, N_LETTERS)
    target[0, :3] = torch.tensor([0.5, 0.25, 0.25])
    expected = sum(w * rce(logits[:1], n_opt[:1], torch.tensor([k])) for k, w in enumerate((0.5, 0.25, 0.25)))
    assert torch.allclose(soft_ce(logits[:1], n_opt[:1], target[:1]), expected, atol=1e-6)
    # mixed batch: hard rows keep exactly the label loss
    is_soft = torch.tensor([True, False, False, False])
    mixed = row_loss(logits, n_opt, y, (torch.where(is_soft[:, None], target, onehot), is_soft))
    assert torch.equal(mixed[1:], rce(logits, n_opt, y)[1:]) and torch.isfinite(mixed).all()
    assert torch.equal(row_loss(logits, n_opt, y, None), rce(logits, n_opt, y))

    q = {"type": "choice", "instructions": "Which?", "criteria": {"a": None, "b": None, "c": None}}
    rows = [
        {"id": "s", "source": "x", "family": "choice", "state": "t", "question": q, "label": 0, "target": [0.6, 0.3, 0.1]},
        {"id": "h", "source": "x", "family": "choice", "state": "t", "question": q, "label": 2},
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    soft, hard = read_rows(path)
    assert soft.target == pytest.approx((0.6, 0.3, 0.1)) and hard.target is None
    items = [Item(soft, np.arange(4, dtype=np.int32), 3, 0), Item(hard, np.arange(4, dtype=np.int32), 3, 2)]
    t, mask = soft_targets(items, N_LETTERS)
    assert mask.tolist() == [True, False] and t[0, :3].tolist() == pytest.approx([0.6, 0.3, 0.1]) and t[1, 2] == 1.0 and t[:, 3:].sum() == 0
    assert soft_targets(items[1:], N_LETTERS) is None
    path.write_text(json.dumps({**rows[0], "target": [0.6, 0.3]}) + "\n")
    with pytest.raises(ValueError, match="target"):
        read_rows(path)
