"""Fine-tune HRM-Text as a typed-decision model.

The objective is cross-entropy over the valid answer letters at the last position,
read after the final H cycle and (optionally, `aux_weight`) after the first one.
Master weights stay fp32; on CUDA the body runs under bf16 autocast, the head and the
loss in fp32. One optimizer step sees `batch` examples, split into length-sorted
micro-batches whose gradients accumulate.

`train(cfg)` writes into `cfg.out_dir`:

    config.json     the settings and data fingerprints (guards resume)
    metrics.jsonl   one line per optimizer step and one per dev evaluation
    status.json     state ("running", "complete", "early_stopped", "aborted"), reason, best step, sanity gate
    last.pt         fp32 weights, optimizer and trainer state at the latest evaluation (for resume)
    best.pt         fp32 weights with the lowest dev_xfer final-cycle NLL
    model/          the best weights in bf16 with the tokenizer; load with `HrmBackend(path)`
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from davout import metrics
from davout.backends.base import LETTERS
from davout.backends.hrm import MODEL_ID, HrmBackend, _pick_device
from davout.train.format import BINARY_FAMILIES, Item, chunks, collate, micro_batches, pack, read_rows, render, soft_targets, tokenize

N_LETTERS = len(LETTERS)
DEV_SETS = ("dev_in", "dev_xfer")
BEST_SET = "dev_xfer"  # the best checkpoint and the stop rules follow this set's final-cycle NLL

# Settings that change the training trajectory; a resumed run must match them.
_RESUME_FIELDS = (
    "base", "lr", "warmup", "steps", "batch", "max_batch_tokens", "aux_weight", "weight_decay",
    "betas", "clip", "eval_steps", "seed", "lr_min_ratio", "max_tokens", "eval_at_start",
)  # fmt: skip


@dataclass
class TrainConfig:
    """One fine-tuning run."""

    data_dir: str
    out_dir: str
    base: str = MODEL_ID  # model id or path of the starting weights
    lr: float = 1e-5  # peak learning rate
    warmup: int = 30  # linear warmup steps
    steps: int | None = None  # optimizer steps the schedule spans; None = one epoch
    batch: int = 64  # examples per optimizer step
    max_batch_tokens: int = 8192  # rows x longest row per micro-batch
    aux_weight: float = 0.0  # weight of the cycle-1 loss (0 = final cycle only)
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    clip: float = 30.0  # global gradient-norm clip
    eval_steps: tuple[int, ...] = (100, 250, 500, 750, 1000, 1250, 1500)
    seed: int = 0
    lr_min_ratio: float = 0.1  # the cosine ends at lr * lr_min_ratio
    device: str = "auto"
    grad_checkpointing: bool = True
    base_dev_xfer_nll: float | None = None  # reference for the gate; None = the value measured at step 0
    max_steps: int | None = None  # stop (and export) after this many steps; the schedule is unchanged
    max_tokens: int = 512  # longer prompts are dropped
    eval_at_start: bool = True  # evaluate the untouched base weights as step 0
    gate_step: int = 250  # abort if dev_xfer NLL at this evaluation exceeds base + gate_margin
    gate_margin: float = 0.02
    sanity_tol: float = 0.01  # the saved model must reproduce the in-loop dev NLL within this
    early_stop: bool = True  # stop when dev_xfer NLL rises at two consecutive evaluations

    def __post_init__(self) -> None:
        self.betas = (float(self.betas[0]), float(self.betas[1]))
        self.eval_steps = tuple(sorted({int(s) for s in self.eval_steps}))
        if self.batch < 1 or self.max_batch_tokens < 1:
            raise ValueError("batch and max_batch_tokens must be >= 1")
        if self.lr <= 0 or self.warmup < 0 or self.aux_weight < 0:
            raise ValueError("lr must be > 0; warmup and aux_weight must be >= 0")
        for name in ("steps", "max_steps"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if any(s < 1 for s in self.eval_steps):
            raise ValueError("eval_steps must be >= 1")


# -- schedule, forward, loss --------------------------------------------------------------


def schedule(step: int, peak: float, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """Learning rate at optimizer step `step` (1-based): linear warmup, then Sapient's cosine."""
    if step <= warmup:
        return peak * step / warmup
    p = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * p)))


def _autocast(device: Any) -> Any:
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def cycle_states(model: Any, ids: Any, mask: Any) -> list[Any]:
    """Last-position hidden state after each H cycle, post-norm and with gradient: [rows, hidden] each.

    `ids` must be left-padded so that position -1 is every row's answer position.
    """
    states: list[Any] = []

    def grab(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        states.append(hidden[:, -1])

    hook = model.model.H_module.register_forward_hook(grab)
    try:
        with _autocast(ids.device):
            # model.model, not model: the wrapper would compute rows x tokens x vocab logits.
            model.model(input_ids=ids, attention_mask=mask, token_type_ids=mask, use_cache=False)
    finally:
        hook.remove()
    if len(states) != model.config.H_cycles or len(states) < 2:
        raise RuntimeError(f"expected {model.config.H_cycles} (>= 2) H cycles, saw {len(states)}")
    return states


def forward_cycles(model: Any, ids: Any, mask: Any, letter_ids: Any) -> list[Any]:
    """fp32 logits over the 26 answer letters after each H cycle: [rows, 26] each."""
    head = model.lm_head
    w = head.weight[letter_ids].float()  # only these rows get gradient
    b = None if head.bias is None else head.bias[letter_ids].float()
    return [F.linear(s.float(), w, b) for s in cycle_states(model, ids, mask)]


def rce(logits: Any, n_opt: Any, y: Any) -> Any:
    """Per-row cross-entropy over the first `n_opt` letters; the other letters are masked out."""
    invalid = torch.arange(logits.shape[-1], device=logits.device)[None, :] >= n_opt[:, None]
    return F.cross_entropy(logits.masked_fill(invalid, float("-inf")), y, reduction="none")


def soft_ce(logits: Any, n_opt: Any, target: Any) -> Any:
    """Per-row cross-entropy against a distribution over the first `n_opt` letters.

    The log-softmax runs over the valid letters only; the masked letters (log-probability
    -inf, target 0) are zeroed before the product so they cannot turn the sum into NaN.
    """
    invalid = torch.arange(logits.shape[-1], device=logits.device)[None, :] >= n_opt[:, None]
    logp = F.log_softmax(logits.masked_fill(invalid, float("-inf")), dim=-1)
    return -(target * logp.masked_fill(invalid, 0.0)).sum(-1)


def row_loss(logits: Any, n_opt: Any, y: Any, soft: tuple[Any, Any] | None) -> Any:
    """Per-row loss: `rce` on the label, replaced by `soft_ce` on rows that carry a soft target."""
    hard = rce(logits, n_opt, y)
    if soft is None:
        return hard
    target, is_soft = soft
    return torch.where(is_soft, soft_ce(logits, n_opt, target), hard)


def train_step(
    model: Any, opt: Any, params: list[Any], chunk: Sequence[Item], cfg: TrainConfig, letter_ids: Any, pad_id: int
) -> dict[str, Any]:
    """One optimizer step over `chunk`; the step is skipped if the gradient is not finite."""
    device = letter_ids.device
    model.train()  # gradient checkpointing is only active in train mode
    total = final = first = 0.0
    padded = real = 0
    batches = micro_batches(chunk, cfg.max_batch_tokens)
    for mb in batches:
        ids, mask, n_opt, y = collate(mb, pad_id, device)
        logits = forward_cycles(model, ids, mask, letter_ids)
        soft = soft_targets(mb, N_LETTERS, device)
        loss_final = row_loss(logits[-1], n_opt, y, soft).sum()
        if cfg.aux_weight > 0:
            loss_first = row_loss(logits[0], n_opt, y, soft).sum()
            loss = (loss_final + cfg.aux_weight * loss_first) / len(chunk)
        else:
            loss_first = rce(logits[0].detach(), n_opt, y).sum()  # logged only
            loss = loss_final / len(chunk)
        loss.backward()
        total += loss.item()
        final += loss_final.item()
        first += loss_first.item()
        padded += ids.numel()
        real += int(mask.sum())
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.clip))
    finite = math.isfinite(grad_norm) and math.isfinite(total)
    if finite:
        opt.step()
    opt.zero_grad(set_to_none=True)
    return {
        "loss": total if math.isfinite(total) else None,
        "loss_final": _finite(final / len(chunk)),
        "loss_cycle1": _finite(first / len(chunk)),
        "grad_norm": _finite(grad_norm),
        "finite": finite,
        "micro_batches": len(batches),
        "padded_tokens": padded,
        "tokens": real,
    }


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


# -- dev evaluation -----------------------------------------------------------------------


def cycle_names(n: int) -> list[str]:
    return [f"c{i + 1}" for i in range(n - 1)] + ["final"]


def summarize(
    probs: np.ndarray,
    n_opt: Sequence[int],
    y: Sequence[int],
    families: Sequence[str],
    sources: Sequence[str],
    mass: np.ndarray | None = None,
) -> dict[str, Any]:
    """NLL, accuracy, Brier and ECE (15 bins, T=1) over all rows, per family and per source.

    `probs` is [rows, 26], normalised over each row's first `n_opt` letters. Groups made
    only of Yes/No rows also get AUROC (positive = the Yes target); `mass`, when given,
    is the full-vocabulary probability on the valid letters and is averaged per group.
    """
    y_arr = np.asarray(y)

    def group(idx: list[int]) -> dict[str, Any]:
        out = metrics.ragged_summarize([probs[i, : n_opt[i]] for i in idx], [int(y_arr[i]) for i in idx])
        if all(families[i] in BINARY_FAMILIES for i in idx):
            auroc = metrics.auroc(probs[idx, 0], (y_arr[idx] == 0).astype(int))
            out["auroc"] = _finite(auroc)
        if mass is not None:
            out["letter_mass"] = float(mass[idx].mean())
        return out

    def by(keys: Sequence[str]) -> dict[str, Any]:
        index: dict[str, list[int]] = {}
        for i, k in enumerate(keys):
            index.setdefault(k, []).append(i)
        return {k: group(index[k]) for k in sorted(index)}

    out = {"all": group(list(range(len(y_arr)))), "family": by(families), "source": by(sources)}
    out["all"].pop("auroc", None)  # only meaningful per family
    return out


@torch.no_grad()
def evaluate(model: Any, items: Sequence[Item], letter_ids: Any, pad_id: int, max_batch_tokens: int) -> dict[str, Any]:
    """Per-cycle dev metrics: `{"c1": {...}, "final": {...}}`, each as `summarize` returns."""
    device = letter_ids.device
    model.eval()
    n_cycles = model.config.H_cycles
    probs = np.zeros((n_cycles, len(items), N_LETTERS))
    mass = np.zeros((n_cycles, len(items)))
    order = sorted(range(len(items)), key=lambda i: len(items[i]))
    head = model.lm_head
    pos = 0
    for mb in pack([items[i] for i in order], max_batch_tokens):
        idx = order[pos : pos + len(mb)]
        pos += len(mb)
        ids, mask, n_opt, _ = collate(mb, pad_id, device)
        invalid = torch.arange(N_LETTERS, device=device)[None, :] >= n_opt[:, None]
        for c, state in enumerate(cycle_states(model, ids, mask)):
            bias = None if head.bias is None else head.bias.float()
            full = F.linear(state.float(), head.weight.float(), bias)  # last position only: [rows, vocab]
            letters = full[:, letter_ids].masked_fill(invalid, float("-inf"))
            mass[c, idx] = (torch.logsumexp(letters, -1) - torch.logsumexp(full, -1)).exp().cpu().numpy()
            probs[c, idx] = torch.softmax(letters.double(), -1).cpu().numpy()
    if not np.isfinite(probs).all():
        raise FloatingPointError("non-finite dev probabilities")
    return _summarize_cycles(probs, items, mass)


def _summarize_cycles(probs: np.ndarray, items: Sequence[Item], mass: np.ndarray | None) -> dict[str, Any]:
    n_opt = [it.n_opt for it in items]
    y = [it.y for it in items]
    families = [it.row.family for it in items]
    sources = [it.row.source for it in items]
    return {
        name: summarize(probs[c], n_opt, y, families, sources, None if mass is None else mass[c])
        for c, name in enumerate(cycle_names(len(probs)))
    }


def served_metrics(backend: Any, items: Sequence[Item]) -> dict[str, Any]:
    """The same per-cycle metrics, computed through the serving path (`backend.read`)."""
    outs = backend.read([render(it.row) for it in items])
    n_cycles = len(outs[0].cycle_logits)
    probs = np.zeros((n_cycles, len(items), N_LETTERS))
    for i, (it, out) in enumerate(zip(items, outs)):
        for c, row in enumerate(out.cycle_logits):
            z = np.asarray(row, dtype=float)
            z = np.exp(z - z.max())
            probs[c, i, : it.n_opt] = z / z.sum()
    return _summarize_cycles(probs, items, None)


def _nlls(per_cycle: dict[str, Any]) -> dict[str, float]:
    return {name: m["all"]["nll"] for name, m in per_cycle.items()}


# -- stop rules ---------------------------------------------------------------------------


@dataclass
class StopRules:
    """The abort and early-stop rules; all state is plain data so it survives a resume.

    Abort: two non-finite steps; the mean train loss of a 50-step window ending at step
    150 or later exceeds the mean of steps 1-50; dev_xfer final NLL at the gate step is
    above base + margin. Early stop: dev_xfer NLL rises at two consecutive evaluations.
    """

    base_nll: float | None = None
    gate_step: int = 250
    gate_margin: float = 0.02
    window: int = 50
    loss_from: int = 150
    max_nonfinite: int = 2
    losses: list[float | None] = field(default_factory=list)  # per step; None = non-finite (skipped)
    nonfinite: int = 0
    evals: list[list[float]] = field(default_factory=list)  # [step, dev_xfer final NLL]

    def _window_mean(self, end: int) -> float | None:
        xs = [x for x in self.losses[end - self.window : end] if x is not None]
        return sum(xs) / len(xs) if xs else None

    def on_step(self, step: int, loss: float | None, finite: bool) -> str | None:
        """Record the train loss of `step`; returns the reason to abort, if any."""
        self.losses.append(loss if finite else None)
        if not finite:
            self.nonfinite += 1
            if self.nonfinite >= self.max_nonfinite:
                return f"{self.nonfinite} non-finite steps (latest at step {step})"
        if step >= self.loss_from and step % self.window == 0:
            ref, cur = self._window_mean(self.window), self._window_mean(step)
            if ref is not None and cur is not None and cur > ref:
                return (
                    f"mean train loss over steps {step - self.window + 1}-{step} ({cur:.4f}) "
                    f"exceeds its step-{self.window} value ({ref:.4f})"
                )
        return None

    def on_eval(self, step: int, nll: float) -> tuple[str | None, str | None]:
        """Record dev_xfer final NLL at `step`; returns (reason to abort, reason to stop early)."""
        self.evals.append([step, nll])
        if not math.isfinite(nll):
            return f"non-finite dev_xfer NLL at step {step}", None
        if step == self.gate_step and self.base_nll is not None and nll > self.base_nll + self.gate_margin:
            return (
                f"dev_xfer final NLL {nll:.4f} at step {step} is above base {self.base_nll:.4f} "
                f"+ {self.gate_margin}; restart at lr 3e-6"
            ), None
        if len(self.evals) >= 3:
            (_, a), (s1, b), (s2, c) = self.evals[-3:]
            if a < b < c:
                return None, f"dev_xfer NLL rose at two consecutive evaluations (steps {int(s1)} and {int(s2)})"
        return None, None


# -- files --------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, body: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _save(path: Path, body: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(body, tmp)
    os.replace(tmp, path)  # a crash mid-write leaves the previous checkpoint intact


def _say(msg: str) -> None:
    print(f"[train] {msg}", file=sys.stderr, flush=True)


def _fingerprint(cfg: TrainConfig, data: dict[str, Any]) -> dict[str, Any]:
    conf = json.loads(json.dumps(asdict(cfg)))  # tuples -> lists, as they come back from disk
    return {**{k: conf[k] for k in _RESUME_FIELDS}, "data": data}


def _check_resume(old: dict[str, Any], new: dict[str, Any], out: Path) -> None:
    diff = [f"{k}: {old.get(k)!r} -> {new[k]!r}" for k in new if old.get(k) != new[k]]
    if diff:
        raise RuntimeError(
            f"{out} was started with different settings ({'; '.join(diff)}); use another --out directory or delete it"
        )


def _keep_metrics(path: Path, step: int) -> None:
    """Drop metric lines after `step` (steps a resumed run is about to repeat)."""
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        try:
            if json.loads(line)["step"] <= step:
                kept.append(line)
        except (ValueError, KeyError):  # a half-written last line from a crash
            continue
    path.write_text("".join(line + "\n" for line in kept))


def _peak_gib(device: Any) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / 2**30


def _free(device: Any) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -- the run ------------------------------------------------------------------------------


def train(cfg: TrainConfig) -> dict[str, Any]:
    """Run (or resume) the fine-tune described by `cfg`; returns the final status (also in status.json)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(cfg.out_dir)
    data_dir = Path(cfg.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_path, best_path = out / "last.pt", out / "best.pt"
    status_path, metrics_path, model_dir = out / "status.json", out / "metrics.jsonl", out / "model"

    files = {name: data_dir / f"{name}.jsonl" for name in ("train", *DEV_SETS)}
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"{path} not found; build the data first with `davout train build-data`")
    fingerprint = _fingerprint(cfg, {name: _sha256(path) for name, path in files.items()})
    config_path = out / "config.json"
    old = json.loads(config_path.read_text()).get("fingerprint") if config_path.exists() else None
    if last_path.exists():
        if old is None:
            raise RuntimeError(f"{out} holds a checkpoint but no config.json; use another --out directory or delete it")
        _check_resume(old, fingerprint, out)
    prior = json.loads(status_path.read_text()) if status_path.exists() else {}
    if prior.get("state") == "aborted" and old == fingerprint:  # the same settings would abort the same way
        _say(f"{out} holds an aborted run ({prior.get('reason')}); use another --out directory or delete it")
        return prior
    if not last_path.exists():
        # nothing to resume: leftovers of a run that never reached its first checkpoint are stale
        for stale in (best_path, metrics_path, status_path):
            stale.unlink(missing_ok=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _pick_device(torch, cfg.device)

    try:
        tok = AutoTokenizer.from_pretrained(cfg.base, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(cfg.base, dtype=torch.float32, local_files_only=True)
    except OSError as e:
        raise RuntimeError(f"could not load {cfg.base!r} from local files; run `davout download` first") from e
    model = model.to(device)
    if cfg.grad_checkpointing:
        model.gradient_checkpointing_enable()
    # Tokenisation only (`_encode`). Never `read` on this backend: its head copy goes stale as the model trains.
    encoder = HrmBackend(model=model, tokenizer=tok, max_tokens=cfg.max_tokens)
    letter_ids = torch.tensor(encoder.letter_ids, device=device)
    pad_id = encoder._pad_id

    t0 = time.perf_counter()
    sets: dict[str, list[Item]] = {}
    dropped: dict[str, int] = {}
    for name, path in files.items():
        sets[name], gone = tokenize(read_rows(path), encoder)
        dropped[name] = len(gone)
        if not sets[name]:
            raise ValueError(f"{path}: every row exceeds max_tokens={cfg.max_tokens}")
    train_items = sets.pop("train")
    epoch_steps = len(train_items) // cfg.batch
    total = cfg.steps if cfg.steps is not None else epoch_steps
    if total < 1 or total > epoch_steps:
        raise ValueError(
            f"{total} steps of {cfg.batch} need {total * cfg.batch} examples; "
            f"{files['train']} has {len(train_items)} usable ones (one epoch, no repeats)"
        )
    stop_at = min(total, cfg.max_steps) if cfg.max_steps is not None else total
    _say(
        f"{len(train_items)} train rows ({dropped['train']} dropped as too long), "
        + ", ".join(f"{len(sets[n])} {n}" for n in DEV_SETS)
        + f"; tokenised in {time.perf_counter() - t0:.1f}s; {total} steps, stopping at {stop_at}; device {device}"
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params, lr=0.0, betas=cfg.betas, weight_decay=cfg.weight_decay, fused=device.type == "cuda"
    )
    rules = StopRules(base_nll=cfg.base_dev_xfer_nll, gate_step=cfg.gate_step, gate_margin=cfg.gate_margin)
    state: dict[str, Any] = {"step": 0, "best": None, "stop": None, "base": None, "padded_tokens": 0, "seconds": 0.0}

    if last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        state = ckpt["trainer"]
        saved = state.pop("rules")
        rules = StopRules(**{**saved, "base_nll": rules.base_nll if rules.base_nll is not None else saved["base_nll"]})
        del ckpt
        _free(device)
        _keep_metrics(metrics_path, state["step"])
        _say(f"resumed from step {state['step']}")
    _write_json(
        config_path,
        {
            "config": asdict(cfg),
            "fingerprint": fingerprint,
            "rows": {"train": len(train_items), **{n: len(sets[n]) for n in DEV_SETS}},
            "dropped_too_long": dropped,
            "total_steps": total,
            "trainable_parameters": sum(p.numel() for p in params),
            "L_bp_cycles": list(model.model.L_bp_cycles_padded),  # fixed at construction; left as loaded
        },
    )

    def status(run_state: str, reason: str | None = None, **extra: Any) -> dict[str, Any]:
        body = {
            "state": run_state,
            "reason": reason,
            "step": state["step"],
            "total_steps": total,
            "stop_at": stop_at,
            "aux_weight": cfg.aux_weight,
            "lr": cfg.lr,
            "best": state["best"],
            "base": state["base"],
            "base_dev_xfer_nll": rules.base_nll,
            "nonfinite_steps": rules.nonfinite,
            "padded_tokens": state["padded_tokens"],
            "train_seconds": state["seconds"],
            "peak_memory_gib": _peak_gib(device),
            **extra,
        }
        _write_json(status_path, body)
        return body

    def run_eval(step: int) -> dict[str, Any]:
        t = time.perf_counter()
        record = {"kind": "eval", "step": step}
        for name in DEV_SETS:
            record[name] = evaluate(model, sets[name], letter_ids, pad_id, cfg.max_batch_tokens)
        record["seconds"] = time.perf_counter() - t
        log.write(json.dumps(record, allow_nan=False) + "\n")
        log.flush()
        _say(
            f"eval step {step}: "
            + "; ".join(
                f"{name} " + " ".join(f"{c} nll {m['all']['nll']:.4f} acc {m['all']['accuracy']:.3f}" for c, m in record[name].items())
                for name in DEV_SETS
            )
            + f" ({record['seconds']:.0f}s)"
        )
        return record

    aborted: str | None = None
    with metrics_path.open("a", encoding="utf-8") as log:
        if state["step"] == 0 and cfg.eval_at_start and state["base"] is None:
            record = run_eval(0)
            state["base"] = {name: _nlls(record[name]) for name in DEV_SETS}
            if rules.base_nll is None:
                rules.base_nll = state["base"][BEST_SET]["final"]
            rules.on_eval(0, state["base"][BEST_SET]["final"])
        status("running")

        batches = list(chunks(train_items, cfg.batch))
        speeds: list[float] = []
        while state["step"] < stop_at and state["stop"] is None and aborted is None:
            step = state["step"] + 1
            lr = schedule(step, cfg.lr, cfg.warmup, total, cfg.lr_min_ratio)
            for group in opt.param_groups:
                group["lr"] = lr
            t = time.perf_counter()
            info = train_step(model, opt, params, batches[step - 1], cfg, letter_ids, pad_id)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - t
            state["step"] = step
            state["padded_tokens"] += info["padded_tokens"]
            state["seconds"] += seconds
            speeds.append(info["padded_tokens"] / seconds)
            line = {
                "kind": "train", "step": step, "lr": lr, **info,
                "seconds": seconds, "tok_s": speeds[-1], "peak_memory_gib": _peak_gib(device),
            }  # fmt: skip
            log.write(json.dumps(line, allow_nan=False) + "\n")
            log.flush()
            if not info["finite"]:
                _say(f"step {step}: non-finite gradient, step skipped")
            if step % 10 == 0 or step == stop_at:
                mem = "" if line["peak_memory_gib"] is None else f" peak {line['peak_memory_gib']:.1f} GiB"
                shown = " ".join(
                    f"{label} {'nan' if info[key] is None else format(info[key], '.4f')}"
                    for label, key in (("loss", "loss"), ("final", "loss_final"), ("cycle1", "loss_cycle1"), ("gnorm", "grad_norm"))
                )
                _say(f"step {step}/{total} {shown} lr {lr:.2e} {np.median(speeds[-10:]):.0f} tok/s{mem}")
            aborted = rules.on_step(step, info["loss"], info["finite"])
            if aborted is not None:
                break

            if step in cfg.eval_steps or step == stop_at:
                try:
                    record = run_eval(step)
                except FloatingPointError as e:
                    aborted = f"{e} at step {step}"
                    break
                nll = record[BEST_SET]["final"]["all"]["nll"]
                if state["best"] is None or nll < state["best"]["dev_xfer_nll"]:
                    state["best"] = {
                        "step": step,
                        "dev_xfer_nll": nll,
                        "nll": {name: _nlls(record[name]) for name in DEV_SETS},
                    }
                    _save(best_path, {"model": model.state_dict(), "step": step})
                aborted, early = rules.on_eval(step, nll)
                if early is not None and cfg.early_stop:
                    state["stop"] = early
                # fp32 weights + optimizer moments + trainer state, so a resumed run continues exactly
                _save(
                    last_path,
                    {"model": model.state_dict(), "optimizer": opt.state_dict(), "trainer": {**state, "rules": asdict(rules)}},
                )
                if aborted is None:
                    status("running")

    if aborted is not None:
        _say(f"ABORTED at step {state['step']}: {aborted}")
        return status("aborted", aborted)
    if state["best"] is None:
        raise RuntimeError("no evaluation ran, so there is no checkpoint to export")

    # Export the best weights in bf16, then check them through the serving path.
    tok_s = state["padded_tokens"] / state["seconds"] if state["seconds"] else None
    peak = _peak_gib(device)
    opt = params = encoder = None  # noqa: F841 - release the optimizer moments before reloading weights
    _free(device)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    if best["step"] != state["best"]["step"]:
        raise RuntimeError(f"{best_path} holds step {best['step']}, expected step {state['best']['step']}")
    model.load_state_dict(best["model"])
    del best
    model = model.to(torch.bfloat16)  # in place: the fp32 master lives on only in best.pt / last.pt
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)
    model = None
    _free(device)
    gate = sanity_gate(model_dir, sets, state["best"]["nll"], cfg)
    _say(
        f"sanity gate {'PASSED' if gate['pass'] else 'FAILED'}: max final-cycle dev NLL difference "
        f"{gate['max_final_diff']:.4f} (tolerance {cfg.sanity_tol})"
    )
    done = "early_stopped" if state["stop"] is not None else "complete"
    result = status(
        done, state["stop"], model_dir=str(model_dir), sanity_gate=gate, tok_s=tok_s, peak_memory_gib=peak
    )
    _say(f"{done} at step {state['step']}; best step {state['best']['step']} -> {model_dir}")
    return result


def sanity_gate(
    model_dir: Path, sets: dict[str, list[Item]], in_loop: dict[str, dict[str, float]], cfg: TrainConfig
) -> dict[str, Any]:
    """Reload the saved model through a fresh `HrmBackend` and compare dev NLL with the in-loop values.

    Passes when the final-cycle NLL of every dev set is within `cfg.sanity_tol`; the
    earlier cycles are reported but do not gate.
    """
    backend = HrmBackend(
        str(model_dir), device=cfg.device, max_tokens=cfg.max_tokens, batch_size=32, max_batch_tokens=cfg.max_batch_tokens
    )
    served = {name: _nlls(served_metrics(backend, sets[name])) for name in DEV_SETS}
    diff = {name: {c: abs(served[name][c] - in_loop[name][c]) for c in served[name]} for name in DEV_SETS}
    worst = max(diff[name]["final"] for name in DEV_SETS)
    info = backend.info()
    del backend
    _free(_pick_device(torch, cfg.device))
    return {
        "pass": bool(worst <= cfg.sanity_tol),
        "tolerance": cfg.sanity_tol,
        "max_final_diff": worst,
        "in_loop_nll": in_loop,
        "served_nll": served,
        "abs_diff": diff,
        "served_dtype": info["dtype"],
        "served_device": info["device"],
    }
