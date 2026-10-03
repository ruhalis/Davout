"""Run a scorer over a task's calibration and test examples, one decision per call.

Every decision is appended to `<out_dir>/<run_id>/raw.jsonl` as soon as it is
scored, so an interrupted run resumes where it stopped. Nothing here imports
torch until a model-backed scorer is actually built.
"""
from __future__ import annotations

import gc
import json
import math
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from davout import __version__
from davout.bench.tasks import TASKS, Example, TaskData, load_jsonl_task, load_task, to_demo
from davout.schema import NoulQuestion
from davout.scorer_nli import DEFAULT_URL as DEFAULT_OPENJEV_URL

BACKENDS = ("hrm", "openjev")
SHOTS_MODES = ("zero", "generic", "task")
DEFAULT_K = {"zero": 0, "generic": 3, "task": 5}
MIN_TASK_SHOTS = 8  # shots sampled per task, so runs with different k share their first shots
PROGRESS_EVERY = 25

# Config fields that change what a run measures; a resumed run must match them.
_RESULT_FIELDS = (
    "backend", "task", "jsonl", "shots_mode", "k", "prefix_lm",
    "max_tokens", "shortlist", "n_calib", "n_test", "seed", "model",
)  # fmt: skip


@dataclass
class RunConfig:
    """One benchmark run: a backend, a task and a prompting configuration."""

    backend: str  # "hrm" or "openjev"
    task: str  # a registered task name, or the name given to a JSONL file
    shots_mode: str = "generic"  # "zero", "generic" (built-in demos) or "task" (demos from the task)
    k: int | None = None  # number of shots; default 3 for generic, 5 for task
    prefix_lm: bool = True  # bidirectional prefix attention (False = causal)
    device: str = "auto"
    batch_size: int = 8
    max_tokens: int = 4096
    shortlist: int = 10
    openjev_url: str = DEFAULT_OPENJEV_URL
    n_calib: int = 300
    n_test: int = 300
    seed: int = 0
    out_dir: str = "results"
    jsonl: str | None = None  # path of the user's own decisions; replaces the task registry
    model: str | None = None  # HRM model id or checkpoint path; None = the stock HRM-Text-1B
    tag: str | None = None  # names a non-default model in the run id; required with `model`

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.model is not None and self.backend != "hrm":
            raise ValueError("model only applies to the hrm backend")
        if (self.model is None) != (self.tag is None):
            raise ValueError("model and tag go together: a run with a non-default model needs a tag for its run id")
        if self.tag is not None and not re.fullmatch(r"[A-Za-z0-9_.]+", self.tag):
            raise ValueError(f"tag must be letters, digits, '_' or '.', got {self.tag!r}")
        if self.shots_mode not in SHOTS_MODES:
            raise ValueError(f"shots_mode must be one of {SHOTS_MODES}, got {self.shots_mode!r}")
        if self.backend == "openjev":  # an NLI classifier takes no demonstrations
            self.shots_mode, self.k = "zero", 0
        if self.k is None:
            self.k = DEFAULT_K[self.shots_mode]
        if self.shots_mode == "zero":
            self.k = 0
        if self.k < 0 or (self.shots_mode != "zero" and self.k == 0):
            raise ValueError(f"k must be >= 1 for shots_mode {self.shots_mode!r}, got {self.k}")
        if self.n_calib < 0 or self.n_test < 0:
            raise ValueError("n_calib and n_test must be >= 0")

    @property
    def shots_label(self) -> str:
        return "zero" if self.shots_mode == "zero" else f"{self.shots_mode}{self.k}"

    @property
    def attention(self) -> str | None:
        """Attention mode for HRM ("prefix" or "causal"); None where it does not apply."""
        if self.backend != "hrm":
            return None
        return "prefix" if self.prefix_lm else "causal"

    @property
    def run_id(self) -> str:
        """Deterministic id, e.g. `hrm-boolq-task5-prefix-s0` or `openjev-boolq-s0`.

        A run with a non-default model carries its tag: `hrm-ftA-boolq-zero-prefix-s0`.
        """
        task = re.sub(r"[^A-Za-z0-9_.]+", "_", self.task).strip("_") or "task"
        parts = [self.backend] + ([self.tag] if self.tag else []) + [task]
        if self.backend == "hrm":
            parts += [self.shots_label, self.attention or ""]
        return "-".join(parts + [f"s{self.seed}"])


class LazyScorer:
    """Builds the real scorer on first use, so a finished run never loads a model."""

    def __init__(self, factory: Callable[[], Any]) -> None:
        self._factory = factory
        self._scorer: Any = None

    def _get(self) -> Any:
        if self._scorer is None:
            self._scorer = self._factory()
        return self._scorer

    def score(self, state: Any, questions: Sequence[Any], demos: Any = None) -> list[Any]:
        return self._get().score(state, questions, demos=demos)

    def info(self) -> dict[str, Any]:
        return scorer_info(self._get())


def build_backend(cfg: RunConfig) -> Any:
    """The HRM backend for `cfg` (imports torch and loads the weights)."""
    from davout.backends.hrm import HrmBackend

    which = {} if cfg.model is None else {"model_id_or_path": cfg.model}
    return HrmBackend(
        device=cfg.device, batch_size=cfg.batch_size, max_tokens=cfg.max_tokens, prefix_lm=cfg.prefix_lm, **which
    )


def build_scorer(cfg: RunConfig, backend: Any = None) -> Any:
    """The scorer `cfg` asks for; pass `backend` to reuse an already loaded HRM backend."""
    if cfg.backend == "openjev":
        from davout.scorer_nli import NliScorer

        return NliScorer(cfg.openjev_url)
    from davout.scorer import LetterScorer

    if backend is None:
        backend = build_backend(cfg)
    shots = cfg.k if cfg.shots_mode == "generic" else 0
    return LetterScorer(backend, shots=shots, shortlist=cfg.shortlist)


def scorer_info(scorer: Any) -> dict[str, Any]:
    """`scorer.info()`, or its class name and backend info when it has none."""
    if hasattr(scorer, "info"):
        return dict(scorer.info())
    out: dict[str, Any] = {"name": type(scorer).__name__}
    backend = getattr(scorer, "backend", None)
    if backend is not None and hasattr(backend, "info"):
        out["backend"] = backend.info()
    return out


def load_task_data(cfg: RunConfig) -> TaskData:
    """The task `cfg` names: a JSONL file if `cfg.jsonl` is set, else a registered task."""
    n_shots = max(MIN_TASK_SHOTS, cfg.k or 0)
    if cfg.jsonl:
        return load_jsonl_task(cfg.jsonl, cfg.task, cfg.n_calib, cfg.n_test, n_shots, cfg.seed)
    return load_task(cfg.task, cfg.n_calib, cfg.n_test, n_shots, cfg.seed)


def read_raw(path: Path) -> list[dict[str, Any]]:
    """Rows of a raw.jsonl; a half-written last line (from a crash) is dropped from the file."""
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        path.write_bytes(data)
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]


def _expected_logits(example: Example) -> int:
    q = example.question
    return 2 if isinstance(q, NoulQuestion) else len(q.criteria)


def _check_resume(cfg: RunConfig, run_dir: Path) -> None:
    """Refuse to append to a run that was started with different settings."""
    path = run_dir / "config.json"
    if not path.exists() or not (run_dir / "raw.jsonl").exists():
        return
    old = json.loads(path.read_text()).get("config", {})
    new = asdict(cfg)
    old.setdefault("model", None)  # runs recorded before `model` existed used the stock model
    diff = [f"{k}: {old.get(k)!r} -> {new[k]!r}" for k in _RESULT_FIELDS if k in old and old[k] != new[k]]
    if diff:
        raise RuntimeError(
            f"{run_dir} was started with different settings ({'; '.join(diff)}); "
            "use another --out directory or delete it"
        )


def _write_config(cfg: RunConfig, run_dir: Path, task: TaskData, info: dict[str, Any], done: bool) -> None:
    path = run_dir / "config.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = {
        "run_id": cfg.run_id,
        "config": asdict(cfg),
        "task": {
            "name": task.name,
            "kind": task.kind,
            "n_shots": cfg.k if cfg.shots_mode == "task" else 0,
            "n_calib": len(task.calib),
            "n_test": len(task.test),
        },
        "scorer": info or previous.get("scorer", {}),
        "davout_version": __version__,
        "created_utc": previous.get("created_utc", now),
        "updated_utc": now,
        "complete": done,
    }
    path.write_text(json.dumps(body, indent=2, default=str) + "\n")


def run(cfg: RunConfig, scorer: Any = None, task_data: TaskData | None = None) -> Path:
    """Score every calibration and test example of the task; returns the run directory.

    One example per `score` call on purpose: the recorded latency is the cost
    of a single decision. Examples already in raw.jsonl are skipped.
    """
    task = task_data if task_data is not None else load_task_data(cfg)
    demos = None
    if cfg.shots_mode == "task":
        if len(task.shots) < cfg.k:
            raise ValueError(f"task {task.name!r} has {len(task.shots)} shots, fewer than k={cfg.k}")
        demos = [to_demo(e) for e in task.shots[: cfg.k]]

    run_dir = Path(cfg.out_dir) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _check_resume(cfg, run_dir)
    raw_path = run_dir / "raw.jsonl"
    done = {row["id"] for row in read_raw(raw_path)}
    todo = [(split, ex) for split, exs in (("calib", task.calib), ("test", task.test)) for ex in exs]
    pending = [(split, ex) for split, ex in todo if ex.id not in done]
    tag = f"[{cfg.run_id}]"
    if not pending:
        print(f"{tag} nothing to do: all {len(todo)} examples are already scored", file=sys.stderr)
        if not (run_dir / "config.json").exists():
            _write_config(cfg, run_dir, task, {}, True)
        return run_dir

    if scorer is None:
        scorer = build_scorer(cfg)
    _write_config(cfg, run_dir, task, scorer_info(scorer), False)

    def call(ex: Example) -> Any:
        raws = scorer.score(ex.state, [ex.question], demos={0: demos} if demos is not None else None)
        if len(raws) != 1:
            raise RuntimeError(f"scorer returned {len(raws)} results for 1 question")
        return raws[0]

    call(pending[0][1])  # warm-up, not timed and not recorded
    print(f"{tag} {len(pending)} to score ({len(done)} already done) -> {raw_path}", file=sys.stderr)
    total_ms = 0.0
    with raw_path.open("a", encoding="utf-8") as f:
        for n, (split, ex) in enumerate(pending, 1):
            t0 = time.perf_counter()
            raw = call(ex)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            logits = [float(x) for x in raw.logits]
            if len(logits) != _expected_logits(ex) or not all(math.isfinite(x) for x in logits):
                raise RuntimeError(f"scorer returned bad logits for {ex.id}: {logits!r}")
            row = {
                "id": ex.id,
                "split": split,
                "label": ex.label,
                "kind": task.kind,
                "logits": logits,
                "cycle_logits": raw.cycle_logits,
                "prompt_tokens": int(raw.prompt_tokens),
                "truncated": bool(raw.truncated),
                "stage": raw.stage,
                "latency_ms": latency_ms,
            }
            if getattr(raw, "stage1_logits", None) is not None:  # two-stage choice: the shortlist is its top `shortlist`
                row["stage1_logits"] = [float(x) for x in raw.stage1_logits]
            f.write(json.dumps(row, allow_nan=False) + "\n")
            f.flush()
            total_ms += latency_ms
            if n % PROGRESS_EVERY == 0 or n == len(pending):
                print(f"{tag} {n}/{len(pending)} ({split})  {total_ms / n:.0f} ms/decision", file=sys.stderr)
    _write_config(cfg, run_dir, task, scorer_info(scorer), True)
    return run_dir


# -- suite ------------------------------------------------------------------------------

# (shots_mode, k, prefix_lm): the default matrix for the HRM theory questions.
HRM_MATRIX: tuple[tuple[str, int, bool], ...] = (
    ("zero", 0, True),
    ("generic", 3, True),
    ("task", 5, True),
    ("task", 5, False),
)


def suite_configs(base: RunConfig, tasks: Iterable[str] | None = None) -> list[RunConfig]:
    """The default run matrix for `base.backend` on each task (all registered tasks by default).

    HRM: zero-shot, generic-3 and task-5 with prefix attention, plus task-5
    with causal attention. OpenJev: one run per task.
    """
    names = list(tasks) if tasks is not None else list(TASKS)
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s) {unknown}; available: {', '.join(TASKS)}")
    if base.backend == "openjev":
        return [replace(base, task=t, jsonl=None) for t in names]
    return [
        replace(base, task=t, jsonl=None, shots_mode=mode, k=k, prefix_lm=prefix)
        for mode, k, prefix in HRM_MATRIX
        for t in names
    ]


def run_suite(
    cfgs: Sequence[RunConfig],
    build: Callable[[RunConfig], Any] = build_backend,
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Run every config; returns (run directories, [(run_id, error)] for the runs that failed).

    HRM runs with the same model and attention mode share one loaded backend; only one
    backend is kept in memory at a time. A failed run does not stop the rest.
    """
    done: list[Path] = []
    failed: list[tuple[str, str]] = []
    live: dict[str, Any] = {"key": None, "backend": None}

    def backend_for(cfg: RunConfig) -> Any:
        key = (cfg.model, cfg.prefix_lm, cfg.device, cfg.batch_size, cfg.max_tokens)
        if live["key"] != key:
            live["backend"] = None
            gc.collect()
            live["backend"], live["key"] = build(cfg), key
        return live["backend"]

    ordered = sorted(cfgs, key=lambda c: not c.prefix_lm)  # stable: prefix runs first
    for cfg in ordered:
        if cfg.backend == "hrm":
            scorer = LazyScorer(lambda cfg=cfg: build_scorer(cfg, backend=backend_for(cfg)))
        else:
            scorer = LazyScorer(lambda cfg=cfg: build_scorer(cfg))
        try:
            done.append(run(cfg, scorer=scorer))
        except Exception as e:  # noqa: BLE001 - one bad run must not stop the suite
            print(f"[{cfg.run_id}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append((cfg.run_id, f"{type(e).__name__}: {e}"))
    return done, failed
