"""Command-line interface: serve, download, ask, bench, calibrate, train."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _env(name: str, default: Any = None) -> Any:
    value = os.environ.get(name)
    return default if value in (None, "") else value


def _add_model_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default=_env("DAVOUT_DEVICE", "auto"))
    p.add_argument("--model", default=_env("DAVOUT_MODEL"), help="model id or checkpoint path (default: HRM-Text-1B)")
    p.add_argument("--shots", type=int, default=_env("DAVOUT_SHOTS"), help="built-in few-shot examples per prompt (default: 0)")
    p.add_argument("--calibration", default=_env("DAVOUT_CALIBRATION"))
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=8)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="davout", description="Open typed-decision API.")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--host", default=_env("DAVOUT_HOST", "127.0.0.1"))
    serve.add_argument("--port", type=int, default=_env("DAVOUT_PORT", 8766))
    _add_model_flags(serve)
    serve.add_argument("--max-wait-ms", type=int, default=8000)

    sub.add_parser("download", help="download the model snapshot")

    ask = sub.add_parser("ask", help="answer one request from a file or stdin")
    ask.add_argument("file", nargs="?", default=None)
    _add_model_flags(ask)

    bench = sub.add_parser("bench", help="benchmark a backend on labelled decisions")
    bench.set_defaults(bench_help=bench.print_help)
    bsub = bench.add_subparsers(dest="bench_command")
    run = bsub.add_parser("run", help="score one task with one configuration")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--task", help="a registered task (see `davout bench tasks`)")
    source.add_argument("--jsonl", help="your own labelled decisions, one JSON object per line")
    run.add_argument("--name", help="task name for --jsonl (default: the file name)")
    run.add_argument("--shots-mode", choices=("zero", "generic", "task"), default="generic")
    run.add_argument("-k", type=int, default=None, help="number of shots (default: 3 generic, 5 task)")
    run.add_argument("--causal", action="store_true", help="causal attention instead of a bidirectional prefix")
    _add_bench_flags(run)
    suite = bsub.add_parser("suite", help="run the default matrix of configurations on each task")
    suite.add_argument("--tasks", help="comma-separated task names (default: all registered tasks)")
    _add_bench_flags(suite)
    report = bsub.add_parser("report", help="compare runs in a markdown report")
    report.add_argument("run_dirs", nargs="+", metavar="RUN_DIR", help="run directories, or a results directory")
    report.add_argument("--out", default="results/report.md")
    bsub.add_parser("tasks", help="list the registered tasks")

    calibrate = sub.add_parser("calibrate", help="fit a serving calibrator from benchmark runs")
    calibrate.add_argument("run_dirs", nargs="+", metavar="RUN_DIR", help="run directories, or a results directory")
    calibrate.add_argument("--out", required=True, help="where to write the calibration JSON")

    train = sub.add_parser("train", help="fine-tune the model on typed decisions")
    train.set_defaults(train_help=train.print_help)
    tsub = train.add_subparsers(dest="train_command")
    build = tsub.add_parser("build-data", help="build the training and dev sets")
    build.add_argument("--out", required=True, help="data directory to write")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--scale", type=float, default=1.0, help="fraction of the full example counts")
    build.add_argument("--sources", help="comma-separated source names (default: all)")
    build.add_argument("--recipe", default="v1", help="training mix: v1 (96,000 rows, default), cand_v1 (candidate-stage experiment) or cand_mix_v1 (cand_v1 plus 20,000 v1 rows)")
    trun = tsub.add_parser("run", help="fine-tune on a built data directory; resumes an interrupted run")
    trun.add_argument("--data", required=True, help="directory with train.jsonl, dev_in.jsonl, dev_xfer.jsonl")
    trun.add_argument("--out", required=True, help="run directory (checkpoints, metrics, the exported model)")
    trun.add_argument("--aux-weight", type=float, default=0.0, help="weight of the cycle-1 loss (default: 0)")
    trun.add_argument("--lr", type=float, default=1e-5, help="peak learning rate")
    trun.add_argument("--warmup", type=int, default=30, help="linear warmup steps")
    trun.add_argument("--lr-min-ratio", type=float, default=0.1, help="the cosine ends at lr times this (1 = constant after warmup)")
    trun.add_argument("--steps", type=int, default=None, help="optimizer steps of the schedule (default: one epoch)")
    trun.add_argument("--batch", type=int, default=64, help="examples per optimizer step")
    trun.add_argument("--max-batch-tokens", type=int, default=8192, help="rows x longest row per micro-batch")
    trun.add_argument("--base", default=None, help="model id or path to start from (default: HRM-Text-1B)")
    trun.add_argument("--base-dev-xfer-nll", type=float, default=None, help="base dev_xfer NLL for the step-250 gate (default: measured at step 0)")
    trun.add_argument("--max-steps", type=int, default=None, help="stop and export after this many steps (smoke runs)")
    trun.add_argument("--eval-steps", default=None, help="comma-separated evaluation steps (default: 100,250,500,...,1500)")
    trun.add_argument("--seed", type=int, default=0)
    trun.add_argument("--device", default=_env("DAVOUT_DEVICE", "auto"))
    return parser


def _add_bench_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=("hrm", "openjev"), required=True)
    p.add_argument("--n-calib", type=int, default=300)
    p.add_argument("--n-test", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results", help="directory that holds one sub-directory per run")
    p.add_argument("--device", default=_env("DAVOUT_DEVICE", "auto"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--shortlist", type=int, default=10)
    p.add_argument("--openjev-url", default=_env("OPENJEV_URL", "http://127.0.0.1:8765"))
    p.add_argument("--model", default=None, help="HRM model id or checkpoint path (default: HRM-Text-1B); needs --tag")
    p.add_argument("--tag", default=None, help="name of the --model in run ids, e.g. ftA -> hrm-ftA-boolq-zero-prefix-s0")


def _engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "model": args.model,
        "shots": args.shots,
        "calibration": args.calibration,
        "max_tokens": args.max_tokens,
        "batch_size": args.batch_size,
    }


def _cmd_serve(args: argparse.Namespace) -> int:
    from davout import server

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    kwargs = _engine_kwargs(args)

    def factory() -> Any:
        from davout.engine import build_engine

        return build_engine(**kwargs)

    server.serve(
        factory,
        host=args.host,
        port=int(args.port),
        api_key=_env("DAVOUT_API_KEY"),
        max_wait_ms=args.max_wait_ms,
    )
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from davout.backends.hrm import download

    print(download())
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    from davout.schema import ValidationError, parse_request

    try:
        if args.file:
            with open(args.file, encoding="utf-8") as f:
                body = json.load(f)
        else:
            body = json.load(sys.stdin)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    try:
        request = parse_request(body)
    except ValidationError as e:
        print(f"{e.path}: {e.message}", file=sys.stderr)
        return 2

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from davout.engine import build_engine

    from davout.backends.base import PromptTooLongError

    engine = build_engine(**_engine_kwargs(args))
    try:
        result = engine.answer(request)
    except PromptTooLongError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


def _bench_config(args: argparse.Namespace, **overrides: Any) -> Any:
    from davout.bench.runner import RunConfig

    kwargs: dict[str, Any] = {
        "backend": args.backend,
        "n_calib": args.n_calib,
        "n_test": args.n_test,
        "seed": args.seed,
        "out_dir": args.out,
        "device": args.device,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "shortlist": args.shortlist,
        "openjev_url": args.openjev_url,
        "model": args.model,
        "tag": args.tag,
    }
    kwargs.update(overrides)
    return RunConfig(**kwargs)


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from pathlib import Path

    from davout.bench import runner

    if args.name and not args.jsonl:
        print("error: --name only applies to --jsonl", file=sys.stderr)
        return 2
    cfg = _bench_config(
        args,
        task=args.task or args.name or Path(args.jsonl).stem,
        jsonl=args.jsonl,
        shots_mode=args.shots_mode,
        k=args.k,
        prefix_lm=not args.causal,
    )
    print(runner.run(cfg))
    return 0


def _cmd_bench_suite(args: argparse.Namespace) -> int:
    from davout.bench import runner

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()] if args.tasks else None
    base = _bench_config(args, task="suite")
    done, failed = runner.run_suite(runner.suite_configs(base, tasks))
    for run_dir in done:
        print(run_dir)
    for run_id, why in failed:
        print(f"failed: {run_id}: {why}", file=sys.stderr)
    return 1 if failed else 0


def _cmd_bench_report(args: argparse.Namespace) -> int:
    from davout.bench import report

    out = report.write_report(report.find_run_dirs(args.run_dirs), args.out)
    print(out.read_text())
    print(f"wrote {out}", file=sys.stderr)
    return 0


def _cmd_bench_tasks(args: argparse.Namespace) -> int:
    from davout.bench.tasks import TASKS
    from davout.schema import to_text

    for spec in TASKS.values():
        splits = f"shots={spec.shots_split} calib={spec.calib_split} test={spec.test_split}"
        n = 2 if spec.kind == "noul" else len(spec.question.criteria)
        print(f"{spec.name:<10} {spec.kind:<6} {n:>2} classes  {spec.dataset} [{spec.config}]  {splits}")
        print(f"{'':<10} {to_text(spec.question.instructions)}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from davout.bench import report

    cal = report.export_calibrator(report.find_run_dirs(args.run_dirs), args.out)
    print(json.dumps(cal.to_json(), indent=2))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _cmd_train_build(args: argparse.Namespace) -> int:
    from davout.train.data import build

    sources = [s.strip() for s in args.sources.split(",") if s.strip()] if args.sources else None
    result = build(args.out, seed=args.seed, scale=args.scale, sources=sources, recipe=args.recipe)
    if isinstance(result, (dict, list)):
        print(json.dumps(result, indent=2, default=str))
    elif result is not None:
        print(result)
    return 0


def _cmd_train_run(args: argparse.Namespace) -> int:
    from davout.train.loop import TrainConfig, train

    kwargs: dict[str, Any] = {
        "data_dir": args.data,
        "out_dir": args.out,
        "aux_weight": args.aux_weight,
        "lr": args.lr,
        "warmup": args.warmup,
        "lr_min_ratio": args.lr_min_ratio,
        "steps": args.steps,
        "batch": args.batch,
        "max_batch_tokens": args.max_batch_tokens,
        "base_dev_xfer_nll": args.base_dev_xfer_nll,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "device": args.device,
    }
    if args.base:
        kwargs["base"] = args.base
    if args.eval_steps:
        kwargs["eval_steps"] = tuple(int(s) for s in args.eval_steps.split(",") if s.strip())
    result = train(TrainConfig(**kwargs))
    print(json.dumps(result, indent=2, allow_nan=False))
    gate = result.get("sanity_gate") or {}
    return 0 if result.get("state") in ("complete", "early_stopped") and gate.get("pass") else 1


_TRAIN_HANDLERS = {"build-data": _cmd_train_build, "run": _cmd_train_run}

_BENCH_HANDLERS = {
    "run": _cmd_bench_run,
    "suite": _cmd_bench_suite,
    "report": _cmd_bench_report,
    "tasks": _cmd_bench_tasks,
}


def _guarded(handler: Any, args: argparse.Namespace) -> int:
    """Run a bench or train handler; expected failures become a one-line error and exit status 1."""
    try:
        return handler(args)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "bench":
        handler = _BENCH_HANDLERS.get(args.bench_command)
        if handler is None:
            args.bench_help()
            return 2
        return _guarded(handler, args)
    if args.command == "calibrate":
        return _guarded(_cmd_calibrate, args)
    if args.command == "train":
        handler = _TRAIN_HANDLERS.get(args.train_command)
        if handler is None:
            args.train_help()
            return 2
        return _guarded(handler, args)
    handlers = {"serve": _cmd_serve, "download": _cmd_download, "ask": _cmd_ask}
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 2
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
