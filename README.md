# Davout

A Jev-style typed-decision system built on Sapient's
[HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B), plus a benchmark
harness to find out how well that works in practice.

You send a `state` and a set of typed questions. Each question gets one forward
pass, the next-token logits over the answer letters are read, and you get back
typed answers with probabilities. No text is generated.

The request and response follow TypeSafe's Jev API (`POST /v1/systemone`), so
code written against Jev ports with a URL change. It is a functional analog,
not a copy: TypeSafe has not published Jev's architecture or training method.
The background and the theory this project tests are in
[HRM vs. Iterative Transformer for a Jev-Style Decision Model](HRM%20vs.%20Iterative%20Transformer%20for%20a%20Jev-Style%20Decision%20Model.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and about 2.4 GB of disk for the weights.

```bash
uv sync
```

```bash
uv run davout download
```

```bash
uv run davout ask examples/quickstart.json
```

```bash
uv run davout serve
```

`serve` listens on `http://127.0.0.1:8766` and never downloads anything.

## API

`POST /v1/systemone`

```json
{
  "state": "Hi, I've been trying to connect my Stripe account for 3 days and the integration keeps failing. I'm losing sales. Please help ASAP.",
  "model": "davout-hrm-text-1b",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this",
      "criteria": {
        "billing": "Payment or subscription issues",
        "technical": "Bugs or integration problems",
        "sales": "Pricing or account questions"
      }
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated the customer appears",
      "criteria": ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"]
    },
    "is_urgent": {
      "type": "noul",
      "instructions": "The message conveys urgency or time-sensitivity"
    }
  }
}
```

| Type | `criteria` | Answer |
| --- | --- | --- |
| `choice` | map of option name to description (or `null`), 2 to 255 options | `choice`, `confidence`, `probabilities` |
| `score` | ordered list of 2 to 10 level descriptions, low to high | `score` (expected level, 0-based), `confidence`, `legend`, `probabilities` |
| `noul` | optional `{"true": ..., "false": ...}` | `noul`, the probability that the statement holds |

- `state`, `instructions` and descriptions may be strings, objects or arrays.
- `confidence` is `(n * peak - 1) / (n - 1)`, which reproduces the examples in Jev's docs.
- The response also carries `usage.input_tokens` (output is always 0) and `timing.total_ms`.
- Errors are JSON `{"error": ...}`: 400 bad JSON, 422 invalid request (with `path`), 503 loading or busy.
- `GET /health` and `GET /v1/models` report status and the loaded model.

Server flags: `--port`, `--device`, `--shots`, `--calibration`, `--max-tokens`,
`--batch-size`; or `DAVOUT_PORT`, `DAVOUT_DEVICE`, `DAVOUT_SHOTS`,
`DAVOUT_CALIBRATION`, `DAVOUT_API_KEY` (bearer token).

## How it works

1. **Prompt.** Each question becomes one prompt in the multiple-choice layout
   HRM-Text was trained on, wrapped in its `direct` condition:
   `<|im_start|><|object_ref_start|>{state}\n\nQuestion: …\nA. …\nB. …\nAnswer:<|im_end|>`.
   Score levels and Noul's Yes/No are presented as options too.
2. **Readout.** The whole prompt is the bidirectional prefix of the PrefixLM.
   The logits at `<|im_end|>` over the letter tokens `A`, `B`, … are the raw
   scores for the options.
3. **Few-shot priming.** The model card says zero-shot is noticeably weaker, so
   three built-in generic examples are prepended by default (`--shots 0` turns
   them off). They triple the prompt length.
4. **Large Choice sets.** HRM has 26 single-token letters. A Choice with more
   options runs in two stages, as Jev describes for itself: every option is
   judged independently with a Yes/No prompt, then the 10 best go through a
   normal letter readout.
5. **Calibration.** Raw logits pass through a `Calibrator`: one temperature
   each for Choice, two-stage Choice and Score, and Platt scaling for Noul.
   Without a calibration file it is the identity.

Questions are isolated: each one sees only the state and its own text.

## Benchmark

The harness measures accuracy, calibration (ECE, NLL, Brier) and latency on
labelled decisions, before and after calibration, and answers three questions
the report raises about HRM:

- Does the second H cycle matter? Logits are captured after every H cycle in
  the same forward pass, so early exit is scored for free.
- Does few-shot priming matter? Runs compare zero-shot, generic 3-shot and
  task 5-shot.
- Does bidirectional prefix attention matter? `--causal` reruns with plain
  causal attention.

The baseline arm is [OpenJev](https://huggingface.co/AlexWortega/openjev)
(Qwen3.5-0.8B NLI) through its sidecar in `~/projects/forum`, fed the same
states and questions.

```bash
uv sync --extra eval
```

```bash
uv run --extra eval davout bench suite --backend hrm --out results
```

```bash
uv run --extra eval davout bench suite --backend openjev --out results
```

```bash
uv run davout bench report results --out results/report.md
```

```bash
uv run davout calibrate results/hrm-*-generic3-prefix-s0 --out calibration.json
```

- Start the baseline's sidecar first with `npm run openjev` in `~/projects/forum`.
- Tasks: `boolq`, `sms_spam`, `sst2` (Noul); `ag_news`, `banking77` (Choice);
  `yelp` (Score). `davout bench tasks` lists them. Datasets download from
  Hugging Face on first use; Yelp is about 320 MB.
- `banking77` has 77 options, so every decision takes 78 forward passes. Use
  `--tasks` to leave it out or `--n-calib 100 --n-test 100` to shrink a run.
- Runs resume: raw logits are appended to `results/<run>/raw.jsonl`, and
  reports are recomputed from those files without rerunning the model.
- Your own decisions: one JSON object per line,
  `{"state": ..., "question": {"type": ..., "instructions": ..., "criteria": ...}, "label": ...}`,
  then `davout bench run --backend hrm --jsonl my.jsonl --shots-mode task -k 5`.

## First measurements

Measured on an Apple M4 with 16 GB (bf16 on MPS), uncalibrated, 1 October 2026.
These are single requests, not a benchmark.

| Request | Input tokens | Time |
| --- | --- | --- |
| Model load | | about 7 s |
| Quickstart (3 questions), 3 shots | 766 | 2.4 s |
| Quickstart (3 questions), `--shots 0` | 220 | 0.76 s |
| One Noul on a short message, 3 shots / 0 shots | | 0.7 s / 0.2 s |
| 30-option Choice (two-stage), 3 shots | 8,159 | 22.6 s |

- **Quickstart answers match a human reading:** technical (0.98),
  "Frustrated but civil" (0.74), urgent (0.86).
- **Jev's published Noul example** ("Is the customer asking for a human
  agent?", six messages): rank correlation with Jev's values is 0.60 at
  3 shots and 0.37 at zero shots. Davout's range is compressed (highest 0.65
  against Jev's 0.99) and it rates "Thanks, that fixed it!" at 0.35 against
  Jev's 0.02.
- **Noul answers move a lot with the built-in examples.** One test question
  went from 0.77 at zero shots to 0.07 at three. The benchmark's shot
  comparison exists to settle which default is right.
- **The first H cycle carries little signal.** Its letter logits are nearly
  flat, so early exit after one cycle looks unpromising before any numbers.
- **Batching gives almost no speedup on MPS**, so questions in one request run
  at roughly sequential cost on this machine.

## Differences from Jev

- **No shared state encoding.** Jev reads the state once for all questions.
  HRM-Text's prompt attention is bidirectional, so nothing can be cached and
  every question re-encodes the state.
- **No calibration training yet.** Jev is trained for calibrated outputs.
  Davout calibrates after the fact; fine-tuning with a proper scoring rule is
  the next step once the benchmark gives a baseline.
- **Score levels see each other.** Jev judges each level independently; here
  the levels are options of one prompt.
- **Limits.** 4,096 tokens per prompt (the state is cut from the left to fit),
  English only, and no code in the training data.

## Layout

| Path | Contents |
| --- | --- |
| `src/davout/schema.py`, `answers.py` | Request validation and Jev-shaped answers |
| `src/davout/prompts.py`, `scorer.py` | Prompt layout, letter readout, two-stage Choice |
| `src/davout/backends/hrm.py` | HRM-Text-1B loading, batching, per-cycle logits |
| `src/davout/calibrate.py`, `metrics.py` | Temperature and Platt scaling; ECE, NLL, Brier, AUROC |
| `src/davout/engine.py`, `server.py`, `cli.py` | Request handling, HTTP server, `davout` command |
| `src/davout/scorer_nli.py` | OpenJev baseline arm |
| `src/davout/bench/` | Tasks, runner, report |

Tests: `uv run pytest -q`.
