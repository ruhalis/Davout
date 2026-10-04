# Davout

Davout is an open, independent reimplementation of the idea behind Jev,
TypeSafe AI's closed "System One" typed-decision model. Jev takes a `state`
(text or JSON) and a set of typed questions (yes/no probability, a choice
among up to 255 options, or a 2-to-10-level score) and returns calibrated
probabilities instead of generated text. Davout serves the same request and
response shape (`POST /v1/systemone`) on top of Sapient's
[HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B), and ships a
benchmark harness that compares it with the open
[OpenJev](https://huggingface.co/AlexWortega/openjev) baseline, a fine-tuning
pipeline, and a teacher soft-label distillation pipeline.

**Status: research prototype.** It runs and is tested, but HRM-Text-1B trails
OpenJev on large option sets and is several times slower there; see
[Results](#results) and [Limitations](#limitations). It is a functional analog,
not a copy: TypeSafe has not published Jev's architecture or training method.
The background and the theory this project tests are in
[HRM vs. Iterative Transformer for a Jev-Style Decision Model](docs/design-hrm-vs-iterative-transformer.md).

## Contents

- [How it works](#how-it-works)
- [Results](#results)
- [Quickstart](#quickstart)
- [API](#api)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Repository layout](#repository-layout)
- [Research notes](#research-notes)
- [Differences from Jev](#differences-from-jev)
- [Limitations](#limitations)
- [Licence](#licence)
- [Acknowledgements](#acknowledgements)

## How it works

| Type | `criteria` | Answer |
| --- | --- | --- |
| `noul` | optional `{"true": ..., "false": ...}` | `noul`, the probability that the statement holds |
| `choice` | map of option name to description (or `null`), 2 to 255 options | `choice`, `confidence`, `probabilities` |
| `score` | ordered list of 2 to 10 level descriptions, low to high | `score` (expected level, 0-based), `confidence`, `legend`, `probabilities` |

Each question gets one forward pass, the next-token logits over the answer
letters are read, and typed answers with probabilities come back. No text is
generated.

1. **Prompt** ([`prompts.py`](src/davout/prompts.py)). Each question becomes
   one prompt in the multiple-choice layout HRM-Text was trained on, wrapped in
   its `direct` condition:
   `<|im_start|><|object_ref_start|>{state}\n\nQuestion: …\nA. …\nB. …\nAnswer:<|im_end|>`.
   Score levels and Noul's Yes/No are presented as options too.
2. **Readout** ([`scorer.py`](src/davout/scorer.py),
   [`backends/hrm.py`](src/davout/backends/hrm.py)). The whole prompt is the
   bidirectional prefix of the PrefixLM. The logits at `<|im_end|>` over the
   letter tokens `A`, `B`, … are the raw scores for the options.
3. **Few-shot priming.** Prompts are zero-shot by default. `--shots N` prepends
   N built-in generic examples (up to 4); the model card recommends few-shot,
   but the benchmark found no accuracy gain and three examples triple the
   prompt length.
4. **Large Choice sets, two stages** ([`scorer.py`](src/davout/scorer.py)).
   HRM has 26 single-token letters. A Choice with more options runs in two
   stages, as Jev describes for itself: every option is judged independently
   with a Yes/No prompt, then the 10 best go through a normal letter readout.
5. **Calibration** ([`calibrate.py`](src/davout/calibrate.py)). Raw logits pass
   through a `Calibrator`: one temperature each for Choice, two-stage Choice
   and Score, and Platt scaling for Noul. Without a calibration file it is the
   identity.
6. **Confidence** ([`answers.py`](src/davout/answers.py)). `confidence` is
   `(n * peak - 1) / (n - 1)`, which reproduces the examples in Jev's docs.

Questions are isolated: each one sees only the state and its own text.

## Results

### Zero-shot benchmark against OpenJev

Run on 1 October 2026 on an RTX 5090 (bf16), one seed, 300 calibration and 300
test decisions per run. With 300 test decisions, accuracy differences under
about 0.05 are within noise. The full table is in
[results/report.md](results/report.md).

HRM-Text-1B zero-shot against the OpenJev baseline (0.8B v2s-long checkpoint,
revision `f004f37`), both after calibration:

| Task | Type | HRM accuracy | OpenJev accuracy | HRM ECE | OpenJev ECE | HRM p50 | OpenJev p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BoolQ | Noul | 0.873 | 0.723 | 0.048 | 0.054 | 32 ms | 19 ms |
| SST-2 | Noul | 0.920 | 0.813 | 0.035 | 0.029 | 27 ms | 16 ms |
| SMS spam | Noul | 0.933 | 0.890 | 0.042 | 0.088 | 26 ms | 17 ms |
| AG News | Choice, 4 options | 0.830 | 0.830 | 0.088 | 0.096 | 31 ms | 32 ms |
| Yelp stars | Score, 5 levels | 0.530 | 0.393 | 0.073 | 0.064 | 33 ms | 34 ms |
| Banking77 | Choice, 77 options | 0.350 | 0.730 | 0.073 | 0.155 | 469 ms | 113 ms |

- **HRM is ahead on reading-style judgments** (BoolQ, SST-2, Yelp), slightly
  ahead on SMS spam (AUROC 0.97 against 0.91), level on AG News, and far behind
  on Banking77, where the two-stage path is both slow and inaccurate.
- **HRM is slower.** OpenJev's p50 latency is lower on the Noul tasks and about
  four times lower on Banking77.
- **Question wording decides whether a Noul works.** Asked to judge the bare
  statement "The message is spam.", neither model beat always answering "not
  spam" (AUROC 0.57 for HRM, 0.61 for OpenJev). Asking "Is this text message
  spam?" and describing both outcomes in `criteria` gives the numbers above.
  The wording was chosen on the test messages and then confirmed on 972 fresh
  ones (AUROC 0.96). Describing the "no" outcome helps most.
- **On a private set of message-routing decisions, OpenJev is the better
  model.** Run through that evaluation's own hypotheses and thresholds, HRM
  matched OpenJev on whether a message needs a reply and was clearly worse at
  picking which agent should answer (right first choice in 58% of cases
  against 84%). The hypotheses were written and tuned for OpenJev, and HRM was
  not fine-tuned. That data is not public.
- **Calibration after the fact is enough to reach ECE under 0.1** for HRM on
  every task. Raw HRM output is already close on Noul and Choice (ECE 0.04 to
  0.13) and poor on Score (0.29).
- **Few-shot examples do not help.** Zero-shot, generic 3-shot and task 5-shot
  agree within noise on every task, while the examples cost 1.3 to 5 times the
  latency. The server therefore defaults to zero-shot.
- **The first H cycle is close to useless.** Reading the answer after cycle 1
  gives chance or majority-class accuracy on BoolQ, SST-2, SMS spam and Yelp,
  and 0.57 against 0.83 on AG News, so early exit is not viable without
  training for it. Banking77 records no per-cycle answers.
- **Bidirectional prefix attention is essential.** With plain causal attention,
  accuracy falls to 0.38 on AG News, 0.24 on Yelp and 0.07 on Banking77, so the
  state cannot be encoded once and shared between questions.
- **OpenJev's latency is for its reference kernels**; its optional fast kernels
  were not installed on the benchmark machine.
- **The baseline is probably understated for some uses.** Davout builds its own
  NLI hypotheses for OpenJev ([`scorer_nli.py`](src/davout/scorer_nli.py))
  rather than OpenJev's trained templates; see
  [docs/research/02-openjev.md](docs/research/02-openjev.md) for this and other
  caveats about the comparison.
- Dataset text was used as published, including literal `\n` and broken HTML
  entities in Yelp and AG News.

### Fine-tuning (v1)

Results from 1–2 October 2026, zero-shot on the same 300 test rows as above.
Arm A trains the final answer only; arm B adds the first-cycle loss. Full
tables are in [docs/results/finetune-v1.md](docs/results/finetune-v1.md) and
[finetune/analysis/](finetune/analysis/).

| Task | Base | Arm A | Arm B | OpenJev |
| --- | --- | --- | --- | --- |
| Banking77 | 0.350 | 0.623 | 0.613 | 0.730 |
| Yelp stars | 0.530 | 0.580 | 0.580 | 0.393 |
| SMS spam | 0.933 | 0.963 | 0.957 | 0.890 |
| AG News | 0.830 | 0.820 | 0.810 | 0.830 |
| BoolQ | 0.873 | 0.870 | 0.870 | 0.723 |
| SST-2 | 0.920 | 0.920 | 0.913 | 0.813 |
| Mean raw NLL | 0.953 | 0.674 | 0.707 | 0.806 |

- **Fine-tuning improved the probabilities more than the answers.** Accuracy
  rose clearly only on Banking77 (+0.27) and marginally on SMS spam and Yelp.
  Raw loss fell on BoolQ, AG News, Yelp and Banking77, and Yelp's raw ECE went
  from 0.29 to 0.13. Raw loss on SMS spam rose slightly; calibration removes
  that.
- **Arm A is the one to keep.** Arm B matches it on accuracy but has slightly
  worse raw loss on four tasks.
- **Early exit is still not usable.** The first-cycle loss lifts the first
  cycle from chance to 0.45–0.83 on held-out tasks, but it stays 0.08 to 0.17
  below the full model. Exiting only when the first cycle is confident lets
  44% of decisions stop early for a 0.018 drop in accuracy, a potential 22%
  compute saving that the code does not yet implement.
- **Wording sensitivity is reduced, not cured.** With the bare statement "The
  message is spam." AUROC rose from 0.57 to 0.81, but accuracy is still at the
  majority rate.
- **On the private routing evaluation the fine-tuned model is much better than
  the base and not distinguishable from OpenJev** in utility when each model
  uses thresholds tuned for it. At the router's existing thresholds it is
  worse, it still picks the right first agent less often (61% against 71% on
  test), and it costs about 2.7 times the compute per message.

Limits on these claims:

- The largest gains are on tasks with a near-domain training source: other
  intent sets for Banking77, Amazon reviews for Yelp. SMS spam has none.
- The learning rate was chosen in a pilot that read benchmark test rows.
- One seed, 300 test rows per task, and the two arms were exported at
  different steps (1,250 and 1,000).

### Teacher distillation

Distilling a 4-bit `Qwen/Qwen3.6-27B` teacher's soft labels on synthetic,
non-banking intent decisions raised Banking77 accuracy from 0.614 (arm A) to
0.660 on 1,000 test rows (+0.046, 95% CI [+0.026, +0.066]), mostly by improving
the second stage of the two-stage Choice, while AG News and Yelp accuracy held.
Public hard-label data, larger shortlists and mined hard negatives did not
help. This is still below OpenJev's 0.73 (measured on a different 300-row
split), and a Banking77 decision still costs about 470–520 ms against
OpenJev's 113 ms. Details, null results and caveats:
[docs/results/distillation.md](docs/results/distillation.md). The raw outputs
of those runs are not committed.

### Measurements on a Mac

Measured on an Apple M4 with 16 GB (bf16 on MPS), uncalibrated, 1 October 2026.
These are single requests, not a benchmark, taken when the default was three
examples; "3 shots" below means `--shots 3`.

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
  went from 0.77 at zero shots to 0.07 at three.
- **Batching gives almost no speedup on MPS**, so questions in one request run
  at roughly sequential cost on this machine.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), Python 3.11 to 3.13, and about
2.4 GB of disk for the HRM-Text-1B weights. Runs on CUDA, Apple MPS or CPU.

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

`download` fetches the model snapshot from Hugging Face. `ask` answers one
request from a file (or stdin) and prints the response. `serve` listens on
`http://127.0.0.1:8766` and never downloads anything. Both take `--model` to
load a fine-tuned checkpoint instead of the base model.

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

- `state`, `instructions` and descriptions may be strings, objects or arrays.
- The response also carries `usage.input_tokens` (output is always 0) and `timing.total_ms`.
- Errors are JSON `{"error": ...}`: 400 bad JSON, 422 invalid request (with `path`), 503 loading or busy.
- `GET /health` and `GET /v1/models` report status and the loaded model.

Server flags: `--host`, `--port`, `--device`, `--model`, `--shots`,
`--calibration`, `--max-tokens`, `--batch-size`, `--max-wait-ms`; or the
environment variables `DAVOUT_PORT`, `DAVOUT_DEVICE`, `DAVOUT_SHOTS`,
`DAVOUT_CALIBRATION`, `DAVOUT_API_KEY` (bearer token).

## Reproducing the experiments

All GPU numbers in this repository come from one NVIDIA RTX 5090 (32 GB).
Datasets download from Hugging Face on first use. Checkpoints are not in this
repository.

### Benchmarks

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
uv run davout calibrate results/hrm-*-zero-prefix-s0 --out calibration.json
```

- **The OpenJev baseline** is reached over HTTP through a local OpenJev
  sidecar, which is not part of this repository. Davout sends
  `POST /classify` with `{"text": ["Premise: …\nHypothesis: …", ...]}` and
  expects, per text, the three raw NLI logits (contradiction, entailment,
  neutral) in `embedding` (see [`scorer_nli.py`](src/davout/scorer_nli.py)).
  The default URL is `http://127.0.0.1:8765`; change it with `--openjev-url`
  or `OPENJEV_URL`, and set `OPENJEV_API_KEY` if the sidecar needs a bearer
  token. The committed results used the `AlexWortega/openjev` 0.8B v2s-long
  checkpoint at revision `f004f37`.
- Tasks: `boolq`, `sms_spam`, `sst2` (Noul); `ag_news`, `banking77` (Choice);
  `yelp` (Score). `davout bench tasks` lists them. Yelp is about 320 MB.
- `banking77` has 77 options, so every decision takes 78 forward passes. Use
  `--tasks` to leave it out or `--n-calib 100 --n-test 100` to shrink a run.
  `--shortlist` sets the stage-1 shortlist size (default 10).
- The suite answers three questions about HRM: whether the second H cycle
  matters (logits are captured after every H cycle in the same forward pass,
  so early exit is scored for free), whether few-shot priming matters
  (zero-shot, generic 3-shot and task 5-shot), and whether bidirectional prefix
  attention matters (`--causal` reruns with plain causal attention).
- Runs resume: raw logits are appended to `results/<run>/raw.jsonl`, and
  reports are recomputed from those files without rerunning the model.
- Your own decisions: one JSON object per line,
  `{"state": ..., "question": {"type": ..., "instructions": ..., "criteria": ...}, "label": ...}`,
  then `davout bench run --backend hrm --jsonl my.jsonl --shots-mode task -k 5`.

### Fine-tuning

`davout train` fine-tunes all 1.18B parameters with log loss over the answer
options only, the "calibration training" step of a Jev analog.

```bash
uv run --extra eval davout train build-data --out data/train_v1
```

```bash
uv run --extra eval davout train run --data data/train_v1 --out checkpoints/ftA
```

```bash
uv run --extra eval davout bench run --backend hrm --task boolq --model checkpoints/ftA/model --tag ftA --out results-ftA
```

- **Data:** 96,000 examples from 23 public datasets converted to Noul, Choice
  and Score questions with randomised wording, option order and criteria. The
  six benchmark datasets are excluded, as are banking intents. The build
  manifest is in [finetune/train_v1/manifest.json](finetune/train_v1/manifest.json).
- **Recipe:** one epoch, 1,500 steps of 64 examples, AdamW at 1e-5, fp32
  weights with bf16 autocast and gradient checkpointing. On an RTX 5090 it
  takes about an hour and 24 GB.
- **`--aux-weight 0.5`** adds the same loss on the answer after the first H
  cycle, to train early exit (arm B).
- Training configs and metrics for both arms are in [finetune/](finetune/).

### Teacher distillation

The teacher needs the optional `teacher` extra (bitsandbytes and accelerate;
4-bit loading needs a CUDA GPU). The pipeline writes synthetic intent
decisions, has the teacher label them, and mixes the soft-labelled rows into a
training build. Every step appends to its output and resumes.

```bash
uv sync --extra eval --extra teacher
```

```bash
uv run --extra eval --extra teacher davout train synth-generate --out data/synth --domains v1
```

```bash
uv run --extra eval --extra teacher davout train synth-flag --taxonomies data/synth/taxonomies.jsonl --out data/synth/flags.jsonl
```

```bash
uv run --extra eval davout train synth-decisions --dir data/synth --model checkpoints/ftA/model --flags data/synth/flags.jsonl
```

```bash
uv run --extra teacher davout teacher --jsonl data/synth/decisions.jsonl --out data/synth/teacher.jsonl
```

```bash
uv run davout train soft-rows --out data/teacher_rows --synth-rows data/synth/decisions.jsonl --synth-teacher data/synth/teacher.jsonl
```

```bash
uv run --extra eval davout train build-data --recipe teacher_intent_v1 --extra data/teacher_rows --out data/train_teacher_v1
```

```bash
uv run --extra eval davout train run --data data/train_teacher_v1 --base checkpoints/ftA/model --lr 3e-6 --out checkpoints/ftA-teacher
```

- `synth-generate --domains v2` writes taxonomies for a second, wider list of
  support desks; recipe `teacher_intent_v2` takes the larger combined set.
- `soft-rows` also takes gold-labelled rows (`--gold-rows`, mixed 0.5 gold +
  0.5 teacher when `--gold-teacher` is given) and rows carried over from an
  earlier build (`--carry`).
- `davout train mine-negatives` ranks each utterance's wrong labels with a
  model's stage 1, for the hard-negative recipe `intent_hn_v1`
  (`build-data --recipe intent_hn_v1 --negatives ...`).
- Other recipes: `cand_v1` and `cand_mix_v1` (candidate-stage experiments).
  The default build (`v1`) still reproduces `train_v1`.

### Tests

```bash
uv run pytest -q
```

Tests marked `model` need the real HRM-Text-1B weights.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/davout/schema.py`, `answers.py` | Request validation and Jev-shaped answers |
| `src/davout/prompts.py`, `scorer.py` | Prompt layout, letter readout, two-stage Choice |
| `src/davout/backends/hrm.py` | HRM-Text-1B loading, batching, per-cycle logits |
| `src/davout/calibrate.py`, `metrics.py` | Temperature and Platt scaling; ECE, NLL, Brier, AUROC |
| `src/davout/engine.py`, `server.py`, `cli.py` | Request handling, HTTP server, `davout` command |
| `src/davout/scorer_nli.py` | OpenJev baseline arm (HTTP client for the sidecar) |
| `src/davout/teacher.py` | Teacher scorer (Qwen3.6-27B, 4-bit) for evaluation and soft labels |
| `src/davout/bench/` | Tasks, runner, report |
| `src/davout/train/` | Training-data builder, recipes, synthetic data, hard-negative mining, fine-tuning loop |
| `results/` | Committed zero-shot benchmark runs (HRM and OpenJev) and [report](results/report.md) |
| `results-ftA/`, `results-ftB/` | Benchmark runs of the two fine-tuned arms |
| `finetune/` | Training configs, metrics, data manifest and analysis scripts for the v1 fine-tune |
| `docs/results/` | [Fine-tune v1 report](docs/results/finetune-v1.md), [distillation log](docs/results/distillation.md) |
| `docs/research/` | Research notes on Jev and its clones |
| `docs/design-hrm-vs-iterative-transformer.md` | Design note: why HRM for a Jev-style model |
| `examples/` | Example request |
| `tests/` | Unit tests |

## Research notes

Compiled in early October 2026 from public sources (vendor docs, model cards,
repositories, papers, discussion threads). Benchmark numbers in them are mostly
self-reported by each project and were not reproduced here unless stated.
Each claim is tagged VERIFIED, REPORTED or INFERRED.

- [01 – Jev, the original](docs/research/01-jev-original.md): what TypeSafe AI
  has published about Jev's API, question types, calibration and training.
- [02 – OpenJev](docs/research/02-openjev.md): code-level reading of
  OpenJev, its training data, results, and caveats about Davout's baseline.
- [03 – Kev](docs/research/03-kev.md): code-level reading of Kev, another
  open Jev-style model family.
- [04 – Census of code hosts](docs/research/04-census-code-hosts.md): Jev
  clones found on GitHub, Hugging Face, package registries and other code hosts.
- [05 – Community census](docs/research/05-census-community.md): clones,
  commercial products and research found through community discussion, news
  and papers.
- [06 – Training data](docs/research/06-training-data.md): what each clone was
  trained on, including distillation from Jev.

## Differences from Jev

- **No shared state encoding.** Jev reads the state once for all questions.
  HRM-Text's prompt attention is bidirectional, so nothing can be cached and
  every question re-encodes the state.
- **Calibration training is a short fine-tune, not Jev's method.** Jev is
  trained for calibrated outputs with a method TypeSafe has not described.
  Davout offers log-loss fine-tuning plus calibration after the fact.
- **Score levels see each other.** Jev judges each level independently; here
  the levels are options of one prompt.

## Limitations

- 4,096 tokens per prompt (the state is cut from the left to fit), English
  only, and no code in the training data.
- Large Choice sets are slow: a 77-option decision takes 78 forward passes
  (about 470–520 ms on an RTX 5090), against OpenJev's 113 ms p50.
- HRM trails OpenJev on Banking77 even after fine-tuning and distillation
  (0.660 against 0.73), and on the private routing evaluation.
- Every result is from one seed, mostly with 300 test decisions per task;
  differences under about 0.05 are noise.
- Early exit after the first H cycle is measured but not implemented.
- The OpenJev sidecar used for the baseline is not included, and Davout's NLI
  hypotheses for it are its own.

## Licence

The code in this repository is licensed under the [Apache License 2.0](LICENSE).

Model weights are not distributed here. HRM-Text-1B and the Qwen teacher
weights are subject to their own licences. The v1 training mix includes
`facebook/anli` (CC-BY-NC-4.0) and DailyDialog (CC-BY-NC-SA-4.0), so
checkpoints trained on it inherit non-commercial terms; check the licence of
every dataset in [finetune/train_v1/manifest.json](finetune/train_v1/manifest.json)
before using a trained checkpoint.

## Acknowledgements

- [HRM-Text-1B](https://huggingface.co/sapientinc/HRM-Text-1B) by Sapient
  Intelligence, the base model.
- [OpenJev](https://huggingface.co/AlexWortega/openjev) by AlexWortega, the
  open baseline.
- Jev by [TypeSafe AI](https://typesafe.ai), whose API shape Davout follows.

Davout is an independent project and is not affiliated with or endorsed by
TypeSafe AI, Sapient or the OpenJev authors.
