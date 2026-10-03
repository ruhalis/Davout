# Kev (jaredpalmer/kev): how it works

Researched 2026-10-02. Code read from a shallow clone of `https://github.com/jaredpalmer/kev` at commit `84847f0` (2026-10-01), at `scratchpad/kev/`. Nothing was installed or run.

Labels: **VERIFIED** means read in code or a primary source. **REPORTED** means the project claims it and I did not check it. **INFERRED** means my own reading.

## 1. Identification

- **Davout repo:** a whole-word, case-insensitive grep for "kev" finds nothing. Davout's README describes Jev as `POST /v1/systemone`, with `noul` / `choice` / `score` question types, and names OpenJev as its baseline.
- **GitHub search:**
  - `gh search repos "kev jev"` returned `goodboybeau/system-one-playground`. Its README links "[Kev](https://github.com/jaredpalmer/kev) by Jared Palmer (Apache-2.0)" as one of the "new wave of decision models" next to Laya, Decider and Jev, and describes it as "Qwen3.5 base + LoRA + pointer head" (VERIFIED).
  - `gh search repos kev` (sorted by update) turned up only derivatives besides the main repo: `jtatman/kev-code-verify` ("Jev-style decision models fine-tuned to judge cheap-model code") and `jonpol01/d1a` ("built on Kev"). The other hits are unrelated: CISA KEV vulnerability-catalog tools and "Kevin" repos. `HermannAI/kev-jev-seo` has no description and 2 stars.
- **Target confirmed:** `jaredpalmer/kev`. Its description is "Jev-like family of decision models built on top of Qwen3.5/3.8". Its topics include `jev`. The README says "The API matches TypeSafe's System One… point their Python SDK at your local server" (VERIFIED).
- **Hugging Face:**
  - The project's own weights: `jaredpalmer/kev-0.8b`, `kev-4b`, `kev-9b`, `kev-27b`, plus the legacy `kev-0.5b`, `kev-0.6b` and `kev-8b`.
  - Third-party conversions (derivatives, not separate projects): `ggml-org/Kev-4B-GGUF`, `onnx-community/kev-4b-ONNX`, `RoderickQiu/kev-4b-mlx-8bit`, `FluidInference/kev-0.8b-coreml`, `TheCulliganMan/kev-27b-nf4`, `espetro/kev-*-gguf`.
- **PyPI:** the `kev` package (0.10.0) is an unrelated Redis/S3 ORM. Kev's own open PR #185 adds a warning for this name clash. I did not search npm, HN, Reddit or X; GitHub and Hugging Face were conclusive.

There is one candidate and no real ambiguity.

## 2. Author, license, popularity, activity

All VERIFIED via the GitHub and Hugging Face APIs.

- **Author:** Jared Palmer (@jaredpalmer). The README says it was "Built with Devin".
- **License:** Apache-2.0. The Qwen bases are also Apache-2.0.
- **Popularity:** 8,236 stars and 528 forks.
- **Activity:** created 2026-09-17, last pushed 2026-10-01. There are 213 issues and PRs in about two weeks, which is a very high cadence driven by agents.
- **Hugging Face downloads (likes):** kev-4b 14.1k (97), kev-0.8b 10.5k (33), kev-9b 3.5k (60), kev-27b 624 (22).
- **Releases:**
  - `v0.1.0` "kev-0.5b" (2026-09-17)
  - `kev-family` (2026-09-20, superseded)
  - `kev-1.0` (2026-10-01): the four checkpoints frozen as one family, with SHA-256-checked adapter tarballs. The 27B is on the Hub only.
- **Research log:** `PLAN.md` is a 3,185-line log of "rounds" (round 29 is the latest). It uses preregistered selection rules and runs a locked test once per release.

## 3. Base model, size, weights vs. wrapper

- **These are fine-tuned weights, not a wrapper around an API** (VERIFIED, `kev/model.py`).
  - The backbone is loaded with `AutoModelForCausalLM(...).model`, without the LM head: "we never generate text".
  - A trained `PointerHead` sits on top of the backbone.
- **Checkpoints** (README table, VERIFIED as documented; weights not inspected):

| Model | Base | What is trained | Temperature |
|---|---|---|---|
| Kev-0.8B | Qwen3.5-0.8B-Base | LoRA r16 + pointer head | 2.35 |
| Kev-4B | Qwen3.5-4B-Base | LoRA r16 + pointer head | 2.41 |
| Kev-9B | Qwen3.5-9B-Base | LoRA r16 + pointer head | 2.19 |
| Kev-27B | Qwen3.8-27B, post-trained (not a `-Base` checkpoint) | Full fine-tune, 51 GB bf16 | 1.32 |

- **Older generations:** Qwen3-0.6B/4B/8B and Qwen2.5-0.5B.
- **LoRA settings** (VERIFIED, `model.py:265-275`):
  - Rank 16, `lora_alpha=2r`, dropout 0.05.
  - Targets are q/k/v/o and gate/up/down. On hybrid Qwen3.5 bases the Gated DeltaNet projections are added too: `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`.

## 4. Input format

How requests are serialized (VERIFIED, `kev/model.py:9-127` and `kev/api.py`):

- **Delimiters reuse existing Qwen special tokens**, so no new embeddings are trained:
  - `<|fim_prefix|>` → `<state>`
  - `<|fim_middle|>` → `<q>`
  - `<|box_start|>` / `<|box_end|>` → `<opt>` / `</opt>`
  - `<|fim_suffix|>` → `<decide>`
- **Sequence layout:** `<state> state_text` followed, for each question, by `<q> instructions <opt> option1 </opt> <opt> option2 </opt> … <decide>`.
- **JSON flattening:** `render()` in `api.py` turns state, instructions and criteria from JSON objects or arrays into labelled text: `key: value` lines and `- item` lists.
- **Delimiter escaping:** `user_tokens()` rewrites `<|name|>` in user text to `<¦name¦>`, so callers cannot forge delimiter tokens.
- **Question-type mapping** (`api.py:to_record`):
  - `noul` → two options, `["no[: false-desc]", "yes[: true-desc]"]`. The answer is p(yes).
  - `choice` → options written as `"name"` or `"name: description"`.
  - `score` → the ordered level descriptions as options.
  - The question id is never shown to the model.
- **Limits:**
  - Up to 255 options per question.
  - Training uses a state of up to 384 tokens, a branch of up to 1,024 and a packed record of up to 2,048.
  - Serving accepts a state of up to 65,536 tokens plus 8,192 per question. A longer state gets a 422 unless `KEV_TRUNCATE_STATES=1` is set (VERIFIED, `model.py:14-28, 130-142`).

## 5. Decision mechanism

- **Pointer readout, not token logits or generation** (VERIFIED, `model.py:205-223`).
  - `PointerHead` projects the `<decide>` hidden state with `q: Linear(d, 256)` and each option's `</opt>` hidden state with `k: Linear(d, 256)`.
  - Each option's logit is the scaled dot product of the two projections.
  - A softmax over the options gives the probabilities.
  - Because `<decide>` comes last, it attends to every option.
- **Question isolation:**
  - **Attention-only bases (Qwen3):** one packed sequence with a block-causal mask (`branch_mask_batch`). A token may attend to a key only if the key is causal and either in the state or in the token's own question. Position ids restart after the state for every question.
  - **Hybrid Qwen3.5/3.8 bases (every current Kev):** Gated DeltaNet layers are recurrent and ignore masks, so each question runs as its own row: state plus that question. The state's KV/recurrent cache is computed once and reused for every row (`rows_of`, `probs_and_prefix`, `kev/shared_prefix.py`).
  - Parity tests in `tests/test_model.py` check that the packed and row forms agree.
  - Inference passes are capped at a budget of 16,384 tokens (`rows_per_pass`).
- **Option-order sensitivity:**
  - An optional `option_isolation` mode gives exact permutation invariance: each option is its own sub-branch with shared positions. It works only on packed-mask (attention-only) bases, so the released hybrid models do not use it.
  - The README says option order can still change answers (VERIFIED).
- **Supported types:** `noul`, `choice` and `score`, all built on the one pointer primitive. There is no free-form or numeric-regression type (VERIFIED).

## 6. Confidence, calibration, abstention, routing, batching

- **Calibration** is a single temperature T per checkpoint. It divides the pointer logits at eval time only, so the argmax never changes (VERIFIED, `PointerHead.forward`).
  - T is fitted by minimizing micro mean NLL over a 121-point log grid from 0.25 to 4 (VERIFIED, `kev/metrics.py:272-284`).
  - `KEV_TEMPERATURE=1.0` returns the raw probabilities.
  - `kev/calibrate.py` reports what a per-workload temperature refit would do, including an out-of-fold estimate.
- **Confidence formulas** are copied from TypeSafe's reference adapter, `system-one-adapter` 0.2.1 (VERIFIED, `api.py:120-160`):
  - Choice: `(p_max - 1/K)/(1 - 1/K)`.
  - Score: `max(0, 1 - E|level - mode|/D)`.
  - The `score` value itself is the expected level index.
  - Probabilities are rounded to 4 decimals.
- **Abstention:** there is no explicit abstain output.
  - Training augmentation adds "None of the above"-style options, as the correct answer (`p_none` 0.1) or as a distractor (`p_none_distract` 0.12), and also adds distractor options (`p_distract` 0.15) (VERIFIED, `kev/data.py:48-53, 320-340`).
  - "Unknowable" eval records measure over-confidence (REPORTED: Kev-9B is ≥0.9-confident on 0% of them vs. Jev's 9%).
- **Routing:** no model routing. The user routes on thresholds (INFERRED from the README).
- **Batching and caching** (VERIFIED, `kev/serve.py:28-134`):
  - One model thread drains a queue and runs whatever is waiting as one batch (`probs_batch`).
  - CUDA graphs are used on hybrid backbones.
  - An LRU state-prefix cache means a repeated state pays only for its question rows.
- **Precision:** serving runs in bf16. Published numbers come from the fp32 path (REPORTED: within about 0.03–0.05).
- **Dates:** `KEV_DATE_FACTS=1` appends deterministic sentences giving the day difference between date pairs found in the state (VERIFIED, `api.py:62-91`).

## 7. Training data and procedure

- **Loss:** cross-entropy over the option logits, or soft-target CE when a question carries a target (VERIFIED, `train.py:41-61`).
  - Optional terms exist: label smoothing, Brier, focal, an ordinal RPS term for score (`--ord_w`), a symmetric permutation-KL across option orders (`--perm_kl`), and a KL anchor toward the frozen base's zero-shot distribution.
  - The README says the released models don't use `perm_kl` or `ord_w` (REPORTED).
- **Augmentation:** options are always permuted, plus the none-of-the-above and distractor insertions above (VERIFIED, `data.py:augment`).
- **Base set `decision-v7`** (REPORTED, README):
  - 10,000 examples from ten public datasets. The code's training sources include banking77, boolq, agnews, mnli, sst5 and yelp (VERIFIED, `data.py:157`).
  - 896 generated policy examples.
  - 1,680 examples from 60 generated rule structures.
- **0.8B / 4B / 9B recipe** (REPORTED):
  - 2 epochs, LoRA r16, lr 1e-4 (0.8B) or 5e-5 (4B and 9B).
  - Then follow-up fine-tunes with `--init_from`: day-count and evidence-removed cases, documents, and skill data, with states up to 7,552 tokens.
- **27B recipe** (REPORTED):
  - Full fine-tune for one epoch on 8×H200 at lr 2e-6.
  - Corpus of 145,840 records with states up to 32k, including licensed task families and generated long-document, tool-routing, agent-log and guardrail data.
  - The result is weight-averaged 0.85/0.15 with the earlier LoRA 27B.
- **Distillation:** the README says "No Jev outputs were used for training" (REPORTED). Jev is called only for evaluation through Vercel AI Gateway (`kev/jev.py`, VERIFIED).
- **User fine-tuning:** JSONL in the API shape plus a `label` per question, run with `kev.train --data … --init_from jaredpalmer/kev-4b`. The `kev-finetune` agent skill runs this on Modal.

## 8. API surface vs. Jev

All VERIFIED from `kev/api.py` and the README.

- **Request shape:** `POST /v1/systemone` with `{state, model, questions:{id:{type, instructions, criteria}}}`. This mirrors Jev.
- **Response shape:** `answers`, `usage`, `latency_ms` and an `x-typesafe-request-id` header.
- **Auth:** optional bearer token via `KEV_API_KEY`.
- **Compatibility:** the README says the official `typesafe_sdk` (`TypeSafeClient(base_url=…)`) works unchanged, and `tests/test_api.py` runs TypeSafe's example requests through it (REPORTED).
- **Extensions not in Jev:** `/v1/systemone/permute`, `/v1/systemone/separate`, and `/v1/models` with checkpoint details.
- **`usage.output_tokens`** counts the serialized answer tokens, not generated tokens.

## 9. Reported benchmarks

All REPORTED from the README and `docs/claims.json`, which CI checks against committed reports.

**How they were measured:**

- **Frozen suites:**
  - `decision-v7`: trained sources.
  - `transfer-v4`: 764 new-source records (QNLI, SciQ, PAWS, MMLU, Emotion, TweetEval, held-out policies and rules).
  - `transfer-v9`: transfer-v4 plus MMLU-Pro, buried records and unknowable records.
  - `breadth-v1`: 14 held-out datasets scored with the community Decision Index.
- **Procedure:** dev/test splits, with the test read once per released model. Evaluation is in fp32. Jev was run only on the dev sets.

**Accuracy on new sources** (dev / test):

| Model | New-source accuracy |
|---|---|
| Kev-0.8B | 0.648 / 0.697 |
| Kev-4B | 0.817 / 0.838 |
| Kev-9B | 0.820 / 0.852 |
| Kev-27B | 0.851 / 0.889 |
| Jev | 0.857 (dev only) |

**Other results:**

- **Decision Index on breadth-v1 test:** Kev-0.8B 23.3, 4B 38.0, 9B 41.0, 27B 52.3; Jev 54.0.
- **Coverage at a 5% error budget, out of domain:** Kev 4B/9B/27B 0.52–0.69, Kev-0.8B 0.14, Jev 0.70.
- **Confident errors** (wrong with p ≥ 0.9): Kev-9B 2.4%, Jev 3.7%.
- **Knowledge:** MMLU-Pro 27B 0.675 vs. Jev 0.840. MMLU 9B 0.73 vs. Jev 0.90.
- **Latency:** Kev-4B answers 6 questions in 18.1 ms on an H100 (about 101 req/s) and 41.5 ms on an L40S. On an M5 with MLX it takes 721 ms for a new state and 136 ms when the state is cached.
- **Third-party playground** (system-one-playground, M1 Max): Kev-4B is 76/90/86% on its three sets, and Kev-0.8B answers in 40–125 ms.
- **Removed suites:** `typesafe-v1` (TypeSafe's public evals), WANLI and scienthoon were dropped as "unsound as gates".

## 10. Differences from Jev, and weaknesses

**Differences:**

- **Basis of the design:** Kev's architecture follows a third party's reverse-engineering, Archer Hume's "Jev's Architecture Unmasked" (https://archerhume.com/posts/jevs-architecture-unmasked). That post itself labels much of it inference: shared-state encoding, isolated question branches, and a pointer- or final-position head.
- **Training objective (INFERRED):** Jev reportedly trains with RLCD ("RL for Calibrated Decisions"). Kev uses supervised CE plus post-hoc temperature scaling.
- **Base model (INFERRED):** Jev is presumably frontier-scale and likely MoE. Kev uses small open Qwen bases.
- **Hybrid bases (VERIFIED):** Kev cannot share one packed pass across questions on its Qwen3.5/3.8 bases. It emulates "encode the state once" with a reused prefix cache and per-question rows.
- **Context:** Kev serves states up to 64k tokens, against Jev's 32k as cited in the code comment at `model.py:15`. But it validates only 8k for the small models.

**Known weaknesses** (REPORTED, `docs/releases/kev-1.0.md` "Known limitations"):

- Large gaps out of domain for the small sizes: 13–31 Decision Index points below Jev.
- A single temperature cannot reorder confidences, so Kev automates fewer decisions than Jev at a 5% error budget.
- Knowledge is bounded by the base model.
- Date arithmetic is weak (deadline-policy accuracy 0.35/0.65/0.725 for 0.8B/4B/9B vs. Jev's 0.95).
- Kev-0.8B scores below chance on When2Call tool routing.
- Kev-27B v2 is overconfident on long contracts (CUAD ECE 0.053).
- Option-order sensitivity remains.
- The v2 checkpoints were selected with earlier dev reads known, so their test margins are optimistic.

**Open issues and PRs:**

- #170: SIGSEGV on ROCm with no `--device` flag.
- #207: Windows portability.
- #184: quantized 27B serving.
- #166: MLX prefix reuse.

## Key URLs

- https://github.com/jaredpalmer/kev
- https://github.com/jaredpalmer/kev/releases/tag/kev-1.0
- https://huggingface.co/collections/jaredpalmer/kev-6aad9d0ea49f2589665e07cd
- https://huggingface.co/jaredpalmer/kev-4b
- https://huggingface.co/spaces/jaredpalmer/kev
- https://huggingface.co/datasets/jaredpalmer/kev-suites
- https://archerhume.com/posts/jevs-architecture-unmasked
- https://github.com/goodboybeau/system-one-playground (independent comparison)
