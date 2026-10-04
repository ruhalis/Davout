# OpenJev (AlexWortega/openjev): how it works, code-level

Researched 2026-10-02. Labels: **VERIFIED** = read in code or a primary file in the repo; **REPORTED** = stated in the model card, RESULTS file, a third-party page, or the Jev vendor's marketing, not checked in code; **INFERRED** = my reasoning from the code.

Paths below are relative to the Hugging Face model repo `AlexWortega/openjev` at commit `a204480` (2026-10-01). The code was read from a local clone without weights.

---

## 1. Which project this is

### The canonical repo
- **Hugging Face model repo:** https://huggingface.co/AlexWortega/openjev. It holds the weights, code, results and model card in one repo. I found no GitHub mirror; the GitHub search API was rate-limited, so I could not check exhaustively. The card links no GitHub repo. **VERIFIED** (HF API, repo tree)
- **Demo Space:** https://huggingface.co/spaces/AlexWortega/openjev. Gradio app serving the 0.8B v2s-long checkpoint, last modified 2026-09-21. **VERIFIED** (HF API) and **REPORTED** (card)
- **Author:** HF user "AlexWortega" (display name "Wortega"), with 169 models, 56 datasets and 244 followers. All 66 commits are by "Wortega"; two community PRs were merged. **VERIFIED** (`git log`, HF API)
- **License:** MIT, from the card metadata. **VERIFIED**
- **Popularity:** 631 likes. The HF API shows 0 downloads, probably because most weights sit in subfolders. Trending score 65. **VERIFIED** (HF API)
- **Dates:** created 2026-09-16; last commit 2026-10-01 (`a204480`, the FinCalc-NLI eval, PR #4). **VERIFIED**
- **Is it the right project?** Yes, it is a Jev clone and not a name collision:
  - The card title is "Qwen3.5 trained as jev model".
  - It benchmarks against "Jev 1.13" on JevBench.
  - `code/serving/decisions_api.py` reimplements the Jev / OpenRouter Decisions request schema (`POST /v1/systemone`).
  - HF discussion #2 asks "Can we use this Open Jev model as an open-source alternative of Jev by TypeSafe Ai?" The author answers "yes exactly".
  - This is also the OpenJev that Davout's benchmark uses as its baseline (see `README.md` and `src/davout/scorer_nli.py`), served through a local OpenJev sidecar.
  
  **VERIFIED**

### Release history (from `git log`) — VERIFIED
| date | version / event |
|---|---|
| 2026-09-16 | v1: `qwen3.5-4b-nli`, a plain AllNLI fine-tune. Card, code, Flappy/Doom demos |
| 2026-09-17 | 35B-A3B latent + MLP heads; WebQL results |
| 2026-09-19 | v2: `qwen3.5-4b-nli-v2` (text + images, large mixture); `qwen3.5-35b-a3b-nli` (LoRA) |
| 2026-09-20 | `qwen3.5-0.8b-nli-v2s-long` (4k ctx, faithfulness/IF/false-premise + long docs); PR #1 shares the prefix cache across hypotheses |
| 2026-09-21 | SGLang serving package (`f004f37`, **the revision pinned by the local OpenJev sidecar used for Davout's baseline**) |
| 2026-09-23 | v5: `qwen3.5-4b-nli-v5` ("typed decisions"), `-2b-nli-v5`, `-0.8b-nli-v5`; RESULTS-v5.md; contamination manifest |
| 2026-09-24 | `qwen3.5-4b-nli-v5-nvfp4` (ModelOpt NVFP4 PTQ + QAD) |
| 2026-09-27 | `qwen3.5-0.8b-nli-v5` **removed** (`a298f27`). Discussion #5 asked why; the author replied "returned", but the folder is not in the tree at HEAD `a204480`. The card still links it, so that link is broken. |
| 2026-09-28 | Image Decisions gateway (`/v1/systemone` with `image_data`) |
| 2026-10-01 | FinCalc-NLI eval (community PR #4) |

There are no tagged releases. Versions exist only as subfolder names.

### Other projects called "OpenJev" (searched; not analysed in depth)
A Hugging Face search for `openjev` / `open-jev` returns about 50 repos. The ones that matter for a clone survey:
- **`openjev/openjev`** (org account, **CC-BY-NC-4.0**, 94 likes, created 2026-09-20). This is a **different project with the same name**. It is a Qwen3.5-based generative model that reads one score per option "at the first output position" in a single pass per question, with up to 52 options. It claims 84.0% vs Jev's 85.4% on its own 10k-question set. It has GGUF, MLX and FP8 variants. **REPORTED** (its card)
- Others: `ZefanCai/Open-Jev-{2B,9B,27B}`, `apus-ailab/APUS-OpenJev-v1-*`, `com-kotobalabs/open-jev-deberta-v3-large`, `IamBusy/OpenJev-0.6B`, `cainai/OpenJev-Qwen3.5-0.8b`, and more. **VERIFIED** (HF search API, names only)

Everything below is about **AlexWortega/openjev**.

---

## 2. Base model and size — fine-tuned weights, not an API wrapper

- **All checkpoints are fully fine-tuned Qwen3.5 backbones with a new 3-way classification head.** The architecture is `Qwen3_5ForSequenceClassification`, with labels `0=contradiction, 1=entailment, 2=neutral` and the template stored in `config.nli_template = "Premise: {premise}\nHypothesis: {hypothesis}"`. **VERIFIED** (`qwen3.5-4b-nli-v5/config.json`, `code/train.py:28-33`)
- **v5 4B backbone:** hidden size 2560 and 32 layers. The layers form a hybrid pattern: 3 `linear_attention` (gated-delta) layers per 1 `full_attention` layer. Max positions are 262144. **VERIFIED** (`config.json` text_config)
- **Pooling:** the hidden state of the last non-pad token (right padding) goes through `model.score`, a linear layer from d to 3. **VERIFIED** (`modeling_openjev.py:65-86`)
- The vision tower is kept. v2 and later were trained with image premises. **VERIFIED** (`train.py` `freeze_vision`, `DataCollatorNLIMM`)

| checkpoint | init | data | max_len | bs×accum | lr | LoRA |
|---|---|---|---|---|---|---|
| `qwen3.5-4b-nli` (v1) | Qwen/Qwen3.5-4B | AllNLI 120k | 256 | 32×1 | 2e-5 | no |
| `qwen3.5-4b-nli-v2` | Qwen/Qwen3.5-4B | mix (200k sampled) | 1024 | 32×1 | 2e-5 | no |
| `qwen3.5-0.8b-nli-v2s-long` | ckpt `0.8b-nli-v2s-jev` | stage3 mix 40k | 4096 | 2×16 | 1e-5 | no |
| `qwen3.5-2b-nli-v5` | ckpt `2b-nli-v4` | nli_v5 mix 200k | 2048 | 8×4 | 1e-5 | no |
| `qwen3.5-4b-nli-v5` | ckpt `4b-nli-v4` | nli_v5 mix (all) | 2048 | 8×2 | 1e-5 | no |
| `qwen3.5-35b-a3b-nli` | Qwen/Qwen3.5-35B-A3B | AllNLI 60k | 256 | 8×4 | 2e-5 | yes (r16) |

All runs: 1 epoch, cosine schedule, 3% warmup, wd 0.01, bf16. **VERIFIED** (`*/train_result.json`, `code/train.py:376-400`)

The intermediate checkpoints v3, v4 and `v2s-jev` are **not published**. So v5 cannot be reproduced exactly from the repo. **VERIFIED** (absent from the tree)

---

## 3. Input format

### Per-pair serialization (what the model sees)
```
Premise: {state window}
Hypothesis: The answer to "{instructions}" is {label}: {criterion text}
```
- The hypothesis template is `HYPOTHESIS` / `TEMPLATE`: `code/openjev_decide.py:31`, `code/serving/decisions_api.py:21`. The same string is `JEV_TEMPLATES[0]` in `code/data_mix.py:748`. A second training template, `'Decision for "{instr}": {label} — {crit}'`, is used at random during training only. **VERIFIED**
- **One pair per option.** The question is never shown with all options together. **VERIFIED**
- Each type maps to options as follows (`decisions_api.py:67-91`). **VERIFIED**
  - **noul** → two hypotheses, label `no` with `criteria.false` and label `yes` with `criteria.true`. With no criteria the defaults are just "no"/"yes".
  - **choice** → one hypothesis per key of `criteria` (the description is used as `crit`), up to 64 options.
  - **score** → one hypothesis per level, labelled `"0"`, `"1"`, … with the level description, up to 64 levels.
  - Template drift: `code/eval_jevbench.py:41` uses `is level {i}: {c}` for score, while the API and `openjev_decide` use `is {i}: {c}`. **VERIFIED**
- **State** is passed as a string; objects and arrays are JSON-dumped. **VERIFIED**
- **Long states** are split into 24,000-character windows with 2,000 characters of overlap. Every (window, option) pair is scored, and P(entail) is **max over windows** before normalisation (`openjev_decide.py:70-86`, `decisions_api.py:60-64,148`). The API rejects states over 110,000 characters (~32k tokens) with HTTP 413. **VERIFIED**
- **Images** (v2 and later, via the gateway): the premise contains `<<IMG>>`. The gateway replaces it with Qwen's `<|vision_start|><|image_pad|>…<|vision_end|>`, and every option shares the image. **VERIFIED** (`code/serving/README.md`, `decisions_server.py` imports `image_inputs`) and **REPORTED** (README)
- The JevBench adapter path (`openjev_decide.OpenJev._rubric`) takes the rubric back out of `instructions` after the marker `"\nAllowed answers and rubric: "` followed by JSON, because the JevBench harness appends the rubric there. **VERIFIED** (`openjev_decide.py:30,58-68`)

---

## 4. Decision mechanism

It is an **NLI cross-encoder with a classification head, scored once per option and normalised over the options.** There is no letter or logit readout over vocabulary tokens, no generation, no grammar, and no constrained decoding. **VERIFIED**

1. Each (window, option-hypothesis) pair is scored: softmax over the 3 classes, keep **P(entailment)** (`modeling_openjev.py:80-86`, `decisions_api.softmax_ent:123-128`).
2. Take the max over windows.
3. Divide by the sum over options. If the sum is 0, use a uniform distribution (`decisions_api.assemble:140-160`). Note that this is **renormalised P(entail)**, not a softmax over option logits. The contradiction and neutral mass is thrown away.
4. Outputs by type:
   - **noul** → `{"noul": p[true]}`.
   - **choice** → `choice = argmax`, plus `probabilities` and `confidence`.
   - **score** → `score = Σ i·p_i`, the **expected level** (a float), plus `probabilities`, `confidence` and `legend`.
5. Supported types: noul, choice and score only. **There is no multi-label, number or free-form type.** **VERIFIED** (`QTYPES`, `decisions_api.py:28`)

**Order invariance** follows from the structure: each option is a separate pass, so option order only enters the final normalisation. The repo measured 0/231 label flips under reversal and shuffle, with a largest Δp of 1.8e-7 in fp32 (`code/order_test.py`, RESULTS-v5.md:72-84). The mechanism is **VERIFIED**; the numbers are **REPORTED**.

**Cost scales with options.** An N-option question over W windows needs N·W forward passes. Mitigations:
- **Shared-prefix batching** (`modeling_openjev.py:88-140`, PR #1). For 3 or more hypotheses over one premise, the common token prefix is prefilled once. Its cache, including Qwen3.5's recurrent linear-attention state, is copied with `reorder_cache` into a batch of suffix continuations. **VERIFIED**
- `decide()` in `openjev_decide.py:84` calls plain `ce.predict(pairs)`, **not** `predict_hypotheses`. So the JevBench adapter path does **not** use the prefix cache; only `rerank` and the local sidecar used for Davout's baseline do. **VERIFIED**
- The SGLang gateway sends pairs in chunks of `CLASSIFY_BS=32` concurrently to SGLang `/classify`, and SGLang does its own radix prefix caching. **VERIFIED** (`decisions_server.py:35,56-80`). The radix caching is **INFERRED** from how SGLang works.
- **There is no two-stage path for large Choice sets**: everything is one pass per option, capped at 64 options in the API. **VERIFIED**

### Optional prompt styles (gateway)
`OPENJEV_PROMPT_STYLE` ∈ {`trained` (default), `unquoted`, `direct_noul`}. `direct_noul` scores a criteria-less noul as a single hypothesis = the bare instruction, with P(true) = P(entail), so neutral mass counts against "true". **VERIFIED** (`code/serving/prompt_templates.py`)

### Other heads (not part of the typed-decision path)
`LatentMLPHead`: a frozen backbone latent feeds an MLP (d→512→1, GELU, dropout 0.1), trained with soft BCE (eps 0.1, positive reweighting) and early stopping on grouped per-question accuracy. Published only for the 35B model, one head per benchmark (`mlp_heads_35b/`). This is per-task supervised and not zero-shot. **VERIFIED** (`modeling_openjev.py:163-261`)

---

## 5. Confidence, calibration, abstention, batching, routing

- **Confidence** = `1 − H(p)/log(n)`, i.e. 1 minus normalised entropy. The code comments: "OpenRouter does not publish its definition; this is an approximation." **VERIFIED** (`decisions_api.py:131-137`). This differs from the formula Davout reverse-engineered from Jev's docs examples, `(n·peak−1)/(n−1)` (Davout README). **INFERRED**: OpenJev's confidence values will not match Jev's.
- **Calibration:** **none fitted.** The probability rule is declared as `normalized_entailment_v1`, "with no calibration fitted on the benchmark examples" (README, 2026-09-28 section). The code has no temperature or Platt step. **VERIFIED** (no calibrator in `decisions_api.py`)
- **Calibration as a training objective:** plain 3-class cross-entropy. There is no RL step and no calibration loss. The only calibration-like signal is the soft labels from teacher-vote distillation (§6). **VERIFIED**
- **Abstention:** none. There is no "none of the above" or abstain output. The API simply 400s on bad input. **VERIFIED**
- **Batching:** pairs are batched (`bs=32`), as is the prefix-cache path. The gateway enforces `MAX_INFLIGHT=64` (429 above that). Limits are 64 questions per request and 64 options per question. **VERIFIED**
- **Determinism:** bit-identical across repeats in fp32 (REPORTED). The forward pass is deterministic, so this is **INFERRED** plausible.
- **Routing:** there is no model router or cascade inside OpenJev. Routing exists only as a use case (for example, using OpenJev as a message router in a downstream application). **VERIFIED** (no such code)
- **API surface of the gateway:**
  - Routes: `POST /api/alpha/decisions`, `/api/v1/systemone` and `/v1/systemone`, all with the same schema; plus `GET /health`.
  - Auth is an optional Bearer `API_KEY`.
  - The response includes `usage.input_tokens`, `output_tokens: 0`, and `cost` from `PRICE_PER_MTOK`.
  
  **VERIFIED** (`decisions_server.py:1-10,137-142`)

---

## 6. Training data and procedure

**Objective:** standard HF `Trainer`, 3-way cross-entropy on (premise, hypothesis, label) rows. Over-long rows are truncated **on the premise side** so the hypothesis survives (`train.py:286-301`). Length-grouped sampling. **VERIFIED**

**Typed decisions are converted to NLI rows** by `jev_rows` (`data_mix.py:769-776`): the gold option's hypothesis gets label *entailment* and up to `n_neg` wrong options get *contradiction*. This is how "typed decision" ability is trained into a 3-class head. **VERIFIED**

**Mixture stages:**
- **v1:** SNLI + MNLI (AllNLI), 120k rows. **VERIFIED** (`train.py:40-49`, train_result)
- **v2** (~1.3M rows per RESULTS-v5; the run sampled 200k). `data_mix.py` subcommands **VERIFIED**; row count **REPORTED**:
  - `text`: SNLI/MNLI/ANLI/WANLI/FEVER-NLI/LingNLI/ConTRoL/SciTail/QNLI/bAbI-NLI plus a "haystack".
  - `images`: a VQAv2 stream turned into claims, including spatial claims.
  - `agentic`/`agentic2`/`agentic_if`/`agentic_distill`: xLAM, AgentTraj-L, AgentInstruct, When2Call, Mind2Web, ToolACE, APIGen-MT, Agent-FLAN, tau-style traces.
  - `longdoc`: GovReport, QASPER.
- **v2s:** adds `faith` (MiniCheck C2D/D2C, RAGTruth train at sentence level), `ifollow` (argilla ifeval-like, labelled by the IFEval checker, with IFEval prompts filtered out) and `bullshit` (FalseQA plus synthetic category-error questions). **VERIFIED** (`data_mix.py:508-745`)
- **`jevfmt`** (typed-decision format, `data_mix.py:779-900`). **VERIFIED**
  - CLINC-150 and Banking77 intents, each item with 6 options (5 random wrong + gold), template "The user's message is about: {o}".
  - xLAM tool selection.
  - Synthetic policy-rule and severity items.
  - SciQ adequacy.
  - **And, by default, the JevBench public items, repeated 8×** (`--jevbench-repeat 8`; skipped only with `--no-jevbench`). The code comment says: "The user asked for these to be trained on". RESULTS-v5 says `--no-jevbench` "used for every v3/v4/v5 build". The build commands are not published, so **that claim is REPORTED, not verifiable**.
- **v4:** adds ~27k rows distilled from a reasoning teacher (`code/distill_teacher.py`). **VERIFIED** (script) / **REPORTED** (row count)
  - `write`: a teacher LLM generates scenarios from 9 family descriptions × 20+ domains. Default model Qwen/Qwen3.5-9B; the card says Qwen3.5-4B.
  - `solve`: k=6 sampled thinking chains per item. Chains that hit the budget are cut off and forced to commit with `</think>\n\nFINAL:` (budget forcing).
  - `rows`: one row per (vote, option), entail if the vote chose it, else contradict. The expectation over votes acts as a **soft label** under ordinary CE. Items whose top-vote share falls below `--min-agree` are dropped.
  - There is also an `AzureGen` backend (Azure OpenAI Responses API, "a reasoning deployment such as gpt-5.5").
  - Plus complex IF formats (`ifcomplex`) and `hardfmt`.
- **v5:** adds 5,998 "computation-heavy items written and solved by gpt-5.5" (temporal, long policy, multi-hop, judge). Teacher self-agreement is 0.921 (**REPORTED**). Alongside that, `code/hard_gen.py` builds the same families procedurally, with labels **computed** by `datetime` arithmetic and rule walking and checked by a `selftest()` (**VERIFIED**). The repo does not make clear which source supplied which rows. **INFERRED**
- **v5 also adds the "panel": TRAIN and TEST splits** of MMLU, ARC-E/C, GSM8K, HellaSwag, WinoGrande, GPQA-diamond, ContractNLI, NLI4CT, ESCI, API-Bank, CLINC-150 test and **Banking77 test**. **VERIFIED** (`qwen3.5-4b-nli-v5/panel_manifest.json`, `data_mix.build_panel`). This contradicts the `BANNED` list (`data_mix.py:40-48`), which still bans MMLU, ARC and others for earlier versions. The panel builder bypasses it, which the card discloses.
- **Build step** (`data_mix.build_final:1616-1690`). **VERIFIED**
  - Optional replay of the stage-1 mixture with per-group quotas.
  - Class balance by downsampling to `balance_ratio ×` the smallest class.
  - Leakage filter against MNLI validation pairs only.
  - A 2k validation split.

**Distillation from Jev outputs:** **none found.** No code calls the TypeSafe or OpenRouter Jev API to label data. The teachers are Qwen3.5 and gpt-5.5. **VERIFIED** (grep of `code/`)

---

## 7. Public API vs Jev's

| aspect | Jev (TypeSafe) | OpenJev (AlexWortega) |
|---|---|---|
| endpoint | `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest` (REPORTED, DataCamp) | `/v1/systemone`, `/api/v1/systemone`, `/api/alpha/decisions` (VERIFIED) |
| request | `state`, `model`, `questions{id: {type, instructions, criteria}}` | same shape (VERIFIED); also `image_data` (base64) |
| types | noul, choice, score (REPORTED) | noul, choice, score (VERIFIED) |
| noul output | probability | `{"type":"noul","noul":p}` (VERIFIED) |
| choice output | choice + confidence + probabilities | same keys; confidence = 1−normalised entropy (VERIFIED) |
| score output | score + confidence + legend + probabilities (per Davout) | expected level (float), confidence, probabilities, legend (VERIFIED) |
| usage | input-token billing ($0.042/M), free output (REPORTED) | `usage.input_tokens`, `output_tokens: 0`, `cost` from env price (VERIFIED) |
| errors | — | `{"error":{"code","message"}}`, 400/413/429 (VERIFIED) |
| Python | — | `OpenJev.from_pretrained(...).decide(state, [{"type","instructions","options"}])`, the JevBench `local_openjev` adapter contract (VERIFIED) |

Overall the wire format mirrors Jev closely. The semantics differ in confidence, calibration, option limits and the lack of a two-stage choice path.

---

## 8. Reported benchmarks vs Jev

All **REPORTED** (README.md, RESULTS-v5.md). The harness code exists (`code/eval_jevbench.py` and others), but I did not run it.

- **JevBench v1.2, 231 public items, using the benchmark's own harness:** 4B v5 scores **0.814** vs **Jev 1.13 0.866**.
  - By tier: easy 1.000/1.000, standard 0.986/0.986, hard **0.622 vs 0.730**.
  - Hard families: adversarial 1.00, routing_hard 1.00, trap 1.00, multi_hop 0.72, probability 0.60, long_policy 0.58, ambiguous 0.57, judge_hard 0.53, tradeoff 0.50, **temporal_numeric 0.27**.
  - Comparison systems on the same items: gemini-3.1-flash-lite 0.87, SemIf (Qwen3.5-4B) 0.81, open-alternative-jev 0.74, system-one-open 0.73, open-jev-deberta-v3-large 0.52. These numbers come from JevBench's own per-task leaderboard file (`eval_jevbench.py:30`).
  - The held-out judge tier was not measured, so no JevBench Score is claimed.
  - Caveat: the distill teacher's own docstring says "the public JevBench items double as the teacher's exam (`--tasks hard.jsonl`)" (`distill_teacher.py:13-14`). The hard-family descriptions it generates from are visibly modelled on JevBench family names. That is a soft form of benchmark targeting even if no item leaked. **INFERRED**
- **WebQL null detection:** 0.8B v2s AUROC 0.826 and 4B v2 0.831, vs **Jev 1.13 0.983**. v5 was not measured.
- **NLI and judging** (no Jev comparison): MNLI 0.896/0.899, ANLI r3 0.627, LLM-AggreFact 0.754 bAcc, RAGTruth AUROC 0.932, HaluBench 0.937, LLMBar 0.834.
- **Weaknesses the author measured** (`code/eval_security.py`):
  - As a shell-command safety reviewer: 0.600 accuracy, catching 37% of deny-worthy commands.
  - Prompt injection: one injected line in the state drops JevBench accuracy from 0.833 to 0.467 (150 items) and raises deny-worthy-allowed from 17% to 85%.
- **Davout's own benchmark** (the benchmark results in `README.md` and `results/report.md`; 300 test items per task, RTX 5090) used the **0.8B v2s-long** checkpoint at revision `f004f37`, **not v5**:
  - OpenJev scored BoolQ 0.723, SST-2 0.813, SMS spam 0.890, AG News 0.830, Yelp 0.393, Banking77 0.730, with p50 latency 16–113 ms.
  - Caveats (VERIFIED in Davout's `src/davout/scorer_nli.py:46-65` and in the local sidecar's request handling):
    - Davout builds its **own hypotheses** (`'Regarding "{ins}", the correct answer is "{name}": {desc}.'`, and a **single** hypothesis for noul). These are not OpenJev's trained `'The answer to "{instr}" is {label}: {crit}'` with a yes/no pair. The baseline is therefore probably understated for v5-style use. **INFERRED**
    - The 0.8B v2s-long lineage went through a `v2s-jev` stage. If that stage included `jevfmt`, it trained on the **Banking77 train split** in a 6-option intent format, which would flatter OpenJev's Banking77 number. The v5 panel goes further and includes the **Banking77 test split**, so any future Davout comparison against v5 on Banking77 or CLINC is contaminated. **VERIFIED** for v5 (manifest); **INFERRED** for v2s-long.

---

## 9. Differences from the original Jev

As far as public information allows:

1. **Architecture:**
   - Jev's is undisclosed. TypeSafe describes "a parallel sampler that generates all outputs in a single query", and observers suspect an open-weight LLM base. **REPORTED** (DataCamp, DigitalOcean)
   - OpenJev is a **3-class NLI cross-encoder** on Qwen3.5 (0.8B/2B/4B/35B-A3B), scored **one forward pass per option** with a classification head. **VERIFIED**
2. **Calibration:**
   - Jev claims "Reinforcement Learning for Calibrated Decisions (RLCD)". **REPORTED**
   - OpenJev uses plain cross-entropy (with teacher-vote soft labels for part of v4) and **no post-hoc calibration**. Its probabilities are renormalised P(entail), which is not a calibrated categorical by construction. **VERIFIED**
3. **Confidence formula:** differs, as described in §5. **VERIFIED** for OpenJev; Jev's formula is **REPORTED** via Davout.
4. **Large option sets:**
   - Jev reportedly uses a two-stage path (per the Davout README).
   - OpenJev has none: it is linear in options, capped at 64.
5. **Questions per call:**
   - Jev evaluates all questions "in parallel in a single pass". **REPORTED**
   - OpenJev runs one pass per (question, option, window). It amortises with prefix caching only on some paths.
6. **Context:**
   - OpenJev windows long states at 24k characters and takes the max over windows. This is a heuristic ("a claim supported by any window is supported by the document") and cannot combine evidence across windows. **VERIFIED** / **INFERRED**
7. **Multimodal:** OpenJev v2 and later accept images in the premise. Jev's image support is not covered by the sources I read.
8. **Accuracy:** behind Jev on JevBench hard (0.62 vs 0.73) and especially on temporal or numeric reasoning (0.27) and WebQL null detection (0.83 vs 0.98). **REPORTED**
9. **Training data:** no Jev outputs are used. Teachers are Qwen3.5 and gpt-5.5, plus large public NLI, agentic and faithfulness corpora. **VERIFIED**

---

## 10. Weaknesses and open issues

**Issue tracker** (HF discussions, 5 total, all closed or merged; **VERIFIED**):
- #1 prefix-cache PR (merged).
- #2 "alternative to Jev?" ("yes exactly").
- #3/#4 FinCalc fine-tune and eval (merged). Base 4B v1 scores 31.9% on numeric claims.
- #5 "why was 0.8B removed?" Answered "returned", but `qwen3.5-0.8b-nli-v5/` is absent at HEAD, and the card still links it.

**Code-level issues I found:**
- **Possible hypothesis truncation (INFERRED, untested).** `OpenJevCrossEncoder` defaults to `max_len=4096` (`modeling_openjev.py:42`) and the tokenizer truncates on the right (`tokenizer_config.json`: `truncation_side: right`). The `openjev_decide` window is 24,000 characters (~6k tokens; its comment claims an "8k-token encoder"). For states of roughly 16k–24k characters, the tail of each pair, **which is the hypothesis**, would be cut off. All options would then get near-identical inputs and a near-uniform distribution. `eval_jevbench.py` passes `max_len=8192`, so the published JevBench numbers would not show this. The SGLang gateway path does not truncate in the gateway.
- `decide()` does not use the shared-prefix path, so the Python adapter costs N full forwards per question. **VERIFIED**
- Score-type hypothesis wording differs between the eval harness and the serving API (`is level {i}` vs `is {i}`). **VERIFIED**
- Confidence is admittedly an approximation of Jev's undisclosed formula. **VERIFIED** (code comment)
- **Contamination:**
  - The v5 panel includes test splits of MMLU, ARC, GSM8K, GPQA, Banking77, CLINC and others. **VERIFIED**
  - The JevBench public items are included in `jevfmt` **by default** unless `--no-jevbench` is passed. The author says the flag was used for v3–v5, but the build invocations are unpublished. **VERIFIED** for the code default; **REPORTED** for its use.
- **Not hardened against prompt injection**, and weak as a security guard (author-measured). **REPORTED**
- **Not reproducible end to end:** the v3/v4/v2s-jev intermediate checkpoints, the build commands and the gpt-5.5 data are not published. **VERIFIED** (absent)
- Throughput depends on the custom SGLang package (`code/sglang_openjev/`), because SGLang has no Qwen3.5 sequence-classification class. Davout notes its OpenJev latencies used the reference kernels; the optional fast kernels were not installed.

---

## Key URLs
- Model repo (code, weights, card): https://huggingface.co/AlexWortega/openjev
- RESULTS-v5: https://huggingface.co/AlexWortega/openjev/blob/main/RESULTS-v5.md
- Typed-decision adapter: https://huggingface.co/AlexWortega/openjev/blob/main/code/openjev_decide.py
- Decisions API logic: https://huggingface.co/AlexWortega/openjev/blob/main/code/serving/decisions_api.py
- Demo Space: https://huggingface.co/spaces/AlexWortega/openjev
- Discussions: https://huggingface.co/AlexWortega/openjev/discussions
- JevBench (referenced): https://github.com/fstandhartinger/jevbench
- Namesake (different project): https://huggingface.co/openjev/openjev
- Jev background: https://www.datacamp.com/blog/system-one-models-jev , https://www.digitalocean.com/resources/articles/what-is-jev
