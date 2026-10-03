# Jev (TypeSafe AI): ground truth

Research date: 2 October 2026. Read-only; nothing in the Davout repo was changed.

Labels: **VERIFIED** = read on a TypeSafe primary source in this session (site, docs, blog, legal, GitHub org, or a TypeSafe staff member's own words in a transcript or thread). **REPORTED** = secondary source, not confirmed by a primary one. **INFERRED** = my reading of the evidence, or a third party's guess.

Disambiguation: Davout's README and its theory note (`HRM vs. Iterative Transformer for a Jev-Style Decision Model.md`) mean TypeSafe AI's Jev, API `POST /v1/systemone`, with question types Choice, Score and Noul. Searching "TypeSafe AI Jev" gives clean results with no name collisions. (TypeSafe AI is not the Scala company Typesafe Inc., now Lightbend.)

---

## 1. What Jev is

| Claim | Status | Source |
| --- | --- | --- |
| TypeSafe AI, Inc. is based in San Francisco. Founders: Diogo Almeida (CEO; co-invented RLHF and InstructGPT at OpenAI, previously Google Brain), Sasha Sheng (COO, ex-Meta/FAIR), Erik Gafni (CTO). | VERIFIED | https://typesafe.ai/team |
| Jev was announced on **15 Sep 2026** in early access, after "two years in stealth". It is the "first System One Model". | VERIFIED | https://typesafe.ai/blog/introducing-system-one-models-and-jev |
| Positioning: "a new class of frontier models built to make fast, structured decisions that software can use directly". It returns typed values plus probabilities, never text. Use cases include classify, route, score, extract-by-selection, guardrails, and "smart if-statements". | VERIFIED | launch post; https://docs.typesafe.ai/concepts/system-one |
| Named after William Stanley Jevons. "System One" refers to Kahneman's System 1. | VERIFIED | launch post FAQ |
| The CEO accepted "generalized zero-shot classifier that takes an option set at runtime" as a description, adding "yes and can do many of those in parallel". He also said it is "a general model", needs "no training at all", and should be called "zero-shot" rather than "instruction-tuned". | VERIFIED (CEO's HN account `CompleteSkeptic`, who wrote "CEO here"; his blog is completeskeptic.com) | https://news.ycombinator.com/item?id=49717558 (comments 49719101, 49719245, 49720799, 49718407) |
| "Jev is neither small nor an LLM." | VERIFIED | launch post FAQ "Is Jev just a smaller LLM?" |
| Not a chat or coding model. It cannot replace the LLM behind Claude Code, Cursor and similar tools. | VERIFIED | https://docs.typesafe.ai/introduction/coding-agents |
| Current model `jev-1.13.0`; aliases `jev-latest` and `jev-preview` both point to it. Cookbooks cite `jev-1.12` runs from Aug 2026, so earlier versions existed before launch. | VERIFIED | https://docs.typesafe.ai/models ; https://docs.typesafe.ai/llms-full.txt |
| Funding: a $40M seed led by DCVC (about $200M post-money), and talks on raising more than $1B at more than a $10B valuation. | REPORTED | https://dealroom.co/news/151032-typesafe-exits-stealth-with-40m-seed-to-build-ai-for-software-not-people/ ; https://cryptobriefing.com/typesafe-ai-billion-dollar-funding-jev-model/ |
| Launch HN thread: about 1,989 points and 520 comments. | VERIFIED (HN Algolia API) | https://news.ycombinator.com/item?id=49717558 |

### Public vs. closed

| Item | Status |
| --- | --- |
| Hosted API (`https://api.typesafe.ai/v1/systemone`, `GET /v1/models`), console and Playground | Public (early access, waitlist). VERIFIED: https://docs.typesafe.ai/api |
| Docs, including the full dump at `llms-full.txt`, cookbooks and the "jaggedness" page | Public. VERIFIED: https://docs.typesafe.ai/llms.txt |
| Python and JS SDKs; agent skills repo; `system-one-adapter-python` (MIT, a drop-in client backed by LLM APIs, used for TypeSafe's own LLM baselines) | Public. VERIFIED: https://github.com/typesafe-ai/system-one-adapter-python |
| Workflow evals site | Public. VERIFIED: https://evals.typesafe.ai/ |
| Weights | **Closed.** One set of weights serves every account, with no fine-tuning or LoRA per customer. VERIFIED: https://docs.typesafe.ai/models |
| Architecture, parameter count, base model | **Not disclosed.** The CEO said "architecture is close to the chest for now, but we have talked about writing a paper." VERIFIED: HN comment 49718824 |
| Paper on RLCD | **None.** Asked "You've never actually published a paper…?", Almeida answered "No, not yet." VERIFIED: https://www.latent.space/p/jev (transcript, about 00:25:05) |
| Model card | No model card in the Hugging Face sense. The docs "Models" page lists price, limits, context and input modality only. VERIFIED |

---

## 2. How it works technically

### 2a. Interface: what TypeSafe has published

- **Request:** `{state, model, questions}`. `state` is a string, a JSON object or an array of text. `questions` is a map from an id you choose to a Question. **The question id is not sent to the model.** VERIFIED: https://docs.typesafe.ai/api
- **`instructions`** and every criterion can be a string, object or array. In structured criteria, "the model sees the names along with the values". VERIFIED: https://docs.typesafe.ai/primitives/choice ; https://docs.typesafe.ai/primitives/advanced
- **Question types** (VERIFIED, API reference):
  - **Noul** ("short for bernoulli", per the CEO in HN comment 49718407). Optional `criteria: {true, false}`. Returns `noul`, a probability in [0,1] that the answer is yes.
  - **Choice.** `criteria` maps each option to a description or null, **at most 255 options**. Returns `choice` (the argmax), `probabilities` (summing to 1) and `confidence`.
  - **Score.** `criteria` is an ordered array of 2 to 10 level descriptions. Returns `score` (the probability-weighted level, which can fall between levels), `legend`, `probabilities` and `confidence`.
- **Score levels are judged independently.** The docs say "each level is judged on its own against the state" and "The model doesn't see a level's number or its neighbours." VERIFIED: https://docs.typesafe.ai/primitives/score
- **Questions are isolated.** "Every question … is evaluated independently", one answer is never context for another, and the state is "ingested once" with all questions evaluated "in parallel". VERIFIED: https://docs.typesafe.ai/primitives ; https://docs.typesafe.ai/models
- **Large Choice sets:** "For the higher cardinality choices, we do a 2 stage-system of scoring independently then making an explicit choice." VERIFIED: launch post (Wikiracing section). The docs give no threshold or shortlist size.
- **Confidence** is a fixed statistic computed from the returned distribution, not a separate model output (VERIFIED, https://docs.typesafe.ai/confidence):
  - Choice: `(p_max − 1/n) / (1 − 1/n)`.
  - Score: `max(0, 1 − Σ p_i·|i − m| / MAD_unif)`, where m is the modal level and MAD_unif is the same spread for a uniform distribution.
  - Noul has no confidence. The docs suggest `|2p − 1|` if you want one.
- **Output tokens:** the price is "FREE", but example responses still report non-zero `output_tokens` (18 to 212). VERIFIED: API reference and Choice page. Billing is per input token only.
- **Errors:** 401, 422, 429, 529. VERIFIED: API reference.

### 2b. Mechanism behind the interface: mostly undisclosed

| Claim | Status | Source |
| --- | --- | --- |
| "a new model architecture, parallel sampler for maximum efficiency, and training method we call Reinforcement Learning for Calibrated Decisions (RLCD)". The architecture and the sampler are not described. | VERIFIED | launch post |
| Sampling is "Parallel. Generates all outputs in a single query", unlike sequential token generation. | VERIFIED | launch post |
| "strings (and all sequential data structures) are not allowed at all - this is how we make sure all outputs can be computed in parallel (thus no output token cost)". | VERIFIED (CEO) | HN comment 49719122 |
| The CEO said constrained decoding "make[s] models dumber" because "simply masking logits is insufficient". This implies Jev is **not** an LLM with grammar or logit masking, though he did not say what it is instead. | VERIFIED (statement) / INFERRED (implication) | HN comment 49718849 |
| "it is just a model, no harness yet … a structured data model, but technically not a language model". | VERIFIED (CEO) | HN comment 49718437 |
| "There's a reason why we don't call them decision models … we have stuff in the tank." | VERIFIED (CEO) | Latent Space transcript |
| Jev is "Transformer-based". | REPORTED only (Chinese explainer). No TypeSafe source says this. | https://www.woshipm.com/?p=6467184 |
| The `typesafe-ai` GitHub org has forks of `vllm` (May 2025) and `LLaDA` ("Large Language Diffusion Models", forked Jul 2025, with no new commits). This hints, weakly, at interest in non-autoregressive or diffusion-style language models and vLLM serving. It is **not** evidence of Jev's architecture. | VERIFIED (the forks exist) / INFERRED (meaning) | https://github.com/typesafe-ai |
| A Latent Space show-notes item is titled "blending transformers and classifiers". It links to a third-party X post, not a TypeSafe statement. | VERIFIED (link exists) / INFERRED | https://www.latent.space/p/jev |

### 2c. Training

| Claim | Status | Source |
| --- | --- | --- |
| RLCD is a post-training path parallel to RLHF and RLVR. Its contract: "does not generate text", "returns decisions and probabilities", and higher probability should mean a higher chance of being correct (group calibration). | VERIFIED | https://docs.typesafe.ai/introduction/machine-learning-primer |
| The docs show "Pretrained language models branch into … RLCD decision-model path" (figure alt text), which suggests Jev starts from a **pretrained language model**. | VERIFIED (figure text) / INFERRED (that Jev literally does) | same page |
| Almeida describes RLCD as a new task or "North Star" rather than a specific algorithm ("Just like DPO and all of its descendants also do RLHF"). No reward function is disclosed. | VERIFIED | Latent Space transcript, about 00:25 |
| Data: "TypeSafe is primarily a data research lab … We make all the data ourselves." Asked "all your data is synthetic", Almeida did not dispute it ("synthetic, so what"). Humans with taste review and regenerate it. The data is not trained on customer data. | VERIFIED | launch post FAQ; Latent Space transcript, about 00:22 to 00:24; https://docs.typesafe.ai/models |
| The acronym RLCD collides with an unrelated 2023 paper, "RLCD: Reinforcement Learning from Contrastive Distillation" (Yang et al.). | REPORTED (woshipm). The paper exists at arXiv 2307.12950, which I did not fetch this session. | https://www.woshipm.com/?p=6467184 |

### 2d. Usage mode, limits, latency, cost

- **Zero-shot, prompt-time only.** You customise Jev through `state`, `instructions` and `criteria`, never weights. VERIFIED: https://docs.typesafe.ai/models. The docs do not prescribe few-shot examples; descriptions may carry `examples` fields inside structured criteria. VERIFIED: Choice page.
- **Context:** "64k tokens per request; 32k tokens for `state` plus the longest question." VERIFIED: https://docs.typesafe.ai/models. This settles the open question in Davout's theory note, which relied on a secondary "32K" figure.
- **Input:** text only. English is primary; other languages work, but less well. VERIFIED: models page.
- **Rate limits:** 100K tokens/s and 40 requests/s, "adjusting dynamically". VERIFIED: models page.
- **Price:** $0.042 per million input tokens; output is free. VERIFIED: models page and launch post.
- **Latency:** "70ms-500ms" end to end, "40x-200x faster" than frontier LLMs on System-One-shaped queries, measured "from our laptops on the West Coast". VERIFIED (vendor claim): launch post.
- **Determinism:** no seed. Robustness is preferred over determinism, and future quantisation is not ruled out. VERIFIED: Latent Space transcript.
- **Known weaknesses ("jaggedness", jev-1.13):** literal reading; counting and arithmetic; date comparison; multi-hop indirection; context rot from irrelevant state; adversarial or injected content; contradictory instructions and criteria; no structural invariants (for example, P(refund)=0.72 and P(not refund)=0.47 on the same ticket, and Noul vs Choice on the same question giving 0.22 vs 0.01); poor generation. VERIFIED: https://docs.typesafe.ai/model-jaggedness/jev-1.13

---

## 3. Reported benchmarks

- **No public-benchmark results, by policy.** "We deliberately chose *not* to publish performance against *public* benchmarks." New evals are to be dated snapshots that are "immediately retired". VERIFIED: launch post FAQ; https://typesafe.ai/blog/antibenchmaxxing (11 Sep 2026)
- **Workflow evals** (https://evals.typesafe.ai/), VERIFIED:
  - Four in-house workflows: Security Incidents, Agent Trace Observability, Invoice Processing, Customer Service.
  - Reference labels are the average of GPT-6 Astra and Claude Fable 5.1 at high thinking. Accuracy is agreement with that consensus, plotted against cost and time.
  - LLM baselines run through TypeSafe's "System One LLM" wrapper.
  - Headline: "193.6x faster, 444.6x cheaper", which TypeSafe says is "on the higher end of real world gains".
  - Caveats TypeSafe states itself: the workflows were built by its own capabilities team, and the reference biases toward OpenAI and Anthropic.
- **Hallucination 0%:** "not empirical. Schema matching is guaranteed." VERIFIED: launch post
- **Cookbook numbers** (single tasks, VERIFIED, https://docs.typesafe.ai/llms.txt):
  - CLERC legal re-ranking: top-1 rises from 5% to 18% and top-10 from 38% to 62% over BM25.
  - GDPR 13-question batch: 12.2x cheaper and 10x faster than one call per question.
  - Further examples cover SEC filings in 75 industry groups, Hermes skill selection over 182 skills, and beer-catalogue entity alignment.
- **Third-party "JevBench"** (benchmarkheaven.com): Jev 75.3 against the SemIf clone at 74.6. REPORTED: https://x.com/airesearch12/status/2101311992984199580. Almeida "has rejected publicly" JevBench-style efforts, according to Latent Space show notes.

---

## 4. Licensing, terms, and statements about clones

- **Master Customer Agreement** (last updated 23 Sep 2026), §2.3 License Restrictions, VERIFIED, https://typesafe.ai/legal/mca. The customer may not:
  - "(b) use the Services or any Output … to perform model distillation, train a model to imitate the output of the Services, or develop (or to facilitate the development of) a similar or competing product or service";
  - "(c) reverse engineer … or attempt to access or derive … the underlying ideas, algorithms, structure, or organization";
  - "(d) … create derivative works".

  Breach allows immediate suspension. TypeSafe assigns Output ownership to the customer. **Implication for clones (INFERRED):** a clone trained on Jev outputs, or a harness that calibrates to Jev, would breach the MCA if built by an API customer. Building a look-alike API from public docs is not addressed by these clauses.
- **Terms of Use** (site, last updated 19 Sep 2026): a personal, revocable licence to use the site. VERIFIED: https://typesafe.ai/legal/terms
- **Benchmarking clause:** in Latent Space, Swyx refers to a benchmarking restriction from the preview period that was not removed at launch. Almeida: "obviously we're not stopping people from do[ing that]… I asked them to check in with the lawyers." VERIFIED (transcript, about 00:17:30 to 00:17:54). I found no benchmarking clause in the current MCA text, but I did not read it exhaustively.
- **On clones** (VERIFIED, Latent Space, https://www.latent.space/p/jev, 21 Sep 2026):
  - The show notes contrast real use cases with "the 55th low effort clone of Jev's API" and link to AINews, "Here are 6 clones of Jev" (https://www.latent.space/p/ainews-here-are-6-clones-of-jev-in).
  - Swyx: "There's like 50 Jev clones." Almeida: "Let's say that there is competition", and he frames the race as "intelligence per dollar".
  - Swyx, on clones: "you're still beating every single clone of you out there." Almeida: "I don't care about the benchmarks."
  - Almeida mentions talk of "doing open source", but "We have absolutely no time for anything else right now."
- I found **no formal TypeSafe statement, blog post or legal action about specific clones** (OpenJev/SemIf, Kev, Von and others).

---

## 5. Unknown, or only inferred by third parties

1. **Architecture.** Encoder, decoder, diffusion or a custom design; how the "parallel sampler" works; whether options and levels are scored by a per-option head, by logit readout, or by pairwise or independent passes. Only the statements above are public. Third-party pieces that call Jev "Transformer-based", or compare it to a "Large Classification Model" or to GLiNER/DeBERTa, are guesses.
2. **Size and base model.** Not disclosed beyond "neither small nor an LLM".
3. **RLCD.** The reward, the algorithm and the calibration target (Brier? log loss?) are not public, and there is no paper.
4. **Training data.** Described only as self-made and largely synthetic, reviewed by humans.
5. **Two-stage Choice details.** The threshold and shortlist size are unknown. That Score levels are judged independently is VERIFIED; how a Score distribution is normalised from independent level judgments is unknown.
6. **Why `output_tokens` is non-zero** in responses when output is free and "no text" is generated.
7. **Independent calibration measurements** (ECE or reliability curves) against public labels. TypeSafe publishes none; any such numbers come from third parties.
8. **Hardware, cost basis and whether the price is subsidised.** TypeSafe itself says "We can't prove it isn't subsidized."
9. **Davout README claims checked.**
   - The confidence formula `(n·peak − 1)/(n − 1)` matches the docs. VERIFIED.
   - "Jev reads the state once for all questions" matches the models page. VERIFIED.
   - "Jev judges each Score level independently" matches the Score page. VERIFIED.
   - "Two-stage for large Choice sets" matches the launch post. VERIFIED.
   - Davout's limit of 2 to 255 Choice options and 2 to 10 Score levels matches the API reference. VERIFIED.
   - Davout's `usage.output_tokens = 0` differs from Jev, whose examples report non-zero output tokens.

## Key URLs

- Launch post: https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Docs index / full dump: https://docs.typesafe.ai/llms.txt , https://docs.typesafe.ai/llms-full.txt
- API: https://docs.typesafe.ai/api · Models: https://docs.typesafe.ai/models · Confidence: https://docs.typesafe.ai/confidence · Score: https://docs.typesafe.ai/primitives/score · Jaggedness: https://docs.typesafe.ai/model-jaggedness/jev-1.13 · AI primer (RLCD): https://docs.typesafe.ai/introduction/machine-learning-primer
- Evals: https://evals.typesafe.ai/ · Anti-benchmark post: https://typesafe.ai/blog/antibenchmaxxing · Bitterest lesson: https://typesafe.ai/blog/bitterest-lesson
- MCA (anti-distillation clause): https://typesafe.ai/legal/mca · Team: https://typesafe.ai/team · GitHub: https://github.com/typesafe-ai
- HN launch thread (CEO comments): https://news.ycombinator.com/item?id=49717558
- Latent Space interview (21 Sep 2026): https://www.latent.space/p/jev
