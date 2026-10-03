# Jev analogs: census from community discussion, commercial products and research

Compiled 2026-10-02. Scope: venues other than direct GitHub/HF/registry sweeps (Hacker News, Reddit, blogs, news, vendor sites, arXiv), following links into repos where a discussion points at one. Davout itself is excluded.

Evidence labels:
- **VERIFIED**: read on the primary source (the project's own repo README, vendor blog or paper page).
- **REPORTED**: read on a secondary source (round-up article, news story, HN comment, third-party benchmark).
- **INFERRED**: my own conclusion from the sources.

Caveat on method: pages were read with a fetch tool that returns a model-written summary of each page, plus raw `curl` for a few READMEs, the HN Algolia API, the arXiv API and the JevBench leaderboard HTML. Numbers below are as the summaries or raw text gave them. Before citing a number externally, re-open the page.

---

## 0. Headline findings

1. **There are far more than 10 analogs.** Here is the count by source:
   - The **awesome-jev full catalog** (Amal-David) has a section "Research and independent reproductions (231)". Filtering that section for reproductions and compatible servers gives about 60 projects. The other entries are Jev users or benchmarks of Jev. (VERIFIED, catalog downloaded.)
   - **JevBench v1.5.4** (Benchmark Heaven) ranks 106 of 112 roster systems. 60 of them meet its "Jev-class" cost and latency cap, and dozens are tagged "Jev rebuild". (VERIFIED, leaderboard HTML parsed.)
   - The HN Algolia API returned more than 40 distinct Jev-like model and server launches between 2026-09-15 and 2026-10-02. (VERIFIED.)
   - This report profiles **26 analogs** in detail (section 2) and lists a long tail in section 6.
2. **Several compilations already exist** (section 1). The most useful ones:
   - The awesome-jev catalog.
   - The JevBench leaderboard.
   - Five or more "open-source Jev alternatives" round-up articles (Apidog, ScriptByAI, Pinggy, DataCamp, Geeky Gadgets).
3. **"OpenJev" is not one project.** At least seven unrelated projects use the name:
   - TheoLeeCJ/openjev, later renamed **SemIf**. This one was on HN as openjev.com with 722 points.
   - razorback16/openjev (DiffusionGemma), also served commercially at api.codiv.ai.
   - GPT-AGI/OpenJev, xingwudao/OpenJev, zhangcy122/OpenJev and zhihz/openjev.
   - **HF AlexWortega/openjev** (Qwen3.5-0.8B NLI). This is the one Davout benchmarks against.

   Several other projects also share the name Kev (jaredpalmer/kev, arjun988/Kev, "Kev on Workers AI"). Analysts of "OpenJev" and "Kev" should pin which one they mean. (VERIFIED from catalog, HN titles and READMEs.)
4. **Big-company entrants in the first two weeks:**
   - OpenAI "Decisions API" on Luna (limited preview, announced 2026-09-30).
   - Cloudflare **Clef / Clef-flash** (open weights, 2026-10-01).
   - AWS Strands **Decider 2B** (2026-10-01).
   - Liquid AI **d1** (hosted).
   - Fastino **GLiNER2.5-Decide**.
   - Nace.AI **Drex** (closed).
   - Edgeless/Privatemode (GLM-5.3-Flash logit readout).
   - Bespoke Labs **Nimble**.
   - PostHog **Jeeves**.
   - Nokia Applied Research **AnyJev**.
   - Stanford/NVIDIA **CLM-8B**.
5. **TypeSafe's only public statement about imitators** that I found came from CEO Diogo Almeida, quoted by TechCrunch about OpenAI's Decisions API. He joked it was "the beginning of the clone wars" and said "Intelligence is the hard part". (REPORTED, TechCrunch 2026-09-30.) I found no takedowns or licensing disputes against clones (section 5).

---

## 1. Existing round-ups and comparisons (the most useful finds)

| Compilation | Who / date | What it contains | Label |
|---|---|---|---|
| **awesome-jev full catalog** <https://github.com/Amal-David/awesome-jev/blob/main/docs/CATALOG.md> | Amal-David; undated, continuously updated | About 850 KB. Sections: "Research and independent reproductions (231)", "Community directories (56)" and others. Each entry carries an evidence grade (primary-source-reviewed / community-indexed / readme-matched). The README's "Independent Models" section highlights AnyJev, CUA-S1-FORMS, jevlike, OpenJev SGLang and Open Medical Jev, with the warning "compatible outputs do not establish equivalent quality". | VERIFIED |
| **Other awesome lists** (56 listed in the catalog) | various | Includes OmniJev/awesome-jev-gallery ("Papers, open reproductions and independent evaluations"), Yifan-Lan/awesome-jev-robustness, heyjunpenn (485 projects), kydlikebtc (805 examples), Li-Evan (3,400+), awesome-jev-zh, jev-radar. | VERIFIED (listing only, not each list) |
| **JevBench v1.5.4** <https://benchmarkheaven.com/jev-models> | Benchmark Heaven (attributed to "Florian S." by regolo.ai, REPORTED) | 1,624 decisions per system (904 open + 720 sealed). Capability Score = mean of Intelligence and Calibration. "Jev-class" means at most 2× Jev 1.13.0's cost and median latency. 60 Jev-class systems are ranked, with labels such as "Jev rebuild", "system-one-open", "Raw-logit control" and "Reranker". Also runs Image JevBench and AudioJevBench. | VERIFIED |
| **Decision Index** | Linked from Von's README as github.com/apolinario/decision-index. Nace.AI's Drex page describes "Decision Index 0.2.1" as its own measurement over 38 benchmarks, chance-corrected. | Leaderboard on 38 decision benchmarks. Jev 1.13.0 scores 57.91. | REPORTED; ownership unclear |
| Apidog, "Top Jev Open Source Alternatives" <https://apidog.com/blog/openjev-open-source-jev-alternatives/> | Ashley Innocent, 2026-09-18 | OpenJev (TheoLeeCJ), mini-jev, jevlike, Parallel Constrained Decoding HF Space, vLLM PR #57250 | REPORTED |
| ScriptByAI, "9 Best Open-Source Jev Alternatives" <https://www.scriptbyai.com/jev-open-source-alternatives/> | 2026-09-22 | Laya, Kev, SemIf, jeff (logan-markewich), OpenJev (razorback16), LocalJev (githubnext), OpenJev Verdict, NanoJev, jevlike | REPORTED |
| Pinggy, "Best Open Source Jev Alternatives" <https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/> | 2026-09-23 | Laya, Kev, SemIf, NanoJev, jevlike, plus Von, OpenJev, openjev-sglang and GLiNER2, with jabr benchmark numbers | REPORTED |
| DataCamp, "Top 7 Open-Source TypeSafe Jev Alternatives" <https://www.datacamp.com/blog/top-open-source-jev-alternatives> | Abid Ali Awan, 2026-09-23 | Laya, Nimble, Kev, SemIf, Rizzo Flow, Von, NanoJev (mechanisms, no numbers) | REPORTED |
| Geeky Gadgets, "Over 20 Open Jev Alternatives…" <https://www.geeky-gadgets.com/open-jev-models-alternatives/> | Julian Horsey, 2026-09-23 | Write-up of a Sam Witteveen YouTube video: SemIf/OpenJev, Nimble, Decider, DiffusionGemma, NanoJev, Laya. No numbers. | REPORTED |
| regolo.ai, "Jev and system one models: benchmarks, open-source alternatives…" | regolo.ai | Laya, regolo ModernBERT capability classifier, OpenJev (razorback16) | REPORTED |
| knolli.ai "8 Best Jev AI Alternatives", opentweet.io "Jev alternatives: seven ways to make a typed decision", developersdigest "Laya vs Kev vs Ollaya" | various | Surfaced in search but not read | not read |
| arXiv 2609.32160, "Typed Decision Models: An Early Evidence Audit and Evaluation Checklist" | Tang and Zheng, 2026-09-26 | Audits 28 papers posted 19–24 Sep about Jev itself, not about clones. Finds "the typed readout itself has not shown an independent accuracy advantage". Gains are latency and cost. Proposes a 14-item checklist. | VERIFIED (abstract) |
| arXiv 2609.30216, "Jev in the Wild" | Ling, Xue, Ye, 2026-09-24 | 2,170 public Jev projects on GitHub as of 2026-09-22. The abstract does not break out clones. | VERIFIED (abstract) |

---

## 2. Analogs: one section each

Kev (jaredpalmer) and OpenJev are covered in depth by other researchers. They appear here only briefly, for the comparisons and the name-collision note.

### 2.1 Laya (Convai Innovations)
- **Who / URL:** Nandakishor M / Convai Innovations. Code: <https://github.com/NandhaKishorM/laya>. Checkpoints: HF `convaiinnovations/laya`, `laya-multilingual`, `laya-typed-decisions`. Site: laya.convaiinnovations.com. Open, Apache-2.0. Launched on HN 2026-09-19 as "I built non-autoregressive decision models with RL a year ago" (1,362 points, 318 comments). (VERIFIED README and HN API)
- **Relationship to Jev:** independent, wire-compatible with `/v1/systemone`. The author claims it is prior art from a year earlier (see the controversy below). (VERIFIED)
- **How it works:** non-autoregressive encoder.
  - Checkpoints: ModernBERT-large 421M (English, 512 tokens) and mmBERT-base 322M (100+ languages, up to 8,192 tokens), plus a fine-tuned `laya-typed-decisions` checkpoint.
  - Scores state, question and options together in one pass. A router picks the checkpoint by script.
  - Ships uncalibrated: "both checkpoints are over-confident as shipped". Temperature fitting is offered. (VERIFIED README)
- **Numbers:**
  - Self-reported: typed-decisions 2,000-decision benchmark, fine-tuned 0.766 vs base 0.362. (VERIFIED)
  - README "Laya (with routing) vs Jev": argmax accuracy 0.766 vs Jev 0.727, soft accuracy 0.580 vs Jev 0.471 in Jev's favour. The README notes the Jev numbers are "published, never measured here". (VERIFIED)
  - Speed: about 33–40 ms per question. (VERIFIED/REPORTED)
- **Third-party results:**
  - Pinggy: Banking77 0.425 vs Jev 0.870. (REPORTED)
  - jabr v2: Laya 0.583 vs Jev 0.966. (REPORTED)
  - JevBench: Laya multilingual Capability 23.0, rank 58 of 60. (VERIFIED)
  - Von README table: Laya composite 30.3 vs Jev 63.3. (VERIFIED)
  - gazelle93 stress tests: large drop at K=128 and high order sensitivity (section 3). (REPORTED)
  - Raschka: IMDb Laya 92.33 vs Jev 96.47. (REPORTED)
  - arXiv 2609.33843 independent reproduction: systematically under-confident (gap −0.214). Temperature T=0.469 reduces ECE from 0.204 to 0.037. (VERIFIED abstract)
- **Controversy:** the "prior art" claim.
  - HN commenters said the earlier SalesRLAgent work had target leakage (outcome in the input state), and called the blog post AI-generated. (REPORTED, HN thread 49765348)
  - John Curcio's rebuttal "Laya's Prior Art Claim Is Absurd" (2026-09-22) argues SalesRLAgent is binary sales-conversion regression with leakage, shares nothing with Jev's schema-driven design, and has a deleted `generate_dataset.py`. (VERIFIED blog)
- **Ecosystem:** Laya has the largest community of the open alternatives:
  - Runtimes: laya-mlx, laya-coreml, laya-mps, Laya on Lambda (laymbda).
  - **Ollaya** ("Ollama for decision models", HN 616 points).
  - Hosted on Vercel AI Gateway.
  - Wrappers: sys1 (Rust), laya-studio.
  - (VERIFIED HN titles)

### 2.2 Kev (Jared Palmer), brief
- <https://github.com/jaredpalmer/kev>, Apache-2.0. HN 2026-09-21, 462 points. The base is Qwen3.5 (0.8B/4B/9B) with a rank-16 LoRA and a pointer head; earlier it was Qwen2.5-0.5B. It speaks `/v1/systemone`. (REPORTED, DataCamp/Pinggy; HN titles VERIFIED)
- **Head-to-head:**
  - Opper, 362 fresh items:
    - Kev-4B roughly matches Jev 1.13 on accuracy: arXiv 95.0 vs 96.9, StackExchange 98.3 vs 97.5, GitHub 93.9 vs 95.1.
    - Calibration is worse.
    - Jev adds about 257 fixed input tokens per request.
    - Latency: Kev 220 ms vs Jev 275 ms.
    - (REPORTED, opper.ai, Jose Sabater, 2026-09-25)
  - Pinggy: Kev-9B 0.822 macro vs Jev 0.857; MMLU 0.74 vs 0.90. (REPORTED)
  - JevBench lists only kev 0.6B (39.0) and kev 0.5B (32.5) among Jev-class systems. (VERIFIED)
- **Name collisions:** arjun988/Kev (Ollama-based engine) and "Kev – A Jev-Compatible API on top of DiffusionGemma running on Workers AI" (kev.workers-ai-mle.workers.dev). (VERIFIED catalog and HN)

### 2.3 SemIf (formerly OpenJev, TheoLeeCJ)
- <https://github.com/TheoLeeCJ/SemIf> (also SemIf-OpenJev), demo openjev.com. MIT. On HN as "OpenJev" 2026-09-18 (722 points, 296 comments). (VERIFIED)
- **Relationship:** API-compatible clean-room reimplementation of the interface. The README says it "reproduces that **interface pattern** with open models; it does not reproduce Jev's undisclosed model or training" and that it "was formerly called OpenJev. It is not affiliated with or endorsed by TypeSafe… No infringement is intended." (VERIFIED) The reason for the rename is not stated. A trademark precaution is plausible (INFERRED).
- **Mechanism:**
  - Frozen open model (Qwen3.5-4B by default; also MiniCPM5-2B and a Qwen3.8-27B EXL3 bridge).
  - Direct option-logit readout, no training.
  - Per-workload temperature scaling.
  - Runs in the browser through WebGPU. (VERIFIED)
- **Numbers:**
  - Self-reported, Qwen3.5-4B: authored balanced accuracy 0.813, perturbation 0.766, TypeSafe-subset modal agreement 0.845 vs Jev 0.883 (102 aligned rows; "we did not run a live Jev endpoint").
  - Qwen3.8-27B bridge: 0.958 on 144 authored rows.
  - Speed: 1.023 s for 21 binary criteria vs 5.332 s generating JSON on an RTX 3090. (VERIFIED)
  - JevBench: Capability 67.6, rank 18. (VERIFIED)
  - Fastino benchmark: 56.4%. (REPORTED)
- **Community:** pasqualepillitteri.it says SemIf and Kev together passed 4,000 GitHub stars within days. (REPORTED) HN criticism: "These jev-copy projects are all vibecoded, only mimic the shape of output" (prodigycorp). (REPORTED)

### 2.4 OpenJev (razorback16) / Codiv
- <https://github.com/razorback16/openjev>, Apache-2.0. Also offered hosted at `api.codiv.ai` (README example). (VERIFIED)
- **Mechanism:** DiffusionGemma 26B-A4B read as a diffusion canvas with single-token answer slots, adapted from vLLM PR #57250.
  - Up to 255 options, image inputs, and an OpenAI-compatible chat endpoint.
  - Also serves CLM-v0.1, JevK5 and Laya behind the same API. (VERIFIED)
- **Relationship:** API-compatible wrapper/server. It accepts the model names `jev-latest`/`jev-preview` so that TypeSafe SDK defaults work. (VERIFIED)

### 2.5 OpenAI Decisions API (on Luna), closed commercial competitor
- Announced at OpenAI Dev Day, reported by TechCrunch 2026-09-30 (Tim Fernholz) and The New Stack 2026-09-29 (Frederic Lardinois; article body not retrievable). (REPORTED)
- **Mechanism:** a predefined option set is given to the Luna model. Altman: "By focusing the model on that choice, we can make it extremely fast" while keeping image understanding and language breadth. Limited preview, no pricing published. (REPORTED)
- **TypeSafe response:** Almeida called it "the beginning of the clone wars" and said "Intelligence is the hard part". (REPORTED, TechCrunch)
- Earlier analysis: Arcturus Labs, "OpenAI is well positioned to fast-follow Jev" (HN 328 points, 2026-09-22). Not read. (VERIFIED title only)

### 2.6 Clef / Clef-flash (Cloudflare), open-weight commercial competitor
- <https://blog.cloudflare.com/clef-decision-models/>, Michelle Chen, 2026-10-01. HN 483 points. Apache-2.0 on HF. (VERIFIED blog)
- **Mechanism:**
  - Clef is based on Qwen3.8-27B; Clef-flash on Qwen3.5-9B.
  - "Non-autoregressive, prefill-only pass" with two-stage attention routing in which each choice extracts context.
  - Post-training uses label-smoothed cross-entropy plus a Brier loss. Cloudflare also describes an RLCD objective of its own and launches an RL fine-tuning platform (AI Gateway capture, Workers AI rollouts, Trainer).
  - "Fully Jev-API compatible". Adds vision and 64k context. (VERIFIED)
- **Self-reported vs Jev:**
  - BFCL 98.47/98.76 vs 95.75.
  - BANKING77 macro-F1 94.20/90.93 vs 79.74.
  - CLINC150+OOS 97.43/66.77 vs 89.27.
  - Median latency 209/39 ms vs 524 ms.
  - Wins 3 of 4 workflow evaluations.
  - (VERIFIED as Cloudflare's claim)

### 2.7 Strands Decider 2B (AWS Strands Agents)
- <https://strandsagents.com/blog/introducing-strands-decider/>, Marc Brooker, Mike Chambers, Fabio Nonato de Paula, 2026-10-01. Open source, with weights, data and scripts released. (VERIFIED)
- **Mechanism:** Qwen2.5-2B with the LM head removed and replaced by a pointer head (about 1M parameters). It "scores the hidden state at each option position against the hidden state at the `<answer>` position". Rank-16 LoRA. (VERIFIED)
- **Numbers:** JevBench public "3rd of 33 in the 2B class". Latency about 115 ms on an RTX 3090. API compatibility is not confirmed. (VERIFIED as claim)

### 2.8 Liquid AI d1
- <https://docs.liquid.ai/lfm/models/decision-models>. Hosted endpoint `/decisions/v1/systemone`, docs refer to a `TypeSafeClient` SDK, model id `d1:free`. Returns calibrated probabilities "with zero generated tokens". Base model, size, licence and benchmarks are not disclosed. (VERIFIED docs) The model probably comes from the LFM family (INFERRED). JevBench lists an unrelated community "lev-350m" built on LiquidAI/LFM2.5-350M. (VERIFIED)

### 2.9 GLiNER2.5-Decide (Fastino)
- <https://huggingface.co/fastino/GLiNER2.5-Decide>, Apache-2.0. 340M DeBERTa-v3-large (from gliner2-large-v1), schema-driven multi-head classification in one pass. Also comes in 1B and multilingual 287M variants. (VERIFIED model card)
- **Release date conflict:** the card summary gave "July 24, 2025", which is probably the GLiNER2 arXiv date. Soniqo says the model was released 2026-09-24. (REPORTED)
- **Numbers:** on Fastino's own fast-decisions set (17 domains × 300): Decide 60.2%, Decide-1B 59.6%, JevK5 57.6%, multi 56.7%, SemIf 56.4%, GLiFormer 49.0%, Laya Router 46.6%. No Jev number. (VERIFIED card / REPORTED soniqo.audio 2026-09-27)
- Raschka notes that GLiNER has existed for about 3 years and "Jev is definitely stronger". (REPORTED)

### 2.10 Drex (Nace.AI), closed
- <https://www.nace.ai/drex>. Proprietary, "under 10B" parameters, one forward pass, probability output. (VERIFIED page)
- **Self-reported:**
  - Decision Index 0.2.1 (2026-09-28): Drex 1.5 58.28 vs Jev 1.13.0 57.91, Surogate Rune v3 57.44, Decider chat·Gemma 57.33.
  - JevBench (231 items): Drex 86.2% vs Jev 87.0%.
  - Drex reads 1,170 median tokens per decision vs Jev 367.
  - (VERIFIED as claim)

### 2.11 Privatemode / Edgeless Systems: GLM-5.3-Flash as a decision model
- <https://www.privatemode.ai/blog/system-one-from-glm-flash>, Johannes Hötter and Marko Rosenmüller, 2026-09-24. HN 138 points. (VERIFIED)
- **Mechanism:**
  - Plain prompt plus prefill (`choice_index:`), then logit readout on the unmodified model.
  - Uses vLLM `allowed_token_ids`, `logprob_token_ids` and `continue_final_message`.
  - Sold as confidential-computing inference.
- **Vs Jev, 28 text datasets:** each wins 10, and the remaining 8 are within 1 pp (median gap 0.7 pp, not significant). Latency 180 ms in Germany vs Jev 264 ms. Cost €62 vs €16 per million decisions. Adds vision (70.2% on scanned documents). (VERIFIED as claim)

### 2.12 Bespoke Nimble (Bespoke Labs)
- <https://github.com/bespokelabsai/nimble>, announced 2026-09-18 on X ("open data, open model, open recipe for an open Jev"), checkpoint 2026-09-24. Available in Ollama ("Ollama now supports Jev-style decision models", ollama.com/library/nimble, 2026-09-30). (VERIFIED)
- **Mechanism:** Qwen3.5-9B, rank-16 LoRA. Logits over the allowed candidate tokens. Contrastive data curation: pairs in which one changed fact flips the answer. 2,676 examples, 10 domains. States "we did not distill from Jev". (VERIFIED)
- **Numbers:** 324 held-out items: Nimble 90.12% vs Jev 1.13.0 93.21% vs base Qwen3.5-9B 66.36%. 106 ms median on an H100. (VERIFIED)
- **Related research:** PACT (arXiv 2609.35865, Yida Lin) evaluates on a 324-item holdout and reaches 84.6% vs a "published baseline" of 85.2%. That baseline is probably Nimble's set (INFERRED). metask-jev claims "metask-jev-4b beats Bespoke Nimble-9B". (REPORTED, catalog)

### 2.13 Jeeves (PostHog)
- <https://github.com/PostHog/jeeves>, MIT. HN 2026-09-29, 242 points, "Reasoning improves Jev-like decision models". (VERIFIED)
- **Mechanism:**
  - Qwen3.5-9B with LoRA and a pointer head. Generates a reasoning chain, then scores options by query-key dot product.
  - Training: SFT on 19,126 questions, then CISPO RL (9,992 questions, 8 rollouts).
  - Drop-in `/v1/systemone` with a reasoning option. (VERIFIED)
- **Self-reported vs Jev:**
  - Test overall 0.889 vs 0.857.
  - JevBench public 0.935 vs 0.866; hard 0.865 vs 0.730.
  - Worse on knowledge: MMLU 0.793 vs 0.900, MMLU-Pro 0.739 vs 0.840.
  - Because it reasons, it is not latency-equivalent to Jev (INFERRED).

### 2.14 Decider (Mark Marosi / Mapika)
- <https://github.com/Mapika/decider>, Apache-2.0, updated 2026-09-29. Independent. (VERIFIED)
- **Mechanism:**
  - Qwen3.5-2B/4B-Base and 35B-A3B-Base fine-tunes.
  - Reads letter logits at each answer slot.
  - Data: about 95 public datasets plus labels from a Qwen3.5-27B teacher. "Nothing was distilled from Jev."
  - TypeSafe SDKs work with `TYPESAFE_BASE_URL`. (VERIFIED)
- **Numbers:**
  - Decision Index: Jev 57.91 vs decider-35b-a3b 47.11.
  - JevBench hard: Jev 0.730 vs decider-4b v2 0.676.
  - JevBench v1.5.4 Capability: decider-4b v2 70.7 (rank 11).
  - Ollaya thread (jonmagic): "Decider-4b v2 scores 64.13, Jev 1.13 scores 63.29" on an earlier JevBench scale.
  - (VERIFIED / REPORTED)

### 2.15 JevK5 (Alibi Serikbay)
- <https://github.com/allebee/jevk5>, HF `alibiserikbay/JevK5` (4B) and `JevK5-9B`, Apache-2.0. (VERIFIED)
- **Mechanism:** Qwen3.5-4B/9B with a distilled LoRA, merged. Uses SemIf's option-logit readout. **Distilled from Qwen3.6-27B and GPT-6 Luna** (17,408 teacher questions in v0.3), not from Jev. There is also a separate JevK5-Lite (437M DeBERTa). (VERIFIED)
- **Numbers:**
  - Self-reported: v0.2 was "second of 76 systems and first among open entrants (62.04; Jev 1.13.0: 63.29)" on a JevBench-style score.
  - JevBench v1.5.4: JevK5 v0.3 Capability 72.3 (rank 7), Intelligence 56.3 vs Jev 72.0, Calibration 88.3 vs Jev 88.0.
  - Truncation hurts long-context hard items. (VERIFIED)
- JevBench also lists a derivative, Plumb-4B (JevK5 v0.2 + LoRA). (VERIFIED)

### 2.16 Von (wfzyx)
- <https://github.com/wfzyx/von>, Apache-2.0. HN 2026-09-21 "Sub-15ms, non-autoregressive, local drop-in alternative to TypeSafe Jev". (VERIFIED)
- **Mechanism:** 395M ModernBERT encoder with OptionMarker representations and direct Choice/Noul/Score logits. Post-hoc calibration. Byte-compatible `/v1/systemone`. Runs on CPU (OpenVINO). (VERIFIED)
- **Numbers:**
  - Self-reported jabr v2: 71.5% macro over 49 tasks / 869 cases.
  - ViZDoom: 9.38 kills vs Jev 5.62. (REPORTED via OrcaRouter)
  - README table: Jev composite 63.3; Laya 30.3.
  - The README honestly states "Out of domain the ranking itself breaks". (VERIFIED)
- **Third-party:** OrcaRouter (Alistair Wren, 2026-09-23) reports Von at 26.7% on commit-type classification and 0.513 AUC on binary tasks out of domain. Its verdict: "well calibrated on the distribution it was trained on and useless on a distribution it has never seen". JevBench: Von Capability 41.7, Intelligence 0.0 (chance-corrected). (REPORTED / VERIFIED)

### 2.17 NanoJev (TianyuCodings)
- <https://github.com/TianyuCodings/NanoJev>, MIT. "A nano replica of Jev". (REPORTED / catalog VERIFIED)
- **Mechanism:** Qwen3-0.6B with dedicated decision heads: set attention for Choice, sigmoid for Noul, ordered levels for Score. Trained on 18,760 questions from Maze, Snake, ViZDoom Basic and Predict Position. (REPORTED)
- **Vs Jev (games, self-reported):** ViZDoom Basic 128/128 vs Jev 56/128. Predict Position 27/128 vs 11/128. Solves a 50×50 maze in 225 attempts vs Jev 2,738. (REPORTED, Pinggy) This is in-domain training against zero-shot Jev (INFERRED).

### 2.18 jevlike (vinnylarouge)
- <https://github.com/vinnylarouge/jevlike>, MIT. HN 2026-09-16 "Reverse-engineered Jev-like model" (169 points). It was one of the earliest. (VERIFIED)
- **Mechanism:** a small scorer in which each option's query vector attends over the encoded context, followed by a shared scorer and softmax. Byte embeddings trained from scratch, or a frozen HF encoder. The user trains it on their own labels. (REPORTED)
- **Numbers:** about 98% on synthetic menus. 26–29% top-1 on Wikispeedia vs 8% shuffled. The chess checkpoint loses 48 of 50 games to Stockfish level 0. Apidog rates it "lowest faithfulness… no noul or score, no calibration claim". (REPORTED)

### 2.19 mini-jev (r-ms)
- <https://github.com/r-ms/mini-jev>, MIT. Frozen Qwen3-4B-Instruct-2507, option-letter logits.
- Preregistered CLINC150 study, 6,750 paired observations: JSON generation 0.909 vs letter readout 0.907 (−0.22 pp, 95% CI [−1.44, +1.04]). About 4× faster.
- Its caveat: letter shares are "a ranking with a confidence gap, not calibrated probabilities". (REPORTED, Apidog)

### 2.20 AnyJev (Nokia Applied Research)
- <https://github.com/nokia-applied-research/AnyJev> and arXiv 2610.00831, "AnyJev Technical Report" (Zhang, Yang, Shi, … Liang Wu), 2026-09-30. (VERIFIED)
- **Mechanism:** restricts the next-token distribution to option tokens on any pretrained LM. Divides out label priors estimated from unlabeled inputs and averages over cyclic rotations of the options to remove position bias. No gradient updates.
- **Numbers:** order-flip rate drops from 0.33 to 0.14–0.18, accuracy improves on 11 of 11 models, and a stopping rule gives 2.2× throughput on vLLM. (VERIFIED abstract) The awesome-jev entry notes "L0 is not calibrated… letter readout supports up to 26 options". (VERIFIED catalog)

### 2.21 CLM-8B (Contrastive-LM; Stanford and NVIDIA per VentureBeat)
- <https://github.com/Contrastive-LM/CLM>. HN 2026-09-25 "Contrastive Language Models: A Fast, Generalizable System One Model". VentureBeat "Stanford and Nvidia's open Jev-like model". The article returned a 404 when fetched. (VERIFIED titles)
- **Mechanism:** contrastive heads (2 × 9.4M) over Qwen3-8B. (VERIFIED, razorback16/openjev README)
- **Numbers:** IMDb 82.90 vs Jev 96.47 (REPORTED, Raschka). JevBench Capability 30.0, rank 53 (VERIFIED).

### 2.22 Jeff (firelex), with a note on a name collision
- <https://github.com/firelex/jeff>. HN 2026-09-28, 573 points, "Jev-compatible 0.8B decision models, trained at home, ~30 ms". (VERIFIED title)
- **Community:**
  - One user reported "70% vs 94%" against Jev on their own use cases.
  - Another: "i've tried all the 'open source' me too Jevs—they all suck" (zergrush).
  - prodigycorp: "Their edge is in their synthetic data".
  - A positive report: a file sorter that "works like a charm".
  - (REPORTED, HN 49883844)
- **Name collision:** ScriptByAI lists a different "jeff" (github.com/logan-markewich/jeff, 400M GLiFormer classifier, MIT). There is also Alurith/jeff, a CLI that uses Jev and is not a clone. (REPORTED / VERIFIED)

### 2.23 Jevstiller (tomerglick57), distillation of Jev itself
- <https://github.com/tomerglick57/Jevstiller>, blog jevstiller.pages.dev. HN 2026-09-29 (66 points). Covered by The Register, "Open-source tool distills Jev so you can run it locally" (2026-09-29). (VERIFIED titles and blog)
- **Mechanism:**
  - Multinomial logistic regression on frozen bge-small embeddings, trained on Jev's full probability outputs. Retrains every 2,000 answers.
  - Clopper–Pearson bound guarantees agreement with Jev of at least X% (for example 98%), with a 2% audit slice kept live.
  - Coverage costs 4–8 pp. (VERIFIED)
- This is the clearest **distillation-from-Jev** analog. No ToS dispute surfaced (section 5).

### 2.24 this-that-model-1.0 (FLock.io)
- arXiv 2609.23886 (Cheng, Dai, Sun, 2026-09-20), HF `flock-io/this-that-model-1.0`. 2B model that reads the hidden state at a designated position. 30.9 ms on a consumer GPU. (VERIFIED abstract)
- **Claim vs Jev:** 0.941 accuracy and 0.042 Brier on a 68-question third-party set, "substantially outperforms" hosted Jev. Weak on multi-step arithmetic (0.560). The sample is very small (INFERRED).

### 2.25 Dyad (academic)
- arXiv 2609.36116 (Zhan, Wang, … Johansson, Yue, 2026-09-28). Adds an environment-conditioned action encoder that embeds candidate actions in parallel and scores them against the LLM's internal state. Trained with a frozen LLM or jointly with RL. +3.80% on ALFWorld with a 9B model. (VERIFIED abstract) It is an academic relative rather than an API clone (INFERRED).

### 2.26 Cua S1 / CUA-S1-FORMS (trycua)
- <https://github.com/trycua/cua>, HF `cua-ai/cua-s1-forms`. Show HN 2026-09-19 "A System One Model for Computer Use" (95 points). MIT research release. A form-filling specialist that scores structured UI decisions. (VERIFIED)
- r/JevAgents post title: "Cua's 2.8MB model beats Jev on form-filling eval". (REPORTED, title only)

---

## 3. Comparison table

Accuracy figures come from different benchmarks and are **not comparable across rows**. "JevBench Cap." is the v1.5.4 Capability Score; Jev 1.13.0 = 80.0.

| # | Analog | Who | Open? | First seen | Relation | Base / size | Mechanism | Self-reported vs Jev | JevBench Cap. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Laya | Convai Innovations | Apache-2.0 | 09-19 | independent, API-compatible; prior-art claim | ModernBERT-large 421M / mmBERT 322M | encoder scoring head, temperature | 0.766 vs 0.727 argmax (Jev numbers not measured) | 23.0 (multilingual) |
| 2 | Kev | Jared Palmer | Apache-2.0 | 09-19/21 | clean-room, API-compatible | Qwen3.5 0.8/4/9B + LoRA | pointer head | Opper: ≈ parity on accuracy, worse calibration | 39.0 (0.6B) |
| 3 | SemIf (ex-OpenJev) | TheoLeeCJ | MIT | 09-18 | interface reimplementation | frozen Qwen3.5-4B etc. | option-logit readout | 0.845 vs 0.883 subset agreement | 67.6 |
| 4 | OpenJev (razorback16) / Codiv | razorback16 | Apache-2.0 + hosted | ~09-17 | API-compatible server | DiffusionGemma 26B-A4B | diffusion canvas slot readout | "+/- a few points" (HN) | n/a; djev 76.4 is another DiffusionGemma rebuild |
| 5 | OpenAI Decisions API | OpenAI | closed | 09-30 | commercial competitor | Luna | undisclosed | none | – |
| 6 | Clef / Clef-flash | Cloudflare | Apache-2.0 + hosted | 10-01 | commercial, API-compatible | Qwen3.8-27B / Qwen3.5-9B | prefill-only parallel choice scoring; Brier + RLCD-like | Banking77 F1 94.2 vs 79.7 | – |
| 7 | Strands Decider 2B | AWS Strands | open | 10-01 | competitor | Qwen2.5-2B + LoRA | pointer head | JevBench 2B class 3rd/33 | – |
| 8 | d1 | Liquid AI | hosted | ~09-29 | commercial, API-compatible path | undisclosed | undisclosed | none | – |
| 9 | GLiNER2.5-Decide | Fastino | Apache-2.0 | 09-24 | competitor | DeBERTa-v3-large 340M | schema multi-head encoder | no Jev number | GLiNER2.5 small 32.5 |
| 10 | Drex 1.5 | Nace.AI | closed | 09-25 | commercial competitor | <10B | one-pass probabilities | DI 58.28 vs 57.91; JevBench 86.2 vs 87.0 | – |
| 11 | Privatemode GLM-5.3-Flash | Edgeless Systems | hosted | 09-24 | commercial, prompt+logit | GLM-5.3-Flash | prefill + logit readout | parity on 28 datasets | – |
| 12 | Nimble | Bespoke Labs | open | 09-18 | open recipe | Qwen3.5-9B + LoRA | candidate-token logits | 90.1 vs 93.2 | – |
| 13 | Jeeves | PostHog | MIT | 09-29 | API-compatible, reasoning | Qwen3.5-9B + LoRA | CoT + pointer head, SFT + CISPO RL | JevBench public 0.935 vs 0.866 | – |
| 14 | Decider | Mark Marosi | Apache-2.0 | ~09-22 | API-compatible | Qwen3.5 2B/4B/35B-A3B | letter logits | DI 47.1 vs 57.9 | 70.7 |
| 15 | JevK5 | Alibi Serikbay | Apache-2.0 | ~09-24 | API-compatible, distilled from Qwen/Luna | Qwen3.5-4B/9B + LoRA | option-logit (SemIf) | 62.04 vs 63.29 (older JevBench) | 72.3 |
| 16 | Von | wfzyx | Apache-2.0 | 09-21 | drop-in API | ModernBERT 395M | OptionMarker heads | jabr 71.5% (own) | 41.7 |
| 17 | NanoJev | TianyuCodings | MIT | ~09-20 | nano replica | Qwen3-0.6B | per-type decision heads | games: beats Jev in-domain | – |
| 18 | jevlike | vinnylarouge | MIT | 09-16 | "reverse-engineered" scorer | bytes or frozen encoder | query-attention scorer | none | – |
| 19 | mini-jev | r-ms | MIT | ≤09-18 | interface study | frozen Qwen3-4B | letter logits | JSON ≈ letters on CLINC150 | – |
| 20 | AnyJev | Nokia Applied Research | open | 09-25/30 | academic + code | any LM | option-token restriction + prior/rotation correction | none | – |
| 21 | CLM-8B | Contrastive-LM (Stanford/NVIDIA per VB) | open | 09-25 | academic | Qwen3-8B + heads | contrastive heads | IMDb 82.9 vs 96.5 (Raschka) | 30.0 |
| 22 | Jeff | firelex | open | 09-28 | API-compatible | 0.8B | – | user: 70% vs 94% | – |
| 23 | Jevstiller | tomerglick57 | open | 09-29 | **distillation of Jev** | bge-small + logistic regression | per-task student with agreement bound | ≥98% agreement with Jev | – |
| 24 | this-that-model-1.0 | FLock.io | open | 09-20 | academic | 2B | hidden-state readout | 0.941 on 68 Qs, beats Jev | – |
| 25 | Dyad | academic (Johansson, Yue et al.) | paper | 09-28 | academic relative | 9B LLM + action encoder | parallel action scoring, RL | n/a | – |
| 26 | CUA-S1 | Cua | MIT | 09-19 | domain-specific competitor | small | form decision scorer | "beats Jev on form-filling" (title) | – |

---

## 4. Published head-to-head results

Ordered roughly by independence from the analog's author.

1. **JevBench v1.5.4** (Benchmark Heaven, third party). (VERIFIED, parsed)
   - Columns: Capability / Intelligence / Calibration.

   | Rank | System | Base | Capability | Intelligence | Calibration |
   |---|---|---|---|---|---|
   | 1 | Jev 1.13.0 | – | 80.0 | 72.0 | 88.0 |
   | 2 | Winnow-12B Q8 "Jev rebuild" | Gemma-4-12B-it | 79.3 | **74.4** | 84.1 |
   | 3 | Cygnet | blockbrain, frozen Gemma-4-12B | 79.0 | 71.1 | 87.0 |
   | 4 | Surogate Rune 26B-A4B v3 | – | 79.0 | 69.7 | 88.3 |
   | 5 | Jev-Omni | – | 76.5 | 70.5 | 82.6 |
   | 6 | djev | Maisa, DiffusionGemma | 76.4 | 72.3 | 80.4 |
   | 7 | JevK5 v0.3 | – | 72.3 | – | – |
   | 11 | decider-4b v2 | – | 70.7 | – | – |
   | 15 | metask-jev-4b | – | 68.1 | – | – |
   | 18 | SemIf | – | 67.6 | – | – |
   | 40 | Von | – | 41.7 | – | – |
   | 43 | kev 0.6B | – | 39.0 | – | – |
   | 53 | CLM-8B | – | 30.0 | – | – |
   | 58 | Laya multilingual | – | 23.0 | – | – |

   - Raw-logit controls: Qwen3-4B-Instruct direct logits 53.5; Qwen3-0.6B direct logits 16.0.
   - Reading: Jev still leads overall. Gemma-4-12B-based rebuilds are within about 1 point, and one beats it on Intelligence. Encoder-only clones score near zero chance-corrected Intelligence.
2. **Opper, Jev vs Kev-4B** (Jose Sabater, 2026-09-25), 362 fresh items.
   - Accuracy is near parity: 96.9/97.5/95.1 vs 95.0/98.3/93.9.
   - Kev is worse calibrated.
   - Jev adds about 257 fixed input tokens per request, making short queries about 12× costlier.
   - (REPORTED)
3. **gazelle93, "decision-models-under-pressure"**: Jev vs Laya vs five open encoders (deberta-base/large, bge-large, gte-large, gliclass). (REPORTED)
   - At K=128: Jev 60%, Laya 39%, best open 41%.
   - Hard distractors: Jev −10% vs Laya −35%.
   - Option shuffle changes the answer on 14.6% (Jev) vs 49.4% (Laya).
   - Jev is non-deterministic on 4.3% of identical calls.
   - Laya's calibration error rises from 0.032 to 0.573 at large K.
4. **Raschka** (2026-09-29): IMDb Jev 96.47, Laya 92.33, CLM 82.90. Judgement: "none of them achieves the same level of performance as Jev on such a breadth of tasks". (REPORTED)
5. **Rafe and Das, arXiv 2610.00346** (2026-09-29): eight decision-model checkpoints from six families, including Jev, vs trained classifiers and LLMs. (VERIFIED abstract)
   - Small trained classifiers win on intent when labels exist.
   - Most decision models beat zero-shot classifiers.
   - A staged cascade (intent model, then Jev) matches accuracy at about 43% of the cost.
   - A 5% risk threshold still let Jev accept 31% of out-of-scope requests.
6. **jabr v2** (Von's own benchmark, cited by Pinggy): Jev 0.966, Von 0.704, GLiNER2 0.698, Laya 0.583. Von's own figure is 71.5%. (REPORTED)
7. **Fastino fast-decisions**: GLiNER2.5-Decide 60.2, JevK5 57.6, SemIf 56.4, Laya Router 46.6. No Jev number. (REPORTED/VERIFIED)
8. **Decision Index 0.2.1**: Drex 58.28, Jev 57.91, Surogate Rune v3 57.44, Decider-chat Gemma 57.33, decider-35b-a3b 47.11. (REPORTED via vendors)
9. **Vendor and self-reported, versus Jev:**
   - Cloudflare Clef: beats Jev on 4 of 5 benchmarks and on latency.
   - Privatemode GLM-5.3-Flash: parity on 28 datasets.
   - Nimble: 90.1 vs 93.2.
   - Jeeves: beats Jev on JevBench public and hard, loses on MMLU.
   - SemIf: 0.845 vs 0.883.
   - Laya: 0.766 vs 0.727 argmax, but loses on soft accuracy.
   - NanoJev: beats Jev on in-domain games.
   - this-that-model: beats Jev on 68 questions.
   - (VERIFIED as claims)
10. **Other harnesses seen, not read:**
    - instax-dutta/sysone-bench ("Laya vs Jev head-to-head on identical inputs").
    - dhruvmehra/jevbench (Jev vs LLMs, fine-tuned BERT, Laya, zero-shot NLI).
    - YidiDev/jev-benchmark (Jev vs Claude Haiku 4.5 vs OpenJev).
    - AbdelStark/jev-benchmarks.
    - A r/machinelearningnews post: "Benchmarked 14 open models, mostly decision models… a naive Bayes baseline…".
    - (VERIFIED titles only)

---

## 5. Controversy, licensing and TypeSafe statements

- **TypeSafe on imitators:** the only quote found is Almeida on OpenAI's Decisions API: "the beginning of the clone wars", and "Intelligence is the hard part". (REPORTED, TechCrunch 2026-09-30)
  - I found no TypeSafe statement about the open clones. HN summaries note no TypeSafe staff replies in the OpenJev, Kev or Jeff threads. (REPORTED)
- **Takedowns:** the github/dmca repo directories `2026/09` and `2026/10` contain no filenames mentioning TypeSafe or Jev. (VERIFIED, filename-level check only; mirror commits found by search returned 403.)
- **Terms of service:** the "Typesafe Terms of Use" on tostracker.app is dated 2024-10-22 and covers website use only. It is very likely the unrelated Scala-era Typesafe/Lightbend site, not TypeSafe AI's API terms (INFERRED). I could not locate TypeSafe AI's API terms on distillation.
- **Distillation provenance:**
  - Projects stating they did not distill from Jev: Nimble and Decider. (VERIFIED)
  - JevK5 distills from Qwen3.6-27B and GPT-6 Luna. (VERIFIED)
  - Jevstiller explicitly distills Jev outputs per task. (VERIFIED)
- **Naming:** SemIf renamed from OpenJev and added a "not affiliated… no infringement is intended" disclaimer. (VERIFIED) Most clones carry similar disclaimers (razorback16 OpenJev, JevK5). (VERIFIED)
- **Laya prior-art dispute:** see 2.1. HN largely rejected the claim, citing target leakage. (REPORTED/VERIFIED rebuttal)
- **Data policy:** an HN commenter (prodigycorp) called TypeSafe's data retention "draconian" and framed the clones partly as "opportunism". (REPORTED)

---

## 6. Long tail: further analogs seen (titles or catalog entries, not read in depth)

From HN titles (VERIFIED as titles):
- Ollaya (ollaya.dev; runs Laya, Kev and Decider behind a TypeSafe-compatible API)
- OpenJev SGLang (ekzhang)
- open-jev (daseinlabs, Gemma 3 4B MLX)
- Gevva0 (Gemma 26B logit scoring)
- Eikos
- Reflex (lateos-ai)
- TinyJev
- Go-System-One (rcarmo)
- Blink (sqliteai, 452 KB C/WASM)
- AT0M (pienomial, 60M Rust)
- Gutsy (0.8B CPU)
- Jeva.cpp (llama.cpp fork with a Jev-compatible API)
- verdict (khimaros, llama-server to System One)
- typed-lm (Rust)
- Bongard-mini (T5)
- Nodd
- Julia-1 (supersoniclabs)
- VisionLaya / Peekaboolean (vision)
- "Trained KV cache bank turns any LLM into Jev like Model"
- "Your Language Model Is Already a Decision Model"
- Lev (yogthos, classifier/LLM escalation)
- Tacet (144M model that "ties Laya")
- vLLM PR #57250 (DiffusionGemma Jev-like mode)
- Parallel Constrained Decoding HF Space (drinkmoonshine)
- "A local alternative to Jev – 94% on Banking77" gist
- OllamaMQ, onesie, Decisions-API aggregators (decisionapi.org was unreachable; decisions-api.dev)

From the awesome-jev catalog (VERIFIED as listings):
- agent-jev (AgentJev-0.6B), AnyDecisionModel (Swift, mattt), cu-Jev (CUDA), JevForge, LitJev, local-jev, metask-jev, minojev (frozen Qwen3-1.7B + head), open-alternative-jev, open-jev-typed-decision-engine (150M), open-spark-jev, OpenJev ×4 (xingwudao, GPT-AGI, zhangcy122, zhihz), openjev-multimodal, PocketJev (iPhone, Qwen3-VL), poorjev, qwen-rlcd, reflex (kshetrajna12), sys1 (Rust, Laya), system-one (sgoedecke), system-one-gemma (Gemma 3 270M), system-one-open (Gemma 4 E2B), Verdict-open-jev / openJev-verdict-2.0 (151M ModernBERT; claims 77.10% accuracy, beating Jev and Laya on "LocalLLaMA/typed-decisions"), jev_local, Open Medical Jev.

From JevBench entries (VERIFIED as listings):
- Winnow-12B, Cygnet, Surogate Rune, Jev-Omni, djev (Maisa), Hopper, Malkuth, Manchego, jqv, spark-s1, Decision 4B (FlyMyJev), Imajev, typecastlm, Nemotron Diffusion 8B, Certo, lev-350m, Deem, decision-machine-1 ("Closed decision API").

From Reddit titles via Pullpush (REPORTED, titles only):
- "Vev: Jev-like decision models with vision — 4B/9B"
- eu/jev (EU-hosted System One, jev.bevel.software)
- "Mercury released Mercury Decide, I benchmarked it" (Inception; unconfirmed by search)
- An unnamed "system 1.5 model" from an Indian startup claiming #1 on Image JevBench
- Smart.NET

## 7. Near-misses (excluded, and why)

- **Jev users, not analogs:** jevals (Openlayer), Jev-Leftpad, jevchat, Jev Ultrafast (browser-use), Jevmem, jev-code-reviewer, typesafe-computer-use, Jeff (Alurith) CLI, Jevopt, Sezwhere. These call Jev.
- **"Jev in 25 Lines of Python"** (nobodywho.ai, Duarte O. Carmo, HN 691 points): a parody or demo of logit readout on Qwen3-0.6B via llama-cpp, not a maintained project. It is useful as the canonical statement of the "you don't need Jev" position.
- **"A single function Jev-like wrapper for LLMs"** (allanrbo blog) and **"Jev can't be calibrated"** (Alex Molas): commentary, not products.
- **GPT-6 Luna / Astra:** general LLMs used as comparisons. Only the Decisions API counts as an analog.
- **The tostracker "Typesafe" ToS:** a different company (see section 5).
- **Davout:** excluded by instruction.

## 8. Community consensus (synthesis, INFERRED from the sources above)

1. **"The interface is easy, the model is not."**
   - Nearly every source says the clones copy the `/v1/systemone` contract and a logit or pointer readout, not Jev's undisclosed model, data or RLCD.
   - The HN view is that Jev's edge is synthetic data and productisation. Comments: "Their edge is in their synthetic data"; "Ideas are cheap; execution including marketing is what matters".
2. **Accuracy:**
   - On easy, in-distribution classification, 4B–12B open rebuilds are close to Jev: Opper Kev parity, Privatemode parity, the JevBench top three within about 1 point.
   - On breadth, hard items, large option sets, knowledge (MMLU) and robustness to distractors and reordering, Jev still leads: Raschka, gazelle93, JevBench Intelligence, Decision Index.
   - Small encoder clones (Laya, Von, Verdict) are fast but collapse out of domain.
3. **Calibration is the main differentiator and the main point of contention.**
   - Jev scores high on JevBench calibration, and Opper found Kev drifts.
   - Molas argues no model can be calibrated on *your* distribution.
   - Several clones ship temperature fitting for exactly that reason (SemIf, Laya, Davout-style).
4. **Speed and cost are not a moat.** Local clones match or beat Jev's latency, while Jev's fixed per-request token overhead makes short calls relatively costly.
5. **Vendor benchmarks conflict.**
   - Clef, Drex, Jeeves, NanoJev and this-that-model each report beating Jev on chosen sets.
   - Independent sets (JevBench, gazelle93, Raschka) do not confirm a general win.
   - The arXiv audit (2609.32160) concludes the typed readout itself shows no independent accuracy advantage.

## 9. Search log (for reproducibility)

- **WebSearch:**
  - "Typesafe AI Jev typed decisions"
  - "Jev" "systemone" API typesafe
  - OpenJev open source Jev alternative
  - jabr classifier benchmark…
  - Jev TypeSafe site:news.ycombinator.com
  - reddit LocalLLaMA Jev clone
  - TypeSafe Almeida clones response
  - TypeSafe ToS distillation takedown
  - OpenAI Luna decision API
  - Mercury Decide
  - "system 1.5" Jev
  - Product Hunt Jev alternative
  - bsky Jev clone
- **HN Algolia API** (stories and comments since 2025-09-01): jev, typesafe, openjev, systemone, semif, jev-like, jev-compatible, decision model, laya, von jev, jev alternative, jev distill.
- **arXiv API:** the "typed decision" query returned 20 papers from Sep–Oct 2026. Other queries were empty or rate-limited. Read abstracts: 2610.00831, 2609.36116, 2609.35865, 2609.23886, 2610.00346, 2609.32160, 2609.33843, 2609.30216.
- **Reddit:** direct fetch was blocked. Pullpush gave one page of titles, then returned a rate-limit notice saying it does not provide free scraping for agents, so I stopped using it. r/LocalLLaMA threads (including the Laya "I literally built the Jev architecture" post) were **not read**.
- **Not covered directly:** X/Twitter, Bluesky, Mastodon, Discord, Product Hunt pages, YouTube descriptions, Semantic Scholar, Papers with Code. They were seen only through secondary references.
- **Unreachable:** decisionapi.org (DNS), VentureBeat CLM article (404), The New Stack article body (not in fetched content), pasqualepillitteri.it (socket hang-up).
