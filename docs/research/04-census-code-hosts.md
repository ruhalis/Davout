# Census of Jev clones and imitations on code and model hosts

Researched 2026-10-02. The venues are GitHub, Hugging Face (models, datasets and Spaces), PyPI, npm, crates.io, the Ollama library, GitLab and Codeberg. Kaggle turned up nothing.

**Scope.** A project counts if it (a) explicitly ties itself to Jev, TypeSafe AI or the `/v1/systemone` wire format, or (b) clearly reproduces Jev's interface: Noul, Choice and Score questions over a state, answered with probabilities and no generated text. Projects that only call the real hosted Jev API (clients, SDKs, agents, plugins) are excluded; there are hundreds of them. Davout is excluded. **OpenJev** (`AlexWortega/openjev`) and **Kev** (`jaredpalmer/kev`) are covered in depth elsewhere (see `02-openjev.md` and `03-kev.md`) and only listed here.

**Evidence labels.** **VERIFIED** means the mechanism was read in code, a model card, `config.json` or a repo doc. **REPORTED** means only the README or tagline was read. Where a section says "model card", the claim is the author's own and was not run or reproduced. Nothing was installed or run. Benchmark numbers are author-reported unless noted.

## Headline

- **The ecosystem is far beyond 10 projects.** This census documents **64 entries (63 distinct projects) in detail**, not counting OpenJev and Kev. The entry gap is because `lawrence3699/jev-style` packages chaoliangUNSW's Jev-Style models. One more project (GLiNER2.5-Decide) was examined and excluded. Every one was created between 2026-09-16 and 2026-10-01, in the two weeks after Jev's announcement on 2026-09-15.
- **Two independent trackers confirm the scale.**
  - The *Jev Decision Index* Space (`huggingface.co/spaces/multimodalart/jev-decision-index`, 372 likes) benchmarks 70 open entries against Jev 1.13 on 42 tasks.
  - The `AnotiaWang/awesome-decision-models` list (597 stars) catalogues about 30 open models and inference techniques.
  - Projects named in those trackers but not documented here are listed under "Further leads" at the end.
- **"OpenJev" is not one project.** At least six unrelated projects use the name:
  - `AlexWortega/openjev`, the NLI cross-encoder that is Davout's baseline;
  - `openjev/openjev`, a 27B letter-readout model from a different author, CC-BY-NC;
  - `TheoLeeCJ/SemIf-OpenJev`, renamed SemIf;
  - `razorback16/openjev`;
  - `ekzhang/openjev-sglang`;
  - `ejhshen/OpenJev`;
  - `zhihz/openjev`.

  Name collisions also exist for "AnyJev" (Nokia's project versus llm2jev's old name) and "Jeff" (Gestalt-Lab versus the GLiFormer entry in Von's table).
- **An HF org named "TypeSafeAI" is not verifiable as the vendor.** `huggingface.co/TypeSafeAI` is unverified, and its contents are re-uploads (a StepFun model, a copy of the Open-Jev dataset) plus Qyvos, a head-only fine-tune. Treat it as a probable name-squat.

## Families of approach

1. **Logit readout over option letters from a decoder LLM, with no training ("wrappers").** These format the state, question and lettered options as a multiple-choice prompt, run one prefill and softmax the next-token logits over the letter tokens. Most add temperature calibration, option-order rotation or averaging, and shared-prefix KV reuse for many questions over one state.
   - Members: AnyJev, SemIf, openjev-sglang, simple-jev, cygnet-recipe, jev-rs, llm2jev, Lichen, mini-jev, sarvam-jev, zhihz/openjev, reflex, Meanblock/JEV-CPU.
   - Variants: LitJev scores the teacher-forced log-likelihood of the option text instead of letters; razorback16/openjev reads answer slots from a *diffusion* LM canvas (DiffusionGemma).
   - This is Davout's own approach.
2. **The same letter or label readout after a LoRA or full fine-tune.** These keep the LM head and train on public classification and QA sets, synthetic questions or teacher distributions.
   - Members: decider, WebJev, OneJev, Vev, eikos, StartLux, JevK5, Nimble, rizzo-flow, snap, StandardOne, Winnow, APUS-OpenJev, openjev/openjev, Jeff, Jev-Style/jev-style (per-option yes-minus-no at a verdict slot), jevos (yes/no logits only) and Tev1. Tev1 actually generates one letter.
3. **The LM head replaced by a decision head on a decoder backbone.** The variants are:
   - a pointer head comparing the answer-position hidden state with each option's last token (Kev, Strands Decider, NeoHorse-Jev-4B, tinyjev);
   - a per-candidate scalar head (Open-Jev, pngwn scorer);
   - a set or permutation-equivariant head (NanoJev, agent-jev, mini-Jev, ejhshen OpenJev, RSI-Jev);
   - a fixed 24 to 256-way "slot" or verbalizer head (AutoTrust JEV, OpenThai-SystemOne, pplx-decider, Matilda, imajev, Jev-Omni);
   - a joint schema head (Cloudflare Clef).
4. **Encoder (BERT-family) classifiers.** One pass packs the state with all options, and option-slot or [MASK]-marker logits give the distribution. Members: Laya (ModernBERT, the most-starred clone, with a large downstream runtime ecosystem), Von, openJev-verdict-2.0, open-jev-deberta-v3-large, bekko, Qyvos, and stuntd (per-site heads distilled onto Laya).
5. **Cross-encoder, NLI, embedding and late-interaction scoring per option.** One pass per option, with no joint context across options:
   - OpenDecision (zero-shot NLI), modernbert-ja-310m-jev (cross-encoder), JevEmbed (cosine similarity) and tasksource-jev-nano (ColBERT MaxSim);
   - OpenJev (AlexWortega) also belongs here.
6. **Distillation from Jev itself.** This is rare. AutoTrust JEV-9B/27B is the only clear case: 498k rows of Jev 1.13 output distributions bought through OpenRouter. stuntd distils the user's own Jev, OpenAI or Anthropic traffic into local heads. jevos evaluates against Jev labels, but its training provenance is unclear. Most others explicitly say they had no Jev access or used other teachers (Qwen3.6-27B, GLM, GPT-6 Luna).

The **lineage between clones** is visible in several places:
- NeoHorse ships Kev's runtime and pointer head;
- tinyjev reproduces Kev, and its v2 follows Tev1;
- JevK5 borrows SemIf's readout;
- Meanblock/JEV-CPU is a CPU port of SemIf;
- WebJev reuses decider's recipe;
- sys1, stuntd and several servers (Decis, laya-server, laya.cpp, laya-rs, laya-mlx and laya-coreml) build on Laya;
- sarvam-jev vendors SemIf fixtures;
- Qyvos trains on the Open-Jev dataset.

**About 85% of the projects ship a `/v1/systemone`-compatible server.**

## Comparison table

| Name | Base model | Mechanism | Trained / wrapper | License | Status |
|---|---|---|---|---|---|
| *OpenJev (AlexWortega), known* | Qwen3.5-0.8B/2B/4B/35B-A3B | NLI cross-encoder, one pass per option | trained | MIT | active (see 02-openjev.md) |
| *Kev (jaredpalmer), known* | Qwen3.5/3.8 0.5B-27B | LoRA + pointer head | trained | Apache-2.0 | active (see 03-kev.md) |
| Laya | ModernBERT-large 421M / mmBERT-base 322M (encoders) | encoder option-slot logits, one pass; RLCD-trained | trained (RL with proper scoring rules); also Jev-compatible server | Apache-2.0 | active |
| Open-Jev (Zefan Cai) | Qwen3.5-2B / 9B / 27B + LoRA r8 | per-candidate scalar head on last-token hidden, softmax over candidates | trained (synthetic/public rows, 80k-149k) | MIT code / Apache-2.0 adapters | active |
| APUS-OpenJev-v1 | Qwen3.5-4B / 9B / 35B-A3B | LM-head logits over short candidate labels; selectable early-exit depth | trained + /v1/systemone gateway (typesafe-sdk compatible) | MIT (own) / Apache-2.0 | abandoned/unclear (last 2026-09-23) |
| AutoTrust JEV-9B / 27B | Qwen3.5-9B / Qwen3.8-27B + LoRA r16 | 24-slot verbalizer head, single prefill over A-P/0-5/false-true tokens | distilled from Jev 1.13 output distributions (498k rows) | Apache-2.0 | active |
| NeoHorse-Jev-4B | NeoHorse-1-4B (Qwen3.5-4B lineage, multimodal) | pointer decision head over options, prefill-only; Kev-derived runtime | trained (data not stated) + /v1/systemone endpoint | Apache-2.0 | recent (2026-09-28) |
| Jev-Omni | Gemma-4-12B-it | classifier head (<=256 options, tested <=20); text/image/audio/video | fine-tuned on 30k questions (data not stated) | Apache-2.0 | abandoned/unclear (2026-09-25) |
| open-jev-deberta-v3-large | DeBERTa-v3-large | one-pass encoder, 3-layer span head over option pooled tokens, softmax per question | trained on public gold labels (Banking77, SST-5, BoolQ) | Apache-2.0 | active |
| Jev-Style Decision (chaoliangUNSW) | Qwen3.5-2B (v3), 0.8B | per-option yes-minus-no logit at verdict slot, block attention, 25.6k ctx | full fine-tune, 181k rows, 58 sources + /v1/systemone server | Apache-2.0 | active |
| JevK5 | Qwen3.5-4B / 9B / 2B + LoRA r16 (merged) | answer-letter next-token logits (SemIf readout), knockout for >16 options | distilled from Qwen3.6-27B and GPT-6 Luna + 26 public train sets; /v1/systemone server | Apache-2.0 | active |
| OpenThai-SystemOne | Qwen3.5-0.8B-Base (Thai CPT) | 256-way slot head at answer tokens (abstain slot); Ollama build uses letters | trained (~2-3M public+synthetic examples) + /v1/systemone server | Apache-2.0 | active |
| NanoJev | Qwen3-0.6B | per-candidate encoding + attention set head, softmax (Choice); sigmoid (Noul) | trained (own game data) | MIT | slowing (last commit 09-21) |
| Von | ModernBERT-large 395M | [MASK] option markers + scoring head, order-invariant attention | trained, encoder; /v1/systemone server | Apache-2.0 | active |
| decider | Qwen3.5-0.8B/2B/4B/35B-A3B-Base, Gemma-4-12B | option-letter logit readout at answer slots, fitted temperature | trained (public data + 27B teacher); /v1/systemone server | Apache-2.0 | active |
| openJev-verdict-2.0 | ModernBERT-base 149.6M (GLiClass) | [MASK] marker pointer head + separate MLP correctness head | trained (typed-decisions soft targets) | Apache-2.0 (GitHub: NOASSERTION) | dormant (09-20) |
| imajev | Qwen3.5-2B/4B/9B (VLM) | 255-code option readout + trained `unknown`; LoRA | trained (~1M decisions, open teachers); /v1/systemone superset with images | Apache-2.0 | active |
| RSI-Jev | Qwen3.5-2B/0.8B-Base | option scorer + cross-attention MLP readout + confidence head; RL stage | trained (agent-driven research loop); /v1/systemone server | MIT code / Apache-2.0 weights | active |
| eikos | Qwen3.5-4B, Qwen3.8-27B | option-letter logit readout (up to 588 options), LoRA merged | trained (GLM teacher, finance focus); /v1/systemone server | MIT | active |
| StartLux-Decision | Qwen3.5-family (6 sizes 0.8B-35B-A3B; base not named) | per-question option-letter logit readout | trained (data mostly not described); /v1/systemone server | Apache-2.0 code / CC BY-NC 4.0 weights | active |
| tinyjev | Qwen3-0.6B/4B-Base, Qwen3.5-4B (v2) | pointer decision head (Kev design); letter logits via Ollama (v2) | trained (Kev/Tev1 data); /v1/systemone server | MIT | active |
| Nimble | Qwen3.5-9B | one-token answer codes, logit softmax; LoRA | trained (2.7k synthetic contrastive); hosted API | none on GitHub / Apache-2.0 weights | slowing (09-24) |
| jevlike | byte encoder from scratch or frozen Qwen2.5-0.5B | option-query attention over context, softmax (Choice only) | trained starter, no API | MIT | abandoned (single push) |
| Strands Decider | Qwen3.5-2B-Base | pointer head (hidden at `<answer>` vs option last token), LoRA | trained (public + generated, 4B teacher KL); /v1/systemone server | Apache-2.0 | active |
| OneJev | Qwen3.5-0.8B/4B/9B, Qwen3.8-27B (multimodal) | option-letter logit readout, CE+Brier | trained (99k public Qs); /v1/systemone server | Apache-2.0 | active |
| Vev | Qwen3.5-4B/9B (VLM) | answer-token logits (Yes/No, letters, digits), LoRA + KL anchor | trained (42 public sources); /v1/systemone server | Apache-2.0 code / CC BY-NC 4.0 weights | active (new) |
| jevos | ~1B (likely MiniCPM5; not stated) | yes/no "0"/"1" logits only; choice/score via per-option/per-level yes/no | trained (provenance unclear, some Jev labels in eval); /v1/systemone C++ server | MIT | active |
| WebJev | Qwen3.5-35B-A3B-Base | decider recipe: option-letter logits, full FT (experts frozen) | trained (3.2M records incl. 64k web); /v1/systemone server | Apache-2.0 | static (one-day push 09-29) |
| reflex | Qwen3.5-4B frozen (also 0.8B/2B/27B) | option-letter logits, 2 option orders averaged | wrapper, no training in shipped config; /v1/systemone server | MIT | dormant (09-22) |
| jev-style | Qwen3.5-0.8B/2B | per-option " yes"-" no" logit at verdict slot, block attention | trained (181k rows, 58 sources); /v1/systemone server | Apache-2.0 | low activity (09-27) |
| Jeff 1 | Qwen3-4B-Instruct-2507 | LM label-token probability readout (first-token or whole-sequence), LoRA | trained (human fact-check labels); /v1/systemone server | Apache-2.0 | abandoned (09-19) |
| AnyJev | Qwen3 (1.7B-32B); Qwen2.5-7B in serve example | letter readout, rotation-averaged (L0), temperature (L1), closed-form hidden-state head (L2) | wrapper; L2 fits small heads on 100-300 labels | Apache-2.0 | active |
| SemIf (OpenJev) | Qwen3.5-4B (27B EXL3, MiniCPM5-2B, Qwen3-0.6B variants) | logit readout over option letters A-P, prefix reuse, per-workload temperature | wrapper | MIT | active (slowing) |
| openjev-sglang | Qwen3.6-35B-A3B NVFP4 on SGLang | N+1 one-token calls, label logprobs over letters, up to 64 options | wrapper | not stated | superseded by SGLang native endpoint |
| razorback16/openjev | DiffusionGemma 26B-A4B (also hosts Laya, Verdict, CLM, JevK5) | diffusion-canvas read of answer slots, entropy-gated re-reads | wrapper (other models bundled are trained by others) | Apache-2.0 | active |
| simple-jev | Gemma-4-26B-A4B (demo), Qwen3.5/3.6/3.8, Gemma-4-12B; Laya backend | prefix-KV letter-logit readout, per-architecture prompt policies | wrapper; RFDT distillation scripts included | Apache-2.0 | active |
| rizzo-flow | Spark-X2.5-4B (1.7B option) + own LoRA | letter logits on llama.cpp, abstain slot, temperature scaling | fine-tuned (LoRA, 28k public questions) | Apache-2.0 | active (slowing) |
| mini-jev | Qwen3-4B-Instruct-2507 | letter-logit read vs grammar JSON study | wrapper (measurement study) | MIT | dormant |
| sarvam-jev | sarvam-1 (2B base) | letter logits with shared-state KV | wrapper (LoRA calibration head only planned) | not stated | dormant |
| cygnet-recipe | Gemma-4-12B-it on vLLM 0.30.0 | masked letter logprobs, one temperature (T=3.4) | wrapper | MIT | active |
| jev-rs | any GGUF/OpenAI-logprob backend (Qwen3-4B in examples) | n_probs letter logprobs, per-bucket temperature, rotations, ensemble stacker | wrapper (also Jev API client) | Apache-2.0 | active |
| OpenDecision | ModernBERT-large-zeroshot-v2.0 (DeBERTa-v3-large optional) | zero-shot NLI entailment per option (HF pipeline) | wrapper over existing NLI model | Apache-2.0 | active (preview) |
| snap | snap1-2b (MiniCPM5-2B fine-tune); Qwen3.8-4B, Spark-4B | letter logits, shared-prefix batched decode, per-option probes beyond 26 | fine-tuned (default) plus wrapper | MIT | active |
| sys1 | Laya (ModernBERT encoder by Convai) | encoder classifier head, Rust/candle server | serving runtime for others' trained model | Apache-2.0 | active |
| stuntd | frozen Laya encoder + per-site heads | distils provider/Jev answers into local heads, confidence-gated fallback | trained heads (distillation) | Apache-2.0 | active |
| llm2jev | any chat model (Qwen3.5-2B example; Qwen3.5/3.6/3.8 tested) | one-position label logprobs after Answer: prefill, global temperature | wrapper | MIT | low activity |
| LitJev | Qwen3.8-27B default (Qwen family) | teacher-forced option-text log-likelihood (letters optional) | wrapper (experimental System Two head) | Apache-2.0 | active |
| Lichen | Gemma-4-26B-A4B (QAT/NVFP4) | label softmax, repeated state, doubled rotated options, temperature + disagreement shrink | wrapper | MIT | active |
| OpenJev (ejhshen) | Qwen3.5-4B backbone + Option Set Interactor head | trained set-wise decision head, permutation-equivariant | trained (SFT + REINFORCE-A) | MIT | active (new) |
| openjev (zhihz) | Qwen3-4B-Instruct-2507 | single next-token label softmax, 2-8 candidates | wrapper | not asserted | dormant |
| pngwn/system-one-qwen3.5-4b-scorer | Qwen3.5-4B-Base | per-option sequence-classification scalar head, softmax over options | trained (LoRA + head, 12.9k questions) | CC-BY-NC-4.0 | inactive since 2026-09-16 |
| openjev/openjev | Qwen3.8-27B (per NOTICE) | option-letter logit readout at first token + temperature calibration; /v1/systemone shim | trained + wrapper | CC-BY-NC-4.0 (code Apache-2.0) | active (2026-09-30) |
| Meanblock/JEV-CPU | Qwen3-0.6B | SemIf letter-logit readout on CPU | CPU port of SemIf, no training | MIT | one-shot, 2026-09-19 |
| aimeigaoshou/agent-jev | Qwen3-0.6B (no LM head) | permutation-equivariant candidate head, softmax per question | trained (SFT + RLCD, coding completion) | Apache-2.0 | active (2026-09-26) |
| samatv256/mini-Jev | Qwen3-0.6B (NF4) | LoRA + set-style decision head, per-option scores | trained (interim, 10.6k/50k steps) | Apache-2.0 | active, mid-training (2026-10-01) |
| TypeSafeAI/Qyvos | SupersonicLabs/Julia-1 (ModernBERT-small, 140M) | frozen encoder + 3.7M head, per-option scorer | head-only fine-tune on Open-Jev data | not stated | new, 2026-09-30 |
| Maincode/matilda-jev-v1 | Qwen3.5-style 26.1B VLM | 255-option readout replacing LM head; /v1/systemone server | trained (data not described) | Apache-2.0 | active (2026-10-02) |
| hotchpotch/bekko-system-one-v0-400m | Ettin reranker 400M (ModernBERT) | shared-prefix encoder, pooled candidate heads | trained (8.4M judgments, full FT) | not finalized | active, v0 (2026-09-30) |
| tasksource/tasksource-jev-nano-v0 | LateOn (ModernBERT ColBERT, 149M) | late-interaction MaxSim, learned temperature | trained (512k decisions, no teacher) | Apache-2.0 | inactive since 2026-09-28 |
| argos1111/modernbert-ja-310m-jev | modernbert-ja-310m | cross-encoder score per candidate, softmax; /v1/systemone via Jev Local | trained (94k Japanese questions) | CC-BY-SA-4.0 | inactive since 2026-09-19 |
| HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B | Qwen3-Embedding-0.6B | embedding cosine similarity to options, softmax; noul sigmoid | LoRA-trained (1.6M questions) | Apache-2.0 | inactive since 2026-09-26 |
| Cloudflare/clef | Qwen3.8-27B (+vision) | joint schema transformer head over final hidden states, per-question softmax | trained; Jev/SystemOne API-compatible | Apache-2.0 | active (2026-10-01) |
| perplexity-ai/pplx-decider-v1-27b | Qwen3.8-27B | 255-option readout replacing LM head, T=2.21; TypeSafe-API server | full fine-tune (73k rows); wrapper server | Apache-2.0 (code MIT) | active, new (2026-10-01) |
| togethercomputer/Tev1-4B-experimental | Qwen3.5-4B | LM head emits one option letter (generative) | LoRA SFT (37.8k examples), not Jev runtime | not stated | one-off, 2026-09-23 |
| StandardThinking/StandardOne-8B | Ministral-3-8B-Instruct-2512 | letter logits via SGLang + per-type temperature; jev-adapter /v1/systemone | LoRA merged (520k rows) + wrapper | Apache-2.0 | active (v2, 2026-09-26) |
| EldanRing/Winnow-12B | Gemma 4 12B IT | answer-token logits, shared-prefix forks, llama.cpp /v1/systemone | LoRA merged (private data, teacher unnamed) | Apache-2.0 | active (2026-09-21) |

---

## Group A: trained models, Hugging Face first

### Laya
- **URL:** https://github.com/NandhaKishorM/laya ; https://huggingface.co/convaiinnovations/laya-typed-decisions (family: laya, laya-multilingual, laya-typed-decisions) **Author:** Nandakishor M / Convai Innovations **License:** Apache-2.0 **Popularity:** GitHub 29,947 stars; HF laya-typed-decisions 140 likes (per brief) **Dates:** first 2026-09-18 (repo created) / latest 2026-10-01 (last commit)
- **Relationship to Jev:** API-compatible wrapper plus clean-room model. `laya.serve` exposes `POST /v1/systemone` ("Jev-compatible"); no Jev outputs used for training (README says there is no TypeSafe API access). The typed-decisions fine-tune uses a "teacher" from the LocalLLaMA/typed-decisions benchmark, not Jev.
- **How it works:** Non-autoregressive encoder classifier, no generation. `laya` = ModernBERT-large (421M, 512 ctx); `laya-multilingual` = mmBERT-base (322M, up to 8,192 ctx); `laya-typed-decisions` = ModernBERT-large fine-tune (1024 ctx). State, questions and options are packed into one sequence; options occupy "slots" in a fixed head budget (192 tokens English, 256 multilingual) and a per-slot logit gives the distribution, so a whole question set costs one forward pass. noul = two slots [false,true]; choice = N slots; score = ordinal slots, expected level returned. Router picks English vs multilingual checkpoint by script/language detection. Confidence = 1 minus normalised entropy (differs from Jev's formula); `answer_confidence` = max p; per-type temperatures fitted post hoc. Training: RLCD (reinforcement learning with strictly proper scoring rules: log + spherical + RPS, REINFORCE with group-mean baseline, plus soft cross-entropy to teacher distributions). Fine-tune on 1,200 cases / 6,000 decisions, ~4-5 h on 2xT4. Weak with more than ~20 options (Banking77: 0.425).
- **Benchmark claims:** Author-reported, typed-decisions test (400 cases, 2,000 decisions): accuracy 0.766 vs Jev 1.13.0's published 0.727, Brier 0.062 vs 0.148, ECE 0.213 vs 0.144, soft acc 0.471 vs 0.580 (Jev ahead). Base `laya` is only 0.362 on the same set. Routed vs Jev (third-party published Jev numbers): AG News 0.950 vs 0.910, DAIR Emotion 0.595 vs 0.480, Banking77 0.425 vs 0.870 (Jev leads). 33 ms single-question latency on T4. Other authors rank it low: 58.4% on JevBench v1.4.1 public (cited in Jev-Style card); 90% of Jev on decision-models-under-pressure (cited in AutoTrust card).
- **Status:** active (last commit 2026-10-01, v0.3.23, 21 contributors).
- **Evidence:** VERIFIED (mechanism from README and laya-typed-decisions model card; training recipe from card; encoder/slot design read from README, code not inspected).

### Open-Jev (Zefan Cai)
- **URL:** https://github.com/Zefan-Cai/Open-Jev ; https://huggingface.co/ZefanCai/Open-Jev-9B (also Open-Jev-2B, Open-Jev-27B-v1.1; datasets ZefanCai/Open-Jev, Open-Jev-v1.1) **Author:** Zefan Cai **License:** code MIT; adapters Apache-2.0 (upstream Qwen terms apply) **Popularity:** GitHub 384 stars; HF Open-Jev-9B 48 likes (0 downloads shown) **Dates:** first 2026-09-20 / latest 2026-10-02 (last commit; HF 9B card modified 2026-09-20)
- **Relationship to Jev:** clean-room reimplementation. README: "independent implementation inspired by TypeSafe's Jev"; serves an official-shaped `/v1/systemone` (partial compatibility); trained on own synthetic/public data, not Jev outputs.
- **How it works:** Qwen3.5 backbone (2B, 9B; 27B v1.1) with a rank-8 LoRA plus a single `Linear(hidden,1)` scalar head initialised from the Yes-minus-No readout. Candidate scoring: each candidate gets its own input `candidate_prompt(state, question, candidate)`; last-token hidden state goes through the scalar head; softmax across the K candidates of one question. noul = one input, logits [0,s], sigmoid(s); score = rubric levels as candidates, expected index. K candidates cost K forward passes (255 options = 255 sequences), 4,096-token cap per candidate (16K for JevBench runs); optional prefix cache. One saved temperature (9B: 1.897, fitted on 512 rows). Training: 9B one pass over 80,816 rows of "release-v2" (13 synthetic/public task sources: painting geometry, Snake, security incidents, ViZDoom, reasoning controls...), 20,204 steps. 27B v1.1 on 148,639 rows.
- **Benchmark claims:** Author-reported. 9B full held-out: test 97.54% hard accuracy, OOD 91.97%, ECE 0.0077/0.037 (own synthetic splits, no base baseline). JevBench public 231: 2B 150/231, 9B 179/231 (77.49%), 27B v1.1 197/231 (85.28%) vs Jev 200/231 (Jev ahead by three). Latency vs Jev-1.13.0: 85 ms local vs 295 ms API on customer-service case, but slower (1016 ms vs 301 ms) at 1024 tokens / 32 candidates.
- **Status:** active (commit 2026-10-02; stated: new 2B training stopped, 9B retraining not started).
- **Evidence:** VERIFIED (model card, docs/multidomain-training.md in repo).

### APUS-OpenJev-v1
- **URL:** https://huggingface.co/apus-ailab/APUS-OpenJev-v1 (family: 4B, 9B, 35B-A3B, GGUF/MLX variants) **Author:** APUS AI Lab (gumpcheng, zhangxu) **License:** original contributions MIT; Qwen-derived weights Apache-2.0 **Popularity:** 26 likes, 127 downloads **Dates:** first 2026-09-21 / latest 2026-09-23 (HF modified)
- **Relationship to Jev:** API-compatible wrapper plus own model. A "TypeSafe deployment package" serves `POST /v1/systemone` for the official `typesafe-sdk==0.7.0`, with an extra `effort` field. Not stated whether Jev outputs were used in training.
- **How it works:** Qwen3.5-4B / 9B / 35B-A3B decision models with selectable compute depth: `effort=high` runs the full 32-layer 9B, `low` an earlier-exit 16-layer export trained jointly (full path gives a distribution-level signal to the short path). Task plus context plus 2-16 candidate descriptions are mapped to request-local short labels; the runtime reads the language-model output logits for those label tokens (candidate-row projection of the LM head on the low path) and returns scores. Probabilities are explicitly "not calibrated confidence". Training data not stated beyond "reference decisions"; eval set is Frozen80 (Mind2Web, HelpSteer3, BoolQ, MNLI, GoEmotions). Multi-question calls scored separately.
- **Benchmark claims:** Author-reported, own 80-question frozen panel: 35B-A3B 88.75%, 9B 85.0%, 4B 82.5%, Jev API 82.5% (Jev 3-round pooled 82.92%), Laya 68.75%. 1,000-question expanded dev panel: 35B-A3B 82.2%, 9B 81.1%, 4B 80.5%, Jev 77.0%, Laya 50.5%. 9B vLLM P50 25.58 ms vs Jev API 280.8 ms. Card says differences are not statistically significant and the panel was used for model selection.
- **Status:** abandoned/unclear (no updates after 2026-09-23, 9 days before today; sub-model repos not checked).
- **Evidence:** VERIFIED (model card text; deployment gateway described in README, code not inspected).

### AutoTrust JEV-9B / JEV-27B
- **URL:** https://huggingface.co/autotrust/JEV-9B ; https://huggingface.co/autotrust/JEV-27B ; dataset https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3 **Author:** AutoTrust AI **License:** Apache-2.0 **Popularity:** 9B 2,007 downloads / 18 likes; 27B 785 / 23 **Dates:** 9B first 2026-09-23 / latest 2026-09-26; 27B first 2026-09-25 / latest 2026-10-01
- **Relationship to Jev:** distillation from Jev outputs. Card: student of TypeSafe Jev 1.13; 498,010 `yuri_v3` rows carry Jev 1.13's full output distributions collected via OpenRouter. Does not claim /v1/systemone compatibility in the text read (vLLM completions endpoint with a custom prompt).
- **How it works:** "Blocks of Experts": frozen Qwen3.5-9B (27B: Qwen3.8-27B) as System 2 plus a detachable System 1 block: LoRA r=16 on decoder projections (40.2M params) and a 24-slot fp32 decision head initialised from `lm_head` rows for verbalizer tokens (false/true, 0-5, A-P). Prompt: `[kind] k\n[state] ...\n[question] ...\n[options]\nA) ...\n[decision]:`; one prefill, `max_tokens=1`, read the slot logits. noul / choice (2-16 options) / score (0-5). Trained to match teacher distributions (KL); per-kind temperatures ~1.0 (noul 1.002, choice 0.984, score 1.012). Corpus 740,957 rows: Jev-labelled 498k, Open-Jev programmatic labels 94.8k (CC0), 148k placeholder rows down-weighted. ~3 B200-hours, 0.93 epoch.
- **Benchmark claims:** Author-reported on own held-out `test_set_30k`: mean KL to Jev 0.019 (9B) / 0.017 (27B), noul AUROC 0.994, choice top-1 agreement 90.2%, score MAE 0.103, ECE 0.0007. On decision-models-under-pressure (16 options): 90% of Jev accuracy (9B), 96% (27B). HumanEval 70.7% (System 2, identical to base). Single decision ~90 ms on B200 vs hosted Jev 238-301 ms (not like-for-like). Not run on Decision Index or JevBench.
- **Status:** active/recent (27B updated 2026-10-01).
- **Evidence:** VERIFIED (JEV-9B model card incl. training table and quickstart code).

### NeoHorse-Jev-4B
- **URL:** https://huggingface.co/TokenRhythm/NeoHorse-Jev-4B ; code https://github.com/TokenRhythm/NeoHorse/tree/main/jev **Author:** TokenRhythm **License:** Apache-2.0 **Popularity:** 2,259 downloads / 25 likes **Dates:** first 2026-09-23 / latest 2026-09-28 (HF); NeoHorse GitHub last commit 2026-09-26
- **Relationship to Jev:** API-compatible wrapper plus own model, and a derivative of Kev's runtime. Native HTTP service has a "System One-style" `/v1/systemone` endpoint. `model_manifest.json` lists `runtime_origin: https://github.com/jaredpalmer/kev` and a "pointer_head". No claim of training on Jev outputs; training data not stated.
- **How it works:** Finetune of NeoHorse-1-4B (itself derived from Qwen3.5-4B per the license note), unified multimodal backbone (Qwen3.5-4B vision tower kept, not fine-tuned) plus a separate float32 pointer decision head (`pointer_head.safetensors`, head_dim 256). Prefill-only: question/option text is packed into one sequence, option positions are pointed to by the head, softmax over the supplied options (up to ~26 candidates in the author's eval). Choice / Noul / Score; optional single image. `option_isolation: false`, so multiple questions do not have isolated attention. Temperature 1.0; calibration (NLL/Brier/ECE) not reported.
- **Benchmark claims:** Author-reported six-group text aggregate 77.70 (JevBench 75.73, Kev 81.92, OpenJev-text 58.74, Nimble 87.23, VitaminC 77.13, MASSIVE 85.43), ahead of Open-Jev-9B 75.67, Kev-4B 74.25, Laya English 58.24; Open-Jev-9B higher on JevBench (77.13) and OpenJev text (65.39). No direct Jev comparison. 83.26% mean on Nimble/VitaminC/MASSIVE vs 71.76 for the NeoHorse-1-4B base.
- **Status:** recently active (HF modified 2026-09-28, 4 days before today).
- **Evidence:** VERIFIED for architecture type (model card and model_manifest.json); REPORTED for training details (not stated).

### Jev-Omni
- **URL:** https://huggingface.co/akhilaaa3/Jev-Omni **Author:** akhilaaa3 (individual) **License:** Apache-2.0 **Popularity:** 1,402 downloads / 342 likes **Dates:** first 2026-09-20 / latest 2026-09-25
- **Relationship to Jev:** inspired by. Card states it implements the typed-decision interface and "nothing in it was trained on Jev output"; no /v1/systemone claim in the card.
- **How it works:** Merged fine-tune of google/gemma-4-12B-it (`gemma4_unified`), 12B, FP32 weights ~50 GB, BF16 autocast. A classifier head accepts up to 256 options (quality established only to 20). One question per call: `predict(state, question, options, media=..., modality=image|audio|video)`; state is re-sent for each question. Inputs: text, image, audio (cap 30 s), video (16 frames). noul/choice/score with per-option probabilities; no generation. Exact head layout and prompt serialisation not stated. Fine-tuned on "a 30,000-question run"; data not described. Calibration: ECE 0.040 on DecisionBench Medium.
- **Benchmark claims:** Author-reported: DecisionBench Medium (80 scenarios / 293 questions) 87.57% (scenario avg); JevBench matched 195 groups / 231 decisions 86.15% group avg (87.45% micro); MMAU 63.10%; MVBench 53.10%. No Jev row in the card (JevBench numbers are not compared directly). Latency on H200: 83 ms text, 26 ms image.
- **Status:** abandoned/unclear (no change since 2026-09-25; no repo link).
- **Evidence:** REPORTED (model card only; mechanism and training data not described).

### open-jev-deberta-v3-large
- **URL:** https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large ; code https://github.com/kotoba-lang/typed-decisions **Author:** Kotoba Labs / Mithril **License:** Apache-2.0 (base DeBERTa-v3-large MIT) **Popularity:** 2,851 downloads / 70 likes **Dates:** first 2026-09-18 / latest 2026-09-24 (HF); repo commit 2026-09-29
- **Relationship to Jev:** clean-room reimplementation of the "shape". Card: no TypeSafe data or code used; numbers "not comparable" to Jev. No /v1/systemone claim.
- **How it works:** DeBERTa-v3-large encoder (~435M) reads `[CLS] [STATE] state [Q] instructions [OPT] option_1 [OPT] option_2 ... [Q] ... [SEP]` in one pass (512 tokens, state cut to 256). A 3-layer span-scoring head scores each option from `[mean(question tokens); mean(option tokens); product]`; softmax within each question's option group. Types: choice (up to 255 options), score (2-10 levels, expected index), noul. Confidence = max probability after one post-hoc temperature fitted on validation. Trained on public gold labels only (Banking77, SST-5, BoolQ; 18,000 states / 42,000 questions, 1 epoch, 229 s on an H100), CE + Brier loss with option-shuffling/paraphrase/negation augmentation. A marker-token-only head failed to learn per the card.
- **Benchmark claims:** Author-reported, own 1,500-state test: in-domain accuracy 0.854, ECE 0.022; OOD (new instructions/options) 0.690, ECE 0.035. No Jev comparison (explicit disclaimer). Latency 28 ms for 10 questions on H100.
- **Status:** active/recent (repo commit 2026-09-29, 3 days ago; HF card 2026-09-24).
- **Evidence:** VERIFIED (model card gives architecture and data; repo code not inspected).

### Jev-Style Decision (chaoliangUNSW)
- **URL:** https://huggingface.co/chaoliangUNSW/Jev-Style-2B-Decision-v3 ; series v1/v2 2B, v3 0.8B; code https://github.com/lawrence3699/jev-style; site jevstyle.com **Author:** chaoliangUNSW (GitHub: lawrence3699) **License:** Apache-2.0 **Popularity:** v3-2B 175 downloads / 3 likes; v1 GGUF 8,713 downloads / 20 likes; GitHub 11 stars **Dates:** first 2026-09-21 (v1 GGUF) / latest 2026-09-27 (v3-2B created and modified; repo commit 2026-09-27)
- **Relationship to Jev:** API-compatible wrapper plus own model. `jev-style serve` provides a local `/v1/systemone`; the card says no outputs of Jev or any TypeSafe model were used in training.
- **How it works:** Full fine-tune of text-only Qwen3.5-2B (1.88B). Prompt: `State:\n...\nQuestion [type]: ...\nOptions:\n- opt1 ->\n- opt2 ->`; each option gets a verdict slot ` ->` and its score is `logit(" yes") - logit(" no")` at that slot from the tied embedding, with no added parameters. Block attention: input cut into 2,048-token blocks, each attending to all previous plus itself (custom runtimes required: PyTorch, GGUF plus `jev-score-v2`, MLX). 25,600-token input, no option cap (numbered catalogue when options exceed 2,048 tokens). Probabilities = softmax(score/T), one global T=0.828 fitted on 2,000 rows. Training: 458 steps on one A100, 181,449 rows (60M tokens), 58 sources (model-written workflows, MASSIVE, CLINC, BoolQ, BANKING77, SGD, GSM8K, ARC, jailbreak sets, own Mac-agent simulators).
- **Benchmark claims:** Author-reported: JevBench v1.4.1 public 73.6% (170/231; CI 67.6-78.9) vs Jev 86.6%, Decision 2B 75.3%, decider-2b 71.0%, Open-Jev-2B 64.5%, Laya 58.4%; tweet_topic 82.2% vs Jev 79.3% (Jev numbers taken from the elcronos study); fin_topic 61.1% vs Jev 67.0%.
- **Status:** active/recent (2026-09-27, 5 days ago).
- **Evidence:** VERIFIED (model card input format, readout, training).

### JevK5
- **URL:** https://github.com/allebee/jevk5 ; https://huggingface.co/alibiserikbay/JevK5 (also JevK5-9B, -2B, -GGUF) **Author:** alibiserikbay (HF) / allebee (GitHub) **License:** Apache-2.0 **Popularity:** GitHub 128 stars; HF 8,885 downloads / 17 likes **Dates:** first 2026-09-22 / latest 2026-09-28 (repo commit; HF modified 2026-09-25)
- **Relationship to Jev:** distillation from other LLMs (not Jev) plus API-compatible wrapper. `jevk5-serve` accepts TypeSafe-style `/v1/systemone`; card says no Jev output or JevBench item used for training. Readout and prompt are from SemIf (TheoLeeCJ, MIT), so it builds on another clone.
- **How it works:** Qwen3.5-4B (9B and 2B variants) with a merged rank-16 LoRA on attention projections. SemIf protocol: answers lettered; probability = softmax over next-token logits of answer letters, divided by one calibration temperature (4B: 1.22). More than 16 options are read in several passes ("knockout" groups of 16 then a final, second temperature 0.93). One CUDA graph per padded length, ~13 ms on H100. Training (v0.3): 17,408 teacher questions (3,270 from Qwen3.6-27B, 14,138 from GPT-6 Luna, each answered twice and kept only when both answers match) plus 30,052 replay items from train splits of 26 public datasets; 1 epoch over 47,460 rows, CE on letter logits (soft targets where a distribution exists).
- **Benchmark claims:** Author-reported: JevBench v1.2 public hard tier 0.784 (v0.2 0.739, untrained 0.613), easy 1.000, standard 0.944. Independent results cited by author: JevBench v1.4 ranked v0.2 #2 of 76 (62.04 vs Jev 1.13.0 63.29), but on 308 fresh sealed decisions 33.1% vs Jev 36.7%; Jev Decision Index 0.2: 36.31, 15th of 49 (v0.2 weights). v0.3 not yet submitted.
- **Status:** active (last commit 2026-09-28, 4 days).
- **Evidence:** VERIFIED (model card and README mechanism; runtime code not read).

### OpenThai-SystemOne
- **URL:** https://github.com/iapp-technology/openthai-systemone ; https://huggingface.co/iapp/OpenThai-SystemOne (Ollama GGUF: iapp/OpenThai-SystemOne-Ollama) **Author:** iApp Technology (sponsor Siam AI) **License:** Apache-2.0 **Popularity:** GitHub 65 stars; HF 9,737 downloads / 30 likes **Dates:** first 2026-09-20 / latest 2026-10-01 (HF modified); GitHub last commit 2026-09-21
- **Relationship to Jev:** API-compatible wrapper plus own model. Request/response "mirrors TypeSafe's POST /v1/systemone" so TypeSafe SDK code can be pointed at it; Ollama 0.35 also serves `/v1/systemone`. No claim of training on Jev outputs; data are public datasets plus synthetic tasks.
- **How it works:** Text tower of Qwen3.5-0.8B-Base (24 layers, hybrid Gated DeltaNet / attention), vision encoder dropped, continued-pretrained on ~5B Thai-heavy tokens. The 248k-token LM head is replaced with a 256-way slot head: options are introduced by control tokens `<|ts_opt_0|>...<|ts_opt_254|>`; the hidden state at each `<|ts_answer|>` token is projected to 256 logits, unused slots masked, softmax gives the distribution; slot 255 = abstain. Up to 255 options for choice, 2-10 levels for score, noul. Trained on ~2-3M decision examples converted from public Thai/English classification, NLI, QA, rating, agent and tool datasets plus synthetic tasks, with option-order shuffling; then a calibration stage (Brier loss, per-type temperature). The Ollama build instead reads answer letters A-Z (2-26 options, no abstain).
- **Benchmark claims:** Author-reported on Bespoke Nimble's 13 public subsets: macro 74.3 vs Nimble-9B 74.8 and Jev 1.13.0 76.0 (Bespoke-published numbers); ahead of Jev on aegis2, massive, multinli, paws, squad2, helpsteer2; behind on boolq, pubmedqa, summeval-relevance (21.7 vs 35.0), vitaminc. Thai held-out: MASSIVE-th 90.0%, Prachathai topics 98.1%, XNLI-th 77.1%; Banking77 45.4%.
- **Status:** active (HF updated 2026-10-01, v0.3; GitHub idle since 2026-09-21).
- **Evidence:** VERIFIED (model card and GitHub README describe slot-head mechanism; code not read).

## Group B: trained community models and implementations (GitHub-first)

Stars/licence/dates from `gh api` on 2026-10-02; "first" = repo created_at, "latest" = last commit date. Benchmark numbers are author-reported unless stated.

### NanoJev
- **URL:** https://github.com/TianyuCodings/NanoJev (weights: huggingface.co/C-Tianyu/NanoJev) **Author:** Tianyu Chen (TianyuCodings) **License:** MIT **Popularity:** 2,473 stars **Dates:** first 2026-09-17 / latest 2026-09-21
- **Relationship to Jev:** inspired by. Calls itself "a nano replica"; Jev only inspired the state/question/candidate-set interface. docs/TYPESAFE_CONTRACT.md states `type:"noul"` currently fails local validation and full response parity is unfinished, so it does not claim /v1/systemone compatibility. Jev is used as an API baseline, not as a teacher.
- **How it works:** Qwen3-0.6B backbone with decision heads. Each candidate is encoded as its own path (state + question + candidate); end-of-sequence representations go through an attention-based "set" Choice head (2-255 candidates) and a softmax. Noul = sigmoid, Score = probability-weighted level over 2-10 levels. Dict states are serialised with Python `str()`. No decoding. Trained with cross-entropy (hard vs soft targets compared) on 18,760 questions per variant, mostly ViZDoom/Maze/Snake game decisions built from expert episodes and policy pools. The "RLCD" objective is an independent proper-scoring experiment, not a recovered recipe. Calibration not claimed.
- **Benchmark claims:** author-reported on own 274-case game test set vs Jev API and untuned Qwen3-0.6B: Maze 4/10 (Jev 7/10), Snake 8/8 (8/8), ViZDoom Basic 128/128 (56/128), Predict Position 27/128 (11/128). No standard decision benchmark.
- **Status:** slowing/likely dormant (last commit 11 days ago; author points to a new JevHarness repo).
- **Evidence:** VERIFIED (HF model README + config.json, docs/TYPESAFE_CONTRACT.md, docs/RLCD_EXPERIMENT.md).

### Von
- **URL:** https://github.com/wfzyx/von (weights: huggingface.co/wfzyx/von; PyPI/npm `von-sdk`) **Author:** wfzyx **License:** Apache-2.0 **Popularity:** 810 stars **Dates:** first 2026-09-18 / latest 2026-10-02
- **Relationship to Jev:** clean-room reimplementation with an API-compatible server. Claims `/v1/systemone` "byte-compatible with the TypeSafe specification"; says clients work by changing the base URL. Training uses public/synthetic data and states no JevBench items in training; no Jev outputs mentioned.
- **How it works:** ModernBERT-large (395M) encoder, non-autoregressive, 8,192-token context. State, question and all options are packed in one sequence; each option gets a `[MASK]` marker whose final hidden state is scored by a small "Option-Marker" head, softmax over markers. Since v1.2, options attend only to the shared premise and themselves (order-invariant: 0 flips on JevBench hard vs 49.5% before). Noul = 2-option choice; Score = choice over levels with expectation. Loss: listwise softmax CE + Brier on ~290k balanced items plus synthetic two-hop/numeric sets. Confidence via input-conditioned temperature map (fitted on 231 public JevBench items, i.e. in-sample); `von calibrate` refits on user labels. Noul band rule maps P(yes) into 0.8+0.1(p-0.5). Extra "chain-of-options" regex/operator layer for dates and amounts (Von 1.3).
- **Benchmark claims:** JevBench v1.4 table (author-reported): Jev 1.13 composite 63.3 vs Von 1.2 27.5 (hard 0.373 vs 0.741, sealed 0.279 vs 0.367); Von leads only on calibration (75.7) and p50 latency (0.34 s CPU). Von 1.3 with chains: hard 0.441.
- **Status:** active (commit today).
- **Evidence:** VERIFIED (HF model card, config.json, README).

### decider
- **URL:** https://github.com/Mapika/decider (weights huggingface.co/Mapika/decider-*) **Author:** Mark Marosi (Mapika) **License:** Apache-2.0 **Popularity:** 1,031 stars **Dates:** first 2026-09-16 / latest 2026-09-30
- **Relationship to Jev:** clean-room reimplementation, API-compatible. Explicit: "Nothing was distilled from Jev"; training uses ~95 public datasets plus data labelled by a local Qwen3.5-27B teacher. `POST /v1/systemone` is "TypeSafe's wire format, so their SDKs work unchanged".
- **How it works:** decoder LLM fine-tunes: Qwen3.5-0.8B/2B/4B-Base, 35B-A3B-Base (MoE), plus Gemma-4-12B-it with a merged LoRA, plus "decider-chat" wrappers (stock Gemma-4-31B / Qwen3.6-27B + fitted temperature, no training). Prompt: `Context ... Question ... Options ... Answer: (` (state-first) or schema-first (50/50), one answer slot per question. Readout: hidden state at each slot projected on one label token per option (A-J, then K-Z and two-letter tokens, up to 255), softmax at a fitted temperature (per type, per option count for the 31B). Choice/Score/Noul; confidence follows TypeSafe's formula since 1.3.0. Training: cross-entropy SFT, one epoch 1.47M examples / 455M tokens (5.3 h GH200); RL stage (v10) not released. GGUF/llama.cpp, MPS, vLLM serving.
- **Benchmark claims:** JevBench v1.5.2 board (third-party, read 2026-09-29): decider-4b v2 71.3 (#7) vs Jev 1.13 72.1 (#3); decider-2b 45.1; Decision Index: 35B-A3B 47.11 vs Jev 57.91; decider-chat-gemma4-31b 57.33 (#2). Hard tier: Jev 0.730 vs 35B 0.676.
- **Status:** active.
- **Evidence:** REPORTED (README with detailed mechanism and repo layout; code not opened).

### openJev-verdict-2.0
- **URL:** https://github.com/Heman10x-NGU/openJev-verdict-2.0 (weights huggingface.co/heman10x/openJev-verdict-2.0, rlcd-modernbert-151m) **Author:** Heman10x-NGU **License:** Apache-2.0 per README/badge; GitHub reports NOASSERTION (LICENSE file not recognised) **Popularity:** 294 stars **Dates:** first 2026-09-19 / latest 2026-09-20
- **Relationship to Jev:** inspired by ("Inspired by TypeSafe AI's Jev architecture and RLCD"). No `/v1/systemone` compatibility claim; no statement of training on Jev outputs.
- **How it works:** ModernBERT-base (149.6M; train manifest says "ModernBERT-base + GLiClass-v2" for the v1 Verdict) encoder. Sequence `[CLS] <type> question: <instr> [SEP] [MASK]opt0 [MASK]opt1 ... [SEP] <state> [SEP]`; hidden state at each [MASK] (plus a question-type embedding) goes through LayerNorm+MLP to one logit; softmax. Two heads: marker-pointer distribution head and a separate MLP "correctness head" over 7 shape features (p1, margin, entropy, 1/K, type one-hot) for confidence. Per-(type, option-count) temperature. Choice/Score/Noul. Training targets are soft distributions from `LocalLLaMA/typed-decisions` (README calls them reviewer-panel distributions); symmetric KL between option-shuffled twins; 8.8 h on a GTX 1660 Ti. v1.x checkpoint (JevBench) trained on 5,000 generated samples, contexts under 71 tokens.
- **Benchmark claims:** author-reported on LocalLLaMA/typed-decisions test (2,000 decisions): 77.10% top-1 vs Laya 76.60%, Jev 1.13 72.70% (Jev figure is a vendor baseline cited from Laya's suite); Brier 0.0636; correctness-head ECE 0.0144. The earlier Verdict v1.4 on JevBench hard is only 36.9% (README admits no change from v1.0).
- **Status:** abandoned/dormant (last commit 12 days ago). GitHub-tracked weights for verdict2 are Git LFS pointers.
- **Evidence:** VERIFIED (verdict2/model.py, verdict2/data.py, artifacts/train_manifest.json).

### imajev
- **URL:** https://github.com/mohit67890/imajev (weights huggingface.co/mohit67890/imajev-{2b,4b,9b}) **Author:** Mohit Garg (mohit67890), "Claude as co-author on the code" **License:** Apache-2.0 **Popularity:** 205 stars; HF imajev-4b 1,042 downloads, 34 likes **Dates:** first 2026-09-23 / latest 2026-10-01
- **Relationship to Jev:** clean-room reimplementation, API-compatible superset: "Jev's contract, now with images" (`/v1/systemone` plus `images`, `unknown_probability`, `abstained`). Training statement: "No Jev outputs, no paid-API outputs ... used". Credits TypeSafe docs for the request contract and Kev as sibling.
- **How it works:** Qwen3.5-2B/4B/9B + LoRA r16 on language layers (vision tower frozen) + 255-code decision readout in a float32 linear head. Prompt renders state, images (up to 2), question and option list; one prefill, logits of the option codes plus an `unknown` code at the decision position, softmax. Probability for each option also carries `unknown_probability` and `abstained`. Single shipped temperature (4B 1.305) fitted on 150 authored items; optional four-option-order averaging. Training ~1M decisions in 4 stages (36 licence-checked sources incl. ~300k converted image decisions; ~416k pseudo-labelled by its own 9B; 17.9k hard questions from Qwen3.6-27B + gpt-oss-20b agreement; 39.5k soft-target rows from Qwen3.6-35B-A3B teacher plus Eikos rows), then weight-space averaging of adapters. About $676 of GPU.
- **Benchmark claims:** author-reported, with screenshots of board results: JevBench v1.4.2.2 #1 of 91, 67.37 vs Jev 1.13.0 63.29; Image JevBench v0.1.3 #1 of 49, 76.39 vs Jev-Omni 73.10; DecisionBench (eng) #3 of 56, 79.65 vs Jev 71.90. Own runs: JevBench hard 72.1% (4B, as shipped), behind JevK5/Eikos-4B (73.9). typed-decisions test 69.2% (Jev 73.4% per Intern-Decision runs).
- **Status:** active.
- **Evidence:** VERIFIED (HF model card + README technical specification).

### RSI-Jev
- **URL:** https://github.com/Shanghua-Gao/RSI-Jev (weights huggingface.co/shgao/rsi-jev-*) **Author:** Shanghua Gao **License:** code MIT, weights Apache-2.0 (some image data non-commercial) **Popularity:** 43 stars **Dates:** first 2026-09-22 / latest 2026-10-02
- **Relationship to Jev:** clean-room reimplementation, API-compatible: server "speaks Jev's API" (`/v1/systemone`); "Not affiliated with TypeSafe AI". No distillation from Jev stated; trains on public benchmark train splits and Open-Jev items.
- **How it works:** Qwen3.5-2B/0.8B-Base fine-tunes found by an autonomous AI-agent research loop (312 experiments; failures published). Option scorer plus cross-attention features combined by a two-layer MLP readout (v3.0), separate per-question confidence head (forbidden from sharpening score questions) refitted on held-out data. Bottom 8 of 24 layers at 1/10 LR. v3.0 adds a listwise reranking RL stage (Plackett-Luce, NDCG@5 reward, KL to parent); v4.0-VL adds images via frozen vision tower and an asymmetric confidence-penalty RL stage. Data: 266k questions from 36 sources, typed-decisions train split oversampled, 15% general replay.
- **Benchmark claims:** author-reported on its own 15-benchmark suite: v4.0-VL 0.756 (ECE 0.043); typed-decisions 0.662 (v1.0) to 0.796 after in-distribution training. No head-to-head Jev numbers in README.
- **Status:** active (commits today).
- **Evidence:** VERIFIED (versions/v3.0.md, rsijev/README.md).

### eikos
- **URL:** https://github.com/caiovicentino/eikos (weights caiovicentino1/Eikos-{4B,27B}) **Author:** Caio Vicentino **License:** MIT (code, deltas); data CC BY 4.0 **Popularity:** 40 stars **Dates:** first 2026-09-23 / latest 2026-10-01
- **Relationship to Jev:** clean-room reimplementation, API-compatible (`/v1/systemone`, plus agent sessions and images). Jev is used for evaluation only: "measured through its API ... not distilled from".
- **How it works:** LoRA (r64, one epoch, lr 1e-4) merged into Qwen3.5-4B and Qwen3.8-27B (Gated DeltaNet hybrid; vLLM >= 0.30 required). Letter-logit readout (A-Z, AA, AB, ... up to 588 options) with prefix cache; soft cross-entropy on option-letter logits with option-order permutation, rationale loss 0.3, PT/EN view-consistency KL, light JEPA loss. T = 1. Data: items written by GLM-5.3-Flash/Qwen and blind-labelled by teacher GLM-5.3-Flash (kept only if teacher agrees with gold), programmatic finance/trade/rule generators, long-context dossiers; 8-gram decontamination against JevBench. Finance/trade-finance focus. FP8, INT4, MLX builds.
- **Benchmark claims:** author-reported, own harness: JevBench public hard 82.9 (27B) / 72.1 (4B) vs Jev 73.0, Laya 35.1; error at >=90% confidence 2.4% vs Jev 5.9%; 64k-token needle accuracy 88.3 (27B).
- **Status:** active.
- **Evidence:** REPORTED (README layout/recipe + HF card header for base model; code not opened).

### StartLux-Decision
- **URL:** https://github.com/StartLuxLabs/StartLux-Decision (weights huggingface.co/startlux-models/*) **Author:** StartLux Labs **License:** code Apache-2.0; weights CC BY-NC 4.0 (commercial needs a separate licence) **Popularity:** 20 stars **Dates:** first 2026-09-29 / latest 2026-10-01
- **Relationship to Jev:** clean-room reimplementation, API-compatible ("TypeSafe `/v1/systemone` format, so clients written for Jev work unchanged"). Training on Jev outputs: not stated.
- **How it works:** six sizes (0.8B, 2B, 4B, 9B, 27B, 35B-A3B MoE). Base model not named in README; config.json is `Qwen3_5ForConditionalGeneration` (Qwen3.5 hybrid linear-attention), so Qwen3.5-family. Each question is its own prompt with lettered options (A, B, ...; up to 26 per round, wider via extra rounds); answer read from option-letter logits at the answer position after one forward pass; per-type temperatures in `decision_config.json`. System prompt prefix cached; CUDA graphs. Training recipe not described; says data includes public train splits of 14 Decision Index benchmarks (marked in table), their test items filtered out. LoRA finetune script ships.
- **Benchmark claims:** author-reported (own runs, not on public boards): Decision Index 0.2.1: 27B 63.88 vs Jev 57.91; JevBench public correct of 231: 35B-A3B 210, 27B 208, Jev 199; chess match vs Jev 1.13 via API: 58.4% over 256 games.
- **Status:** active (3 days old).
- **Evidence:** VERIFIED (HF config.json, docs/finetuning.md, docs/inference.md); training data REPORTED only.

### tinyjev
- **URL:** https://github.com/ankit-aglawe/tinyjev (weights AnkitAI/TinyJev-0.6B, TinyJev-4B; Ollama parable/tinyjev) **Author:** Ankit Aglawe **License:** MIT **Popularity:** 26 stars **Dates:** first 2026-09-22 / latest 2026-10-01
- **Relationship to Jev:** fork of another clone, in effect: 0.6B/4B v1 reproduce Kev (jaredpalmer/kev) using Kev's training data, evaluation suites and "pointer-head design"; 4B v2 uses Together AI's Tev1 recipe/data. Also ships a System One compatible `/v1/systemone` server (and Ollama endpoint). No Jev distillation stated.
- **How it works:** Qwen3-0.6B-Base (596M) and Qwen3-4B-Base (v1), Qwen3.5-4B (v2). Backbone hidden states fed to a decision head (`head.safetensors`) over options; Choice/Noul/Score, up to 255 options; State flattened with field names kept. Calibration: shipped temperature 1.464 fitted in-distribution. v2: trained on the exact Ollama prompt with Tev1 data plus 1,269 contract questions (ContractNLI).
- **Benchmark claims:** author-reported OpenDecision OD-500 (500 never-seen cases, 25 domains): 0.6B 440/500 (88.0%), 4B v2 490/500 (98.0%), Kev-0.8B 463, Claude Opus 5.5 496. JevBench public: 177/231 (Tev1 176). No direct Jev comparison except a game-lap time (59.0 s vs 59.2 s reported for hosted Jev).
- **Status:** active.
- **Evidence:** REPORTED (README + HF card excerpts; code not opened).

### Nimble
- **URL:** https://github.com/bespokelabsai/nimble (weights bespokelabs/Bespoke-Nimble-9B, -v2) **Author:** Bespoke Labs (Maheswaran Sathiamoorthy) **License:** repo: none recorded by GitHub; HF weights Apache-2.0 **Popularity:** 2,023 stars **Dates:** first 2026-09-18 / latest 2026-09-24
- **Relationship to Jev:** inspired by / clean-room recipe. "We did not distill from Jev"; objective uses hard labels. A hosted API and a TypeSafe-labelling dataset module exist in the repo, but `/v1/systemone` compatibility is not stated. Serving idea taken from a public post on how Jev decodes.
- **How it works:** Qwen3.5-9B + LoRA r16 (lr 5e-5, 1 epoch, 2,048-token prompts; 8,192 in latest release). Each allowed answer gets a one-token code; scorer reads logits of those codes at the answer position, softmax, no JSON generation. Flat schema: enum (1-255 choices) or boolean; Score as ordered levels. Mac MLX scorer shares context prefix across fields; CUDA scorer rescoring per field. Data: 2,676 train / 324 held-out synthetic "contrastive" examples (pairs differing by one fact, checked by separate model calls). Labels are model-checked, not human. Original release fitted T = 2.179; latest checkpoint T = 1.0.
- **Benchmark claims:** author-reported on own 324 held-out examples (162 pairs, 6 source families): Nimble-9B 90.12% reference agreement vs Jev 1.13.0 93.21%, Qwen3.5-9B 66.36%, Qwen3.8-27B 84.88%. Second-party (Vev README) Nimble set: Jev 0.923.
- **Status:** slowing (last commit 8 days ago; the repo says it was built in one day).
- **Evidence:** VERIFIED (README methodology, HF card header).

### jevlike
- **URL:** https://github.com/vinnylarouge/jevlike **Author:** vinnylarouge **License:** MIT **Popularity:** 1,338 stars **Dates:** first 2026-09-16 / latest 2026-09-16 (single push)
- **Relationship to Jev:** inspired by. "An independent starter model with the same input and output shape"; states it did not show equal quality with Jev nor reproduce TypeSafe's method. No `/v1/systemone`.
- **How it works:** Context text + N option texts, softmax over N. Each option becomes a query vector that attends over context tokens; attended vector dotted with option embedding gives its score (shared across options, so N is variable). Default encoder: byte embeddings trained from scratch (context truncated to 192 bytes, option to 32); optional frozen Hugging Face encoder (e.g. Qwen2.5-0.5B) with a rank-256 scorer head. Choice only, no Noul/Score. Training on user JSONL (`context`, `options`, `label`); synthetic and Wikispeedia examples. Reports ECE, but no calibration method.
- **Benchmark claims:** author-reported toy results: ~98% on synthetic menus; Wikispeedia next-click 26% (frozen Qwen2.5-0.5B + head) vs ~8% controls, 29% from-scratch. Doom/chess controller demos (50 games vs Stockfish level 0: 0 wins, 48 losses). No Jev comparison.
- **Status:** abandoned (one push, no later commits).
- **Evidence:** VERIFIED (README architecture and limits; code not opened).

### Strands Decider
- **URL:** https://github.com/strands-labs/strands-decider (weights StrandsAgents/strands-decider-2B-hobson-v19) **Author:** Strands Labs **License:** Apache-2.0 **Popularity:** 96 stars **Dates:** first 2026-09-29 / latest 2026-10-01
- **Relationship to Jev:** clean-room reimplementation, API-compatible (`strands-decider serve` exposes `/v1/systemone`). Training: no Jev outputs stated; teacher is frozen Qwen3.5-4B, plus Qwen3.6-27B/Qwen3.5-397B generated questions.
- **How it works:** Qwen3.5-2B-Base torso, LM head discarded, replaced by a ~1M-parameter pointer head scoring each option by comparing the hidden state at the `<answer>` position with the option's own last-token hidden state (no per-option parameters, no option cap); rank-16 LoRA, fp32 head. Choice, Noul and Score from the same masked softmax; ordinal levels only reversed, never permuted. Loss has three parts, including a KL (weight 1.0) to a frozen Qwen3.5-4B teacher on multi-step rows only. Data: 21 short-task HF datasets, generated document/adequacy/flip questions (Qwen writer, Qwen verifier agreement), replay distributions. ~11 h on one RTX 3090. Confidence headline: answers at >=0.9 right ~95% on unseen short tasks.
- **Benchmark claims:** author-reported JevBench public (231): 0.723 (167/231), Brier 0.342, ECE 0.052; tiers easy/standard/hard 1.000/0.875/0.505. Nearest comparison decider-2b. Notes six retrains vary +/-3.2 tasks.
- **Status:** active. The research log records failed runs with preregistered predictions.
- **Evidence:** REPORTED (README plus docs/architecture.md and data/sources.md excerpts).

### OneJev (OmniJev)
- **URL:** https://github.com/OmniJev/OneJev (weights OmniJev/OneJev-{0.8B,4B,9B,27B}; dataset OmniJev/OneJev-Data) **Author:** OmniJev Team **License:** Apache-2.0 **Popularity:** 108 stars **Dates:** first 2026-09-27 / latest 2026-09-30
- **Relationship to Jev:** clean-room reimplementation, API-compatible ("compatible with TypeSafe System One"; official TypeSafe SDK example included). Code ports Jev's confidence formulas from TypeSafe's MIT `system-one-adapter-python`, verified against published Jev answers (MAE 0.013 Choice, 0.005 Score). Training uses 127 public datasets, no Jev outputs.
- **How it works:** full fine-tune of Qwen3.5-0.8B/4B/9B and Qwen3.8-27B, vision tower frozen, multimodal (screens, photos, video). Prompt (`qev/prompt.py`): system instruction "Respond with only its uppercase letter", state, question, lettered options (A..Z, then two-letter codes, up to 256 slots); shared prefix cached, one suffix per question, answer read from option-label logits. Noul as yes/no, Score as levels. Loss: cross-entropy + Brier over answer probabilities, 1 epoch, lr 5e-6, 16,384-token limit, 99,193 questions (94,707 released: 33k GUI-agent runs, 16.8k text, 13.6k images, 18.9k videos, 12k rules). Calibration via `qev/calibrate.py`.
- **Benchmark claims:** author-reported (charts only; table is an image): vs Jev 1.13 (published text-only scores), Jev-Omni 12B, Qwen3.8-27B thinking on OneJev test set, DecisionBench hard, TypeSafe, MMStar; no numbers in text. Latency on H200: 4B 64 ms/1 question, 104 ms/10.
- **Status:** active/recent (4 days).
- **Evidence:** VERIFIED (qev/prompt.py, qev/answers.py, HF model card, dataset card).

### Vev
- **URL:** https://github.com/Xiaooolong/vev (weights CountingSheep/vev-4b, vev-9b) **Author:** Xiaolong Wang **License:** code Apache-2.0; weights CC BY-NC 4.0 **Popularity:** 9 stars **Dates:** first 2026-09-30 / latest 2026-10-01
- **Relationship to Jev:** clean-room reimplementation, API-compatible: serves `/v1/systemone`; TypeSafe's Python SDK works after changing base URL. States "no Jev outputs were used for training"; says compatible does not mean it behaves like Jev.
- **How it works:** LoRA (r16, all language linear layers; vision tower frozen) on Qwen3.5-4B and 9B, text, images, English and Chinese. Prompt with state, question, lettered options; reads next-token probabilities over answer tokens (Yes/No, option letters, level digits), renormalised. No head (released run had an untrained zero-init pointer head). Training: 2,500 steps, CE + 0.1 Brier + KL anchor (0.3) to base model on unlabelled "anchor" rows; 100k sampled from 762,820 records (42 public sources, 431k image, 331k text, 142k Chinese), options shuffled, negation/distractor augmentation.
- **Benchmark claims:** author-reported, own harness: JevBench public subset 0.766 (4B) / 0.823 (9B) vs Jev 0.861; nimble set 0.707/0.747 vs Jev 0.923; kev transfer-v4 0.776/0.781 vs Jev 0.854; judgekit (Chinese) 0.962 vs Jev 0.962. Versus base Qwen: mostly n.s. on text. Top answer flips on option reversal 13-17% vs <4% for Jev.
- **Status:** active (new).
- **Evidence:** VERIFIED (TRAINING.md, README).

### jevos
- **URL:** https://github.com/feder-cr/jev (binary `jev`; model jevos-v2) **Author:** Federico Elia (feder-cr) with Loris Salsi **License:** MIT **Popularity:** 1,174 stars **Dates:** repo created 2024-08-15 (id 842982339) but git history starts 2026-09-27 ("feat: jevos-1b ..."); 38 commits, latest 2026-10-01; release jevos-v2 2026-09-30
- **Relationship to Jev:** clean-room reimplementation, API wire-compatible ("speaks TypeSafe Jev's wire format"; accepts `model: jev-*`). Caution: held-out choice/score sets include questions with "labels from Jev" (81.2% agreement with Jev on workflow questions; 1,389 rubric score questions "labels from Jev"), so Jev outputs were at least used for evaluation, and possibly for training; training-data provenance is not stated.
- **How it works:** ~1B decoder (README says "1B"; tokenizer patched with a MiniCPM5 pre-tokenizer, so likely MiniCPM5-1B; not stated). Yes/no only at heart: logits of the two answer tokens "0"/"1" via a custom OpenVINO graph. Choice = one yes/no question per option ("Among the candidates, is it this one?"), probability normalised over options; Score = one "is it at this level or higher?" question per level above the lowest, pool-adjacent-violators monotone fix. C++ server, INT8 OpenVINO / GGUF Q4/Q8, 8,192-token context, state cache. Confidence = Jev formula.
- **Benchmark claims:** author-reported, own set of 999 hand-written yes/no questions: jevos-v2 0.803-0.810, Jev 0.927, Laya 0.489; held-out choice 78.8%, score 54% (within one level 82%). Latency 26 ms short / 112 ms long vs Jev 344 ms (laptop CPU vs API).
- **Status:** active (recent). Note: repository identity predates the Jev launch; GitHub shows no earlier history, so the 1,174 stars cannot be attributed to this project era (stargazer endpoint returned 404).
- **Evidence:** VERIFIED (README, export/jevos_graph.py); training provenance REPORTED/unknown.

### WebJev
- **URL:** https://github.com/lexmount/WebJev (weights Lexmount/WebJev-35B-A3B; data Lexmount/WebJev) **Author:** Lexmount **License:** Apache-2.0 **Popularity:** 2 stars **Dates:** first 2026-09-29 / latest 2026-09-29
- **Relationship to Jev:** fork of another clone in method: training "builds on" Mapika/decider (its prompt builder and `full` data recipe at a pinned commit, copied not forked); also reuses ZefanCai/Open-Jev and Nimble data. API-compatible server (`/v1/systemone`, `/api/alpha/decisions`). No Jev outputs in training stated.
- **How it works:** Qwen3.5-35B-A3B-Base (34.7B/3B active). Decider's format: state after `Context:`, options `(A)..`, one answer slot per question; option-letter logits read, cross-entropy. Full fine-tune of all non-expert weights (2.45B of 34.7B), routed experts frozen, Muon+AdamW, one epoch, lr 1e-5, 8xA100, 3.19M records (1.21B tokens): decider mixture, 64k live-web decisions (next operation, element grounding), Open-Jev train splits, knowledge MCQA, Nimble records, teacher-written contrastive pairs (deepseek-v4.1-flash). Specialised for browser agents.
- **Benchmark claims:** author-reported: 125 real-website tasks via jev-ultrafast agent, success 38.5% vs Jev 1.13 16.7%; 8 single-step benchmarks mean 83.72% vs Jev 84.70% (JevBench public 87.88 vs 85.71; MMLU-Pro 69.4 vs 83.4).
- **Status:** one-day push (09-29); static.
- **Evidence:** VERIFIED (train/README.md, train/data/README.md).

### reflex
- **URL:** https://github.com/kshetrajna12/reflex **Author:** kshetrajna12 **License:** MIT **Popularity:** 161 stars **Dates:** first 2026-09-17 / latest 2026-09-22 (pushed 09-27)
- **Relationship to Jev:** clean-room reimplementation ("open re-creation"), API-compatible (`/v1/systemone` same shapes as hosted API). Credits Jev launch post and docs plus a public architecture write-up. Optional `reflex-distill` uses a stronger open model as teacher, not Jev.
- **How it works:** frozen Qwen3.5-4B (default), no training in the recommended `stable` config. Per-question branch off a cached state, with option-letter (A..Z, lettered yes/no pair) next-token logits read in one pass; each question read in two option orders and averaged; Evidence/Criterion framing. Choice up to 26 options, Score 2-10. Optional LoRA/calibration scripts; temperature fit on MMLU (ECE 0.090 to 0.039). Authors report all their fine-tunes (4 LoRA mixes, 27B distillation) lost general judgement and were rejected.
- **Benchmark claims:** author-reported JevBench public: frozen 4B 1.000/0.917/0.685 (easy/std/hard), hard ECE 0.081 vs Jev 1.000/0.986/0.730, ECE 0.031; frozen 27B hard 0.766. Official JevBench v1.2: reflex-4B #5 of 36 (71.7), reflex-27B #22.
- **Status:** slowing/dormant (last commit 10 days ago).
- **Evidence:** REPORTED (README; docs/ARCHITECTURE.md not opened).

### jev-style (lawrence3699)
- **URL:** https://github.com/lawrence3699/jev-style (weights chaoliangUNSW/Jev-Style-{0.8B,2B}-Decision-v3; PyPI `jev-style`) **Author:** lawrence3699 (HF: chaoliangUNSW) **License:** Apache-2.0 **Popularity:** 11 stars **Dates:** first 2026-09-25 / latest 2026-09-27
- **Relationship to Jev:** clean-room reimplementation, API-compatible server ("follows the public systemone request shape"). "No Jev weights, code or outputs are included"; typed-question convention from Laya.
- **How it works:** full fine-tunes of Qwen3.5-0.8B and Qwen3.5-2B (post-trained, not Base). Each option scored at its own verdict slot: option k's score = logit(" yes") - logit(" no") at its ` ->` slot, per option; 25,600-token input, block attention (2B, 2,048-token blocks). MLX, PyTorch, GGUF (custom scorer). One global temperature fitted on own calibration rows. 2B trained on reduced pool: 181,449 rows / 60M tokens, 58 public sources, 458 steps on one A100 40GB. Extras: Claude Code guard, MCP server, six agent skills.
- **Benchmark claims:** author-reported, self-run (not official board): JevBench public 73.6% (2B), 64.1% (0.8B) vs hosted Jev 86.6%; beats Laya 0.8B checkpoint on Banking77 (68.2 vs 49.2), MASSIVE and tweet_topic.
- **Status:** low activity (last commit 5 days ago).
- **Evidence:** VERIFIED (HF 2B model card, README).

### Jeff 1
- **URL:** https://github.com/Gestalt-Lab/jeff (weights GestaltLabs/Jeff-1) **Author:** Gestalt Lab **License:** Apache-2.0 **Popularity:** 5 stars **Dates:** first 2026-09-19 / latest 2026-09-19
- **Relationship to Jev:** clean-room reimplementation, API-compatible local server (`POST /v1/systemone`). Trained on human-labelled fact-check data, and evaluated against live Jev 1.13.0 (Jev predictions on same rows, agreement reported separately). A pre-registered "soft-distill" run toward "the teacher distribution" is pending; the teacher is not identified.
- **How it works:** LoRA adapter on Qwen3-4B-Instruct-2507 (not a head): label scoring using the LM's own next-token distribution over label tokens; first-token readout if first tokens are distinct, otherwise whole-sequence readout. Each question scored separately. Confidence = largest label probability. Choice-only adapter (9,119 rows) then multi adapter (+1,800 Score, +1,200 Noul). Trained on Colab A100; developed by an autoresearch loop.
- **Benchmark claims:** author-reported, 9,730 human-labelled fact-checking examples (FEVER, VitaminC, SciFact, Climate-FEVER), vs live Jev 1.13.0: accuracy 0.8183 vs 0.8283 (Jev ahead, p=0.0033), Brier 0.2839 vs 0.2750, ECE 0.0807 vs 0.0932 (Jeff better). One earlier calibration figure was retracted.
- **Status:** abandoned (one session, 09-19).
- **Evidence:** VERIFIED (PROVENANCE.md, README).

## Group C: wrappers and inference techniques (status relative to 2026-10-02)

### AnyJev
- **URL:** https://github.com/nokia-applied-research/AnyJev **Author:** Jiamu Zhang, Tianze Yang, Yucheng Shi, Liang Wu (Nokia Applied Research; one author at Tencent Hunyuan); README cites an arXiv technical report **License:** Apache-2.0 **Popularity:** 1006 stars / 131 forks **Dates:** created 2026-09-21 / last push 2026-10-02
- **Relationship to Jev:** inspired by. A Python library (`Decider`, `Question.choice/noul/score`) that adds a decision layer to any open LLM; the README states it is not affiliated with TypeSafe and re-uses Jev's published numbers rather than Jev outputs. Not a wire-compatible server (there is a "Jev mode" demo doc). Not the same project as llm2jev, which used to be named AnyJev before 0.5.0.
- **How it works:** Backends: HF transformers and vLLM (generate server, or an embed server for L2). Headline models are Qwen3 (1.7B to 32B), Qwen3-8B for the zero-label table; Qwen2.5-7B in the serve example. One prefill, restricted softmax over lettered option tokens (max 26 options). Levels: raw; L0 (zero labels: averages the K option rotations to remove position bias, divides out a label prior; optional adaptive rotation budget); L1 (temperature scaling, 100-500 labels); L2 (closed-form head on a mid-depth hidden state, 100-300 labels per question, no gradients, truncated model at about 0.7x forward cost). Confidence comes from calibrated probabilities; heads can re-estimate feature mean/scale from unlabelled traffic. No distillation from Jev.
- **Benchmark claims (author-reported):** Qwen3-8B BANKING77-20: order-flip rate 0.230 to 0.073, ECE 0.240 to 0.095, auto-decidable at 5% risk 7.7% to 52.0%. On LocalLLaMA/typed-decisions, L2 heads: Qwen3-1.7B 0.730, 4B 0.786, 8B 0.771, 32B 0.798 vs published Jev 0.727 and Laya 0.768 (not re-run).
- **Status:** active (pushed today).
- **Evidence:** VERIFIED (anyjev/decider.py: levels, softmax, rotation, `fit_head`, `observe`); head and benchmark details REPORTED.

### SemIf (formerly OpenJev)
- **URL:** https://github.com/TheoLeeCJ/SemIf-OpenJev **Author:** TheoLeeCJ (plus community PRs) **License:** MIT **Popularity:** 4655 stars / 331 forks **Dates:** created 2026-09-16 / last push 2026-09-23
- **Relationship to Jev:** clean-room reimplementation of the interface pattern with open models. Its README says it does not reproduce Jev's model or training. It is a CLI/scorer, not an HTTP-compatible server. The README never mentions AlexWortega/openjev on HF, so it cannot be confirmed from here that they differ; by author and repo they are separate projects.
- **How it works:** Frozen Qwen3.5-4B (BF16) is the baseline; a 27B EXL3 bridge, llama.cpp GGUF, MLX, MPS and a WebGPU browser demo are also provided. Input is `state` (string or JSON) plus a question and typed options; each option gets a letter (A to P, max 16 in core.py) and one forward pass reads the option-letter logits. Modes: direct, serial prefix reuse, parallel shared-state suffixes. Native reranker (Qwen3-Reranker-4B) tested as a comparison. Per-workload temperature scaling (fitted on labelled data; out-of-fold ECE 0.068 to 0.038 on its authored set). No training.
- **Benchmark claims (author-reported):** 102-row TypeSafe subset agreement 0.845 (Qwen3.5-4B) vs Jev 0.883 (from TypeSafe's published records, not re-run); authored 144 balanced accuracy 0.813 (4B) and 0.958 (27B); 21 binary decisions in 1.02 s vs 5.33 s for a generated JSON array.
- **Status:** active, but slowing (last push 9 days ago).
- **Evidence:** VERIFIED (src/semif_phase1/core.py: LETTERS, softmax, direct messages); results REPORTED.

### openjev-sglang
- **URL:** https://github.com/ekzhang/openjev-sglang **Author:** ekzhang **License:** not stated (no license detected by GitHub) **Popularity:** 338 stars / 45 forks **Dates:** created 2026-09-17 / last push 2026-10-01
- **Relationship to Jev:** API-compatible wrapper: a server for the TypeSafe `/v1/systemone` HTTP API on an open model, using `jev-latest` as an alias. The README calls it an early experiment and says SGLang now ships a native decisions endpoint built the same way.
- **How it works:** Qwen3.6-35B-A3B (NVFP4 checkpoint, hybrid MoE, 3B active) on SGLang 0.5.19 with Rust frontend and radix cache, deployed on Modal B200. The chat template is rendered once with thinking off; the common prefix is warmed with a 1-token call; each question then makes one `max_new_tokens=1` call with `token_ids_logprob` on its answer labels. Options render as `A: description`; labels are A to Z then verified single-token letter pairs, up to 64 answers. Softmax over labels at configurable temperature (default 1.0). noul = P(yes); choice = argmax plus distribution; score = expected zero-based level. Confidence = 1 - H/log K; the README says probabilities are not calibrated. No training.
- **Benchmark claims:** none stated for decision quality (evals/ folder holds BoolQ and MMLU-Pro probes).
- **Status:** superseded by SGLang's native endpoint; recently pushed but marked as an early experiment.
- **Evidence:** REPORTED (README is detailed; code not read).

### razorback16/openjev (DiffusionGemma)
- **URL:** https://github.com/razorback16/openjev **Author:** razorback16 **License:** Apache-2.0 **Popularity:** 579 stars / 46 forks **Dates:** created 2026-09-18 / last push 2026-09-29
- **Relationship to Jev:** API-compatible wrapper (same wire API; TypeSafe SDK works unchanged). Shares its name with other "openjev" repos; it is not a fork of them. It also serves other people's models behind the same API: Laya, Verdict, CLM and JevK5.
- **How it works:** Default `openjev-latest` is NVIDIA DiffusionGemma 26B-A4B (NVFP4, Apache-2.0), a discrete-diffusion model, via vLLM (pinned commit, vllm PR #57250) or MLX. It builds a canvas in which only answer slots are masked (one token per question) and runs one read-only denoise pass; label logprobs (yes/no, A..., 0..9) are read at those slots. If a slot's entropy is above 0.1 it reads three more times with fresh noise and averages. Confidence = 1 - H/ln K. Optional `steps`, `samples`, `think`, `sequential`, and image inputs (extensions beyond Jev). Up to 255 choices. No training by this project; the small encoder and CLM/JevK5 models are others' trained models.
- **Benchmark claims (author-reported):** same top answer as JevK5's own v0.2 run on all 231 JevBench public items, both scoring 86.6%; 27 ms p50 per request, 57 req/s at 64 concurrency on one RTX PRO 6000. No accuracy for DiffusionGemma stated in the README.
- **Status:** active (last push 3 days ago).
- **Evidence:** VERIFIED (openjev/engine.py: seeded canvas, `logprob_token_ids`, label ids); benchmarks REPORTED.

### simple-jev
- **URL:** https://github.com/featherless-ai/simple-jev **Author:** Featherless AI **License:** Apache-2.0 **Popularity:** 579 stars / 66 forks **Dates:** created 2026-09-18 / last push 2026-10-01
- **Relationship to Jev:** API-compatible wrapper ("takes inspiration from TypeSafe's structured-decision interface"; `/v1/systemone` is an alias of `/v1/classifier`). A public demo API serves Gemma.
- **How it works:** HF Transformers/PyTorch server. Models: reference configs for Qwen3.5-4B, Qwen3.8-27B, Qwen3.6-35B-A3B, Gemma-4-12B and Gemma-4-26B-A4B (demo default); also a Laya backend. Per-question prefix: shared state is prefilled once and the KV cache is copied across question suffixes; the next-token logits on allowed answer labels are normalised by `common/response_scoring.py` (softmax over allowed labels; choice = argmax, score = expectation). Case-sensitive labels A to Z then a to x (up to 255 choices via config). Prompt-format policy (`baseline`, `repeat_state`, `strict_mix_repeat2`, `examples_binary`) auto-selected by architecture and size, with a prompt-search tool. Confidence = max probability, explicitly not calibrated. A separate `RFDT/` folder trains students (teacher-estimate distillation, LoRA, answer-token-logit loss).
- **Benchmark claims:** eval framework with native JevBench scoring and a 477-case selection set; no headline accuracy in the README.
- **Status:** active.
- **Evidence:** VERIFIED (common/response_scoring.py, common/prompt_builder.py); RFDT only REPORTED.

### rizzo-flow
- **URL:** https://github.com/Rizzo-AI-Academy/rizzo-flow **Author:** Rizzo AI Academy **License:** Apache-2.0 **Popularity:** 785 stars / 52 forks **Dates:** created 2026-09-21 / last push 2026-09-25
- **Relationship to Jev:** API-compatible wrapper plus a fine-tune. The README says it was inspired by SemIf, reproduces only the interface, and uses public data without Jev outputs.
- **How it works:** llama.cpp runtime (MLX optional). Default Spark-X2.5-4B (XHToken) with its own merged LoRA, 4B Q8_0 (also 1.7B). Prompt layout: state, then lettered options, closing line "Answer with the letter of the best option"; one position per question, logits over answer letters only (max 26), nothing sampled. Built-in `__insufficient__` abstain option on by default. Training: LoRA r16 on 28,321 questions (tasksource/procedural-typed-decisions, ZefanCai/Open-Jev, Praveenrajus/jev-bench), soft cross-entropy on letter softmax, one epoch. Per-primitive temperature scaling bound to a weights/runtime fingerprint.
- **Benchmark claims (author-reported):** LocalLLaMA/typed-decisions (2,000 decisions): fine-tune accuracy 0.648, ECE 0.112, vs base 0.574 / 0.349 and Jev 1.13.0 0.727 (from dataset card, not re-run). SemIf fixtures: authored144 0.845 (SemIf 0.819), perturbations108 0.946 (SemIf 0.766).
- **Status:** active, slowing (last push 7 days ago).
- **Evidence:** REPORTED for training and numbers; letter-option prompt VERIFIED in src/rizzo_flow/prompts.py.

### mini-jev
- **URL:** https://github.com/r-ms/mini-jev **Author:** r-ms (Mikhail) **License:** MIT **Popularity:** 57 stars / 5 forks **Dates:** created 2026-09-17 / last push 2026-09-18
- **Relationship to Jev:** inspired by. A preregistered measurement study of the interface idea (read option-letter logits vs grammar-constrained JSON), not a server or API clone.
- **How it works:** Qwen3-4B-Instruct-2507 (bf16, frozen), PyTorch with xgrammar for the JSON baseline. Each closed-choice field becomes a lettered multiple-choice question; one forward pass reads the option-letter scores (A to Z; more than 26 needs a two-level code, not built). Shared-prefix KV cache across fields. Decision types measured: enum (choice) and boolean; score not measured. Shares are normalised candidate scores, explicitly not calibrated. No training; a local demo page shows the steps.
- **Benchmark claims (author-reported):** CLINC150 intent, 6750 paired observations: JSON 0.909 vs letters 0.907 (CI covers 0); about 4x faster at 32 tokens; letter beats option-name by 10 pp; asking the model to write probabilities scores 0.346. Run records on the Hub (27,900 decisions).
- **Status:** dormant (no push since 2026-09-18).
- **Evidence:** VERIFIED (minijev/letters.py: 26 letter tables); results REPORTED.

### sarvam-jev
- **URL:** https://github.com/SAGAR-TAMANG/sarvam-jev **Author:** SAGAR-TAMANG **License:** not stated **Popularity:** 58 stars / 10 forks **Dates:** created 2026-09-18 / last push 2026-09-18
- **Relationship to Jev:** inspired by, a port of SemIf-OpenJev's scorer and fixtures (it benchmarks against "openjev's ladder" and a `reference/openjev` checkout) to Sarvam's Indic models. Not a Jev-compatible server.
- **How it works:** sarvamai/sarvam-1 (2B base, BF16, RTX 3060). Options as lettered slots; softmax over letter token ids; the state, instructions and 3 few-shot examples are prefilled once into a KV cache that is replicated across criteria branches (native prefix cache). Choice-style 3-option decisions only so far. Calibration head (LoRA, Brier/log loss) and packed branch attention are planned, not done. Browser demo via wllama.
- **Benchmark claims (author-reported):** 144 authored decisions, mean family balanced accuracy 0.516 (chance 0.333), vs published Qwen3-0.6B 0.440, MiniCPM5-2B 0.686, Qwen3.5-4B 0.813. Shared-state path 5.3x faster than fresh on a Hindi preset, but only 64/72 argmaxes agree in BF16 (drift attributed to a near-indifferent base model).
- **Status:** abandoned/dormant (single day of activity).
- **Evidence:** VERIFIED (src/sarvam_jev/shared.py prefix KV scoring); numbers REPORTED.

### cygnet-recipe
- **URL:** https://github.com/blockbrain-ai/cygnet-recipe **Author:** Blockbrain **License:** MIT (Gemma terms noted in NOTICE.md) **Popularity:** 25 stars / 7 forks **Dates:** created 2026-09-24 / last push 2026-09-29
- **Relationship to Jev:** API-compatible wrapper. A small shim over unmodified vLLM 0.30.0 presents `/v1/systemone`. Frozen model, no fine-tuning, no Jev outputs.
- **How it works:** google/gemma-4-12B-it, one chat request per decision with `max_tokens=1`, thinking off. vLLM masks the answer position to option letters (`structured_outputs.choice`) and returns top-20 logprobs; the shim sums the mass of all tokens spelling each letter (Gemma has duplicates), renormalises, and applies one temperature p^(1/T), T=3.4, fitted on its own generated items, not JevBench. Up to 26 options in the benchmark shim; `decision_server.py` extends to 255 via group-then-winner passes. Confidence = (K·p_max - 1)/(K - 1). choice, noul, score supported.
- **Benchmark claims (author-reported):** JevBench public 231 items via JevBench's CLI: 203/231 (87.9%) on both an A6000 and an L40S (204 on a pinned-revision rerun); 704 input tokens per decision; p50 0.050 to 0.066 s. ECE at T=3.4 vs T=1: 0.067 vs 0.249 on 77 intents, but 0.171 vs 0.074 on 150 intents. No Jev score quoted.
- **Status:** active, small (last push 3 days ago).
- **Evidence:** VERIFIED (shim/cygnet_shim.py: max_tokens=1, top_logprobs, TEMPERATURE).

### jev-rs
- **URL:** https://github.com/yijunyu/jev-rs **Author:** yijunyu **License:** Apache-2.0 **Popularity:** 17 stars / 2 forks **Dates:** created 2026-09-21 / last push 2026-09-28
- **Relationship to Jev:** API-compatible wrapper. A Rust harness (`jev serve` speaks `/v1/systemone`; `jev mcp` exposes a `judge` tool to coding agents). It also has a backend that calls the real Jev/TypeSafe API for `--compare`, so parts of it are a Jev client.
- **How it works:** Backend-agnostic: llama-server (raw `/completion` with `n_predict:1`, `n_probs`), or any OpenAI-compatible endpoint with `logprobs`. No default model; examples use Qwen3-4B GGUF. State is rendered as a shared prefix; each question ends with `Answer:` and lettered options; restricted softmax divided by a fitted temperature per (question type, option count) bucket (`jev calibrate`); `--permutations k` averages option rotations. Adapters for Laya and AgentJev; `jev ensemble` fits a logistic stacker across backends. choice, noul, score.
- **Benchmark claims (author-reported):** own 75-decision dev_tasks: Qwen3-4B accuracy 0.71, ECE 0.244; Qwen3.8-Flash 0.84, ECE 0.108. On typed-decisions test: Qwen3-4B logprob path 0.539 vs Laya 0.766 and AgentJev-0.6B 0.796.
- **Status:** active but low traffic (last push 4 days ago).
- **Evidence:** VERIFIED (src/backend/llamacpp.rs: `/completion`, `n_probs`); numbers REPORTED.

### OpenDecision
- **URL:** https://github.com/deepanwadhwa/OpenDecision **Author:** deepanwadhwa **License:** Apache-2.0 **Popularity:** 57 stars / 3 forks **Dates:** created 2026-09-17 / last push 2026-09-25
- **Relationship to Jev:** API-compatible wrapper. The README calls it the open equivalent of Jev; it offers `POST /v1/systemone` plus its own `/v1/documents/decide`. The mechanism is unlike the other clones: it uses a zero-shot NLI classifier, not a decoder LLM.
- **How it works:** Default MoritzLaurer/ModernBERT-large-zeroshot-v2.0 (DeBERTa-v3-large optional) run through the HF `zero-shot-classification` pipeline: per-candidate entailment scoring with a hypothesis template built from the question and option descriptions. Adds a `Relation` type (supports/contradicts/unknown/conflicted) and document handling (section split, passage retrieval, evidence return); yes/no modes `binary`, `three_way`, `both`. Python 3.13 or later. The README says scores are uncalibrated. No training in the repo.
- **Benchmark claims (author-reported):** synthetic insurance claim: 17/17 required facts retrieved and 10/10 composed decisions (one development case); GDPR document 9 of 10 questions. No comparison to Jev. Doom demo with no win rate.
- **Status:** active, developer preview v0.1.2 (last push 7 days ago).
- **Evidence:** VERIFIED (src/opendecision/engine.py: `pipeline("zero-shot-classification")`, DEFAULT_MODEL, hypothesis_template).

### snap
- **URL:** https://github.com/emnlmn/snap **Author:** emnlmn **License:** MIT **Popularity:** 22 stars / 0 forks **Dates:** created 2026-09-23 / last push 2026-10-01
- **Relationship to Jev:** API-compatible wrapper (Jev wire format, plus extensions). It also ships a fine-tuned default model, so it is partly a trained model.
- **How it works:** Rust binary with llama.cpp built in. Default `snap1-2b`: MiniCPM5-2B fine-tuned for letter readout (training pipeline in TRAINING.md, trained on other workflows only); also runs Qwen3.8-4B and Spark-4B GGUFs. Each question becomes a letter prompt (A to Z); one batched `llama_decode` with a shared-prefix KV tree; read the letter-logit row, merge variant tokens (`A` vs ` A`), softmax; `coverage` reports mass on the letters. More than 26 options (up to 256): one yes/no probe per option, normalised. Extras: `numeric` type, optional abstain slot, layout modes, TOON state serialisation. `snap calibrate` fits temperature per question type, bound to model and prompt version.
- **Benchmark claims (author-reported):** own 301-case suite: snap1-2b 85.0% (balanced 69.8%), Qwen3.8-4B 86.4%; typed-decisions zero-shot accuracy 0.624, ECE 0.043. Comparison table lists snap at "73% on a 4B model" on the TypeSafe public eval vs Jev "~88% self-reported". 48% lower latency than Ollama for 4 to 8 questions.
- **Status:** active (last push yesterday).
- **Evidence:** VERIFIED (src/engine.rs: letters A to Z, softmax at one position); training REPORTED.

### sys1
- **URL:** https://github.com/alvarobartt/sys1 **Author:** alvarobartt **License:** Apache-2.0 **Popularity:** 48 stars / 2 forks **Dates:** created 2026-09-22 / last push 2026-09-30
- **Relationship to Jev:** API-compatible wrapper: a Rust inference server (axum, candle) exposing `/v1/systemone` and `/v1/decide`. It contains no model of its own; it serves Laya, a ModernBERT-based encoder trained by Convai Innovations.
- **How it works:** Supports `convaiinnovations/laya`, `laya-multilingual` and `laya-typed-decisions`. The decisions come from the encoder's classifier head, not from an LLM's next-token logits (src/models/laya.rs wraps a ModernBERT encoder; razorback16's README describes Laya as ModernBERT-large fine-tuned for typed decisions). Token-based dynamic batching, SDPA on CPU/Metal/CUDA, Flash Attention 2/3. Calibration and decision-type details are inherited from Laya and are not stated here. No training or distillation in the repo.
- **Benchmark claims:** none beyond "blazing fast"; no numbers.
- **Status:** active (last push 2 days ago).
- **Evidence:** REPORTED (README); src/models/laya.rs seen only as an encoder wrapper, so mechanism is partly VERIFIED.

### stuntd
- **URL:** https://github.com/bladedevoff/stuntd **Author:** bladedevoff **License:** Apache-2.0 **Popularity:** 56 stars / 1 fork **Dates:** created 2026-09-23 / last push 2026-09-30
- **Relationship to Jev:** distillation from Jev outputs, optionally. In front of the paid Jev API (or OpenAI/Anthropic) it records provider answers and trains local heads to replace them. Without an upstream it is a plain local `/v1/systemone` server on Laya.
- **How it works:** Local proxy speaking `/v1/systemone`, OpenAI Chat Completions and Anthropic Messages. Frozen Laya encoder (convaiinnovations/laya); a per-decision-site small head (about 50 MB) is trained on recorded provider answers (24 epochs, a few thousand rows). A site answers locally only when its calibrated confidence meets `target_agreement` (default 0.99); otherwise the request goes to the provider. Types: choice, noul, score; multi-field JSON objects get one head per field.
- **Benchmark claims (author-reported):** trained heads vs zero-shot Laya: support 99.5/85.5/97.5% vs 69.5/34.5/66.5%; devtools 97.0/100.0% vs 45.0/84.5%; banking `risk` 30.0% to 72.5%. Quotes jevbench: Laya zero-shot 38.2 on banking77 vs Jev 76.4. Agreement is with the teacher, not human labels.
- **Status:** active (v0.1, last push 2 days ago).
- **Evidence:** REPORTED (README; no mechanism code read).

### llm2jev
- **URL:** https://github.com/tic-top/llm2jev **Author:** tic-top **License:** MIT **Popularity:** 7 stars / 0 forks **Dates:** created 2026-09-23 / last push 2026-09-24
- **Relationship to Jev:** API-compatible wrapper. It implements the documented System One wire format and says it is independent. Formerly named AnyJev (before 0.5.0); not the same as Nokia's AnyJev.
- **How it works:** Any chat model on SGLang, vLLM, transformers or MLX; Qwen3.5-2B in examples, parity tested on Qwen3 and Qwen3.6/3.8 up to 35B-A3B. The state is rendered with the model's chat template (thinking off), the question goes in a last user turn with an `Answer:` prefill; the label token logprobs at that one position (`max_new_tokens=1`) are softmaxed with an optional global temperature. Labels A to Z then AA, AB (up to 255), checked at startup to be a single token. All options in one prompt; shared prefix warmed once. Images, video and audio in the state. No training; the README also pitches the readout as an RL policy.
- **Benchmark claims (author-reported):** zero-shot JevBench public: Qwen3.5-27B 0.879 (rank 4 of 50), Qwen3.6-27B 0.866, Qwen3.8-27B 0.840, Qwen3.5-4B 0.740. Estimated v1.4 scores are its own extrapolation.
- **Status:** low activity (last push 8 days ago).
- **Evidence:** VERIFIED (llm2jev/scoring.py: softmax over label logprobs); parity and numbers REPORTED.

### LitJev
- **URL:** https://github.com/zhengxuyu/litjev **Author:** ZhengxuYu **License:** Apache-2.0 **Popularity:** 46 stars / 9 forks **Dates:** created 2026-09-17 / last push 2026-10-01
- **Relationship to Jev:** clean-room reimplementation. It is a "hypothesis-based reproduction from public information", wire-compatible (`/v1/systemone`) with the documented Jev schema; no Jev outputs used.
- **How it works:** Qwen family; default Qwen3.8-27B on one H100, transformers or SGLang backends. The default readout teacher-forces each option's own words after `Answer:` and softmaxes the summed token log-probability (src/litjev/content_readout.py); `--readout coded` restores single-token letter codes. State is prefilled once (SGLang radix cache, or batched on transformers). Questions do not see one another. Optional post-hoc temperature calibration. Experimental "System Two" (`/v1/systemtwo`) uses a trained decision head to route some questions to the backbone's own slow thinking. Up to 10 questions per request.
- **Benchmark claims:** MMLU-Pro direct scoring, Doom and chess from screenshots; no comparison with Jev stated in the README.
- **Status:** active (last push yesterday).
- **Evidence:** VERIFIED (src/litjev/content_readout.py: teacher-forced option summed logprob).

### Lichen
- **URL:** https://github.com/Mushroom-Systems/lichen **Author:** Mushroom Systems **License:** MIT **Popularity:** 58 stars / 3 forks **Dates:** created 2026-09-23 / last push 2026-09-30
- **Relationship to Jev:** API-compatible wrapper ("drop-in replacement"; `/v1/systemone`). The idea is credited to Duarte O. Carmo's "Jev in 25 lines of Python".
- **How it works:** Frozen Gemma-4-26B-A4B (QAT on llama.cpp, NVFP4 on vLLM); other models (Qwen3.6-35B-A3B, Qwen3.5-9B) also tested. One chat prompt per question with labels (letters for choice, Yes/No, digits for score); answer is the next-token distribution over labels. Prompt techniques: state and question written twice (prompt repetition), each option listed twice in rotated order and summed by option. Confidence: logits divided by a fitted temperature (1.25 QAT, 2.25 NVFP4, fitted on its own 98 questions), then pulled toward uniform by the disagreement of the two readings; neither changes the argmax. Images in the state. No training.
- **Benchmark claims (author-reported):** JevBench public 231: Lichen Gemma QAT 0.896 (207/231), NVFP4 0.883, vs Jev 1.13.0 0.866 (200/231); the same model without the prompt techniques 0.874. Median latency 56 ms (vLLM) vs Jev 665 ms.
- **Status:** active (last push 2 days ago).
- **Evidence:** VERIFIED (lichen/method.py: label softmax, permute, repeat options); numbers REPORTED.

### OpenJev (ejhshen)
- **URL:** https://github.com/ejhshen/OpenJev **Author:** ejhshen (weights: shenjunhao/OpenJev-4B on HF) **License:** MIT **Popularity:** 15 stars / 0 forks **Dates:** created 2026-09-29 / last push 2026-09-29
- **Relationship to Jev:** clean-room reimplementation as a trained model. It is not a prompting wrapper, and it is an unrelated project to the other "openjev" repos.
- **How it works:** OpenJev-4B: the text backbone of Qwen3.5-4B plus a lightweight Option Set Interactor and a shared decision head. Options are encoded by shared (permutation-equivariant) option encoders and the head outputs a distribution over a request-defined candidate set, with parallel option branches and shared-prefix reuse (attention and recurrent states). It is trained on OpenJevData-140k in two stages: full-parameter SFT, then REINFORCE-Analysis (16 sampled actions, 10% uniform exploration, outcome rewards, 25% SFT replay, frozen-reference KL). Types: choice (up to 255), noul, score (1 to 10 levels). Training code (verl/FSDP2) included. No distillation from Jev stated.
- **Benchmark claims (author-reported, HF model card):** JevBench public 88.31% (Brier 0.1764) vs Jev 1.13.0 86.15%; MMDM hard 85.57% vs Jev 86.37%; MMDM soft Brier 0.0391 vs Jev 0.1119. Also compared with Decider 4B, JevK5 4B and Intern-Decision-4B. The README itself notes the MMDM split mixes dev and test, so it is not a blind test.
- **Status:** active but brand new (single day of commits).
- **Evidence:** REPORTED (README, repo layout and HF card; model and training code not read).

### openjev (zhihz)
- **URL:** https://github.com/zhihz/openjev **Author:** zhihz **License:** not asserted (GitHub: NOASSERTION) **Popularity:** 35 stars / 4 forks **Dates:** created 2026-09-16 / last push 2026-09-16
- **Relationship to Jev:** inspired by. Local research preview ("Open JEV"), bilingual EN/ZH, with its own `/decide` API (not the Jev wire format). It says it is not affiliated and has no demonstrated lead over Jev. Not related to AlexWortega's HF openjev as far as the README shows.
- **How it works:** Frozen Qwen3-4B-Instruct-2507 (FP16, or MLX 8-bit on Apple Silicon); a Qwen3-0.6B variant is also benchmarked. Single next-token label softmax over distinct single-token labels, thinking off, limited to the supplied candidates (2 to 8 per question; Choice and Binary). Each question is run sequentially and re-encodes the context (shared-context compute is planned). The Score schema is experimental and not exposed. Probabilities are not calibrated. No training.
- **Benchmark claims (author-reported):** fixed development challenge: 4B 212/236 (89.8%), 0.6B 155/236 (65.7%); Belebele-derived reading subset 125/128 for the 4B. No Jev comparison.
- **Status:** abandoned/dormant (one commit on 2026-09-16, 16 days idle).
- **Evidence:** VERIFIED (decisionmaking/instruction.py: label ids, single-token softmax).

## Group D: Hugging Face-hosted models (census as of 2026-10-02)

### pngwn/system-one-qwen3.5-4b-scorer
- **URL:** https://huggingface.co/pngwn/system-one-qwen3.5-4b-scorer **Author:** pngwn **License:** CC-BY-NC-4.0 (ticket data is NC) **Popularity:** ~1,000 downloads / 20 likes **Dates:** created 2026-09-16 / last modified 2026-09-16
- **Relationship to Jev:** clean-room reimplementation. Card says it is "in the shape of TypeSafe's Jev"; no /v1/systemone server, no Jev outputs used. (The card does not mention nanodiff-350m-typed-decisions; not examined.)
- **How it works:** Qwen3.5-4B-Base + LoRA r=16 (all linear layers) + new scalar `score` head (30.5M trainable of 4.24B), via sequence classification. Each (state, question, option) triple is scored separately; per-question logits are softmaxed. Option cap 16 in training, uncapped at test; max_len 384. Choice, yes/no and numeric Score. Confidence is the softmax; temperature T=1.75 fitted on val (ECE 0.135 to 0.044). Trained 2,200 steps on `pngwn/system-one-decisions` (12,913 questions, 9 task families, public datasets), 1 A100, ~3h.
- **Benchmark claims:** author-reported: held-out accuracy 0.707, ECE 0.044 over 9 own task families; 0.748 vs prompted-base 0.679 on 7 families. No comparison with Jev.
- **Status:** one release, not updated since 2026-09-16 (inactive 2 weeks, no sign of abandonment).
- **Evidence:** VERIFIED (README.md; repo ships `system_one.py`, `adapter_config.json`).

### openjev/openjev
- **URL:** https://huggingface.co/openjev/openjev **Author:** org "openjev" (contact support@loopai.com) **License:** weights CC-BY-NC-4.0, helper/serve code Apache-2.0 **Popularity:** 4,582 downloads / 94 likes **Dates:** created 2026-09-20 / last modified 2026-09-30
- **Relationship to Jev:** API-compatible wrapper plus fine-tune. Card: "request and response shapes follow the hosted Jev API"; `helper/shim.py` serves POST /v1/systemone. States "independent project, not affiliated with TypeSafe". Whether hosted-Jev outputs were used for training is not stated.
- **How it works:** Qwen-family 27B VLM (`Qwen3_5ForConditionalGeneration`, hidden 5120; NOTICE names Qwen/Qwen3.8-27B as base), bf16 54 GB. Options get letters; one forward pass reads the logits of those letters at the first output position; helper applies fixed temperature calibration (READOUT_T=0.85, NOUL_T=1.83). Up to 52 options per pass, otherwise grouped rounds; 16k-token prompts, one image; DOM/screenshot agent decisions. Tuned for order-invariance (flips 18.5% to 2.3%). Training data not described. Formats: FP8, MLX 8/4-bit, GGUF.
- **Benchmark claims:** author-reported: 10,000 text questions, 34 public sources, same questions to all: Jev hosted 85.4%, OpenJev 84.0%, base 80.4%, Nimble 9B 75.7%; MiniWoB 39 tasks tied with Jev.
- **Status:** active (updated 2026-09-30).
- **Evidence:** VERIFIED (README.md, NOTICE, config.json; helper not read).

#### AlexWortega/openjev (already known, 3 lines)
- https://huggingface.co/AlexWortega/openjev, MIT, created 2026-09-16, modified 2026-10-01, 631 likes; base Qwen3.5-4B (also 2B, 0.8B, 35B-A3B).
- Mechanism: NLI cross-encoder (entailment/contradiction/neutral), one forward pass per option; JevBench v1.2 public 0.814 vs Jev 1.13 0.866 (author-reported).
- Different architecture, licence and author from openjev/openjev (see conclusion in report).

### Meanblock/JEV-CPU
- **URL:** https://huggingface.co/Meanblock/JEV-CPU **Author:** Meanblock (GitHub leesk212/JEV-CPU) **License:** MIT **Popularity:** 0 downloads / 48 likes **Dates:** created 2026-09-19 / last modified 2026-09-19
- **Relationship to Jev:** fork of another clone: CPU port of TheoLeeCJ/SemIf (formerly OpenJev). Card: "not affiliated with ... TypeSafe, or Jev".
- **How it works:** Qwen3-0.6B (float32, CPU) with unchanged SemIf engine. Prompt: system instruction plus JSON {evidence, criterion, options with letters A, B, ...}; chat template with thinking off; one forward pass, last-position logits gathered at the single-token option letters, softmax over them. Choice-style decisions with an own `/api/decide` endpoint and web UI (not /v1/systemone). No training; probabilities uncalibrated ("calibrate on your workload"). ~1 s per decision.
- **Benchmark claims:** author-reported: demo table of 13 decisions in 8 domains. Quotes SemIf's 0.6B balanced accuracy 0.440 (4B: 0.813).
- **Status:** one-shot release, no update since 2026-09-19.
- **Evidence:** VERIFIED (README.md; code inventory only, `src/semif_phase1/` not read).

### aimeigaoshou/agent-jev
- **URL:** https://huggingface.co/aimeigaoshou/agent-jev **Author:** aimeigaoshou (code: github.com/malevrigs/agent-jev, citation author "malevrigs") **License:** Apache-2.0 **Popularity:** 949 downloads / 38 likes **Dates:** created 2026-09-21 / last modified 2026-09-26
- **Relationship to Jev:** clean-room reimplementation ("0.6B System One Decision Model"; Boolean/Choice/Score). Its invoice number is agreement with a "public teacher argmax", teacher not named in the card (could be hosted Jev; not stated).
- **How it works:** Qwen3-0.6B without the LM head plus a permutation-equivariant candidate head; softmax per question; candidates share a prefix cache (609.7 to 298.9 ms on a 64-option load). Per-type temperatures in `temperatures.json`. This checkpoint is the coding-completion model: SFT then RLCD on executed coding pairs. 2,048-token cap. Served by own `jev_service` on port 8149, plus client.
- **Benchmark claims:** author-reported: coding completion 57.8% (606/1048), AUROC 0.589; invoice 87.2% teacher agreement. No Jev comparison.
- **Status:** active (updated 2026-09-26).
- **Evidence:** VERIFIED (README.md, config.json; code repo not read).

### samatv256/mini-Jev
- **URL:** https://huggingface.co/samatv256/mini-Jev **Author:** samatv256 **License:** Apache-2.0 **Popularity:** 248 downloads / 19 likes **Dates:** created 2026-09-21 / last modified 2026-10-01
- **Relationship to Jev:** clean-room reimplementation, narrowed to tool/action selection. Cites Jev only as the "typed decisions" format; trained partly on the author's `jev-decisions-v1` and evaluated on LocalLLaMA/typed-decisions.
- **How it works:** Qwen3-0.6B (NF4 4-bit) + LoRA adapter + compact decision head (mean pooling, projection 256, 2 set layers); candidates scored permutation-equivariantly and softmaxed. Choice (focus), plus noul/score. 8,192-token branches. Probabilities explicitly uncalibrated. Interim checkpoint at 10,626 of a planned 50,000 updates.
- **Benchmark claims:** author-reported: BFCL function-selection adaptation 96.5% (193/200, not an official score); LocalLLaMA/typed-decisions full test only 34.0% (Choice 27.0, Noul 52.2, Score 25.6); 17+ candidates 4/17.
- **Status:** active, mid-training.
- **Evidence:** VERIFIED (README.md, config.json; `inference.py` not read).

### TypeSafeAI/Qyvos
- **URL:** https://huggingface.co/TypeSafeAI/Qyvos **Author:** HF org "TypeSafe AI" (unverified; see report) **License:** not stated (no license field) **Popularity:** 38 downloads / 0 likes **Dates:** created 2026-09-30 / last modified 2026-09-30
- **Relationship to Jev:** inspired by / third-party derivative. Tagged `open-jev`; trained on the ZefanCai/Open-Jev dataset. The card never claims to be TypeSafe's, and has no /v1/systemone server.
- **How it works:** SupersonicLabs/Julia-1, a ModernBERT-small encoder (22 layers, hidden 384, 140.5M, frozen), plus a new 3.7M head (2 transformer layers, type embedding, per-option MLP scorer). Choice, score and noul; softmax over option scores. Head-only fine-tune, 30,000 rows of Open-Jev `release-v2-redistributable`, soft-target cross-entropy, 3,750 steps, 1.64 GB RSS.
- **Benchmark claims:** author-reported: test accuracy 83.1% vs 83.6% for untuned Julia-1 (within noise); softCE improves 6.7% (calibration). No Jev comparison.
- **Status:** brand new, single upload.
- **Evidence:** VERIFIED (README.md, config.json, file list).

### Maincode/matilda-jev-v1
- **URL:** https://huggingface.co/Maincode/matilda-jev-v1 **Author:** Maincode **License:** Apache-2.0 **Popularity:** 151 downloads / 6 likes **Dates:** created 2026-09-30 / last modified 2026-10-02
- **Relationship to Jev:** API-compatible, independently trained model; name and tags say "jev". Bundled server exposes `POST /v1/systemone` with the Jev response shape. Whether Jev outputs were used is not stated.
- **How it works:** 26.1B Qwen3.5-style multimodal backbone (hidden 5120, linear/full attention mix, same layout as Qwen3.8-27B); LM head removed and replaced by a 255-option readout (codes A..Z, AA, AB, ...). One pass, softmax per question, stored calibration temperature; choice/noul/score, text/JSON plus images. bf16 ~49 GiB. Training data and method not described.
- **Benchmark claims:** author-reported: Decision Index 0.2.1 over 150,317 requests: 59.59 overall (awaiting official submission). The same figure appears in the bekko card as Jev 1.13's Task Avg (59.59), which is suspicious, not explained.
- **Status:** active (updated 2026-10-02).
- **Evidence:** VERIFIED (README.md, config.json, `decision_config.json`, USAGE.txt).

### hotchpotch/bekko-system-one-v0-400m
- **URL:** https://huggingface.co/hotchpotch/bekko-system-one-v0-400m **Author:** hotchpotch **License:** not finalized (card assigns none) **Popularity:** 0 downloads / 3 likes **Dates:** created 2026-09-29 / last modified 2026-09-30
- **Relationship to Jev:** clean-room reimplementation: "same category as TypeSafe AI's Jev"; also builds the S1MB benchmark. No API-compat claim, no Jev outputs.
- **How it works:** cross-encoder/ettin-reranker-400m-v1 (ModernBERT-compatible, 395M), full fine-tune. Shared-prefix encoder: instructions+state encoded once, candidates attend to the prefix but not each other; mean pooling; Choice/Noul/Score heads, softmax per decision; also ranking. 8k context, ONNX browser export (17M, 68M variants). 8.42M judgments from `bekko-system-one-dataset-v0` (153 subsets), 16,517 updates, 25 h.
- **Benchmark claims:** author-reported on S1MB (137 benchmarks): Task Avg 50.60 vs Jev 1.13 59.59; on synthetic generalization 54.48 vs Jev 96.27. Admits 77 subset names overlap train and eval.
- **Status:** active, v0.
- **Evidence:** VERIFIED (README.md).

### tasksource/tasksource-jev-nano-v0
- **URL:** https://huggingface.co/tasksource/tasksource-jev-nano-v0 **Author:** tasksource (Damien Sileo) **License:** Apache-2.0 **Popularity:** 67 downloads / 2 likes **Dates:** created 2026-09-28 / last modified 2026-09-28
- **Relationship to Jev:** clean-room reimplementation; "No teacher model was used".
- **How it works:** lightonai/LateOn (ModernBERT ColBERT-style, ~149M) via PyLate. State+question encoded once to token vectors; each option encoded independently; MaxSim score; softmax with learned temperature 0.53. Permutation-equivariant. Choice, noul, score. Trained on `tasksource/tasksource-jev-typed-decisions` (512k decisions, 519 tasks, 8000 steps, checkpoint picked on unseen tasks).
- **Benchmark claims:** author-reported: unseen-test 49.7%, Fast Decisions 41.3%, classifier-bench v2 macro 57.2%; behind GLiNER2.5 Base on some sets. No Jev comparison.
- **Status:** one release, not updated since 2026-09-28.
- **Evidence:** VERIFIED (README.md, config file list).

### argos1111/modernbert-ja-310m-jev
- **URL:** https://huggingface.co/argos1111/modernbert-ja-310m-jev **Author:** argos1111 (GitHub Argos1111/jev_local) **License:** CC-BY-SA-4.0 **Popularity:** 1,080 downloads / 15 likes **Dates:** created 2026-09-19 / last modified 2026-09-19
- **Relationship to Jev:** API-compatible wrapper (Jev Local serves `/v1/systemone`); card says "unofficial, unrelated to TypeSafe's Jev" and does not reproduce Jev's accuracy.
- **How it works:** sbintuitions/modernbert-ja-310m cross-encoder, `ModernBertForSequenceClassification` with 1 label, CLS pooling. "Question: ... State: ..." paired with each candidate (`label - description`); candidate scores softmaxed (Choice, Score levels, Noul as [true,false]). Listwise cross-entropy, 94,384 questions from JGLUE, JCoLA, JCommonsenseMorality, MASSIVE ja, 2 epochs. Confidence is distribution concentration, not calibrated. 512-token cap.
- **Benchmark claims:** author-reported: JNLI 92.62%, JCommonsenseQA 92.40% (in-distribution) vs LFM2.5-1.2B zero-shot; weaker out-of-distribution (livedoor 35.6% vs 46.8%).
- **Status:** one release, not updated since 2026-09-19.
- **Evidence:** VERIFIED (README.md).

### HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B
- **URL:** https://huggingface.co/HIT-TMG/JevEmbed-Qwen3-Embedding-0.6B **Author:** HIT-TMG (HITsz-TMG/JevEmbed) **License:** Apache-2.0 **Popularity:** 468 downloads / 1 like **Dates:** created 2026-09-26 / last modified 2026-09-26
- **Relationship to Jev:** API-compatible wrapper: requests use the Jev shape (`state`, `questions`, `criteria`, `model`). Training data `JevEmbed-Data` includes `openjev:` IDs, so it likely derives from the Open-Jev dataset (inference).
- **How it works:** Qwen3-Embedding-0.6B, LoRA r=64 on Q/K/V, merged; 1,024-d embeddings; choice/score by cosine similarity of state+question embedding to option embeddings (temperature 0.1), noul by a sigmoid slope of 10. 1,024-token cap. Trained one epoch on 1,601,157 questions, 16 GPUs. Variants exist (KaLM-Embedding V2.5).
- **Benchmark claims:** author-reported on own test split (66,482 questions): 82.30% vs 33.79% for base; choice 84.5, score 69.6, noul 94.4. No Jev comparison.
- **Status:** one release, not updated since 2026-09-26.
- **Evidence:** VERIFIED (README.md).

### Cloudflare/clef
- **URL:** https://huggingface.co/Cloudflare/clef **Author:** Cloudflare (card links blog.cloudflare.com/clef-decision-models; org identity not independently checked) **License:** Apache-2.0 **Popularity:** 18 downloads / 425 likes **Dates:** created 2026-09-30 / last modified 2026-10-01
- **Relationship to Jev:** API-compatible, independently trained: "fully compatible with Jev and SystemOne"; `systemone()` takes a /v1/systemone body and returns the same body. Training on Jev outputs not stated.
- **How it works:** Qwen3.8-27B (with vision encoder) post-trained; a joint schema head (2 routing + 4 transformer layers, width 1024, hidden 5120) reads final hidden states, routes state evidence to each question and scores all options of all questions jointly. Softmax per question; choice/score/noul; text, JSON, images, video; 16,384 tokens. Training data not described. A smaller Clef-Flash exists.
- **Benchmark claims:** author-reported, internal Decision Index 0.2.1: beats Jev on most of ~37 benchmarks but loses on MMLU-Pro (65.9 vs 82.7), BBH (73.7 vs 92.9), GPQA (48.0 vs 78.3). Median latency 209 ms vs 524 ms.
- **Status:** active (updated 2026-10-01).
- **Evidence:** VERIFIED (README.md, `joint_head_config.json`).

### perplexity-ai/pplx-decider-v1-27b
- **URL:** https://huggingface.co/perplexity-ai/pplx-decider-v1-27b **Author:** perplexity-ai (org not verified here) **License:** Apache-2.0 (weights), MIT (`source/`) **Popularity:** 0 downloads / 33 likes **Dates:** created 2026-10-01 / last modified 2026-10-01
- **Relationship to Jev:** API-compatible wrapper: `source/.../server.py` is a FastAPI "TypeSafe decision API" with aliases "jev-latest", "jev-1.13.0"; NOTICE disclaims affiliation with TypeSafe. Labels and states come from public datasets and synthetic data; Jev outputs not mentioned. Code package is named `autojev`.
- **How it works:** Qwen3.8-27B with the LM head replaced by a 255-option decision readout, all retained weights fully fine-tuned (lr 2e-6, 73k rows, update 200 of 286); prompt "State / Question / Options" with letter codes; temperature 2.2076 fitted on 3,500 rows saved in `decision_config.json`. Choice, noul, score, images.
- **Benchmark claims:** author-reported "through the Perplexity API": overall 85.71% vs Jev 84.51% and base Qwen3.8-27B 74.76% across 11 benchmarks; loses to Jev on WinoGrande, BBH, TruthfulQA, JevBench hard (70.3 vs 73.3).
- **Status:** active, just released.
- **Evidence:** VERIFIED (README.md, NOTICE, decision_config.json, training.json, `model.py`, `server.py` grep).

### togethercomputer/Tev1-4B-experimental
- **URL:** https://huggingface.co/togethercomputer/Tev1-4B-experimental **Author:** Together AI (togethercomputer) **License:** not stated ("being finalized") **Popularity:** 2,805 downloads / 27 likes **Dates:** created 2026-09-23 / last modified 2026-09-23
- **Relationship to Jev:** inspired by: "a Jev-inspired experiment, not a non-autoregressive Jev runtime". No /v1/systemone.
- **How it works:** Qwen3.5-4B, LoRA SFT (rank 8, 1 epoch, lr 5e-5, 2,048 tokens), keeps the standard LM head. Prompt JSON {state, question, options 2-24 with letters}; generates a single option letter at temperature 0, thinking off (constrained to option letters in the example client). Choice only, no probabilities, no calibration. 37,840 training examples (language classification, policy, routing, synthetic research), per github.com/togethercomputer/tev1.
- **Benchmark claims:** author-reported dev-set only: 880/1,000 main, 300/300 policy transfer; no baseline, no Jev comparison.
- **Status:** one-off experiment, not updated since 2026-09-23.
- **Evidence:** VERIFIED (README.md; tev1 GitHub README).

### fastino/GLiNER2.5-Decide
**EXCLUDE.** A 340M general GLiNER2 classifier (`classify_text`, label sets at call time; DeBERTa-v3-large span extractor); no mention of Jev, typed decisions, noul/score types or /v1/systemone beyond a leaderboard row against JevK5 and SemIf on fastino's own fast-decisions set (60.2% avg). Borderline: a benchmark competitor, not a Jev clone. Apache-2.0, 38,386 downloads / 295 likes, created 2026-09-23.

### StandardThinking/StandardOne-8B
- **URL:** https://huggingface.co/StandardThinking/StandardOne-8B **Author:** StandardThinking **License:** Apache-2.0 **Popularity:** 2,120 downloads / 0 likes **Dates:** created 2026-09-24 / last modified 2026-09-27
- **Relationship to Jev:** API-compatible wrapper plus fine-tune: `jev-adapter` serves POST /v1/systemone (alias `jev-latest`) over SGLang; trained on public datasets and synthetic data, no Jev outputs mentioned.
- **How it works:** Ministral-3-8B-Instruct-2512 (Mistral3 VLM), LoRA on language-model projections (44.6M trainable), merged. Prompt with State/Question/Options and letters A-Z (max 26); one forward pass, logits over letters read via SGLang; per-type temperatures fitted on non-JevBench data (choice 0.85, noul 0.85, score 0.70); entropy-based confidence. ~520k-row training mixture, 8 languages, plus 3B, FP8 and GGUF variants.
- **Benchmark claims:** author-reported, served: JevBench public hard 54.95% vs Jev 1.13 72.07%; standard 93.06 vs 98.61; wins on own realistic-transfer and probability suites; hard-tier ECE 0.184 vs 0.099.
- **Status:** active (v2, 2026-09-26).
- **Evidence:** VERIFIED (README.md).

### EldanRing/Winnow-12B
- **URL:** https://huggingface.co/EldanRing/Winnow-12B **Author:** EldanRing (inference code github.com/EldanRing/winnow-inference) **License:** Apache-2.0 **Popularity:** 21,482 downloads / 35 likes **Dates:** created 2026-09-20 / last modified 2026-09-21
- **Relationship to Jev:** API-compatible wrapper plus fine-tune: server provides `/v1/systemone` and `/v1/chat/completions`; "not affiliated with TypeSafe". Training includes "teacher-supervised examples"; the teacher is not named (Jev not stated).
- **How it works:** Gemma 4 12B IT, LoRA r=32/alpha 64 on all projections, merged; GGUF (BF16, Q8_0) + vision projector, llama.cpp server. State prefilled once, question branches forked, answer-token logits read without generating; softmax over options; default T=1.0, no fitted calibration; entropy confidence. 64K context. Private dataset (synthetic, teacher and labelled tasks); loss: gold cross-entropy plus teacher cross-entropy when the teacher agrees.
- **Benchmark claims:** author-reported: JevBench public subset 231 items: Q8 85.71% = Jev 1.13 hosted via OpenRouter (85.71%); Kev-v9 clean 81.55% vs Jev 87.00%.
- **Status:** active (last modified 2026-09-21).
- **Evidence:** VERIFIED (README.md).

---

## Further leads (identified, not individually examined)

These are open Jev-style entries listed in the Jev Decision Index (`multimodalart/jev-decision-index`, `data/index.json`, generated 2026-09-28) or in `AnotiaWang/awesome-decision-models`. Neither source's description was checked against the project's own code or card, so the kind and base model below are the tracker's labels.

| Entry | Tracker's kind / base | URL |
|---|---|---|
| system-one-gemma | LoRA + head / gemma-3-270m | https://github.com/akash-kamat/system-one-gemma |
| Bosun v3.1 (0.6B, 1.7B) | LoRA + head / Qwen3 | https://huggingface.co/Hanno-Labs/bosun-v3.1-1.7b |
| CLM-v0.1-8B | full FT / Qwen3-8B | https://huggingface.co/Contrastive-LM/CLM-v0.1-8B |
| Decision 1.0 (Eos, Kai, Lex, Lux, Nox, Sol) | head/adapter / Qwen3.5, mmBERT | https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B |
| djev | diffusion readout / DiffusionGemma | https://github.com/Davipar/djev-dev |
| open-jev (JoshuaSP) | diffusion readout / DiffusionGemma | https://github.com/JoshuaSP/open-jev |
| Hopper (G) 1.2 | LoRA / Qwen3.5-4B | https://huggingface.co/HopitAI/hopper-g |
| jeff (logan-markewich; distinct from Gestalt-Lab Jeff 1) | inference technique / GLiFormer-large | https://github.com/logan-markewich/jeff |
| Jevfire | inference technique / Qwen3.8-27B | https://github.com/kikoncuo/jevfire |
| Jobe | inference technique / Qwen3.5-4B | https://github.com/MantisShrimpdev/jobe |
| LFM2.5 RLCD (350M, 2.6B), Qwen-2.5-1B-RLCD | RLCD fine-tunes (Laya recipe) | https://huggingface.co/monotykamary/LFM2.5-2.6B-RLCD |
| Lumma-Fev (0.1B, 0.6B) | full FT / LoRA | https://huggingface.co/FrontiersMind/Lumma-fev-0.6b |
| Metask-Jev-4B | LoRA / Qwen3.5-4B | https://huggingface.co/wayfind/metask-jev-4b-policy-mix |
| MoJev | head / Qwen3.5-0.8B | https://huggingface.co/MoLeMo-Lab/mojev |
| openvons | inference technique / Qwen3-4B | https://github.com/genai-craft/openvons |
| Surogate Rune 26B-A4B | full FT / Gemma-4 | https://huggingface.co/surogate/rune-26b-a4b-GGUF |
| Solomon v1.1 | head / Qwen3.8-27B | https://huggingface.co/DoccyHealth/Solomon |
| this-that 1.2 | full FT of Mapika/decider-2b | https://huggingface.co/flock-io/this-that-model-1.2 |
| Xor | full FT / Qwen3.6-35B-A3B | https://huggingface.co/juspay/xor |
| lev | LoRA + head / Qwen3.5-4B | https://huggingface.co/interfaze-ai/lev |
| Intern-Decision (0.8B, 2B, 4B) | full FT / Qwen3.5 | https://huggingface.co/internlm/Intern-Decision-4B |
| Julia 1 (Qyvos's base) | full FT / mmBERT-small | https://huggingface.co/SupersonicLabs/Julia-1 |
| Jebadiah 27B | LoRA / Qwen3.8-27B | https://huggingface.co/frontier-infra/jebadiah-27b |
| Lavoir | full FT / ModernBERT-large | https://huggingface.co/moganai/lavoir |
| JPT (0.8B, 4B, 9B) | LoRA / Qwen3.5 | https://huggingface.co/kirp/jpt-4b |
| Jet v6.2 | LoRA / Qwen3.5-4B | https://huggingface.co/michaljach/jet |
| PlayJev, OneJev's game sibling | Qwen3.5-0.8B letter readout | https://github.com/OmniJev/PlayJev |
| jevmlx, PocketJev, jev-visual, TetraJev | MLX / iPhone / visual / dual-reader techniques | https://github.com/bnsd55/jevmlx · https://github.com/NullPo-jp/PocketJev · https://github.com/hr98w/jev-visual · https://github.com/FeiLiuEM/tetrajev |
| psyb0t/decidealot (Codeberg) | local decision models | https://codeberg.org/psyb0t/decidealot |
| actbro/minicpm5-2b-jev (Ollama) | MiniCPM5-2B build | https://ollama.com/actbro/minicpm5-2b-jev |

**Hosted, closed decision models from other vendors.** These are listed in awesome-decision-models and speak `/v1/systemone`, but they are services, not hosted code or weights:
- meraGPT Decider 1;
- Upstage Solar Decide;
- Liquid AI d1;
- Respan Span-01.

**Runtimes for others' models** were not counted as separate clones:
- laya-mlx (6.7k stars), laya-coreml, laya.cpp, laya-rs, receptron/laya (npm), 1Panel laya-server, chaitin/Decis, arbiter, openjev.cpp, cu-Jev;
- HarnessRouter/SystemOneHarness;
- npm `open-jev`, which runs "the open-jev model" in the browser;
- crates `jev-bridge`, which turns an OpenAI-compatible API into a Jev-style scorer;
- Ollama `library/nimble`, `library/tev1` and `iapp/openthai-systemone`.

## Near-misses and exclusions

- **fastino/GLiNER2.5-Decide** (295 likes): a general GLiNER2 label classifier with no Jev or typed-decision interface. It appears only on clone leaderboards. Excluded; see the Group D section.
- **Clients and integrations of the real Jev API**, which reproduce nothing. Examples:
  - on GitHub: browser-use/jev-ultrafast, jev-chat, fast-jev-compaction, jev-mcp, jev-router, jevgrep, pi-jev and Jevbridge;
  - on crates.io: `jev`, `typesafe-jev`, `kunobi-jev` and `jev-sdk`;
  - on PyPI: `jev`, a decorator that compiles functions into Jev queries;
  - on npm: more than 25 `jev-*` packages.
- **Awesome lists**, used only as indexes: awesome-jev (yibie, cobanov, heyjunpenn, hellogumbo), awesome-typesafe-jev, awesome-decision-models and awesome-jev-projects.
- **Benchmarks and datasets**, which are not clones: JevBench (fstandhartinger), jev-benchmarks (AbdelStark), JEVals, the Jev Decision Index, imaddde867/jev-position-test, `SargeDev/jev-distill-corpus-v3` (the source of AutoTrust's distillation), `ZefanCai/Open-Jev` and `tasksource/tasksource-jev-typed-decisions`.
- **Name collisions with nothing to do with Jev:** PyPI `kev` (a 2010s key-value ORM), PyPI `openjev` 0.0.1 (an empty placeholder by balys), Codeberg `jevko-*` (a syntax project), and many pre-2026 HF "jev*" user models (jeveuxaider, Jevil and others). `ruvnet/ruvector-typesafe-*` uses "typesafe" in its name only; it was not examined.
- **Repurposed or suspect projects:**
  - `feder-cr/jev` (jevos) is a repo created in 2024 whose history restarts on 2026-09-27; it is included.
  - The `TypeSafeAI` HF org is unverified (see Qyvos).
  - Jev-Omni's 342 likes are high for 1.4k downloads and a sparse card.
  - Laya has 29.9k stars two weeks after creation.
  - Popularity figures on these projects should be treated cautiously.
- **Kaggle:** a web search found no Jev clone notebooks or models; the result was a DataCamp article.
