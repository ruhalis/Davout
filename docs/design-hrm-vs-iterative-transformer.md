# HRM vs. Iterative Transformer for a Jev-Style Decision Model

Oct 1, 2026 · @Arlan Baikurazov

## Summary

For a Jev-style decision model at equal GPU compute, HRM is the better-supported recurrent core for language, though the evidence is split by scale.

- **Puzzles, independent tests.** Two outside ablations found that HRM's two-level hierarchy adds little over a flat recurrent transformer.
- **Language, Sapient's own tests.** At matched training compute, HRM-Text 1B outscored plain transformers up to 3B, a looped transformer and a TRM with shared weights. Nobody has replicated this independently.
- **Practical edge.** HRM-Text-1B is the only pretrained language checkpoint for either design, released under Apache 2.0.

The core is still not what makes Jev work. The typed output interface, calibration training and broad pretrained knowledge matter more, and no source has measured either core on typed decisions or calibration.

The recommended next step is to fine-tune HRM-Text-1B with typed heads and test it against a conventional small pretrained LLM on your own decisions.

## What a Jev-style model has to do

Jev takes a state plus typed questions and returns typed answers with probabilities, with no text generation. TypeSafe AI launched it on 15 September 2026 as its first "System One" model ([launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [docs](https://docs.typesafe.ai/introduction)).

| Question type | Goal | Returns |
| --- | --- | --- |
| Choice | Choose an option from a list | choice, probabilities, confidence |
| Score | Score the state on a rubric | score, probabilities, confidence |
| Noul | Is this statement true? | a value from 0 to 1 |

An analog has to reproduce five properties:

- **Parallel, isolated questions.** Every question is evaluated independently against the same state in one request.
- **Atomic judgments.** The docs advise one well-scoped question at a time, the kind a knowledgeable person settles in a few seconds. Anything needing extended reasoning is split up and recombined in code.
- **Calibrated probabilities.** Jev is trained with Reinforcement Learning for Calibrated Decisions (RLCD), so higher confidence should mean higher accuracy.
- **Latency and price.** TypeSafe states 70 to 500 ms end to end, $0.042 per million input tokens, and free output.
- **Bounded choices.** Choice supports up to 255 options; larger sets use a two-stage process.

TypeSafe says the stack uses a new architecture and a parallel sampler but describes neither. Any analog is therefore functional, not a copy.

## What HRM is

HRM is a recurrent architecture from Sapient Intelligence with two coupled Transformer modules: a slow high-level module (H) and a fast low-level module (L). Five results frame the comparison:

- **Original HRM, June 2025.** A 27M-parameter model trained on about 1,000 examples per task, strong on Sudoku, mazes and ARC-AGI ([paper](https://arxiv.org/abs/2506.21734)). It is a puzzle solver with no language ability.
- **ARC Prize verification, August 2025.** HRM scored 32% on ARC-AGI-1 and 2% on ARC-AGI-2 on hidden tasks. A same-size plain transformer came within about 5 points, and the outer refinement loop drove most of the gain ([analysis](https://arcprize.org/blog/hrm-analysis)).
- **Tiny Recursive Model, October 2025.** One 2-layer network with 7M parameters reached 45% on ARC-AGI-1 and 8% on ARC-AGI-2 ([paper](https://arxiv.org/abs/2510.04871)).
- **HRM-Text paper, May 2026.** Sapient scaled HRM to a 1B language model and reported that it beat plain, looped and TRM baselines at matched training compute ([paper](https://arxiv.org/abs/2605.20613)). The numbers are in the Equal GPU budget section.
- **MBZUAI mechanistic study, September 2026.** A flat recurrent transformer matched the 27M HRM on Sudoku-Extreme, while one-pass models solved almost no puzzles. The authors describe hierarchy as an aid to interpretability, not the source of strength ([paper](https://arxiv.org/abs/2609.22197)).

HRM-Text-1B, released in May 2026, is the only version that handles language ([model card](https://huggingface.co/sapientinc/HRM-Text-1B)).

| Field | HRM-Text-1B |
| --- | --- |
| Parameters | About 1B |
| Hidden size | 1,536 |
| Layers per stack | 16 |
| H cycles × L cycles | 2 × 3 |
| Max sequence length | 4,096 tokens |
| Training data | 40B unique tokens, predominantly English, no code |
| Objective | PrefixLM: prompt tokens attend to each other in both directions |
| License | Apache 2.0 |
| Status | Pre-alignment; not chat-tuned or instruction-tuned |

For classification and structured output, the model card recommends its `direct` mode with 2 to 8 few-shot examples. It says pure zero-shot is noticeably weaker.

## The two architectures

HRM alternates two separately weighted stacks on a nested schedule; an iterative transformer loops one shared stack.

&#91;embedded content: HRM and iterative transformer, side by side · same input and heads, different core\]

Both designs read the same input and feed the same typed heads. Only the recurrent core in the middle differs.

|  | HRM | Iterative transformer |
| --- | --- | --- |
| Networks | Two stacks, H and L, with separate weights | One stack with shared weights |
| States | Two: a slow H state and a fast L state | One in a looped transformer; TRM keeps two on shared weights |
| Schedule | 2 cycles, each 3 L steps then 1 H step | The same stack, looped |
| Training gradient | Truncated: through the last 2 recurrent steps, warmed up to 5 | No settled recipe at language scale |
| Halting | Fixed schedule in HRM-Text; adaptive halting only in the puzzle version | Fixed loop count |

## Equal GPU budget

At matched compute, Sapient's language results put HRM ahead of both single-stack layouts.

The HRM-Text-1B model card gives `H_cycles × (L_cycles + 1)` steps per pass. That is 2 × 4 = 8 runs of a 16-layer stack, or 128 layer-runs at hidden size 1,536. Two single-stack layouts cost the same 128 layer-runs.

| Design | Layout | Parameters | Closest model in Sapient's tests |
| --- | --- | --- | --- |
| HRM | Two 16-layer stacks; L runs 6 times, H runs 2 | About 1B | HRM 1B |
| Iterative A | One 16-layer stack, 8 loops | About 0.5B | TRM 0.6B: shared H/L weights, same schedule |
| Iterative B | One 32-layer stack, 4 loops | About 1B | Looped Transformer 1B, 4 recursions |

The [HRM-Text paper](https://arxiv.org/abs/2605.20613) trained the models below at about 1.0 to 1.1 × 10²¹ training FLOPs each. Scores are percentages.

| Model | DROP | GSM8K | BoolQ | HellaSwag | Winogrande |
| --- | --- | --- | --- | --- | --- |
| HRM 1B | 82.2 | 84.5 | 86.2 | 63.4 | 72.4 |
| TRM 0.6B | 79.9 | 78.5 | 84.2 | 55.1 | 67.0 |
| Transformer 1B | 75.3 | 75.1 | 83.6 | 47.3 | 65.5 |

On DROP, the looped transformer scored 76.2, RINS 79.9, and 3B plain transformers 77.0 (deep) and 74.0 (wide). At equal size the picture differs: TRM 0.6B roughly matched HRM 0.6B but used about twice the training compute. A 1B TRM with fewer recursions trained unstably.

Three limits apply to these numbers:

- They come from Sapient and have not been independently replicated.
- They measure text benchmarks under a small 40B-token training budget, not typed decisions.
- HRM 1B spends about four times the inference compute of a plain 1B transformer per forward pass.

## Pros and cons for Jev-style classification

HRM wins on language-scale evidence and on having a pretrained model; the iterative transformer wins on simplicity and weight memory.

| Criterion | HRM | Iterative transformer |
| --- | --- | --- |
| Pretrained language checkpoint | Yes: HRM-Text-1B, Apache 2.0 | None found; you pretrain it |
| Evidence on puzzles, independent | Hierarchy adds little over a flat recurrent transformer | Matches HRM on Sudoku; TRM beats it on ARC-AGI with 7M parameters |
| Evidence on language, from Sapient | Highest scores at matched training compute | Looped transformer and half-size TRM score lower; TRM trained unstably at 1B |
| Weights at equal compute | About 1B | About 0.5B with shared H/L weights |
| Early exit under a latency budget | After each H cycle; logits from the shallower cycle come at no extra cost | After any loop in a looped transformer; TRM keeps HRM's schedule |
| Training stability | Needs two added techniques: MagicNorm and a warmup on gradient depth | Fewer parts, but no published recipe for stable training at 1B |
| Interpretability | Two separately intervenable states | One state |

Two points decide how much these rows matter for a Jev analog:

- **Jev asks for gut-check judgments.** Its docs push multi-step reasoning into code. Deep latent iteration, the main selling point of both cores, therefore matters less than knowledge breadth and calibration.
- **Adaptive depth is the one edge specific to recurrence.** Easy questions could stop after fewer cycles and hard ones run longer. HRM-Text does not use adaptive halting; its paper calls it a promising direction. No source has tested it on typed decisions.

## Recommended build

Start from an existing pretrained model and add a typed readout. Do not pretrain a new looped core first.

1. **Two candidate bases.** Take HRM-Text-1B and one conventional small instruction-tuned LLM that covers your languages.
2. **Typed readout.** Run one forward pass and read the logits over the answer labels: option labels for Choice, ordered levels for Score, true or false for Noul. Generate no text.
3. **Isolated questions.** Evaluate each question against the state separately and batch them, as Jev does.
4. **Calibration training.** Fine-tune with a proper scoring rule, such as log loss or Brier score, on decision data. Measure calibration error on held-out data.
5. **Head-to-head.** Compare both bases on your own decisions at the same latency budget and keep the winner.

Check three HRM-Text-1B limits before committing to it: English-only training data, no code data and a 4,096-token maximum. Its model card also says pure zero-shot use is noticeably weaker than few-shot, so expect to fine-tune.

Pretraining your own recurrent model is the last resort. The repo's 1B reference run takes 16 H100 GPUs for about 46 hours, roughly $1,472. Fine-tuning needs only a JSONL file of instruction and response pairs.

## Validation experiment

Sapient compared these architectures on text benchmarks; nobody has compared them on typed decisions and calibration. One controlled run closes that gap, and the HRM-Text repo already ships a config for each arm.

| Arm | Repo config | What it isolates |
| --- | --- | --- |
| HRM | `hrm` | Two stacks on a nested schedule |
| Matched TRM | `trm_match_recurrence` | The same recurrence with half the parameters |
| Universal Transformer | `ut` | One weight-shared stack |
| Plain transformer | `transformer` | No recurrence |

1. Pretrain each arm at the same size on the same data.
2. Attach identical typed heads and fine-tune each on the same decision set.
3. Compare accuracy, calibration error and latency at equal compute, over several seeds.

Keep HRM unless a simpler arm matches it on accuracy and calibration within the seed-to-seed spread. If one does, take the simpler model.

The repo prices its 0.6B HRM reference run at 8 H100 GPUs for about 50 hours, roughly $800. Four arms at that size would cost about $3,200 if the baselines train at a similar speed, which the repo does not state.

## Caveats and open questions

Four kinds of claim in this report are weaker than they look.

- **Vendor-reported numbers.** Jev's latency, price and speed-up figures come from TypeSafe, which calls its 193.6× faster and 444.6× cheaper results the high end of real-world gains. HRM-Text's benchmark and ablation numbers come from Sapient.
- **Evidence split by scale.** The independent ablations used 27M-parameter models on puzzles. The only language-scale comparison is Sapient's own. Neither tested typed decisions or calibration.
- **My own estimates.** The equal-compute layouts, the four-times inference cost and the early-exit comparison are my reasoning from published configs, not measurements.
- **Partly illegible tables.** Some cells in the HRM-Text paper's comparison tables did not render in the version I read, so those values are left out.

Four questions stay open:

- Which decisions will the model make, in which languages, and at what latency budget?
- Does recurrence improve calibration, or only accuracy?
- Does HRM's advantage hold when the baselines get a full pretraining budget instead of 40B tokens?
- What is Jev's context window? A secondary source reports 32K tokens; the TypeSafe pages read for this report do not state it.

## Sources

Pages opened for this report:

- [TypeSafe AI: Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe AI docs: Introduction](https://docs.typesafe.ai/introduction)
- [TypeSafe AI home page](https://typesafe.ai/)
- [ARC Prize: The Hidden Drivers of HRM's Performance on ARC-AGI](https://arcprize.org/blog/hrm-analysis)
- [Sapient Intelligence: HRM-Text-1B model card](https://huggingface.co/sapientinc/HRM-Text-1B)
- [Sapient Intelligence: HRM-Text repository](https://github.com/sapientinc/HRM-Text)
- [HRM-Text: Efficient Pretraining Beyond Scaling](https://arxiv.org/abs/2605.20613)
- [Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/abs/2510.04871)
- [Dissecting Hierarchical Reasoning Models: A Mechanistic Study](https://arxiv.org/abs/2609.22197)

Cited from search summaries only, not opened:

- [Hierarchical Reasoning Model, original paper](https://arxiv.org/abs/2506.21734)
- [ThursdAI: TypeSafe AI releases](https://thursdai.news/companies/typesafe-ai), for the 32K context figure
