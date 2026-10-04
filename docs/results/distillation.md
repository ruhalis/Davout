# Teacher distillation for large Choice sets

Experiment log, 2 to 4 October 2026. All runs on one RTX 5090 (32 GB). The raw
outputs of these runs (benchmark directories, checkpoints, teacher
distributions) are **not committed**; the numbers below are copied from the run
logs. The code that produced them is in the repository:
[`src/davout/teacher.py`](../../src/davout/teacher.py),
[`src/davout/train/synth.py`](../../src/davout/train/synth.py),
[`src/davout/train/mine.py`](../../src/davout/train/mine.py) and the recipes in
[`src/davout/train/sources.py`](../../src/davout/train/sources.py).

## Problem

After the first fine-tune ([finetune-v1.md](finetune-v1.md)), arm A (ftA)
reached 0.62 on Banking77 (77 intents, held out from training) against
OpenJev's 0.73. Choices with more than 26 options run in two stages
([`src/davout/scorer.py`](../../src/davout/scorer.py)): stage 1 judges every
option with a Yes/No prompt and shortlists the best 10; stage 2 picks among the
shortlist with one lettered prompt. Errors can come from either stage, so each
run below reports shortlist recall (gold in the shortlist) and
accuracy-given-gold-shortlisted (stage 2 alone) as well as overall accuracy.

## Null results

Pooled calibration and test rows (n = 600) unless noted. None of these changed
accuracy beyond noise.

| Attempt | What changed | Result |
| --- | --- | --- |
| ftA-cand300 | 300 more steps at LR 1.6e-6 on 20k LEDGAR and FewRel candidate-stage rows | 0.645 vs ftA 0.640 |
| Shortlist 20 | No training; stage 1 keeps 20 options instead of 10 | Recall 0.875 → 0.937, but accuracy-given-gold-shortlisted 0.731 → 0.680; accuracy 0.637 |
| candmix | 20k candidate rows + 20k v1 replay, peak LR 1e-5, best step 375 | 0.648; Yelp −4.2 points (p = 0.002) |
| Intent hard negatives | CLINC (non-banking) + MASSIVE, distractors mined from ftA's own stage 1 (`davout train mine-negatives`, recipe `intent_hn_v1`), LR 3e-6 | 1,000 test rows: 0.614 → 0.617 (p = 0.76) |

The intent hard-negative utterances were already in the v1 mix; only the
distractors were new, so this run added little new text.

## Teacher gate

Before building any distillation data, the teacher was checked on the
decisions it would label.

- **Teacher:** `Qwen/Qwen3.6-27B` (Apache-2.0), loaded in 4-bit NF4. Readout as for HRM: next-token logits over
  the option letters, no generation, averaged over two option orders (given and
  shuffled).
- **Banking77 stage 2:** on the 875 test rows where gold is in ftA's
  shortlist, the teacher scored 0.829 against ftA's 0.702 (+0.127, 95% CI
  [+0.099, +0.154]).
- **Yelp:** a tie in accuracy (0.567 vs 0.580), but the teacher is badly
  overconfident (NLL 2.15 vs 0.94). The teacher therefore labels Choice rows
  only.

## Distillation runs

Both recipes continue from ftA and mix teacher soft targets into training.

- **`teacher_intent_v1`** (run "ftA-teacher"): 16.5k teacher-written synthetic
  non-banking intent rows with teacher soft targets, 20k CLINC/MASSIVE rows with
  targets of 0.5 gold + 0.5 teacher, and 20k v1 replay rows. 883 steps, peak LR
  3e-6.
- **`teacher_intent_v2`**: 128k rows: about 75k synthetic rows from 196 intent
  taxonomies, filtered by a teacher check for banking-like intents
  (`davout train synth-flag`), plus 20k CLINC/MASSIVE and 40k v1 replay rows.
  2,001 steps, run at peak LR 3e-6 and 6e-6.

Banking77 with shortlist 10, the same 1,000 test rows for every model:

| Model | Accuracy | Shortlist recall | Accuracy given gold shortlisted | Raw NLL |
| --- | --- | --- | --- | --- |
| ftA | 0.614 | 0.875 | 0.702 | 1.719 |
| ftA-teacher (v1, LR 3e-6) | 0.641 | 0.887 | 0.723 | 1.586 |
| v2, LR 3e-6 | 0.644 | 0.898 | 0.717 | 1.551 |
| v2, LR 6e-6 | **0.660** | 0.894 | **0.738** | **1.511** |

Paired comparisons for the v2 6e-6 arm:

| Against | Accuracy difference | Significance |
| --- | --- | --- |
| ftA | +0.046 | 95% CI [+0.026, +0.066], p = 7e-6 |
| ftA-teacher | +0.019 | p = 0.03 |
| v2 at LR 3e-6 | +0.016 | p = 0.01 |

### Guards

300 test rows each, accuracy / raw NLL:

| Task | ftA | v2, LR 3e-6 | v2, LR 6e-6 |
| --- | --- | --- | --- |
| AG News | 0.820 / 0.525 | 0.833 / 0.616 | 0.843 / 0.607 |
| Yelp stars | 0.580 / 0.944 | 0.610 / 1.007 | 0.607 / 0.977 |

Accuracy held or rose on both guards; raw NLL rose somewhat. Held-out
transfer dev NLL moved from 0.4465 to about 0.46.

## Takeaways

- Public hard-label data did nothing for Banking77; teacher soft labels on
  fresh synthetic rows did.
- The learning rate mattered more than 4.5 times more data: v2 at 3e-6 was
  no better than v1 at 3e-6, while v2 at 6e-6 was.
- At stage 2 the gap to the teacher is about 28% closed (0.702 → 0.738
  against the teacher's 0.829).
- Overall accuracy (0.660) is still below OpenJev's 0.73, and cost per decision
  remains HRM's main weakness: a Banking77 decision takes about 470 to 520 ms
  (about 78 forward passes in the two-stage readout) against OpenJev's 113 ms
  p50, and OpenJev's p50 on the other tasks is 16 to 34 ms (committed benchmark,
  [`results/report.md`](../../results/report.md)).

## Caveats

- One seed for every run.
- Banking77 is a single benchmark; the improvement may not carry to other
  large intent sets.
- OpenJev's 0.73 comes from the 300-row benchmark test split, not the same
  1,000 rows used here.
- Only raw metrics were computed for these runs; calibrated accuracy, ECE and
  NLL were not.
