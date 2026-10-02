## Sanity: the ad hoc script reproduces the benchmark rows and logits for the registered wording

- base sms_spam: 600/600 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- base ag_news: 300/300 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- base sst2: 600/600 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- ftA sms_spam: 600/600 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- ftA ag_news: 300/300 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- ftA sst2: 600/600 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- ftB sms_spam: 600/600 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- ftB ag_news: 300/300 rows match the benchmark run by id, split and label; max |logit difference| 0.0000
- ftB sst2: 600/600 rows match the benchmark run by id, split and label; max |logit difference| 0.0000

## SMS spam: old bare statement vs the registered question + criteria (test n=300)

| wording | model | AUROC [95% CI] | accuracy raw | accuracy at calib-fitted threshold | raw NLL | raw yes-rate (positives) |
|---|---|---|---|---|---|---|
| bare statement "The message is spam." | base | 0.571 [0.477, 0.664] | 0.867 | 0.877 | 0.395 | 0.017 (0.123) |
| bare statement "The message is spam." | ftA | 0.806 [0.726, 0.873] | 0.877 | 0.877 | 0.329 | 0.000 (0.123) |
| bare statement "The message is spam." | ftB | 0.798 [0.716, 0.873] | 0.877 | 0.867 | 0.329 | 0.000 (0.123) |
| registered: question + both criteria | base | 0.971 [0.949, 0.986] | 0.910 | 0.933 | 0.189 | 0.040 (0.123) |
| registered: question + both criteria | ftA | 0.987 [0.972, 0.997] | 0.883 | 0.963 | 0.215 | 0.007 (0.123) |
| registered: question + both criteria | ftB | 0.983 [0.965, 0.996] | 0.877 | 0.957 | 0.240 | 0.000 (0.123) |

- base: AUROC gap (first wording minus registered) -0.399 [-0.492, -0.309]; calibrated accuracy gap -0.057
- ftA: AUROC gap (first wording minus registered) -0.181 [-0.252, -0.117]; calibrated accuracy gap -0.087
- ftB: AUROC gap (first wording minus registered) -0.186 [-0.259, -0.121]; calibrated accuracy gap -0.090

## SST-2: question form vs the registered statement (test n=300)

| wording | model | AUROC [95% CI] | accuracy raw | accuracy at calib-fitted threshold | raw NLL | raw yes-rate (positives) |
|---|---|---|---|---|---|---|
| question "Does the review express a positive sentiment?" | base | 0.974 [0.956, 0.989] | 0.933 | 0.933 | 0.200 | 0.513 (0.520) |
| question "Does the review express a positive sentiment?" | ftA | 0.967 [0.946, 0.984] | 0.923 | 0.927 | 0.273 | 0.503 (0.520) |
| question "Does the review express a positive sentiment?" | ftB | 0.968 [0.949, 0.984] | 0.887 | 0.930 | 0.293 | 0.453 (0.520) |
| registered statement "The review expresses a positive sentiment." | base | 0.966 [0.946, 0.982] | 0.917 | 0.920 | 0.248 | 0.517 (0.520) |
| registered statement "The review expresses a positive sentiment." | ftA | 0.961 [0.938, 0.980] | 0.907 | 0.920 | 0.275 | 0.507 (0.520) |
| registered statement "The review expresses a positive sentiment." | ftB | 0.962 [0.939, 0.980] | 0.880 | 0.913 | 0.304 | 0.447 (0.520) |

- base: AUROC gap (first wording minus registered) +0.009 [+0.003, +0.016]; calibrated accuracy gap +0.013
- ftA: AUROC gap (first wording minus registered) +0.006 [+0.000, +0.012]; calibrated accuracy gap +0.007
- ftB: AUROC gap (first wording minus registered) +0.006 [+0.002, +0.011]; calibrated accuracy gap +0.017

## AG News under three option orders (test n=300, argmax of raw letter logits)

| model | acc registered | acc reversed | acc perm2 | spread (max − min) | choice changes with order (any of 3) | registered vs reversed | registered vs perm2 | reversed vs perm2 | all three agree and correct | majority-vote acc |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0.830 | 0.793 | 0.830 | 0.037 | 0.067 | 0.047 | 0.037 | 0.050 | 0.783 | 0.823 |
| ftA | 0.820 | 0.853 | 0.810 | 0.043 | 0.050 | 0.037 | 0.020 | 0.043 | 0.807 | 0.823 |
| ftB | 0.810 | 0.813 | 0.797 | 0.017 | 0.020 | 0.010 | 0.013 | 0.017 | 0.797 | 0.807 |

Orders: registered = world, sports, business, sci_tech; reversed = sci_tech, business, sports, world; perm2 = business, world, sci_tech, sports.
Letter picked (A/B/C/D share) per order, to show position bias; the gold labels are near-uniform:

- base: registered 0.25/0.26/0.32/0.18; reversed 0.15/0.33/0.25/0.27; perm2 0.30/0.27/0.18/0.26
- ftA: registered 0.25/0.27/0.32/0.17; reversed 0.20/0.30/0.27/0.24; perm2 0.33/0.24/0.16/0.27
- ftB: registered 0.25/0.26/0.33/0.16; reversed 0.16/0.33/0.26/0.25; perm2 0.34/0.24/0.15/0.26
