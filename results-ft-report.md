# Davout benchmark report

## Runs

| run | backend | task | kind | shots | attention | n | accuracy raw → cal | ECE raw → cal | NLL raw → cal | Brier | AUROC | MAE | p50 ms | p95 ms | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hrm-ag_news-zero-prefix-s0 | hrm | ag_news | choice | zero | prefix | 300 | 0.830 → 0.830 | 0.134 → 0.088 | 0.683 → 0.516 | 0.270 | – | – | 31 | 31 | 138 |
| hrm-banking77-zero-prefix-s0 | hrm | banking77 | choice | zero | prefix | 300 | 0.350 → 0.350 | 0.088 → 0.073 | 2.767 → 2.763 | 0.826 | – | – | 469 | 508 | 4803 |
| hrm-boolq-zero-prefix-s0 | hrm | boolq | noul | zero | prefix | 300 | 0.837 → 0.873 | 0.080 → 0.048 | 0.396 → 0.322 | 0.198 | 0.930 | – | 32 | 42 | 205 |
| hrm-sms_spam-zero-prefix-s0 | hrm | sms_spam | noul | zero | prefix | 300 | 0.910 → 0.933 | 0.056 → 0.042 | 0.189 → 0.151 | 0.090 | 0.971 | – | 26 | 27 | 68 |
| hrm-sst2-zero-prefix-s0 | hrm | sst2 | noul | zero | prefix | 300 | 0.917 → 0.920 | 0.041 → 0.035 | 0.248 → 0.233 | 0.131 | 0.966 | – | 27 | 27 | 58 |
| hrm-yelp-zero-prefix-s0 | hrm | yelp | score | zero | prefix | 300 | 0.530 → 0.530 | 0.289 → 0.073 | 1.435 → 1.111 | 0.596 | – | 0.617 | 33 | 51 | 256 |
| openjev-ag_news-s0 | openjev | ag_news | choice | – | – | 300 | 0.830 → 0.830 | 0.097 → 0.096 | 0.568 → 0.564 | 0.284 | – | – | 32 | 33 | 347 |
| openjev-banking77-s0 | openjev | banking77 | choice | – | – | 300 | 0.730 → 0.730 | 0.313 → 0.155 | 1.255 → 1.190 | 0.455 | – | – | 113 | 117 | 3222 |
| openjev-boolq-s0 | openjev | boolq | noul | – | – | 300 | 0.747 → 0.723 | 0.166 → 0.054 | 0.633 → 0.532 | 0.353 | 0.794 | – | 19 | 22 | 183 |
| openjev-sms_spam-s0 | openjev | sms_spam | noul | – | – | 300 | 0.877 → 0.890 | 0.041 → 0.088 | 0.243 → 0.269 | 0.159 | 0.905 | – | 17 | 18 | 62 |
| openjev-sst2-s0 | openjev | sst2 | noul | – | – | 300 | 0.773 → 0.813 | 0.136 → 0.029 | 0.643 → 0.436 | 0.277 | 0.881 | – | 16 | 17 | 37 |
| openjev-yelp-s0 | openjev | yelp | score | – | – | 300 | 0.393 → 0.393 | 0.200 → 0.064 | 1.493 → 1.319 | 0.696 | – | 0.692 | 34 | 39 | 959 |
| hrm-ftA-ag_news-zero-prefix-s0 | hrm | ag_news | choice | zero | prefix | 300 | 0.820 → 0.820 | 0.097 → 0.069 | 0.525 → 0.466 | 0.259 | – | – | 34 | 35 | 138 |
| hrm-ftA-banking77-zero-prefix-s0 | hrm | banking77 | choice | zero | prefix | 300 | 0.623 → 0.623 | 0.064 → 0.067 | 1.761 → 1.730 | 0.535 | – | – | 519 | 566 | 4802 |
| hrm-ftA-boolq-zero-prefix-s0 | hrm | boolq | noul | zero | prefix | 300 | 0.873 → 0.870 | 0.021 → 0.042 | 0.326 → 0.330 | 0.198 | 0.927 | – | 35 | 45 | 205 |
| hrm-ftA-sms_spam-zero-prefix-s0 | hrm | sms_spam | noul | zero | prefix | 300 | 0.883 → 0.963 | 0.104 → 0.021 | 0.215 → 0.101 | 0.058 | 0.987 | – | 27 | 30 | 68 |
| hrm-ftA-sst2-zero-prefix-s0 | hrm | sst2 | noul | zero | prefix | 300 | 0.907 → 0.920 | 0.066 → 0.045 | 0.275 → 0.247 | 0.140 | 0.961 | – | 27 | 30 | 58 |
| hrm-ftA-yelp-zero-prefix-s0 | hrm | yelp | score | zero | prefix | 300 | 0.580 → 0.580 | 0.127 → 0.069 | 0.944 → 0.935 | 0.540 | – | 0.504 | 36 | 57 | 256 |
| hrm-ftB-ag_news-zero-prefix-s0 | hrm | ag_news | choice | zero | prefix | 300 | 0.810 → 0.810 | 0.092 → 0.062 | 0.517 → 0.472 | 0.267 | – | – | 34 | 35 | 138 |
| hrm-ftB-banking77-zero-prefix-s0 | hrm | banking77 | choice | zero | prefix | 300 | 0.613 → 0.613 | 0.085 → 0.057 | 1.812 → 1.726 | 0.531 | – | – | 519 | 565 | 4802 |
| hrm-ftB-boolq-zero-prefix-s0 | hrm | boolq | noul | zero | prefix | 300 | 0.857 → 0.870 | 0.048 → 0.054 | 0.371 → 0.338 | 0.205 | 0.922 | – | 36 | 45 | 205 |
| hrm-ftB-sms_spam-zero-prefix-s0 | hrm | sms_spam | noul | zero | prefix | 300 | 0.877 → 0.957 | 0.103 → 0.030 | 0.240 → 0.109 | 0.062 | 0.983 | – | 27 | 30 | 68 |
| hrm-ftB-sst2-zero-prefix-s0 | hrm | sst2 | noul | zero | prefix | 300 | 0.880 → 0.913 | 0.042 → 0.029 | 0.304 → 0.241 | 0.135 | 0.962 | – | 27 | 30 | 58 |
| hrm-ftB-yelp-zero-prefix-s0 | hrm | yelp | score | zero | prefix | 300 | 0.580 → 0.580 | 0.099 → 0.127 | 0.998 → 0.978 | 0.558 | – | 0.518 | 36 | 57 | 256 |

Metrics are computed on each run's test split. "raw" uses the scorer's logits as they are; "cal" applies temperature scaling (choice, score) or Platt scaling (noul) fitted on the run's calibration split. Brier, AUROC and MAE are the calibrated values (raw when a run has no calibration rows). MAE is the mean distance between the expected level and the true level. Latency is per decision (one `score` call); tokens is the mean prompt length per decision, summed over every prompt or hypothesis the decision needed.

## Early exit: metrics after each H cycle

| run | cycle | accuracy raw → cal | ECE raw → cal | NLL raw → cal | agrees with final |
| --- | --- | --- | --- | --- | --- |
| hrm-ag_news-zero-prefix-s0 | 1 of 2 | 0.573 → 0.573 | 0.197 → 0.063 | 1.124 → 0.956 | 0.607 |
| hrm-ag_news-zero-prefix-s0 | 2 of 2 | 0.830 → 0.830 | 0.134 → 0.088 | 0.683 → 0.516 | 1.000 |
| hrm-boolq-zero-prefix-s0 | 1 of 2 | 0.660 → 0.660 | 0.097 → 0.068 | 0.660 → 0.651 | 0.597 |
| hrm-boolq-zero-prefix-s0 | 2 of 2 | 0.837 → 0.873 | 0.080 → 0.048 | 0.396 → 0.322 | 1.000 |
| hrm-sms_spam-zero-prefix-s0 | 1 of 2 | 0.150 → 0.877 | 0.408 → 0.020 | 0.787 → 0.379 | 0.067 |
| hrm-sms_spam-zero-prefix-s0 | 2 of 2 | 0.910 → 0.933 | 0.056 → 0.042 | 0.189 → 0.151 | 1.000 |
| hrm-sst2-zero-prefix-s0 | 1 of 2 | 0.520 → 0.630 | 0.104 → 0.035 | 0.692 → 0.651 | 0.517 |
| hrm-sst2-zero-prefix-s0 | 2 of 2 | 0.917 → 0.920 | 0.041 → 0.035 | 0.248 → 0.233 | 1.000 |
| hrm-yelp-zero-prefix-s0 | 1 of 2 | 0.220 → 0.220 | 0.061 → 0.071 | 1.558 → 1.558 | 0.040 |
| hrm-yelp-zero-prefix-s0 | 2 of 2 | 0.530 → 0.530 | 0.289 → 0.073 | 1.435 → 1.111 | 1.000 |
| hrm-ftA-ag_news-zero-prefix-s0 | 1 of 2 | 0.467 → 0.467 | 0.215 → 0.155 | 1.093 → 1.025 | 0.497 |
| hrm-ftA-ag_news-zero-prefix-s0 | 2 of 2 | 0.820 → 0.820 | 0.097 → 0.069 | 0.525 → 0.466 | 1.000 |
| hrm-ftA-boolq-zero-prefix-s0 | 1 of 2 | 0.660 → 0.647 | 0.063 → 0.051 | 0.644 → 0.641 | 0.633 |
| hrm-ftA-boolq-zero-prefix-s0 | 2 of 2 | 0.873 → 0.870 | 0.021 → 0.042 | 0.326 → 0.330 | 1.000 |
| hrm-ftA-sms_spam-zero-prefix-s0 | 1 of 2 | 0.130 → 0.870 | 0.420 → 0.059 | 0.769 → 0.343 | 0.013 |
| hrm-ftA-sms_spam-zero-prefix-s0 | 2 of 2 | 0.883 → 0.963 | 0.104 → 0.021 | 0.215 → 0.101 | 1.000 |
| hrm-ftA-sst2-zero-prefix-s0 | 1 of 2 | 0.520 → 0.710 | 0.097 → 0.044 | 0.696 → 0.543 | 0.507 |
| hrm-ftA-sst2-zero-prefix-s0 | 2 of 2 | 0.907 → 0.920 | 0.066 → 0.045 | 0.275 → 0.247 | 1.000 |
| hrm-ftA-yelp-zero-prefix-s0 | 1 of 2 | 0.170 → 0.170 | 0.178 → 0.128 | 1.611 → 1.589 | 0.087 |
| hrm-ftA-yelp-zero-prefix-s0 | 2 of 2 | 0.580 → 0.580 | 0.127 → 0.069 | 0.944 → 0.935 | 1.000 |
| hrm-ftB-ag_news-zero-prefix-s0 | 1 of 2 | 0.723 → 0.723 | 0.156 → 0.093 | 0.762 → 0.672 | 0.870 |
| hrm-ftB-ag_news-zero-prefix-s0 | 2 of 2 | 0.810 → 0.810 | 0.092 → 0.062 | 0.517 → 0.472 | 1.000 |
| hrm-ftB-boolq-zero-prefix-s0 | 1 of 2 | 0.693 → 0.703 | 0.096 → 0.066 | 0.636 → 0.582 | 0.743 |
| hrm-ftB-boolq-zero-prefix-s0 | 2 of 2 | 0.857 → 0.870 | 0.048 → 0.054 | 0.371 → 0.338 | 1.000 |
| hrm-ftB-sms_spam-zero-prefix-s0 | 1 of 2 | 0.877 → 0.877 | 0.056 → 0.023 | 0.373 → 0.357 | 1.000 |
| hrm-ftB-sms_spam-zero-prefix-s0 | 2 of 2 | 0.877 → 0.957 | 0.103 → 0.030 | 0.240 → 0.109 | 1.000 |
| hrm-ftB-sst2-zero-prefix-s0 | 1 of 2 | 0.817 → 0.830 | 0.083 → 0.058 | 0.439 → 0.391 | 0.870 |
| hrm-ftB-sst2-zero-prefix-s0 | 2 of 2 | 0.880 → 0.913 | 0.042 → 0.029 | 0.304 → 0.241 | 1.000 |
| hrm-ftB-yelp-zero-prefix-s0 | 1 of 2 | 0.450 → 0.450 | 0.156 → 0.132 | 1.364 → 1.253 | 0.640 |
| hrm-ftB-yelp-zero-prefix-s0 | 2 of 2 | 0.580 → 0.580 | 0.099 → 0.127 | 0.998 → 0.978 | 1.000 |

Each row reads the answer after that H cycle, with its own calibrator fitted on the calibration split; the last cycle is the model's normal output. "agrees with final" is the share of test decisions whose top answer already matches the final cycle. Two-stage choice runs record no per-cycle logits and are not listed.
