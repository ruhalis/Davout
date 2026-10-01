# Davout benchmark report

## Runs

| run | backend | task | kind | shots | attention | n | accuracy raw → cal | ECE raw → cal | NLL raw → cal | Brier | AUROC | MAE | p50 ms | p95 ms | tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hrm-ag_news-generic3-prefix-s0 | hrm | ag_news | choice | generic-3 | prefix | 300 | 0.827 → 0.827 | 0.118 → 0.079 | 0.647 → 0.509 | 0.265 | – | – | 41 | 41 | 336 |
| hrm-ag_news-task5-causal-s0 | hrm | ag_news | choice | task-5 | causal | 300 | 0.380 → 0.380 | 0.066 → 0.074 | 1.374 → 1.365 | 0.738 | – | – | 59 | 60 | 798 |
| hrm-ag_news-task5-prefix-s0 | hrm | ag_news | choice | task-5 | prefix | 300 | 0.827 → 0.827 | 0.114 → 0.081 | 0.604 → 0.502 | 0.259 | – | – | 65 | 66 | 798 |
| hrm-ag_news-zero-prefix-s0 | hrm | ag_news | choice | zero | prefix | 300 | 0.830 → 0.830 | 0.134 → 0.088 | 0.683 → 0.516 | 0.270 | – | – | 31 | 31 | 138 |
| hrm-banking77-generic3-prefix-s0 | hrm | banking77 | choice | generic-3 | prefix | 300 | 0.333 → 0.333 | 0.072 → 0.075 | 2.867 → 2.868 | 0.830 | – | – | 1264 | 1307 | 19784 |
| hrm-banking77-task5-causal-s0 | hrm | banking77 | choice | task-5 | causal | 300 | 0.073 → 0.073 | 0.019 → 0.012 | 4.183 → 4.190 | 0.974 | – | – | 1635 | 1649 | 28852 |
| hrm-banking77-task5-prefix-s0 | hrm | banking77 | choice | task-5 | prefix | 300 | 0.327 → 0.327 | 0.095 → 0.074 | 3.022 → 3.017 | 0.854 | – | – | 1703 | 1718 | 28854 |
| hrm-banking77-zero-prefix-s0 | hrm | banking77 | choice | zero | prefix | 300 | 0.350 → 0.350 | 0.088 → 0.073 | 2.767 → 2.763 | 0.826 | – | – | 469 | 508 | 4803 |
| hrm-boolq-generic3-prefix-s0 | hrm | boolq | noul | generic-3 | prefix | 300 | 0.830 → 0.863 | 0.071 → 0.046 | 0.392 → 0.327 | 0.204 | 0.926 | – | 41 | 51 | 378 |
| hrm-boolq-task5-causal-s0 | hrm | boolq | noul | task-5 | causal | 300 | 0.557 → 0.663 | 0.072 → 0.063 | 0.687 → 0.614 | 0.425 | 0.655 | – | 82 | 96 | 1252 |
| hrm-boolq-task5-prefix-s0 | hrm | boolq | noul | task-5 | prefix | 300 | 0.833 → 0.877 | 0.059 → 0.059 | 0.381 → 0.337 | 0.203 | 0.925 | – | 100 | 114 | 1252 |
| hrm-boolq-zero-prefix-s0 | hrm | boolq | noul | zero | prefix | 300 | 0.837 → 0.873 | 0.080 → 0.048 | 0.396 → 0.322 | 0.198 | 0.930 | – | 32 | 42 | 205 |
| hrm-sms_spam-generic3-prefix-s0 | hrm | sms_spam | noul | generic-3 | prefix | 300 | 0.850 → 0.877 | 0.065 → 0.050 | 0.402 → 0.363 | 0.211 | 0.647 | – | 33 | 36 | 230 |
| hrm-sms_spam-task5-causal-s0 | hrm | sms_spam | noul | task-5 | causal | 300 | 0.353 → 0.877 | 0.187 → 0.051 | 0.724 → 0.322 | 0.189 | 0.802 | – | 37 | 39 | 393 |
| hrm-sms_spam-task5-prefix-s0 | hrm | sms_spam | noul | task-5 | prefix | 300 | 0.863 → 0.867 | 0.093 → 0.061 | 0.371 → 0.343 | 0.197 | 0.722 | – | 42 | 44 | 393 |
| hrm-sms_spam-zero-prefix-s0 | hrm | sms_spam | noul | zero | prefix | 300 | 0.867 → 0.877 | 0.066 → 0.026 | 0.395 → 0.372 | 0.216 | 0.571 | – | 26 | 27 | 57 |
| hrm-sst2-generic3-prefix-s0 | hrm | sst2 | noul | generic-3 | prefix | 300 | 0.887 → 0.887 | 0.023 → 0.028 | 0.289 → 0.281 | 0.169 | 0.951 | – | 33 | 33 | 231 |
| hrm-sst2-task5-causal-s0 | hrm | sst2 | noul | task-5 | causal | 300 | 0.830 → 0.833 | 0.249 → 0.047 | 0.577 → 0.418 | 0.257 | 0.901 | – | 32 | 36 | 313 |
| hrm-sst2-task5-prefix-s0 | hrm | sst2 | noul | task-5 | prefix | 300 | 0.907 → 0.903 | 0.021 → 0.029 | 0.249 → 0.246 | 0.146 | 0.962 | – | 36 | 41 | 313 |
| hrm-sst2-zero-prefix-s0 | hrm | sst2 | noul | zero | prefix | 300 | 0.917 → 0.920 | 0.041 → 0.035 | 0.248 → 0.233 | 0.131 | 0.966 | – | 27 | 27 | 58 |
| hrm-yelp-generic3-prefix-s0 | hrm | yelp | score | generic-3 | prefix | 300 | 0.527 → 0.527 | 0.287 → 0.075 | 1.373 → 1.090 | 0.585 | – | 0.599 | 43 | 64 | 431 |
| hrm-yelp-task5-causal-s0 | hrm | yelp | score | task-5 | causal | 300 | 0.243 → 0.243 | 0.179 → 0.105 | 1.684 → 1.574 | 0.785 | – | 1.147 | 138 | 158 | 2138 |
| hrm-yelp-task5-prefix-s0 | hrm | yelp | score | task-5 | prefix | 300 | 0.543 → 0.543 | 0.259 → 0.103 | 1.319 → 1.087 | 0.587 | – | 0.608 | 180 | 206 | 2138 |
| hrm-yelp-zero-prefix-s0 | hrm | yelp | score | zero | prefix | 300 | 0.530 → 0.530 | 0.289 → 0.073 | 1.435 → 1.111 | 0.596 | – | 0.617 | 33 | 51 | 256 |
| openjev-ag_news-s0 | openjev | ag_news | choice | – | – | 300 | 0.830 → 0.830 | 0.097 → 0.096 | 0.568 → 0.564 | 0.284 | – | – | 32 | 33 | 347 |
| openjev-banking77-s0 | openjev | banking77 | choice | – | – | 300 | 0.730 → 0.730 | 0.313 → 0.155 | 1.255 → 1.190 | 0.455 | – | – | 113 | 117 | 3222 |
| openjev-boolq-s0 | openjev | boolq | noul | – | – | 300 | 0.747 → 0.723 | 0.166 → 0.054 | 0.633 → 0.532 | 0.353 | 0.794 | – | 19 | 22 | 183 |
| openjev-sms_spam-s0 | openjev | sms_spam | noul | – | – | 300 | 0.877 → 0.877 | 0.081 → 0.031 | 0.430 → 0.369 | 0.214 | 0.611 | – | 17 | 18 | 37 |
| openjev-sst2-s0 | openjev | sst2 | noul | – | – | 300 | 0.773 → 0.813 | 0.136 → 0.029 | 0.643 → 0.436 | 0.277 | 0.881 | – | 16 | 17 | 37 |
| openjev-yelp-s0 | openjev | yelp | score | – | – | 300 | 0.393 → 0.393 | 0.200 → 0.064 | 1.493 → 1.319 | 0.696 | – | 0.692 | 34 | 39 | 959 |

Metrics are computed on each run's test split. "raw" uses the scorer's logits as they are; "cal" applies temperature scaling (choice, score) or Platt scaling (noul) fitted on the run's calibration split. Brier, AUROC and MAE are the calibrated values (raw when a run has no calibration rows). MAE is the mean distance between the expected level and the true level. Latency is per decision (one `score` call); tokens is the mean prompt length per decision, summed over every prompt or hypothesis the decision needed.

## Early exit: metrics after each H cycle

| run | cycle | accuracy raw → cal | ECE raw → cal | NLL raw → cal | agrees with final |
| --- | --- | --- | --- | --- | --- |
| hrm-ag_news-generic3-prefix-s0 | 1 of 2 | 0.500 → 0.500 | 0.173 → 0.096 | 1.218 → 1.077 | 0.507 |
| hrm-ag_news-generic3-prefix-s0 | 2 of 2 | 0.827 → 0.827 | 0.118 → 0.079 | 0.647 → 0.509 | 1.000 |
| hrm-ag_news-task5-causal-s0 | 1 of 2 | 0.247 → 0.247 | 0.105 → 0.009 | 1.436 → 1.386 | 0.053 |
| hrm-ag_news-task5-causal-s0 | 2 of 2 | 0.380 → 0.380 | 0.066 → 0.074 | 1.374 → 1.365 | 1.000 |
| hrm-ag_news-task5-prefix-s0 | 1 of 2 | 0.400 → 0.400 | 0.191 → 0.276 | 1.340 → 1.339 | 0.410 |
| hrm-ag_news-task5-prefix-s0 | 2 of 2 | 0.827 → 0.827 | 0.114 → 0.081 | 0.604 → 0.502 | 1.000 |
| hrm-ag_news-zero-prefix-s0 | 1 of 2 | 0.573 → 0.573 | 0.197 → 0.063 | 1.124 → 0.956 | 0.607 |
| hrm-ag_news-zero-prefix-s0 | 2 of 2 | 0.830 → 0.830 | 0.134 → 0.088 | 0.683 → 0.516 | 1.000 |
| hrm-boolq-generic3-prefix-s0 | 1 of 2 | 0.650 → 0.660 | 0.088 → 0.067 | 0.662 → 0.650 | 0.593 |
| hrm-boolq-generic3-prefix-s0 | 2 of 2 | 0.830 → 0.863 | 0.071 → 0.046 | 0.392 → 0.327 | 1.000 |
| hrm-boolq-task5-causal-s0 | 1 of 2 | 0.660 → 0.660 | 0.039 → 0.066 | 0.640 → 0.647 | 0.377 |
| hrm-boolq-task5-causal-s0 | 2 of 2 | 0.557 → 0.663 | 0.072 → 0.063 | 0.687 → 0.614 | 1.000 |
| hrm-boolq-task5-prefix-s0 | 1 of 2 | 0.660 → 0.660 | 0.059 → 0.066 | 0.648 → 0.650 | 0.607 |
| hrm-boolq-task5-prefix-s0 | 2 of 2 | 0.833 → 0.877 | 0.059 → 0.059 | 0.381 → 0.337 | 1.000 |
| hrm-boolq-zero-prefix-s0 | 1 of 2 | 0.660 → 0.660 | 0.097 → 0.068 | 0.660 → 0.651 | 0.597 |
| hrm-boolq-zero-prefix-s0 | 2 of 2 | 0.837 → 0.873 | 0.080 → 0.048 | 0.396 → 0.322 | 1.000 |
| hrm-sms_spam-generic3-prefix-s0 | 1 of 2 | 0.120 → 0.877 | 0.424 → 0.034 | 0.766 → 0.368 | 0.063 |
| hrm-sms_spam-generic3-prefix-s0 | 2 of 2 | 0.850 → 0.877 | 0.065 → 0.050 | 0.402 → 0.363 | 1.000 |
| hrm-sms_spam-task5-causal-s0 | 1 of 2 | 0.477 → 0.887 | 0.040 → 0.044 | 0.696 → 0.281 | 0.623 |
| hrm-sms_spam-task5-causal-s0 | 2 of 2 | 0.353 → 0.877 | 0.187 → 0.051 | 0.724 → 0.322 | 1.000 |
| hrm-sms_spam-task5-prefix-s0 | 1 of 2 | 0.223 → 0.877 | 0.298 → 0.024 | 0.724 → 0.379 | 0.120 |
| hrm-sms_spam-task5-prefix-s0 | 2 of 2 | 0.863 → 0.867 | 0.093 → 0.061 | 0.371 → 0.343 | 1.000 |
| hrm-sms_spam-zero-prefix-s0 | 1 of 2 | 0.130 → 0.877 | 0.427 → 0.024 | 0.793 → 0.375 | 0.030 |
| hrm-sms_spam-zero-prefix-s0 | 2 of 2 | 0.867 → 0.877 | 0.066 → 0.026 | 0.395 → 0.372 | 1.000 |
| hrm-sst2-generic3-prefix-s0 | 1 of 2 | 0.520 → 0.480 | 0.012 → 0.066 | 0.694 → 0.700 | 0.520 |
| hrm-sst2-generic3-prefix-s0 | 2 of 2 | 0.887 → 0.887 | 0.023 → 0.028 | 0.289 → 0.281 | 1.000 |
| hrm-sst2-task5-causal-s0 | 1 of 2 | 0.520 → 0.517 | 0.057 → 0.024 | 0.697 → 0.691 | 0.510 |
| hrm-sst2-task5-causal-s0 | 2 of 2 | 0.830 → 0.833 | 0.249 → 0.047 | 0.577 → 0.418 | 1.000 |
| hrm-sst2-task5-prefix-s0 | 1 of 2 | 0.520 → 0.540 | 0.037 → 0.036 | 0.695 → 0.693 | 0.520 |
| hrm-sst2-task5-prefix-s0 | 2 of 2 | 0.907 → 0.903 | 0.021 → 0.029 | 0.249 → 0.246 | 1.000 |
| hrm-sst2-zero-prefix-s0 | 1 of 2 | 0.520 → 0.630 | 0.104 → 0.035 | 0.692 → 0.651 | 0.517 |
| hrm-sst2-zero-prefix-s0 | 2 of 2 | 0.917 → 0.920 | 0.041 → 0.035 | 0.248 → 0.233 | 1.000 |
| hrm-yelp-generic3-prefix-s0 | 1 of 2 | 0.180 → 0.180 | 0.119 → 0.043 | 1.642 → 1.602 | 0.023 |
| hrm-yelp-generic3-prefix-s0 | 2 of 2 | 0.527 → 0.527 | 0.287 → 0.075 | 1.373 → 1.090 | 1.000 |
| hrm-yelp-task5-causal-s0 | 1 of 2 | 0.180 → 0.180 | 0.128 → 0.022 | 1.739 → 1.609 | 0.027 |
| hrm-yelp-task5-causal-s0 | 2 of 2 | 0.243 → 0.243 | 0.179 → 0.105 | 1.684 → 1.574 | 1.000 |
| hrm-yelp-task5-prefix-s0 | 1 of 2 | 0.177 → 0.177 | 0.118 → 0.038 | 1.661 → 1.606 | 0.013 |
| hrm-yelp-task5-prefix-s0 | 2 of 2 | 0.543 → 0.543 | 0.259 → 0.103 | 1.319 → 1.087 | 1.000 |
| hrm-yelp-zero-prefix-s0 | 1 of 2 | 0.220 → 0.220 | 0.061 → 0.071 | 1.558 → 1.558 | 0.040 |
| hrm-yelp-zero-prefix-s0 | 2 of 2 | 0.530 → 0.530 | 0.289 → 0.073 | 1.435 → 1.111 | 1.000 |

Each row reads the answer after that H cycle, with its own calibrator fitted on the calibration split; the last cycle is the model's normal output. "agrees with final" is the share of test decisions whose top answer already matches the final cycle. Two-stage choice runs record no per-cycle logits and are not listed.
