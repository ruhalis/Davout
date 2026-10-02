## Cycle 1 vs final (test split)

| task | model | n | c1 acc raw | c1 acc cal | final acc raw | final acc cal | c1 − final (cal) | c1 NLL raw / cal | final NLL raw / cal | agree (raw argmax) | agree (cal pred) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| boolq | base | 300 | 0.660 | 0.660 | 0.837 | 0.873 | -0.213 | 0.660 / 0.651 | 0.396 / 0.322 | 0.597 | 0.680 |
| boolq | ftA | 300 | 0.660 | 0.647 | 0.873 | 0.870 | -0.223 | 0.644 / 0.641 | 0.326 / 0.330 | 0.633 | 0.663 |
| boolq | ftB | 300 | 0.693 | 0.703 | 0.857 | 0.870 | -0.167 | 0.636 / 0.582 | 0.371 / 0.338 | 0.743 | 0.720 |
| sms_spam | base | 300 | 0.150 | 0.877 | 0.910 | 0.933 | -0.057 | 0.787 / 0.379 | 0.189 / 0.151 | 0.067 | 0.890 |
| sms_spam | ftA | 300 | 0.130 | 0.870 | 0.883 | 0.963 | -0.093 | 0.769 / 0.343 | 0.215 / 0.101 | 0.013 | 0.853 |
| sms_spam | ftB | 300 | 0.877 | 0.877 | 0.877 | 0.957 | -0.080 | 0.373 / 0.357 | 0.240 / 0.109 | 1.000 | 0.867 |
| sst2 | base | 300 | 0.520 | 0.630 | 0.917 | 0.920 | -0.290 | 0.692 / 0.651 | 0.248 / 0.233 | 0.517 | 0.623 |
| sst2 | ftA | 300 | 0.520 | 0.710 | 0.907 | 0.920 | -0.210 | 0.696 / 0.543 | 0.275 / 0.247 | 0.507 | 0.730 |
| sst2 | ftB | 300 | 0.817 | 0.830 | 0.880 | 0.913 | -0.083 | 0.439 / 0.391 | 0.304 / 0.241 | 0.870 | 0.857 |
| ag_news | base | 300 | 0.573 | 0.573 | 0.830 | 0.830 | -0.257 | 1.124 / 0.956 | 0.683 / 0.516 | 0.607 | 0.607 |
| ag_news | ftA | 300 | 0.467 | 0.467 | 0.820 | 0.820 | -0.353 | 1.093 / 1.025 | 0.525 / 0.466 | 0.497 | 0.497 |
| ag_news | ftB | 300 | 0.723 | 0.723 | 0.810 | 0.810 | -0.087 | 0.762 / 0.672 | 0.517 / 0.472 | 0.870 | 0.870 |
| yelp | base | 300 | 0.220 | 0.220 | 0.530 | 0.530 | -0.310 | 1.558 / 1.558 | 1.435 / 1.111 | 0.040 | 0.040 |
| yelp | ftA | 300 | 0.170 | 0.170 | 0.580 | 0.580 | -0.410 | 1.611 / 1.589 | 0.944 / 0.935 | 0.087 | 0.087 |
| yelp | ftB | 300 | 0.450 | 0.450 | 0.580 | 0.580 | -0.130 | 1.364 / 1.253 | 0.998 / 0.978 | 0.640 | 0.640 |

## Confidence gate, tau chosen on calib (smallest tau with combined calib accuracy within 0.01 of final-only)

| task | model | tau | calib: exit / combined / final | TEST exit frac | TEST combined acc | TEST final-only acc | Δ acc | rows changed (worse/better) | potential compute saving | ≥50% exit & within 0.01 |
|---|---|---|---|---|---|---|---|---|---|---|
| boolq | base | 0.62 | 0.053 / 0.910 / 0.920 | 0.013 | 0.873 | 0.873 | +0.000 | 0/0 | 0.7% | no |
| boolq | ftA | 0.70 | 0.217 / 0.910 / 0.920 | 0.187 | 0.847 | 0.870 | -0.023 | 7/0 | 9.3% | no |
| boolq | ftB | 0.83 | 0.200 / 0.900 / 0.900 | 0.197 | 0.850 | 0.870 | -0.020 | 6/0 | 9.8% | no |
| sms_spam | base | 0.88 | 0.203 / 0.903 / 0.913 | 0.217 | 0.920 | 0.933 | -0.013 | 6/2 | 10.8% | no |
| sms_spam | ftA | 0.87 | 0.567 / 0.943 / 0.947 | 0.603 | 0.957 | 0.963 | -0.007 | 6/4 | 30.2% | yes |
| sms_spam | ftB | 0.87 | 0.513 / 0.933 / 0.943 | 0.517 | 0.940 | 0.957 | -0.017 | 6/1 | 25.8% | no |
| sst2 | base | 0.73 | 0.047 / 0.947 / 0.953 | 0.047 | 0.913 | 0.920 | -0.007 | 2/0 | 2.3% | no |
| sst2 | ftA | 0.79 | 0.327 / 0.923 / 0.933 | 0.340 | 0.907 | 0.920 | -0.013 | 6/2 | 17.0% | no |
| sst2 | ftB | 0.87 | 0.553 / 0.930 / 0.940 | 0.497 | 0.903 | 0.913 | -0.010 | 3/0 | 24.8% | no |
| ag_news | base | 0.71 | 0.343 / 0.877 / 0.887 | 0.367 | 0.807 | 0.830 | -0.023 | 8/1 | 18.3% | no |
| ag_news | ftA | 0.65 | 0.280 / 0.863 / 0.873 | 0.277 | 0.807 | 0.820 | -0.013 | 5/1 | 13.8% | no |
| ag_news | ftB | 0.78 | 0.617 / 0.857 / 0.867 | 0.630 | 0.807 | 0.810 | -0.003 | 1/0 | 31.5% | yes |
| yelp | base | 0.28 | 0.420 / 0.423 / 0.423 | 0.470 | 0.453 | 0.530 | -0.077 | 64/41 | 23.5% | no |
| yelp | ftA | 0.28 | 0.017 / 0.543 / 0.547 | 0.013 | 0.580 | 0.580 | +0.000 | 0/0 | 0.7% | no |
| yelp | ftB | 0.48 | 0.333 / 0.520 / 0.527 | 0.337 | 0.540 | 0.580 | -0.040 | 12/0 | 16.8% | no |

## Gate curve on TEST at fixed tau: exit fraction / combined accuracy (final-only accuracy in the last column)

| task | model | tau 0.6 | tau 0.7 | tau 0.8 | tau 0.9 | tau 0.95 | always exit (c1 only) | final only |
|---|---|---|---|---|---|---|---|---|
| boolq | base | 0.33 / 0.810 | 0.00 / 0.873 | 0.00 / 0.873 | 0.00 / 0.873 | 0.00 / 0.873 | 0.660 | 0.873 |
| boolq | ftA | 0.51 / 0.767 | 0.19 / 0.847 | 0.05 / 0.863 | 0.01 / 0.870 | 0.00 / 0.870 | 0.647 | 0.870 |
| boolq | ftB | 0.77 / 0.777 | 0.51 / 0.827 | 0.27 / 0.847 | 0.07 / 0.870 | 0.01 / 0.870 | 0.703 | 0.870 |
| sms_spam | base | 1.00 / 0.877 | 1.00 / 0.877 | 0.95 / 0.870 | 0.11 / 0.923 | 0.00 / 0.933 | 0.877 | 0.933 |
| sms_spam | ftA | 0.98 / 0.867 | 0.93 / 0.873 | 0.78 / 0.917 | 0.47 / 0.970 | 0.20 / 0.967 | 0.870 | 0.963 |
| sms_spam | ftB | 0.99 / 0.877 | 0.95 / 0.880 | 0.81 / 0.907 | 0.33 / 0.950 | 0.05 / 0.957 | 0.877 | 0.957 |
| sst2 | base | 0.47 / 0.800 | 0.09 / 0.907 | 0.01 / 0.920 | 0.00 / 0.920 | 0.00 / 0.920 | 0.630 | 0.920 |
| sst2 | ftA | 0.79 / 0.793 | 0.54 / 0.867 | 0.31 / 0.910 | 0.13 / 0.917 | 0.05 / 0.920 | 0.710 | 0.920 |
| sst2 | ftB | 0.89 / 0.860 | 0.77 / 0.890 | 0.63 / 0.900 | 0.44 / 0.907 | 0.31 / 0.907 | 0.830 | 0.913 |
| ag_news | base | 0.47 / 0.780 | 0.37 / 0.807 | 0.24 / 0.820 | 0.14 / 0.830 | 0.06 / 0.830 | 0.573 | 0.830 |
| ag_news | ftA | 0.36 / 0.793 | 0.22 / 0.810 | 0.17 / 0.817 | 0.06 / 0.820 | 0.00 / 0.820 | 0.467 | 0.820 |
| ag_news | ftB | 0.84 / 0.773 | 0.74 / 0.793 | 0.60 / 0.810 | 0.40 / 0.810 | 0.09 / 0.810 | 0.723 | 0.810 |
| yelp | base | 0.00 / 0.530 | 0.00 / 0.530 | 0.00 / 0.530 | 0.00 / 0.530 | 0.00 / 0.530 | 0.220 | 0.530 |
| yelp | ftA | 0.00 / 0.580 | 0.00 / 0.580 | 0.00 / 0.580 | 0.00 / 0.580 | 0.00 / 0.580 | 0.170 | 0.580 |
| yelp | ftB | 0.23 / 0.567 | 0.16 / 0.573 | 0.06 / 0.580 | 0.00 / 0.580 | 0.00 / 0.580 | 0.450 | 0.580 |

## Spec D5 'early exit usable', applied literally

### base
- (a) cycle-1 calibrated accuracy within 0.03 of final: 0 of 5 tasks (none) → NOT met (needs ≥ 4 of 5)
- (b) per task, calib-chosen tau, on test: ≥50% exit with combined accuracy within 0.01 of final on 0 of 5 tasks (none)
  pooled over the 5 tasks (1500 test decisions, per-task calib-chosen tau): exit 0.223, combined accuracy 0.793 vs final-only 0.817 (Δ -0.024) → NOT met on the pooled reading
  best case (tau picked on the test rows themselves, optimistic): possible on 0 of 5 tasks (none)

### ftA
- (a) cycle-1 calibrated accuracy within 0.03 of final: 0 of 5 tasks (none) → NOT met (needs ≥ 4 of 5)
- (b) per task, calib-chosen tau, on test: ≥50% exit with combined accuracy within 0.01 of final on 1 of 5 tasks (sms_spam)
  pooled over the 5 tasks (1500 test decisions, per-task calib-chosen tau): exit 0.284, combined accuracy 0.819 vs final-only 0.831 (Δ -0.011) → NOT met on the pooled reading
  best case (tau picked on the test rows themselves, optimistic): possible on 1 of 5 tasks (sms_spam)

### ftB
- (a) cycle-1 calibrated accuracy within 0.03 of final: 0 of 5 tasks (none) → NOT met (needs ≥ 4 of 5)
- (b) per task, calib-chosen tau, on test: ≥50% exit with combined accuracy within 0.01 of final on 1 of 5 tasks (ag_news)
  pooled over the 5 tasks (1500 test decisions, per-task calib-chosen tau): exit 0.435, combined accuracy 0.808 vs final-only 0.826 (Δ -0.018) → NOT met on the pooled reading
  best case (tau picked on the test rows themselves, optimistic): possible on 2 of 5 tasks (sst2, ag_news)

- ftB final readout vs ftA, mean NLL over these 5 tasks: raw 0.4569 → 0.4860 (Δ +0.0290, NOT within 0.02); calibrated 0.4158 → 0.4276 (Δ +0.0118, within 0.02)
