## Benchmark table (test split, n=300 per task, zero-shot, seed 0)

| task | kind | model | acc cal (raw) | NLL raw (cal) | ECE raw (cal) | AUROC | MAE | p50 ms |
|---|---|---|---|---|---|---|---|---|
| boolq | noul | base | 0.873 (0.837) | 0.396 (0.322) | 0.080 (0.048) | 0.930 | – | 32 |
| boolq | noul | ftA | 0.870 (0.873) | 0.326 (0.330) | 0.021 (0.042) | 0.927 | – | 35 |
| boolq | noul | ftB | 0.870 (0.857) | 0.371 (0.338) | 0.048 (0.054) | 0.922 | – | 36 |
| boolq | noul | openjev | 0.723 (0.747) | 0.633 (0.532) | 0.166 (0.054) | 0.794 | – | 19 |
| sms_spam | noul | base | 0.933 (0.910) | 0.189 (0.151) | 0.056 (0.042) | 0.971 | – | 26 |
| sms_spam | noul | ftA | 0.963 (0.883) | 0.215 (0.101) | 0.104 (0.021) | 0.987 | – | 27 |
| sms_spam | noul | ftB | 0.957 (0.877) | 0.240 (0.109) | 0.103 (0.030) | 0.983 | – | 27 |
| sms_spam | noul | openjev | 0.890 (0.877) | 0.243 (0.269) | 0.041 (0.088) | 0.905 | – | 17 |
| sst2 | noul | base | 0.920 (0.917) | 0.248 (0.233) | 0.041 (0.035) | 0.966 | – | 27 |
| sst2 | noul | ftA | 0.920 (0.907) | 0.275 (0.247) | 0.066 (0.045) | 0.961 | – | 27 |
| sst2 | noul | ftB | 0.913 (0.880) | 0.304 (0.241) | 0.042 (0.029) | 0.962 | – | 27 |
| sst2 | noul | openjev | 0.813 (0.773) | 0.643 (0.436) | 0.136 (0.029) | 0.881 | – | 16 |
| ag_news | choice | base | 0.830 (0.830) | 0.683 (0.516) | 0.134 (0.088) | – | – | 31 |
| ag_news | choice | ftA | 0.820 (0.820) | 0.525 (0.466) | 0.097 (0.069) | – | – | 34 |
| ag_news | choice | ftB | 0.810 (0.810) | 0.517 (0.472) | 0.092 (0.062) | – | – | 34 |
| ag_news | choice | openjev | 0.830 (0.830) | 0.568 (0.564) | 0.097 (0.096) | – | – | 32 |
| yelp | score | base | 0.530 (0.530) | 1.435 (1.111) | 0.289 (0.073) | – | 0.617 | 33 |
| yelp | score | ftA | 0.580 (0.580) | 0.944 (0.935) | 0.127 (0.069) | – | 0.504 | 36 |
| yelp | score | ftB | 0.580 (0.580) | 0.998 (0.978) | 0.099 (0.127) | – | 0.518 | 36 |
| yelp | score | openjev | 0.393 (0.393) | 1.493 (1.319) | 0.200 (0.064) | – | 0.692 | 34 |
| banking77 | choice | base | 0.350 (0.350) | 2.767 (2.763) | 0.088 (0.073) | – | – | 469 |
| banking77 | choice | ftA | 0.623 (0.623) | 1.761 (1.730) | 0.064 (0.067) | – | – | 519 |
| banking77 | choice | ftB | 0.613 (0.613) | 1.812 (1.726) | 0.085 (0.057) | – | – | 519 |
| banking77 | choice | openjev | 0.730 (0.730) | 1.255 (1.190) | 0.313 (0.155) | – | – | 113 |

Mean raw NLL over the six tasks: base 0.9531, ftA 0.6743, ftB 0.7070, openjev 0.8057
Mean calibrated NLL over the six tasks: base 0.8495, ftA 0.6348, ftB 0.6440, openjev 0.7182
Mean calibrated accuracy over the six tasks: base 0.7394, ftA 0.7961, ftB 0.7906, openjev 0.7300

## Paired differences (second minus first), 95% paired-bootstrap CI; * = CI excludes 0

| task | pair | d acc (cal) | McNemar b/c, p | d raw NLL | d cal NLL | d AUROC or d MAE |
|---|---|---|---|---|---|---|
| boolq | base → ftA | -0.003 [-0.030, +0.027]  | 10/9, p=1 | -0.070 [-0.128, -0.018]* | +0.007 [-0.033, +0.046]  | AUROC -0.003 [-0.017, +0.010]  |
| boolq | base → ftB | -0.003 [-0.033, +0.023]  | 10/9, p=1 | -0.025 [-0.077, +0.024]  | +0.016 [-0.026, +0.058]  | AUROC -0.008 [-0.023, +0.007]  |
| boolq | ftA → ftB | +0.000 [-0.013, +0.013]  | 2/2, p=1 | +0.045 [+0.021, +0.071]* | +0.009 [-0.003, +0.020]  | AUROC -0.005 [-0.010, -0.000]* |
| boolq | openjev → ftA | +0.147 [+0.097, +0.200]* | 13/57, p=1.03e-07 | -0.306 [-0.414, -0.202]* | -0.203 [-0.279, -0.127]* | AUROC +0.133 [+0.088, +0.180]* |
| sms_spam | base → ftA | +0.030 [+0.003, +0.057]* | 4/13, p=0.049 | +0.026 [+0.009, +0.044]* | -0.049 [-0.082, -0.018]* | AUROC +0.016 [+0.002, +0.033]* |
| sms_spam | base → ftB | +0.023 [-0.003, +0.050]  | 5/12, p=0.143 | +0.052 [+0.019, +0.087]* | -0.042 [-0.076, -0.009]* | AUROC +0.013 [-0.003, +0.031]  |
| sms_spam | ftA → ftB | -0.007 [-0.020, +0.007]  | 3/1, p=0.625 | +0.026 [+0.006, +0.046]* | +0.007 [-0.003, +0.017]  | AUROC -0.003 [-0.008, +0.001]  |
| sms_spam | openjev → ftA | +0.073 [+0.033, +0.113]* | 8/30, p=0.000472 | -0.028 [-0.068, +0.011]  | -0.168 [-0.213, -0.122]* | AUROC +0.082 [+0.045, +0.125]* |
| sst2 | base → ftA | +0.000 [-0.020, +0.017]  | 4/4, p=1 | +0.027 [-0.021, +0.068]  | +0.014 [-0.009, +0.036]  | AUROC -0.005 [-0.013, +0.002]  |
| sst2 | base → ftB | -0.007 [-0.027, +0.013]  | 6/4, p=0.754 | +0.055 [+0.008, +0.099]* | +0.008 [-0.016, +0.031]  | AUROC -0.004 [-0.012, +0.003]  |
| sst2 | ftA → ftB | -0.007 [-0.020, +0.007]  | 3/1, p=0.625 | +0.029 [+0.012, +0.047]* | -0.006 [-0.015, +0.002]  | AUROC +0.001 [-0.002, +0.004]  |
| sst2 | openjev → ftA | +0.107 [+0.063, +0.150]* | 8/40, p=3.31e-06 | -0.368 [-0.498, -0.249]* | -0.189 [-0.256, -0.120]* | AUROC +0.080 [+0.047, +0.115]* |
| ag_news | base → ftA | -0.010 [-0.037, +0.017]  | 10/7, p=0.629 | -0.158 [-0.242, -0.078]* | -0.050 [-0.097, -0.006]* | – |
| ag_news | base → ftB | -0.020 [-0.050, +0.010]  | 13/7, p=0.263 | -0.167 [-0.258, -0.081]* | -0.044 [-0.092, +0.003]  | – |
| ag_news | ftA → ftB | -0.010 [-0.027, +0.003]  | 4/1, p=0.375 | -0.008 [-0.022, +0.005]  | +0.006 [-0.002, +0.015]  | – |
| ag_news | openjev → ftA | -0.010 [-0.047, +0.023]  | 16/13, p=0.711 | -0.043 [-0.120, +0.034]  | -0.098 [-0.172, -0.028]* | – |
| yelp | base → ftA | +0.050 [+0.000, +0.100]  | 24/39, p=0.0769 | -0.491 [-0.617, -0.367]* | -0.176 [-0.217, -0.136]* | MAE -0.113 [-0.149, -0.078]* |
| yelp | base → ftB | +0.050 [+0.003, +0.100]* | 20/35, p=0.0581 | -0.437 [-0.556, -0.320]* | -0.133 [-0.173, -0.095]* | MAE -0.099 [-0.134, -0.066]* |
| yelp | ftA → ftB | +0.000 [-0.023, +0.023]  | 7/7, p=1 | +0.054 [+0.030, +0.079]* | +0.043 [+0.029, +0.057]* | MAE +0.014 [+0.004, +0.023]* |
| yelp | openjev → ftA | +0.187 [+0.123, +0.253]* | 28/84, p=1.11e-07 | -0.549 [-0.672, -0.428]* | -0.384 [-0.458, -0.311]* | MAE -0.187 [-0.234, -0.142]* |
| banking77 | base → ftA | +0.273 [+0.213, +0.333]* | 15/97, p=7.12e-16 | -1.007 [-1.214, -0.798]* | -1.033 [-1.223, -0.842]* | – |
| banking77 | base → ftB | +0.263 [+0.200, +0.323]* | 15/94, p=3.71e-15 | -0.955 [-1.193, -0.715]* | -1.037 [-1.237, -0.839]* | – |
| banking77 | ftA → ftB | -0.010 [-0.037, +0.017]  | 9/6, p=0.607 | +0.051 [-0.031, +0.135]  | -0.004 [-0.066, +0.057]  | – |
| banking77 | openjev → ftA | -0.107 [-0.160, -0.057]* | 50/18, p=0.000131 | +0.506 [+0.289, +0.727]* | +0.540 [+0.295, +0.777]* | – |

## Spec D5, applied literally

### ftA vs base
- mean raw NLL over six tasks: base 0.9531 → ftA 0.6743 (lower: PASS)
- tasks improved (raw-NLL CI excludes 0 in the better direction, or calibrated accuracy ≥ +0.05): 4 of 6 (PASS; need ≥ 4): boolq (NLL CI<0), ag_news (NLL CI<0), yelp (NLL CI<0 & acc +0.050), banking77 (NLL CI<0 & acc +0.273)
  (by the NLL-CI clause alone: 4 of 6: boolq, ag_news, yelp, banking77)
- tasks worse by more than 0.05 raw NLL: none; worse by more than 0.05 calibrated accuracy: none
  raw NLL change per task: boolq -0.070, sms_spam +0.026, sst2 +0.027, ag_news -0.158, yelp -0.491, banking77 -1.007
- raw ECE up by more than 0.05: 0 task(s) (none); 'overfit' needs ≥ 3
  raw ECE change per task: boolq -0.059, sms_spam +0.048, sst2 +0.025, ag_news -0.037, yelp -0.162, banking77 -0.023

### ftB vs base
- mean raw NLL over six tasks: base 0.9531 → ftB 0.7070 (lower: PASS)
- tasks improved (raw-NLL CI excludes 0 in the better direction, or calibrated accuracy ≥ +0.05): 3 of 6 (FAIL; need ≥ 4): ag_news (NLL CI<0), yelp (NLL CI<0 & acc +0.050), banking77 (NLL CI<0 & acc +0.263)
  (by the NLL-CI clause alone: 3 of 6: ag_news, yelp, banking77)
- tasks worse by more than 0.05 raw NLL: sms_spam (+0.052), sst2 (+0.055); worse by more than 0.05 calibrated accuracy: none
  raw NLL change per task: boolq -0.025, sms_spam +0.052, sst2 +0.055, ag_news -0.167, yelp -0.437, banking77 -0.955
- raw ECE up by more than 0.05: 0 task(s) (none); 'overfit' needs ≥ 3
  raw ECE change per task: boolq -0.032, sms_spam +0.047, sst2 +0.000, ag_news -0.042, yelp -0.190, banking77 -0.003

- ftB final readout vs ftA, mean NLL over six tasks: raw 0.6743 → 0.7070 (Δ +0.0327, NOT within 0.02); calibrated 0.6348 → 0.6440 (Δ +0.0092, within 0.02)
- ftB final readout vs ftA, mean NLL over five single-prompt tasks: raw 0.4569 → 0.4860 (Δ +0.0290, NOT within 0.02); calibrated 0.4158 → 0.4276 (Δ +0.0118, within 0.02)
