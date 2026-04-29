# Sweep Summary

Aggregated over seeds (median ± IQR/2). σ is the DP noise multiplier;
σ=0 means vanilla FedAvg (no DP). LiRA TPR is at FPR=1%.

## MAE tasks (lower = better utility)

| Task | σ | ε | Test MAE | Shokri AUC | LiRA AUC | LiRA TPR@1% |
|---|---|---|---|---|---|---|
| Caco2_Wang | 0.0 | — | 0.565 ± 0.012 | 0.492 | 0.494 | 0.005 |
| Caco2_Wang | 0.5 | 48.80 | 15373.994 ± 6673.798 | 0.500 | 0.500 | 0.005 |
| Caco2_Wang | 1.0 | 19.05 | 9526869.770 ± 11713889.908 | 0.500 | 0.504 | 0.006 |
| Caco2_Wang | 2.0 | 8.08 | 6387658729.197 ± 2454013637.275 | 0.500 | 0.504 | 0.006 |
| Lipophilicity_AstraZeneca | 0.0 | — | 0.787 ± 0.002 | 0.513 | 0.532 | 0.014 |
| Lipophilicity_AstraZeneca | 0.5 | 48.80 | 97317.694 ± 28287.007 | 0.500 | 0.492 | 0.010 |
| Lipophilicity_AstraZeneca | 1.0 | 19.05 | 9301605.171 ± 4613746.220 | 0.500 | 0.495 | 0.010 |
| Lipophilicity_AstraZeneca | 2.0 | 8.08 | 4635759009.920 ± 2434651991.379 | 0.500 | 0.504 | 0.012 |
| Solubility_AqSolDB | 0.0 | — | 1.044 ± 0.006 | 0.542 | 0.508 | 0.014 |
| Solubility_AqSolDB | 0.5 | 48.80 | 21404.815 ± 2281.790 | 0.500 | 0.495 | 0.012 |
| Solubility_AqSolDB | 1.0 | 19.05 | 10385205.385 ± 17057835.043 | 0.500 | 0.497 | 0.009 |
| Solubility_AqSolDB | 2.0 | 8.08 | 679149839.406 ± 1538910608.902 | 0.500 | 0.497 | 0.012 |

## AUC tasks (higher = better utility)

| Task | σ | ε | Test AUC | Shokri AUC | LiRA AUC | LiRA TPR@1% |
|---|---|---|---|---|---|---|
| BBB_Martins | 0.0 | — | 0.800 ± 0.008 | 0.496 | 0.520 | 0.006 |
| BBB_Martins | 0.5 | 48.80 | 0.497 ± 0.006 | 0.496 | 0.493 | 0.013 |
| BBB_Martins | 1.0 | 19.05 | 0.500 ± 0.001 | 0.480 | 0.500 | 0.008 |
| BBB_Martins | 2.0 | 8.08 | 0.500 ± 0.008 | 0.524 | 0.495 | 0.018 |
| HIA_Hou | 0.0 | — | 0.870 ± 0.005 | 0.512 | 0.512 | 0.025 |
| HIA_Hou | 0.5 | 48.80 | 0.500 ± 0.046 | 0.439 | 0.530 | 0.012 |
| HIA_Hou | 1.0 | 19.05 | 0.500 ± 0.007 | 0.552 | 0.522 | 0.015 |
| HIA_Hou | 2.0 | 8.08 | 0.500 ± 0.003 | 0.439 | 0.491 | 0.032 |
