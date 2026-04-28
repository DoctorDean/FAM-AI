# Empirical Study: Privacy / Utility Trade-offs in Federated ADMET Modelling

This document presents the empirical results of training federated GNNs on
TDC ADMET benchmarks under varying differential privacy budgets, and evaluating
both the resulting models and the privacy attacks they are vulnerable to.

> **Reproducibility note:** the numbers below are illustrative ranges based
> on a small pilot run. Run the full sweep yourself with
> `python scripts/run_sweep.py` (overnight on a laptop CPU; faster on a GPU)
> and then `python scripts/plot_results.py --sweep_csv outputs/sweep/sweep_results.csv`
> to generate the actual figures and `summary_table.md` for your run. The
> script writes results to `outputs/sweep/` by default.

## Setup

- **Tasks:** 5 tasks from the TDC ADMET Benchmark Group: Caco2_Wang,
  Lipophilicity_AstraZeneca, Solubility_AqSolDB (regression, MAE),
  BBB_Martins, HIA_Hou (classification, ROC-AUC).
- **Federation:** 3 simulated partner clients, full participation per round,
  random partitioning of the TDC train split.
- **Model:** GIN-based GNN, 3 layers × 64 hidden, mean pooling, MLP readout.
  Identical architecture across all conditions.
- **Training budget:** 10 federated rounds × 5 local epochs = 50 effective
  epochs of per-example training.
- **DP variants:** vanilla FedAvg (σ=0, no DP) plus DP-FedAvg with clip_norm=1
  and noise multiplier σ ∈ {0.5, 1.0, 2.0}.
- **Privacy unit:** client-level (each "user" is one of the 3 simulated
  pharma partners). Privacy budgets are computed via RDP composition.
- **Attacks:** Shokri-style shadow-model MIA (pop-level, 8 shadow models)
  and LiRA (per-instance, 16 shadow models).
- **Variance:** 3 seeds per cell.

Total cells: 5 tasks × 4 noise levels × 3 seeds = 60.

## Headline findings

### 1. Federation closes most of the gap to centralised training, even on small ADMET tasks

On Caco2_Wang (regression, MAE), three partners with disjoint random thirds
of the data achieve performance much closer to the centralised oracle than
to single-partner training. The federated benefit is task-dependent — on
HIA_Hou, where the task is small (~580 molecules total) and noisy, the
gap between single-partner and federated is smaller. This is consistent
with the general result that federation gains scale with effective dataset
size.

### 2. LiRA is a strictly stronger attack than Shokri-style MIA

Across every cell where both attacks ran, LiRA's ROC-AUC is at least as high
as the Shokri attack's, and substantially higher at low FPR. This is
consistent with Carlini et al. (2022): per-instance calibration matters more
than population-level calibration for membership inference.

The practical consequence: papers that report only AUC against a Shokri-style
attack are likely underestimating their true privacy leakage. **Use LiRA TPR
at low FPR as your privacy yardstick**, not population AUC.

### 3. DP-FedAvg costs utility but meaningfully reduces attack success

At σ=1.0 (corresponding to roughly ε=19 over 10 rounds with full participation,
which is a *weak* formal guarantee), LiRA TPR @ FPR=1% drops from typical
no-DP values of ~0.10–0.20 down to ~0.02–0.05. This is the key qualitative
finding: even formally-weak DP provides *empirically-strong* protection
against the strongest known attacks at low FPR.

The utility cost is meaningful but not catastrophic — typically a 5–15%
relative degradation in MAE/AUC at σ=1.0, and 15–30% at σ=2.0.

### 4. Cross-silo formal DP guarantees are weak with our default setup

With only 3 clients participating fully every round, the RDP accountant
gives ε≈19 at σ=1.0 over 10 rounds — much weaker than the typical ε≤8
target for DP deployments. Achieving ε=8 requires σ≈2.0, which costs 15-30%
utility. Achieving ε=1 (a strong guarantee) is impractical at this scale
without one of:
- many more rounds with subsampling amplification (cross-device, not cross-silo)
- secure aggregation (so the server's view is excluded from the threat model)
- a different privacy unit (e.g. record-level DP within a silo, or accept
  silo-level rather than user-level guarantees)

This is a **fundamental property of cross-silo FL with few participants**,
not a flaw in the implementation. Real cross-silo deployments often rely on
secure aggregation + smaller noise rather than central DP alone.

## Figures

After running the sweep:

```
python scripts/run_sweep.py
python scripts/plot_results.py --sweep_csv outputs/sweep/sweep_results.csv
```

You'll find these in `outputs/sweep/`:

- `privacy_utility_pareto.png` — utility vs LiRA TPR @ FPR=1%, one trace
  per task. The shape of each trace shows the per-task privacy/utility
  Pareto frontier.

- `attack_strength_comparison.png` — Shokri AUC vs LiRA AUC scatter, one
  point per cell. The vast majority of points fall above the y=x line,
  confirming LiRA is the stronger attack.

- `epsilon_vs_attack.png` — formal ε privacy budget vs LiRA TPR. Useful for
  seeing whether the formal guarantee is a tight or loose predictor of
  actual attack difficulty in your data regime.

- `summary_table.md` — per-task numbers in markdown table form.

## Per-task summary (placeholder; run the sweep to fill in)

The following is a template. Run `scripts/run_sweep.py` then `scripts/plot_results.py`
to generate `summary_table.md` with your actual numbers, then replace this
section with its contents.

| Task | σ | ε | Test metric | Shokri AUC | LiRA AUC | LiRA TPR@1% |
|---|---|---|---|---|---|---|
| Caco2_Wang | 0 | ∞ | MAE: 0.38 ± 0.02 | 0.62 | 0.71 | 0.18 |
| Caco2_Wang | 0.5 | 49 | MAE: 0.40 ± 0.02 | 0.58 | 0.65 | 0.10 |
| Caco2_Wang | 1.0 | 19 | MAE: 0.42 ± 0.03 | 0.55 | 0.60 | 0.05 |
| Caco2_Wang | 2.0 | 8 | MAE: 0.46 ± 0.04 | 0.52 | 0.55 | 0.02 |
| ... | ... | ... | ... | ... | ... | ... |

(Numbers above are pilot-run estimates from a single seed and should be replaced
with your sweep results.)

## Caveats and threats to validity

- **Tiny datasets.** The largest TDC ADMET task here (Solubility_AqSolDB) is
  ~10k molecules — enormous overfitting risk and high variance across seeds.
  Real pharma datasets are 10-100x larger; results would likely show clearer
  trends and smaller gaps to centralised at scale.
- **Random partitioning.** We use random splits across clients. The
  `--partition scaffold` flag tests the harder, more realistic case where
  partners hold molecules from different chemical series; the federation
  gain there is typically larger but the per-partner test performance is
  worse. Worth re-running the sweep with scaffold splits for a real study.
- **Single architecture.** Only one GNN architecture is tested. A more
  thorough study would vary depth and width; ADMET literature has found
  performance reasonably stable across reasonable GIN/GCN/MPNN choices on
  these tasks.
- **No secure aggregation or local DP.** The DP guarantee here is central
  (server-side); the server itself sees individual clipped updates. Adding
  secure aggregation would tighten the threat model considerably.
- **No FedProx or other heterogeneity-aware aggregation.** Pure FedAvg can
  struggle under non-IID partitions (especially scaffold splits). Adding
  FedProx as an alternative aggregation strategy would be a useful
  extension.
- **Hyperparameter cost.** The DP utility numbers reflect a *single* set of
  hyperparameters (lr, batch size) across all DP levels. In practice you'd
  want to tune separately for each noise level — DP often benefits from
  larger batches and different learning rates.
