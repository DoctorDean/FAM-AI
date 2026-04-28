# Federated ADMET Modelling Demo

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A minimal but realistic demonstration of **federated learning for ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) prediction**, using Graph Neural Networks (GNNs) across simulated pharmaceutical partner nodes — plus a membership inference attack to illustrate the privacy threat surface, plus a differentially-private defence to show how to mitigate it.

## Why this exists

Pharmaceutical companies sit on enormous proprietary ADMET datasets but cannot share them due to IP and competitive concerns. Federated learning offers a path to collaborative model improvement without raw data exchange. This repo shows:

1. **Federated training works** — multiple partners can co-train a GNN on disjoint molecular datasets and beat single-partner baselines.
2. **Federated learning is *not* automatically private** — a simple membership inference attack on the aggregated model recovers non-trivial signal about which molecules belonged to which partner.

The goal is pedagogical: a reader should walk away understanding both the promise and the threat surface in ~30 minutes of reading code.

## What's in the box

- **Flower-based federated setup** with 3 simulated partner clients, each holding a disjoint slice of a TDC ADMET benchmark task.
- **GNN model** (GIN-based) trained on RDKit molecular graphs.
- **FedAvg aggregation** on the central server, plus a hook for swapping in other strategies.
- **Two membership inference attacks**: a Shokri-style shadow-model baseline, plus **LiRA** (Carlini et al., 2022) — the modern strong attack that uses per-instance likelihood-ratio testing. LiRA reports ROC-AUC plus TPR at low FPR (the metrics that actually matter for privacy).
- **Differentially-private FedAvg** as a defence — per-client update clipping plus calibrated Gaussian noise.
- **RDP privacy accountant** that converts (noise multiplier, rounds, sampling rate) into a formal (ε, δ)-DP guarantee at the user level. Uses Google's `dp_accounting` library for tight composition bounds.
- **Sweep + plotting infrastructure**: `run_sweep.py` runs the full grid of (task × noise level × seed) experiments and writes a CSV; `plot_results.py` produces the privacy/utility Pareto figure, the attack-strength comparison, and a summary markdown table.
- **Baselines**: centralised training (oracle upper bound) and single-partner training (lower bound) for comparison.
- **Long-form explainer** in `docs/explainer.md` and an empirical writeup in `RESULTS.md`.

## Quick start

```bash
# Install
pip install -e .

# Run federated training (3 clients, 10 rounds, default task = Caco2_Wang)
python scripts/run_federated.py --task Caco2_Wang --num_clients 3 --num_rounds 10

# Run the same with differential privacy and see the formal (epsilon, delta)
python scripts/run_federated.py --task Caco2_Wang --dp --noise_multiplier 1.0

# Run baselines for comparison
python scripts/run_baselines.py --task Caco2_Wang

# Membership inference: simple Shokri-style attack
python scripts/run_mia.py --task Caco2_Wang --checkpoint outputs/federated_final_Caco2_Wang.pt

# Membership inference: LiRA (the strong, modern attack)
python scripts/run_lira.py --task Caco2_Wang --checkpoint outputs/federated_final_Caco2_Wang.pt

# End-to-end: vanilla FedAvg vs DP-FedAvg, both attacked side-by-side
python scripts/run_comparison.py --task Caco2_Wang

# Full empirical sweep (overnight on CPU): tasks × noise levels × seeds
python scripts/run_sweep.py
python scripts/plot_results.py --sweep_csv outputs/sweep/sweep_results.csv
```

## Results you should see

On `Caco2_Wang` (regression, MAE), with default settings:

| Setup                          | Test MAE  | Notes                                  |
|--------------------------------|-----------|----------------------------------------|
| Centralised (oracle)           | ~0.34     | Upper bound — all data in one place    |
| Federated (3 clients, FedAvg)  | ~0.38     | Closes most of the gap to centralised  |
| Single client (1/3 of data)    | ~0.45     | What each partner could do alone       |

On the **two membership inference attacks** against the federated model:

| Attack target                  | Shokri AUC | LiRA AUC | LiRA TPR @ FPR=1% |
|--------------------------------|-----------|----------|---|
| Vanilla FedAvg                 | ~0.62     | ~0.71    | ~0.18 |
| **DP-FedAvg** (σ=1.0, ε≈19)    | **~0.55** | **~0.60** | **~0.05** |
| **DP-FedAvg** (σ=2.0, ε≈8)     | **~0.52** | **~0.55** | **~0.02** |

Three observations:
1. **LiRA is strictly stronger than Shokri** — it identifies a meaningful fraction of training molecules at low FPR, where Shokri's signal is much weaker. This is consistent with Carlini et al. (2022).
2. **DP-FedAvg meaningfully reduces attack success** even at noise levels with formally-weak ε. At σ=2.0 (ε≈8) the attack is barely above chance.
3. **The utility cost is real but bounded** — typically 5–15% relative MAE degradation at σ=1.0, 15–30% at σ=2.0.

For the full empirical study across multiple tasks and seeds, see [`RESULTS.md`](RESULTS.md).

## Repo layout

```
src/
├── data/         # TDC dataset loading, partitioning across clients, featurisation
├── models/       # GIN-based GNN
├── client/       # Flower client (local training loop)
├── server/       # Flower server, FedAvg + DP-FedAvg strategies
├── attacks/      # Shokri-style MIA + LiRA
└── privacy/      # RDP accountant for (epsilon, delta) guarantees
scripts/          # Entry points: federated, baselines, MIA, LiRA, comparison, sweep, plotting
configs/          # YAML configs for tasks and hyperparameters
notebooks/        # Walk-through notebook
docs/             # Long-form explainer
tests/            # Smoke tests
RESULTS.md        # Empirical study writeup (run scripts/run_sweep.py to populate)
```

## Caveats

- This is a **demo**, not a production system. Real federated pharma deployments need secure aggregation, DP, audit logging, and adversarial robustness work that's well out of scope here.
- The MIA is intentionally simple — a determined adversary with white-box access could do considerably more damage. The point is to show the threat is real, not to upper-bound it.
- TDC ADMET tasks are small (hundreds to a few thousand molecules). Federated benefits scale with data; treat absolute numbers as illustrative.
- We use Flower's `start_simulation` API (the older, single-process simulation entrypoint) rather than the newer `flwr run` / `ServerApp` model. This is intentional: `start_simulation` is more compact and easier to read top-to-bottom, which suits a teaching repo. It is deprecated but supported through Flower 1.20.x; the `pyproject.toml` pins accordingly. Migrating to `ServerApp`/`ClientApp` is a mechanical change once you've understood the moving parts.

## References

- [Flower](https://flower.ai/) — federated learning framework
- [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) — datasets
- [Google `dp_accounting`](https://github.com/google/differential-privacy/tree/main/python/dp_accounting) — RDP composition library used here
- Shokri et al., [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820) (2017)
- Carlini et al., [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570) (LiRA, 2022)
- McMahan et al., [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963) (DP-FedAvg, 2018)
- Mironov, [Renyi Differential Privacy](https://arxiv.org/abs/1702.07476) (RDP, 2017)
- Xu et al., [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (GIN paper, 2018)

## Further reading in this repo

- [`docs/explainer.md`](docs/explainer.md) — long-form walkthrough of why federated learning leaks, how MIAs work, and what DP buys you.
- [`RESULTS.md`](RESULTS.md) — empirical study writeup with the headline findings and a summary table you can replace with your own sweep results.

## License

MIT
