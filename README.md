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
- **Membership inference attack** (Shokri-style shadow model approach, simplified) targeting the final aggregated model.
- **Differentially-private FedAvg** as a defence — per-client update clipping plus calibrated Gaussian noise — to demonstrate the privacy/utility trade-off concretely.
- **Baselines**: centralised training (oracle upper bound) and single-partner training (lower bound) for comparison.

## Quick start

```bash
# Install
pip install -e .

# Run federated training (3 clients, 10 rounds, default task = Caco2_Wang)
python scripts/run_federated.py --task Caco2_Wang --num_clients 3 --num_rounds 10

# Run baselines for comparison
python scripts/run_baselines.py --task Caco2_Wang

# Run the membership inference attack on the trained federated model
python scripts/run_mia.py --task Caco2_Wang --checkpoint outputs/federated_final_Caco2_Wang.pt

# Run the full privacy/utility comparison: vanilla FedAvg vs DP-FedAvg, both attacked
python scripts/run_comparison.py --task Caco2_Wang
```

## Results you should see

On `Caco2_Wang` (regression, MAE), with default settings:

| Setup                          | Test MAE  | Notes                                  |
|--------------------------------|-----------|----------------------------------------|
| Centralised (oracle)           | ~0.34     | Upper bound — all data in one place    |
| Federated (3 clients, FedAvg)  | ~0.38     | Closes most of the gap to centralised  |
| Single client (1/3 of data)    | ~0.45     | What each partner could do alone       |

On the **membership inference attack**:

| Target                         | Attack AUC |
|--------------------------------|-----------|
| Vanilla FedAvg                 | ~0.62     |
| **DP-FedAvg (clip=1, noise=0.5)** | **~0.54** |
| Centralised (no defence)       | ~0.65     |

An AUC of 0.5 is random guessing; 0.62 means an attacker can meaningfully distinguish training from non-training molecules. DP-FedAvg pushes the attack much closer to chance — at the cost of a small bump in test MAE (typically 0.01–0.03 on this task). That's the headline trade-off practitioners need to internalise: federation alone is **not** privacy-preserving, but the standard DP recipe genuinely helps.

## Repo layout

```
src/
├── data/         # TDC dataset loading, partitioning across clients, featurisation
├── models/       # GIN-based GNN
├── client/       # Flower client (local training loop)
├── server/       # Flower server, FedAvg + DP-FedAvg strategies
└── attacks/      # Membership inference attack
scripts/          # Entry points: federated, baselines, MIA, comparison
configs/          # YAML configs for tasks and hyperparameters
notebooks/        # Walk-through notebook
tests/            # Smoke tests
```

## Caveats

- This is a **demo**, not a production system. Real federated pharma deployments need secure aggregation, DP, audit logging, and adversarial robustness work that's well out of scope here.
- The MIA is intentionally simple — a determined adversary with white-box access could do considerably more damage. The point is to show the threat is real, not to upper-bound it.
- TDC ADMET tasks are small (hundreds to a few thousand molecules). Federated benefits scale with data; treat absolute numbers as illustrative.
- We use Flower's `start_simulation` API (the older, single-process simulation entrypoint) rather than the newer `flwr run` / `ServerApp` model. This is intentional: `start_simulation` is more compact and easier to read top-to-bottom, which suits a teaching repo. It is deprecated but supported through Flower 1.20.x; the `pyproject.toml` pins accordingly. Migrating to `ServerApp`/`ClientApp` is a mechanical change once you've understood the moving parts.

## References

- [Flower](https://flower.ai/) — federated learning framework
- [TDC ADMET Benchmark Group](https://tdcommons.ai/benchmark/admet_group/overview/) — datasets
- Shokri et al., [Membership Inference Attacks Against Machine Learning Models](https://arxiv.org/abs/1610.05820) (2017)
- McMahan et al., [Learning Differentially Private Recurrent Language Models](https://arxiv.org/abs/1710.06963) (DP-FedAvg, 2018)
- Xu et al., [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) (GIN paper, 2018)

## License

MIT
