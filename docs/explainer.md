# Federated ADMET, Membership Inference, and DP — Explained

This document walks through what this repo is doing and why, for a reader
who knows ML but not necessarily federated learning, GNNs for chemistry, or
membership inference attacks. By the end you should have an intuition for:

1. Why a pharma company would care about federated learning
2. How GNNs predict molecular properties
3. Why federated learning leaks information even though raw data never leaves
4. How a membership inference attack actually works
5. What differential privacy buys you, and what it doesn't

I'll skip code-level details — read the source for those. The aim here is
the *picture*.

## 1. The pharma collaboration problem

Drug discovery is hugely expensive and almost everything fails. ADMET
properties — Absorption, Distribution, Metabolism, Excretion, Toxicity —
are the cluster of properties that determine whether a candidate molecule
ever makes it through clinical trials. Predicting them well, before you
synthesise and test, would save vast amounts of time and money.

Pharma companies have huge ADMET datasets. They've measured these
properties for hundreds of thousands of compounds over decades of internal
work. But they can't share them, for two reasons:

- **IP and competitive concerns.** The set of molecules a pharma has
  measured is a window into their research programme. Knowing that AstraZeneca
  has a thousand measurements clustered around a particular scaffold tells
  you a lot about what they're working on.
- **Regulatory and legal constraints.** Some data was collected under
  agreements that explicitly prohibit sharing.

So the collaboration question is: *can companies pool their ADMET prediction
power without pooling their molecules?*

That's federated learning's pitch. Instead of moving the data to a central
model, you move the model to each company's data. Each round, the central
"server" sends out the current global model, each company trains it briefly
on their private data, and sends back only the *updated weights* (or an
update relative to the previous weights). The server averages the updates
and sends the new model out for the next round.

The promise is simple and powerful: every company gets a model trained on
the *combined* knowledge of all participants, but no raw molecule ever
leaves any company's premises.

The catch — the focus of this repo — is whether "no raw data leaves" actually
means "no information about the data leaks." It doesn't, by default.

## 2. GNNs for molecular property prediction

A molecule is naturally a graph: atoms are nodes, bonds are edges. Both have
features — atomic number, hybridisation, formal charge, ring membership for
nodes; bond type, conjugation, ring membership for edges.

Graph neural networks (GNNs) operate on this representation directly. The
basic operation is *message passing*: each atom updates its representation
by aggregating messages from its neighbours, applying some non-linearity,
and updating. After K rounds of this, each atom's representation encodes
information about its K-hop neighbourhood. Pool these atom representations
into a single vector for the whole molecule, and feed that to an MLP that
predicts the property of interest.

We use **GIN** (Graph Isomorphism Network) here. GIN's specific message
passing rule is a sum aggregation followed by an MLP, and it's been proved
maximally expressive among message-passing GNNs (Xu et al., 2019). For our
purposes this means: if a property is computable from local atom-and-bond
structure (which most ADMET properties largely are), GIN can in principle
learn it.

In practice on small TDC tasks, the choice of GNN architecture matters less
than people often assume. GIN, GCN, and a basic message-passing network all
give roughly similar performance. What matters more is regularisation,
because these datasets are small.

## 3. Why federation leaks anyway

Here's the crux. Even though Alice's molecules never leave Alice's machine,
the *model updates* she sends back to the server are computed from those
molecules. They have to encode something about the data, otherwise training
wouldn't work.

The information leakage takes a few forms:

- **Update magnitude.** The norm of Alice's update is roughly proportional
  to how much her local data disagrees with the current global model.
  Watching update norms over time tells you which clients have "easy" or
  "hard" data.
- **Direction of updates.** The direction of the update encodes which
  features need adjustment to fit Alice's data. Gradients, even averaged
  ones, are surprisingly informative about the underlying training set.
- **The final aggregated model itself.** This is the threat model we focus
  on in this repo. The final model is a function fit to the union of
  everyone's data. Trained models — including aggregated federated models —
  are systematically more confident on examples they've seen during training
  than on similar but unseen examples. This phenomenon is called
  *memorisation*, and it's what enables membership inference.

## 4. Membership inference, intuitively

A membership inference attack (MIA) is a yes/no test:

> Given a trained model M and a candidate input x, was x in M's training set?

This is meaningful for ADMET because the *list of molecules a partner
contributed* is itself confidential information. If an attacker can run a
membership test against the public global model and recover Alice's training
set, that's a privacy breach: they've learned what Alice was researching.

How does the attack work? The simplest version is just:

- Train a "shadow model" with the same architecture on a population dataset.
- Note: shadow-model loss on training-set examples is systematically *lower*
  than loss on held-out examples. This is overfitting in action.
- Apply this insight to the target model: query molecules with surprisingly
  low loss are probably members.

That's the **Shokri-style attack** in `src/attacks/membership_inference.py`.
It works, but it's crude — it uses a single global threshold for "low loss
implies member."

**LiRA** (Carlini et al., 2022, in `src/attacks/lira.py`) is much more
principled. The insight: different molecules have different *intrinsic*
difficulties. An easy molecule will have low loss whether it was trained on
or not. A hard molecule trained on will still have higher loss than an easy
molecule that wasn't trained on. The Shokri attack confuses these.

LiRA fixes this by training many shadow models, with each candidate molecule
randomly included or excluded from each shadow model's training set. Then
for each query molecule, it compares the actual target model's loss to two
distributions: the loss-when-trained-on distribution and the
loss-when-not-trained-on distribution, *for that specific molecule*. After
a stabilising transform, both distributions are roughly Gaussian. The optimal
test (Neyman-Pearson) is the likelihood ratio. That's all LiRA is.

The reason this matters: at very low false-positive rates — which is what
you actually care about in a privacy context — LiRA dominates. It correctly
identifies a small number of training molecules with very high confidence,
where Shokri-style attacks dilute their signal across all molecules and look
weak in aggregate.

## 5. What differential privacy buys you

Differential privacy (DP) is a formal framework for bounding how much any
single input can affect the output of a randomised computation. The core
definition: an algorithm A is (ε, δ)-DP if for any two datasets that differ
in one record, and any output region S,

> Pr[A(D) ∈ S] ≤ e^ε · Pr[A(D') ∈ S] + δ

ε is the budget — smaller is more private. δ is a small slack term, usually
1/N where N is the dataset size.

The key fact: this guarantee is robust against *all* post-processing. If
the trained model is (ε, δ)-DP with respect to one input, no membership
inference attack — Shokri, LiRA, or anything not yet invented — can do better
than what (ε, δ) allows.

How do you make federated training DP?

For each round:
1. Each client computes their update as usual.
2. Clip each client's update to a fixed L2 norm C (this bounds the sensitivity).
3. Sum the clipped updates on the server.
4. Add Gaussian noise with σ = (z * C) where z is the "noise multiplier."
5. Average and use as the new global model.

Composing T rounds of this mechanism gives a (ε, δ)-DP guarantee for the
final model with respect to one client's contribution (user-level DP). The
math for translating (z, T) into (ε, δ) is the *RDP accountant*; we use
Google's `dp_accounting` library.

What does this cost?
- **Utility.** Clipping plus noise hurts model quality. How much depends on
  your clip norm, noise multiplier, learning rate, and how many rounds you
  can afford to run.
- **Hyperparameter sensitivity.** DP training often wants different learning
  rates and batch sizes than non-DP training.

What does this not protect against?
- **Server-side adversaries.** Our DP setup adds noise on the server. If the
  server itself is compromised, it's already seen the individual clipped
  updates. Closing this gap requires *secure aggregation*: a cryptographic
  protocol where the server only sees the sum of updates, not individuals.
- **Side-channel leaks.** Update norms, timing, network metadata — these
  are outside the DP guarantee.
- **Honest-but-curious clients.** A malicious client can extract information
  about the global model itself, which then leaks information about other
  clients' data through the model.

So DP is one important layer, not the whole stack. The repo demonstrates the
empirical effect — DP-FedAvg's noisy updates make LiRA much harder — and
provides the formal (ε, δ) bound for any given run.

## 6. The cross-silo problem

This repo simulates 3 partner companies. That's representative of real-world
cross-silo federated learning: a small number of large institutions with full
participation each round. It's also where DP is *hardest*.

The reason: most of the privacy-amplification tricks that make DP-SGD
practical for large datasets rely on subsampling — each round, only a
small fraction of records participates, and that random sub-sampling
amplifies privacy. With 3 silos all participating every round, you get no
amplification. So achieving an honest ε=8 over 10 rounds requires a noise
multiplier around 2.0, which costs serious utility.

Real cross-silo deployments handle this with one or more of:
- **Secure aggregation** to remove the server from the threat model entirely
- **Local DP**, where each silo adds enough noise on its own that even the
  server is protected
- **Per-record DP within each silo**, accepting that the *silo itself* is
  protected (i.e. the threat model assumes silo-level adversaries, not
  silo-membership)

Each has its own trade-offs and is somewhat application-specific. This repo
sticks with the simplest central DP setup so the mechanics are clear; future
work would be to add secure aggregation and compare.

## 7. Putting it together

The story this repo tells, end-to-end:

1. Federated learning lets pharmas collaborate on ADMET models without
   sharing molecules.
2. The aggregated model still encodes information about each partner's data,
   measurable through membership inference attacks. Even simple shadow-model
   attacks recover non-trivial signal; modern attacks like LiRA recover much
   more.
3. Differentially-private FedAvg meaningfully suppresses these attacks.
   Even a weak formal guarantee (ε≈19) reduces LiRA TPR @ FPR=1% by a
   factor of 3-5x.
4. Achieving strong formal guarantees in a cross-silo setting (a small
   number of full-participation partners) is genuinely hard and
   typically requires combining DP with secure aggregation.

What's missing for a production deployment? Briefly: secure aggregation,
per-task hyperparameter tuning, communication efficiency improvements like
gradient compression, robustness against malicious clients, audit logging,
and integration with whatever ELN/LIMS the partners actually use.

But the core trade-off — federation gives you collaborative models at the
cost of an attack surface that DP can mitigate but not eliminate — is the
fundamental shape of the problem. Everything else is engineering on top.

## Further reading

- **Federated learning generally:** McMahan, Moore, Ramage, Hampson, Arcas.
  *Communication-Efficient Learning of Deep Networks from Decentralized
  Data.* AISTATS 2017.
- **GNN for molecular property prediction:** Yang et al. *Analyzing Learned
  Molecular Representations for Property Prediction.* JCIM 2019.
- **Membership inference, original:** Shokri, Stronati, Song, Shmatikov.
  *Membership Inference Attacks Against Machine Learning Models.* IEEE S&P 2017.
- **Membership inference, modern:** Carlini, Chien, Nasr, Song, Terzis,
  Tramer. *Membership Inference Attacks From First Principles.* IEEE S&P 2022.
- **DP-FedAvg:** McMahan, Ramage, Talwar, Zhang. *Learning Differentially
  Private Recurrent Language Models.* ICLR 2018.
- **RDP accounting:** Mironov. *Renyi Differential Privacy.* CSF 2017.
- **Secure aggregation:** Bonawitz et al. *Practical Secure Aggregation for
  Privacy-Preserving Machine Learning.* CCS 2017.
