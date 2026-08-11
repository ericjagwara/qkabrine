---
title: 'Qkabrine: A Joint Architecture, Encoding, and Hyperparameter Search Framework for Quantum Machine Learning'
tags:
  - Python
  - quantum computing
  - quantum machine learning
  - AutoML
  - variational quantum circuits
  - neural architecture search
authors:
  - name: Eric Jagwara
    orcid: 0009-0003-4935-3667
    affiliation: 1
affiliations:
  - name: Solid Elf Labs, Uganda
    index: 1
date: 12 August 2026
bibliography: paper.bib
---

# Summary

Variational quantum circuits are the workhorse of near-term quantum machine
learning (QML), but building one that works well requires choosing, largely
by hand, a circuit ansatz, a data encoding scheme, a parameter
initialization strategy, and an optimizer configuration. These choices
interact: a circuit architecture that performs well under angle encoding may
perform poorly under amplitude embedding, and a promising architecture can
still fail to train because of barren plateaus. `qkabrine-automl` is a
Python package that treats this whole configuration space as a single joint
search problem. Given a classification or regression dataset, it searches
across circuit architecture, circuit depth, data encoding, model paradigm
(variational circuit versus quantum kernel), parameter initialization, and
learning rate simultaneously, using one of five interchangeable search
strategies (grid, random, Bayesian optimization, an evolutionary algorithm,
or successive halving). The package also exposes circuit-analysis
diagnostics, including expressibility, entangling capability, and a Data
Quantum Fisher Information Metric (DQFIM) trainability score, so that
candidates can be screened or monitored for barren plateaus during search, and it can
export the best circuit found as OpenQASM 2.0 for deployment on other
toolchains.

# Statement of need

Existing QML libraries such as PennyLane [@bergholm2018pennylane] and Qiskit
Machine Learning [@qiskit2024] provide the primitives needed to build and
train a variational quantum circuit, but they do not search the design
space for the user: the user still selects the ansatz, the encoding, and
the training hyperparameters by hand, typically through manual
trial-and-error or a small hand-rolled grid. Classical AutoML tools such as
Auto-sklearn [@feurer2015autosklearn] and Optuna [@akiba2019optuna] solve an
analogous problem for classical models, but they have no notion of a
quantum circuit ansatz, an encoding scheme, or quantum-specific training
pathologies such as barren plateaus, so they cannot be applied directly to
QML pipelines. Prior quantum architecture search (QAS) work has generally
searched over gate arrangements alone, leaving the encoding and
hyperparameters fixed or hand-tuned separately.

`qkabrine-automl` is aimed at researchers and practitioners who want to
benchmark QML approaches on a new dataset without first becoming circuit
designers, and at QAS researchers who want a reusable search harness rather
than a one-off implementation. By searching architecture, encoding, and
hyperparameters jointly rather than in separate stages, the package avoids
the common failure mode where a strong architecture is discarded because it
was only ever evaluated under a mismatched encoding. Built-in trainability
diagnostics (DQFIM prescreening, barren plateau monitoring with optional
automatic circuit surgery) make the search budget-aware, spending less
effort on candidates that are unlikely to train, and the OpenQASM export
lets a discovered circuit be carried into hardware-facing tooling once the
search is complete.

# Functionality

The core entry point, `QkabrineAutoML`, follows a scikit-learn-like
`fit`/`predict`/`score` interface. A search is configured by choosing a
task (`'classification'` or `'regression'`), a search strategy, and,
optionally, the subset of encodings, qubit counts, and circuit depths to
consider:

```python
from qkabrine_automl import QkabrineAutoML

automl = QkabrineAutoML(
    task="classification",
    n_qubits=4,
    max_layers=2,
    search_strategy="bayesian",
    encodings=("angle", "iqp"),
    feature_reduction="pca",
)
automl.fit(X_train, y_train)
automl.leaderboard()
print(f"Test accuracy: {automl.score(X_test, y_test):.4f}")
print(automl.export_qasm())
```

The search space spans twelve circuit architectures (including
strongly-entangling, hardware-efficient, data-re-uploading, and
QFT-inspired cascading ansätze), four data encodings (angle, angle-YZ, IQP,
and amplitude embedding), five parameter-initialization schemes, and
learning rate, jointly with model paradigm (variational circuit or quantum
kernel plus support vector machine), so that the two dominant approaches to
NISQ-era supervised learning are compared on equal footing rather than
requiring two separate pipelines. True multi-class classification is
supported via `qml.probs()` with a cross-entropy objective, rather than the
one-vs-rest decomposition common in smaller QML libraries.

For diagnosing why a candidate circuit trains poorly, `qkabrine-automl`
exposes `DataQuantumFisherMetric` for pre-training trainability estimates
and `BarrenPlateauMonitor` for gradient-magnitude monitoring during
training, with an option to trigger automatic circuit surgery (pruning
near-identity rotations and simplifying redundant gate pairs) when a
plateau is detected. Optional noise models (depolarizing, bit-flip, and
others) allow the search to account for NISQ hardware conditions rather
than assuming an idealized simulator. Fitted searches can be serialized
with `.save()`/`.load()`, and the best discovered circuit can be exported as
an OpenQASM 2.0 string via `.export_qasm()` for use outside the PennyLane
ecosystem the package builds on.

# Acknowledgements

We thank early users and contributors to the `qkabrine` repository for
testing and feedback.

# References
