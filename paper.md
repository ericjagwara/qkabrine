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

Quantum machine learning (QML) models called variational quantum circuits are
trained on classical data using techniques borrowed from classical machine
learning, but they first require a human to design the circuit by hand:
choosing a gate arrangement, a way of loading classical data into qubits, an
initialization scheme, and training hyperparameters. These choices interact
with one another, so a design that performs well under one data-loading
scheme may fail under another, and a promising-looking circuit can still be
untrainable because of a phenomenon called a barren plateau, where gradients
vanish exponentially as the circuit grows [@mcclean2018barren]. `qkabrine-automl` is a Python package that
automates this design process. Given a labelled dataset for classification
or regression, it searches jointly over circuit architecture, circuit depth,
data-loading scheme, model type (a trainable circuit versus a quantum
kernel), parameter initialization, and learning rate, using one of five
interchangeable search algorithms, and reports the best combination found,
optionally exporting it as OpenQASM 2.0 for use outside the package.

# Statement of need

Existing QML libraries such as PennyLane [@bergholm2018pennylane] and Qiskit
[@qiskit2024] (including its Qiskit Machine Learning module) give researchers
the building blocks needed
to construct and train a variational quantum circuit, but they leave the
circuit design itself to the user. In practice this design step is usually
done by hand, through small ad hoc trials rather than a systematic search,
which makes it easy to under-explore the design space and to draw
conclusions from a circuit and encoding pair that happened to be tried
rather than one that was shown to be a good fit for the data. `qkabrine-automl`
is aimed at two audiences: researchers who want to evaluate what a
reasonably well-tuned quantum model can do on a new dataset without first
becoming circuit designers themselves, and researchers working on quantum
architecture search (QAS) methodology who want a reusable search harness
with several interchangeable strategies rather than a one-off
implementation tied to a single search algorithm.

# State of the field

Classical AutoML tools such as Auto-sklearn [@feurer2015autosklearn] and
Optuna [@akiba2019optuna] solve an analogous configuration-search problem
for classical models, but they have no built-in notion of a quantum circuit
ansatz, a data-encoding scheme, or quantum-specific training pathologies
such as barren plateaus, so applying them to QML requires the user to build
all of that domain logic themselves before the search can even start. Prior
work on quantum architecture search (QAS) has explored several distinct
search strategies for the architecture itself: structure optimization
methods that co-optimize gate placement and parameter values within a
fixed circuit template [@ostaszewski2021structure], gradient-based
differentiable search over a distribution of candidate gate layouts
[@zhang2022dqas], and reinforcement-learning agents that construct a
circuit action-by-action [@kuo2021rlqas]. Despite their different search
mechanics, these methods share a common scope: architecture is searched
while the data-encoding scheme and training hyperparameters are held fixed
or tuned separately in a later stage. This staged approach can discard a
genuinely strong architecture simply because it was only ever evaluated
under a mismatched encoding, or can report an inflated advantage for an
architecture that was tuned more carefully than its competitors.
`qkabrine-automl`'s contribution relative to this landscape is not another
search strategy for architecture in isolation, but a search space and
evaluation harness that makes architecture, encoding, and hyperparameters
co-optimizable in a single search, with the search strategy itself
swappable among five options ranging from exhaustive and random baselines
to Bayesian, evolutionary, and successive-halving methods. It is a
supporting library rather than a single search strategy that
requires the user to write the QAS logic on top of it: existing QML
libraries provide the primitives, existing classical AutoML tools provide
generic search algorithms, and `qkabrine-automl` provides the
domain-specific search space and evaluation logic that connects the two
for the quantum case specifically.

# Software design

The package is organized around a fixed evaluation interface and a
pluggable search strategy: any candidate configuration, meaning an
architecture, encoding, model type, initialization, and learning rate, is
turned into a trainable PennyLane circuit through the same code path
regardless of which search strategy proposed it, so that grid, random,
Bayesian, evolutionary, and successive-halving search can be swapped
without touching the evaluation logic. This separation was chosen because
the main open question in QAS is which search strategy is most
sample-efficient for a given problem size, and answering that requires
holding the evaluation logic fixed while the search strategy varies, rather
than reimplementing evaluation for each strategy. The architecture space
itself spans twelve named ansätze (including strongly entangling,
hardware-efficient, data-re-uploading, and simplified-two-design layouts)
crossed with four data-encoding schemes (angle, angle-YZ, IQP, and
amplitude embedding), so that a search can distinguish an architecture that
underperforms on its own merits from one that was simply paired with an
unsuitable encoding. A second design decision was to treat quantum kernel
methods and variational circuits as directly comparable candidates within
the same search rather than as two separate pipelines, since the
literature has not converged on which paradigm is preferable for a given
dataset size and qubit budget, and forcing an early choice between them
would bias the search before any data has been seen. Trainability
diagnostics, specifically a Data Quantum Fisher Information Metric
estimate [@haug2024dqfim] and a gradient-magnitude barren-plateau monitor
targeting the vanishing-gradient failure mode described by
@mcclean2018barren, are implemented as an optional prescreening and
monitoring layer that sits in front of the evaluation step, so that the
search budget is not spent training circuits that are unlikely to produce
a useful gradient signal in the first place; expressibility and entangling
capability [@sim2019expressibility] are computed alongside these
diagnostics to characterize *why* a candidate circuit trains well or
poorly, not just whether it does. A post-search circuit-surgery pass prunes
near-identity rotation gates and simplifies redundant gate pairs, reducing
the depth of the exported circuit for near-term hardware without requiring
the user to re-run the search. Because all quantum execution goes through
PennyLane, the package can extend to noise models and future
PennyLane-supported hardware backends without changing the search or
evaluation code; the current release already supports simple noise models
(for example depolarizing and bit-flip channels) for noise-aware training
on top of classical simulation.

A minimal end-to-end use of the package is:

```python
from qkabrine_automl import QkabrineAutoML

automl = QkabrineAutoML(task="classification", search_strategy="bayesian")
automl.fit(X_train, y_train)
automl.leaderboard()
print(automl.export_qasm())
```

`fit` runs the joint search and trains the best candidate found;
`leaderboard` reports the ranked configurations that were evaluated; and
`export_qasm` returns the winning circuit as OpenQASM 2.0 for use outside
the package.

# Research impact statement

`qkabrine-automl` is a newly released package, published on PyPI and hosted
on GitHub under the MIT license with an accompanying test suite,
ReadTheDocs documentation, and a Zenodo archival DOI
(10.5281/zenodo.19308532). It does not yet have citing publications or
third-party software integrations to report; its only usage evidence so
far is informal feedback from early testers, acknowledged below, that
shaped several interface decisions during development. Its near-term
significance rests on filling a concrete gap identified above, joint
architecture, encoding, and hyperparameter search for QML, that is not
addressed by combining existing QML and classical AutoML libraries without
substantial additional code, and on being immediately usable and
reproducible: the package installs from PyPI, ships with runnable
classification and regression examples in its documentation, and its
search results can be exported as OpenQASM 2.0 for verification or reuse
outside the package.

# AI usage disclosure

Generative AI (Claude, Anthropic) was used to assist with writing portions
of the `qkabrine-automl` source code and with drafting this paper's text.
All AI-assisted code was written and reviewed by the author, who is
responsible for its correctness; all AI-assisted paper text was reviewed
and edited by the author for factual accuracy before submission, and no
performance claims or benchmark results are included in this paper beyond
what is directly verifiable from the released package and its
documentation.

# Acknowledgements

We thank the early testers of the `qkabrine` package for trying it on
their own datasets and providing feedback that shaped several interface
decisions.

# References
