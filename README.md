# ⚛️ Qkabrine AutoML

> **An open-source framework that unifies intelligent architecture search, quantum kernel methods, and variational circuits in a single AutoML pipeline.**


[![PyPI](https://img.shields.io/pypi/v/qkabrine-automl)](https://pypi.org/project/qkabrine-automl/)
[![Python](https://img.shields.io/pypi/pyversions/qkabrine-automl)](https://pypi.org/project/qkabrine-automl/)
[![Docs](https://readthedocs.org/projects/qkabrine/badge/?version=latest)](https://qkabrine.readthedocs.io)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19308532.svg)](https://doi.org/10.5281/zenodo.19308532)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/ericjagwara/qkabrine/blob/main/LICENSE)

*Created by [Eric Jagwara](https://ericjagwara.online)*

```bash
pip install qkabrine-automl
```

```python
from qkabrine_automl import QkabrineAutoML

automl = QkabrineAutoML(task='classification', search_strategy='bayesian')
automl.fit(X_train, y_train)
automl.leaderboard()
preds = automl.predict(X_test)
print(automl.export_qasm())
```

---

## What Makes This Different

Existing quantum ML tools force you to choose a circuit, choose an encoding, choose a training strategy, and hope it works. Qkabrine AutoML searches across **all of these simultaneously**:

| Dimension | What's Searched |
|---|---|
| **Circuit architecture** | 12 ansätze (strongly entangling, hardware efficient, data re-uploading, cascading, criss-cross, all-to-all, ...) |
| **Circuit depth** | 1 to `max_layers` for each architecture |
| **Data encoding** | Angle, Angle-YZ, IQP, Amplitude embedding |
| **Model paradigm** | Variational circuits *and* quantum kernel + SVM |
| **Parameter init** | Uniform, small, zeros, normal, block (avoids barren plateaus) |
| **Learning rate** | Co-optimized per candidate |

### Key Features

1. **Five search strategies in one API** — grid, random, Bayesian (GP + Expected Improvement), evolutionary (genetic algorithm), and successive halving (HyperBand-inspired).

2. **Joint search over architecture × encoding × hyperparameters** — most QAS tools only search over gate arrangements. We co-optimize the complete pipeline.

3. **Quantum kernels as first-class citizens** — the search considers both variational circuits and quantum kernel + SVM methods, comparing them on equal footing.

4. **Circuit surgery** — post-search pruning of near-identity rotation gates and simplification of redundant gate pairs, reducing circuit depth for NISQ deployment.

5. **True multi-class classification** — `qml.probs()` with cross-entropy loss trains all classes simultaneously.

6. **Expressibility & entangling capability metrics** — KL-divergence expressibility and Meyer-Wallach entanglement for circuit analysis.

7. **Training dynamics** — DQFIM trainability prediction, barren plateau monitoring, and quantum natural gradient optimization.

8. **Noise-aware training** for NISQ hardware readiness + **QASM export** for deployment.

---

## Quickstart

### Binary Classification

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from qkabrine_automl import QkabrineAutoML

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

automl = QkabrineAutoML(
    task='classification',
    n_qubits=4,
    max_layers=2,
    search_strategy='bayesian',
    encodings=('angle', 'iqp'),
    feature_reduction='pca',
)
automl.fit(X_train, y_train)
automl.leaderboard()
print(f"Test accuracy: {automl.score(X_test, y_test):.4f}")
```

### Multi-Class Classification

```python
from sklearn.datasets import load_iris
from qkabrine_automl import QkabrineAutoML

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

automl = QkabrineAutoML(
    task='classification',
    n_qubits=4,
    search_strategy='evolutionary',
)
automl.fit(X_train, y_train)
```

### Regression

```python
automl = QkabrineAutoML(task='regression', search_strategy='bayesian')
automl.fit(X_train, y_train)
```

### Save and Load

```python
automl.save('my_model.pkl')
loaded = QkabrineAutoML.load('my_model.pkl')
loaded.predict(X_test)
```

---

## Search Strategies

| Strategy | Use When |
|---|---|
| `'bayesian'` | Default. Best sample efficiency. GP + Expected Improvement. |
| `'evolutionary'` | Large search spaces. Genetic algorithm with crossover + mutation. |
| `'successive_halving'` | Training is expensive. Start many, halve, double budget. |
| `'grid'` | Small spaces. Exhaustive enumeration. |
| `'random'` | Quick baseline. Budget-controlled sampling. |

---

## Circuit Architectures (12)

| Name | Key Property |
|---|---|
| `strongly_entangling` | High expressibility |
| `hardware_efficient` | Low depth |
| `data_reuploading` | Data re-encoding between layers |
| `simplified_two_design` | Near-Haar coverage |
| `all_to_all` | Maximum connectivity |
| `cascading` | QFT-inspired multi-scale entanglement |
| `criss_cross` | Butterfly-pattern spread |
| `ring_of_cnots` | Dense parameterization |
| `full_rotation` | Maximum rotation freedom |
| `alternating_rx_ry` | Layer diversity |
| `shallow_rx` | Barren-plateau resistant |
| `hadamard_entangling` | Superposition-first |

---

## Training Dynamics

```python
from qkabrine_automl import DataQuantumFisherMetric, BarrenPlateauMonitor

# Predict trainability before training
dqfim = DataQuantumFisherMetric(n_qubits=4)
metrics = dqfim.predict_generalization(circuit_fn, X, n_params=12)
print(f"Trainability: {metrics.trainability_score:.3f}")

# Monitor for barren plateaus during training
monitor = BarrenPlateauMonitor(threshold=1e-7, auto_surgery=True)
```

---

## API Reference

### `QkabrineAutoML`

| Parameter | Default | Description |
|---|---|---|
| `task` | `'classification'` | `'classification'` or `'regression'` |
| `n_qubits` | `None` | Auto-inferred (max 10) |
| `max_layers` | `3` | Max circuit depth |
| `train_steps` | `40` | Gradient steps per candidate |
| `search_strategy` | `'bayesian'` | Search algorithm |
| `encodings` | `('angle',)` | Encodings to search |
| `optimizer` | `'adam'` | `'adam'`, `'sgd'`, `'momentum'` |
| `include_kernels` | `True` | Evaluate quantum kernel methods |
| `cv_folds` | `None` | Cross-validation folds (>= 2) |
| `noise_model` | `None` | `'depolarizing'`, `'bitflip'`, etc. |
| `feature_reduction` | `'pca'` | Dimensionality reduction method |
| `use_dqfim_prescreening` | `False` | Pre-screen via DQFIM |
| `monitor_barren_plateaus` | `False` | Early-stop on vanishing gradients |

### Methods

| Method | Description |
|---|---|
| `.fit(X, y)` | Run architecture search |
| `.predict(X)` | Predict with best model |
| `.predict_proba(X)` | Soft scores (classification) |
| `.score(X, y)` | Accuracy or R² |
| `.leaderboard()` | Print ranked results |
| `.best_circuit_summary()` | Gate-by-gate breakdown |
| `.export_qasm()` | OpenQASM 2.0 string |
| `.save(path)` / `.load(path)` | Serialize / deserialize |

---

## Links

- **Homepage**: [qkabrine.online](https://qkabrine.online)
- **Author**: [Eric Jagwara](https://ericjagwara.online)
- **Lab**: [Solid Elf Labs](https://solidelf.org)
- **Repository**: [github.com/ericjagwara/qkabrine](https://github.com/ericjagwara/qkabrine)
- **Issues**: [github.com/ericjagwara/qkabrine/issues](https://github.com/ericjagwara/qkabrine/issues)

---

## License

MIT — Copyright (c) 2026 Eric Jagwara, Solid Elf Labs
