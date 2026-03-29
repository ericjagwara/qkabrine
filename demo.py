#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
  ⚛️  Qkabrine AutoML — Sample Task Demo
═══════════════════════════════════════════════════════════════════

This script demonstrates Qkabrine AutoML on the Breast Cancer Wisconsin
dataset: 30 features, 569 samples, binary classification.

What happens:
  1. PCA reduces 30 features → 4 qubits
  2. Bayesian search explores 12 ansätze × 2 encodings × 2 depths
  3. Quantum kernel methods are evaluated alongside variational circuits
  4. The best model is selected, its circuit is pruned, and QASM is exported

Run:
  pip install qkabrine-automl
  python demo.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from qkabrine_automl import QkabrineAutoML

# ── 1. Load data ────────────────────────────────────────────────
X, y = load_breast_cancer(return_X_y=True)
X, y = X[:80], y[:80]  # subset for demo speed
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(f"Dataset : Breast Cancer Wisconsin (subset)")
print(f"Features: {X.shape[1]}  |  Train: {len(X_train)}  |  Test: {len(X_test)}")
print()

# ── 2. Run QkabrineAutoML ────────────────────────────────────────
automl = QkabrineAutoML(
    task="classification",
    n_qubits=4,
    max_layers=2,
    train_steps=10,
    search_strategy="bayesian",      # GP + Expected Improvement
    encodings=("angle", "iqp"),       # search over 2 encodings
    feature_reduction="pca",          # 30 features → 4 qubits
    include_kernels=True,             # also try quantum kernel + SVM
    verbose=True,
)

automl.fit(X_train, y_train)

# ── 3. Results ──────────────────────────────────────────────────
automl.leaderboard()

test_acc = automl.score(X_test, y_test)
print(f"  >>> Test Accuracy: {test_acc:.4f} <<<")
print()

# ── 4. Best model details ──────────────────────────────────────
automl.best_circuit_summary()

# ── 5. QASM export (if variational) ────────────────────────────
if automl._best.get("model_type") != "kernel":
    print("\n  ── OpenQASM 2.0 ──")
    print(automl.export_qasm())
