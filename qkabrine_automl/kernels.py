"""
Quantum Kernel Methods for QkabrineAutoML.

Provides quantum kernel estimation (QKE) as an alternative to
variational quantum circuits. Quantum kernels embed data into
Hilbert space and use the inner product of quantum states as
a similarity measure for classical SVMs.

This module implements:
- Fidelity-based quantum kernels
- Trainable quantum kernels with kernel-target alignment
- Multiple embedding circuits for kernel computation

References
----------
- Havlíček et al. (2019). "Supervised learning with quantum-enhanced
  feature spaces." Nature 567.
- Schuld & Killoran (2019). "Quantum machine learning in feature
  Hilbert spaces." PRL 122.
- Huang et al. (2021). "Power of data in quantum machine learning."
  Nature Communications 12.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score


class QuantumKernel:
    """Quantum kernel estimator.

    Computes kernel matrices using quantum circuit fidelity:
        k(x_i, x_j) = |<ψ(x_i)|ψ(x_j)>|²

    Parameters
    ----------
    n_qubits : int
        Number of qubits for the embedding circuit.
    embedding : str
        Embedding circuit type: 'iqp', 'angle', 'hardware_efficient'
    n_layers : int
        Number of embedding layers (for data re-uploading style).
    trainable : bool
        Whether to optimize embedding parameters via kernel-target alignment.
    """

    EMBEDDINGS = ('iqp', 'angle', 'hardware_efficient')

    def __init__(self, n_qubits: int = 4, embedding: str = 'iqp',
                 n_layers: int = 1, trainable: bool = False):
        self.n_qubits = n_qubits
        self.embedding = embedding
        self.n_layers = n_layers
        self.trainable = trainable
        self._train_params = None
        self._qml = None

    def _make_circuit(self):
        """Build the embedding circuit as a PennyLane QNode."""
        import pennylane as qml
        self._qml = qml

        dev = qml.device('default.qubit', wires=self.n_qubits)

        if self.embedding == 'iqp':
            @qml.qnode(dev)
            def kernel_circuit(x1, x2):
                # Encode x1
                for _ in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.Hadamard(wires=i)
                        qml.RZ(x1[i], wires=i)
                    for i in range(self.n_qubits - 1):
                        qml.CNOT(wires=[i, i + 1])
                        qml.RZ(x1[i] * x1[i + 1], wires=i + 1)
                        qml.CNOT(wires=[i, i + 1])
                # Adjoint of encoding x2
                for _ in range(self.n_layers):
                    for i in range(self.n_qubits - 2, -1, -1):
                        qml.CNOT(wires=[i, i + 1])
                        qml.RZ(-x2[i] * x2[i + 1], wires=i + 1)
                        qml.CNOT(wires=[i, i + 1])
                    for i in range(self.n_qubits - 1, -1, -1):
                        qml.RZ(-x2[i], wires=i)
                        qml.Hadamard(wires=i)
                return qml.probs(wires=range(self.n_qubits))

        elif self.embedding == 'angle':
            @qml.qnode(dev)
            def kernel_circuit(x1, x2):
                for _ in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.RX(x1[i], wires=i)
                        qml.RY(x1[i], wires=i)
                for _ in range(self.n_layers):
                    for i in range(self.n_qubits - 1, -1, -1):
                        qml.RY(-x2[i], wires=i)
                        qml.RX(-x2[i], wires=i)
                return qml.probs(wires=range(self.n_qubits))

        elif self.embedding == 'hardware_efficient':
            @qml.qnode(dev)
            def kernel_circuit(x1, x2):
                for _ in range(self.n_layers):
                    for i in range(self.n_qubits):
                        qml.RY(x1[i], wires=i)
                    for i in range(0, self.n_qubits - 1, 2):
                        qml.CZ(wires=[i, i + 1])
                    for i in range(self.n_qubits):
                        qml.RZ(x1[i], wires=i)
                # Adjoint
                for _ in range(self.n_layers):
                    for i in range(self.n_qubits - 1, -1, -1):
                        qml.RZ(-x2[i], wires=i)
                    for i in range(0, self.n_qubits - 1, 2):
                        qml.CZ(wires=[i, i + 1])
                    for i in range(self.n_qubits - 1, -1, -1):
                        qml.RY(-x2[i], wires=i)
                return qml.probs(wires=range(self.n_qubits))
        else:
            raise ValueError(f"Unknown embedding: {self.embedding}")

        return kernel_circuit

    def compute_kernel_matrix(self, X1: np.ndarray,
                               X2: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the quantum kernel matrix K[i,j] = k(X1[i], X2[j]).

        If X2 is None, computes K[i,j] = k(X1[i], X1[j]).
        """
        circuit = self._make_circuit()
        symmetric = X2 is None
        if symmetric:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))

        for i in range(n1):
            start_j = i if symmetric else 0
            for j in range(start_j, n2):
                probs = circuit(X1[i], X2[j])
                fidelity = float(probs[0])  # probability of |00...0>
                K[i, j] = fidelity
                if symmetric and i != j:
                    K[j, i] = fidelity

        return K

    def kernel_target_alignment(self, K: np.ndarray, y: np.ndarray) -> float:
        """Compute kernel-target alignment (KTA).

        KTA = y^T K y / (sqrt(Tr(K^2)) * N)

        Higher KTA indicates the kernel better separates the classes.
        """
        y = np.array(y, dtype=float)
        N = len(y)
        # Normalize labels to ±1
        if len(np.unique(y)) == 2:
            y_binary = 2 * (y - y.min()) / (y.max() - y.min()) - 1
        else:
            y_binary = y - y.mean()
            y_binary /= np.std(y_binary) + 1e-10

        yyt = np.outer(y_binary, y_binary)
        kta = np.sum(K * yyt) / (np.sqrt(np.sum(K ** 2)) * N)
        return float(kta)


class QuantumKernelClassifier:
    """Quantum kernel + classical SVM classifier.

    Wraps QuantumKernel with sklearn's SVC for end-to-end classification.
    """

    def __init__(self, n_qubits: int = 4, embedding: str = 'iqp',
                 n_layers: int = 1, C: float = 1.0):
        self.kernel = QuantumKernel(n_qubits, embedding, n_layers)
        self.C = C
        self._svm = None
        self._X_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.array(X)[:, :self.kernel.n_qubits]
        self._X_train = X
        K_train = self.kernel.compute_kernel_matrix(X)
        self._svm = SVC(kernel='precomputed', C=self.C)
        self._svm.fit(K_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)[:, :self.kernel.n_qubits]
        K_test = self.kernel.compute_kernel_matrix(X, self._X_train)
        return self._svm.predict(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return accuracy_score(y, preds)


class QuantumKernelRegressor:
    """Quantum kernel + classical SVR for regression."""

    def __init__(self, n_qubits: int = 4, embedding: str = 'iqp',
                 n_layers: int = 1, C: float = 1.0):
        self.kernel = QuantumKernel(n_qubits, embedding, n_layers)
        self.C = C
        self._svr = None
        self._X_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.array(X)[:, :self.kernel.n_qubits]
        self._X_train = X
        K_train = self.kernel.compute_kernel_matrix(X)
        self._svr = SVR(kernel='precomputed', C=self.C)
        self._svr.fit(K_train, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)[:, :self.kernel.n_qubits]
        K_test = self.kernel.compute_kernel_matrix(X, self._X_train)
        return self._svr.predict(K_test)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return r2_score(y, preds)
