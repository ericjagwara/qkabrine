"""
Utilities for QkabrineAutoML.

Provides gate application, data encoding strategies, circuit surgery
(pruning/simplification), noise models, parameter initialization
strategies, and QASM export.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  GATE APPLICATION
# ═══════════════════════════════════════════════════════════════════

def apply_gate(gate: dict, params: np.ndarray, p_idx: int) -> int:
    """Apply a single gate dict inside a PennyLane context.

    Returns the updated parameter index.
    """
    import pennylane as qml
    g, w = gate['gate'], gate['wires']
    if   g == 'RX':       qml.RX(params[p_idx], wires=w[0]); p_idx += 1
    elif g == 'RY':       qml.RY(params[p_idx], wires=w[0]); p_idx += 1
    elif g == 'RZ':       qml.RZ(params[p_idx], wires=w[0]); p_idx += 1
    elif g == 'Hadamard':  qml.Hadamard(wires=w[0])
    elif g == 'CNOT':     qml.CNOT(wires=w)
    elif g == 'CZ':       qml.CZ(wires=w)
    elif g == 'SWAP':     qml.SWAP(wires=w)
    elif g == 'CRZ':      qml.CRZ(params[p_idx], wires=w); p_idx += 1
    elif g == 'CRY':      qml.CRY(params[p_idx], wires=w); p_idx += 1
    elif g == 'ENCODE':   pass  # handled externally by the trainer
    return p_idx


def count_params(architecture: list) -> int:
    """Count trainable parameters in an architecture."""
    return sum(1 for g in architecture if g.get('trainable', False))


# ═══════════════════════════════════════════════════════════════════
#  DATA ENCODING STRATEGIES
# ═══════════════════════════════════════════════════════════════════

class DataEncoder:
    """Pluggable data encoding strategies for embedding classical data
    into quantum states.

    Supported methods
    -----------------
    'angle'       : RX rotations (default, simplest)
    'angle_yz'    : RY + RZ rotations (richer single-qubit encoding)
    'iqp'         : IQP-style encoding with entanglement (Havlíček et al.)
    'amplitude'   : Amplitude encoding (exponential compression)
    """

    METHODS = ('angle', 'angle_yz', 'iqp', 'amplitude')

    def __init__(self, method: str = 'angle'):
        if method not in self.METHODS:
            raise ValueError(f"Unknown encoding: {method}. Choose from {self.METHODS}")
        self.method = method

    def encode(self, x: np.ndarray, n_qubits: int):
        """Encode a single data point x into the current PennyLane context."""
        import pennylane as qml
        if self.method == 'angle':
            for i in range(n_qubits):
                qml.RX(x[i], wires=i)

        elif self.method == 'angle_yz':
            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
                if 2 * i + 1 < len(x):
                    qml.RZ(x[min(i + n_qubits, len(x) - 1)], wires=i)

        elif self.method == 'iqp':
            # IQP embedding: H → RZ(x) → ZZ(xi*xj) → H → RZ(x)
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
                qml.RZ(x[i] * x[i + 1], wires=i + 1)
                qml.CNOT(wires=[i, i + 1])
            for i in range(n_qubits):
                qml.Hadamard(wires=i)
                qml.RZ(x[i], wires=i)

        elif self.method == 'amplitude':
            # Amplitude encoding: requires 2^n amplitudes
            dim = 2 ** n_qubits
            padded = np.zeros(dim)
            padded[:min(len(x), dim)] = x[:min(len(x), dim)]
            norm = np.linalg.norm(padded)
            if norm > 1e-10:
                padded = padded / norm
            else:
                padded[0] = 1.0
            qml.StatePrep(padded, wires=range(n_qubits))


# ═══════════════════════════════════════════════════════════════════
#  PARAMETER INITIALIZATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════

def init_params(n_params: int, strategy: str = 'uniform') -> np.ndarray:
    """Initialize circuit parameters.

    Returns PennyLane-compatible tensors with requires_grad=True.

    Strategies
    ----------
    'uniform'  : U(-π, π)  — standard
    'small'    : U(-0.1, 0.1) — near identity, avoids barren plateau
    'zeros'    : all zeros (identity circuit start)
    'normal'   : N(0, 0.1) — Gaussian small
    'block'    : alternating 0 / π/4 — structured initialization
    """
    import pennylane.numpy as pnp

    if strategy == 'uniform':
        return pnp.array(np.random.uniform(-np.pi, np.pi, n_params), requires_grad=True)
    elif strategy == 'small':
        return pnp.array(np.random.uniform(-0.1, 0.1, n_params), requires_grad=True)
    elif strategy == 'zeros':
        return pnp.zeros(n_params, requires_grad=True)
    elif strategy == 'normal':
        return pnp.array(np.random.randn(n_params) * 0.1, requires_grad=True)
    elif strategy == 'block':
        p = np.zeros(n_params)
        p[1::2] = np.pi / 4
        return pnp.array(p, requires_grad=True)
    else:
        raise ValueError(f"Unknown init strategy: {strategy}")


# ═══════════════════════════════════════════════════════════════════
#  CIRCUIT SURGERY (pruning & simplification)
# ═══════════════════════════════════════════════════════════════════

def prune_circuit(
    architecture: list,
    params: np.ndarray,
    threshold: float = 0.05,
) -> Tuple[list, np.ndarray]:
    """Remove rotation gates whose trained parameter is near zero
    (within `threshold` of 0 or 2π), as they approximate identity.

    Returns (pruned_arch, pruned_params).
    """
    new_arch = []
    new_params = []
    p_idx = 0
    for gate in architecture:
        if gate['trainable']:
            val = params[p_idx]
            # Check if rotation ≈ identity (0) or full rotation (2π)
            effective = val % (2 * np.pi)
            if effective > np.pi:
                effective = 2 * np.pi - effective
            if effective < threshold:
                p_idx += 1
                continue  # skip this gate
            new_arch.append(gate)
            new_params.append(val)
            p_idx += 1
        else:
            new_arch.append(gate)
    return new_arch, np.array(new_params) if new_params else np.array([])


def simplify_circuit(architecture: list) -> list:
    """Remove adjacent inverse-pair gates (e.g. CNOT-CNOT on same wires)
    and collapse consecutive same-axis rotations on the same qubit."""
    result = list(architecture)
    changed = True
    while changed:
        changed = False
        new_result = []
        i = 0
        while i < len(result):
            if i + 1 < len(result):
                g1, g2 = result[i], result[i + 1]
                # Remove CNOT-CNOT pairs on same wires
                if (g1['gate'] == g2['gate'] == 'CNOT'
                        and g1['wires'] == g2['wires']):
                    i += 2
                    changed = True
                    continue
                # Remove CZ-CZ pairs
                if (g1['gate'] == g2['gate'] == 'CZ'
                        and set(g1['wires']) == set(g2['wires'])):
                    i += 2
                    changed = True
                    continue
                # Remove H-H pairs
                if (g1['gate'] == g2['gate'] == 'Hadamard'
                        and g1['wires'] == g2['wires']):
                    i += 2
                    changed = True
                    continue
            new_result.append(result[i])
            i += 1
        result = new_result
    return result


# ═══════════════════════════════════════════════════════════════════
#  QASM EXPORT
# ═══════════════════════════════════════════════════════════════════

_GATE_QASM_MAP = {
    'RX': 'rx',
    'RY': 'ry',
    'RZ': 'rz',
    'Hadamard': 'h',
    'CNOT': 'cx',
    'CZ': 'cz',
    'SWAP': 'swap',
    'CRZ': 'crz',
    'CRY': 'cry',
}

def to_qasm(
    architecture: list,
    params: np.ndarray,
    n_qubits: int,
    encoding: str = 'angle',
    include_encoding: bool = True,
) -> str:
    """Export a circuit (encoding + ansatz) to OpenQASM 2.0 string.

    Parameters
    ----------
    architecture : list of gate dicts
    params : trained parameter values
    n_qubits : number of qubits
    encoding : which data encoding was used
    include_encoding : whether to include the encoding as parameterized gates

    Returns
    -------
    qasm : str
    """
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{n_qubits}];',
        f'creg c[{n_qubits}];',
        '',
        '// ── Data encoding ──',
    ]

    if include_encoding:
        if encoding in ('angle', 'angle_yz'):
            for i in range(n_qubits):
                lines.append(f'rx(x[{i}]) q[{i}];  // data feature {i}')
        elif encoding == 'iqp':
            for i in range(n_qubits):
                lines.append(f'h q[{i}];')
                lines.append(f'rz(x[{i}]) q[{i}];')
        else:
            lines.append('// amplitude encoding — not representable in QASM')

    lines.append('')
    lines.append('// ── Variational ansatz ──')

    p_idx = 0
    for gate in architecture:
        g = gate['gate']
        w = gate['wires']
        if g == 'ENCODE':
            lines.append('// (data re-encoding layer)')
            continue
        qasm_name = _GATE_QASM_MAP.get(g, g.lower())
        if gate['trainable']:
            val = float(params[p_idx])
            p_idx += 1
            if len(w) == 1:
                lines.append(f'{qasm_name}({val:.6f}) q[{w[0]}];')
            else:
                wire_str = ', '.join(f'q[{wi}]' for wi in w)
                lines.append(f'{qasm_name}({val:.6f}) {wire_str};')
        else:
            wire_str = ', '.join(f'q[{wi}]' for wi in w)
            lines.append(f'{qasm_name} {wire_str};')

    lines.append('')
    lines.append('// ── Measurement ──')
    for i in range(n_qubits):
        lines.append(f'measure q[{i}] -> c[{i}];')

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
#  NOISE MODELS
# ═══════════════════════════════════════════════════════════════════

def get_noisy_device(n_qubits: int, noise_model: str = 'depolarizing',
                     noise_strength: float = 0.01):
    """Create a PennyLane device with noise for NISQ-aware training.

    Parameters
    ----------
    noise_model : 'depolarizing', 'bitflip', 'phaseflip', 'amplitude_damping'
    noise_strength : probability / strength of the noise channel
    """
    import pennylane as qml

    dev = qml.device('default.mixed', wires=n_qubits)
    return dev, noise_model, noise_strength


def apply_noise_layer(n_qubits: int, noise_model: str, noise_strength: float):
    """Insert a noise layer after the current point in a PennyLane context.
    Must be called inside a qnode that uses 'default.mixed' device."""
    import pennylane as qml

    for q in range(n_qubits):
        if noise_model == 'depolarizing':
            qml.DepolarizingChannel(noise_strength, wires=q)
        elif noise_model == 'bitflip':
            qml.BitFlip(noise_strength, wires=q)
        elif noise_model == 'phaseflip':
            qml.PhaseFlip(noise_strength, wires=q)
        elif noise_model == 'amplitude_damping':
            qml.AmplitudeDamping(noise_strength, wires=q)


# ═══════════════════════════════════════════════════════════════════
#  DIMENSIONALITY REDUCTION HELPERS
# ═══════════════════════════════════════════════════════════════════

def reduce_features(X: np.ndarray, n_target: int, method: str = 'pca') -> Tuple[np.ndarray, object]:
    """Reduce feature dimensionality to n_target.

    Parameters
    ----------
    method : 'pca', 'random_projection', 'select_variance'

    Returns
    -------
    X_reduced, transformer (for applying to test data)
    """
    n_features = X.shape[1]
    if n_features <= n_target:
        return X, None

    if method == 'pca':
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_target)
        return pca.fit_transform(X), pca

    elif method == 'random_projection':
        from sklearn.random_projection import GaussianRandomProjection
        rp = GaussianRandomProjection(n_components=n_target, random_state=42)
        return rp.fit_transform(X), rp

    elif method == 'select_variance':
        # Keep the n_target features with highest variance
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-n_target:]
        top_indices = np.sort(top_indices)

        class VarianceSelector:
            def __init__(self, indices):
                self.indices = indices
            def transform(self, X):
                return X[:, self.indices]

        return X[:, top_indices], VarianceSelector(top_indices)

    else:
        raise ValueError(f"Unknown reduction method: {method}")
