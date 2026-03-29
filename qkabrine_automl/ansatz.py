"""
Quantum circuit ansatz library.

Provides a rich library of parameterized quantum circuit templates (ansätze)
for variational quantum machine learning, along with circuit analysis metrics
(expressibility, entangling capability) used to guide architecture search.

References
----------
- Sim, Johnson & Aspuru-Guzik (2019). "Expressibility and Entangling
  Capability of Parameterized Quantum Circuits." Adv. Quantum Technol.
- Schuld, Sweke & Meyer (2021). "The effect of data encoding on the
  expressive power of variational quantum machine learning models."
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import List, Dict, Callable, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  Gate-dict format used throughout the package
# ═══════════════════════════════════════════════════════════════════
#  { 'gate': str,  'wires': list[int],  'trainable': bool }
#  Trainable gates consume one parameter from the flat param vector.


# ───────────────────────────────────────────────────────────────────
#  1. STRONGLY ENTANGLING  (RZ-RY-RZ + ring CNOT)
# ───────────────────────────────────────────────────────────────────
def strongly_entangling(n_qubits: int, n_layers: int) -> list:
    """Strongly entangling layers (PennyLane-style)."""
    gates = []
    for layer in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
        shift = layer % n_qubits
        for q in range(n_qubits):
            target = (q + shift + 1) % n_qubits
            if target != q:
                gates.append({'gate': 'CNOT', 'wires': [q, target], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
#  2. HARDWARE EFFICIENT  (RY + brickwork CZ)
# ───────────────────────────────────────────────────────────────────
def hardware_efficient(n_qubits: int, n_layers: int) -> list:
    gates = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
        for q in range(0, n_qubits - 1, 2):
            gates.append({'gate': 'CZ', 'wires': [q, q + 1], 'trainable': False})
        for q in range(1, n_qubits - 1, 2):
            gates.append({'gate': 'CZ', 'wires': [q, q + 1], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
#  3. ALTERNATING RX/RY  (alternating rotation + CNOT ladder)
# ───────────────────────────────────────────────────────────────────
def alternating_rx_ry(n_qubits: int, n_layers: int) -> list:
    gates = []
    for l in range(n_layers):
        rot = 'RX' if l % 2 == 0 else 'RY'
        for q in range(n_qubits):
            gates.append({'gate': rot, 'wires': [q], 'trainable': True})
        for q in range(n_qubits - 1):
            gates.append({'gate': 'CNOT', 'wires': [q, q + 1], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
#  4. SHALLOW RX  (minimal depth)
# ───────────────────────────────────────────────────────────────────
def shallow_rx(n_qubits: int, n_layers: int) -> list:
    gates = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'RX', 'wires': [q], 'trainable': True})
        for q in range(n_qubits - 1):
            gates.append({'gate': 'CNOT', 'wires': [q, q + 1], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
#  5. FULL ROTATION  (RX+RY+RZ + ring CNOT)
# ───────────────────────────────────────────────────────────────────
def full_rotation(n_qubits: int, n_layers: int) -> list:
    gates = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            for g in ['RX', 'RY', 'RZ']:
                gates.append({'gate': g, 'wires': [q], 'trainable': True})
        for q in range(n_qubits):
            gates.append({'gate': 'CNOT', 'wires': [q, (q + 1) % n_qubits], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
#  6. HADAMARD ENTANGLING  (H + RZ + CNOT ladder)
# ───────────────────────────────────────────────────────────────────
def hadamard_entangling(n_qubits: int, n_layers: int) -> list:
    gates = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'Hadamard', 'wires': [q], 'trainable': False})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
        for q in range(n_qubits - 1):
            gates.append({'gate': 'CNOT', 'wires': [q, q + 1], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
#  7. SIMPLIFIED TWO-DESIGN  (based on Cerezo et al. 2021)
# ───────────────────────────────────────────────────────────────────
def simplified_two_design(n_qubits: int, n_layers: int) -> list:
    """Approximate 2-design circuit: initial RY layer then alternating
    CZ-brickwork + RY-RZ per qubit."""
    gates = []
    # initial uniform superposition via RY
    for q in range(n_qubits):
        gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
    for layer in range(n_layers):
        # entangling block — even/odd brickwork
        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            gates.append({'gate': 'CZ', 'wires': [q, q + 1], 'trainable': False})
        for q in range(n_qubits):
            gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
    return gates


# ───────────────────────────────────────────────────────────────────
#  8. DATA RE-UPLOADING  (interleaved encode + variational)
# ───────────────────────────────────────────────────────────────────
def data_reuploading(n_qubits: int, n_layers: int) -> list:
    """Data re-uploading circuit (Pérez-Salinas et al. 2020).
    Marks where data should be re-encoded between variational layers
    via a special 'ENCODE' pseudo-gate."""
    gates = []
    for _ in range(n_layers):
        # data encoding marker (handled specially by the trainer)
        for q in range(n_qubits):
            gates.append({'gate': 'ENCODE', 'wires': [q], 'trainable': False})
        # variational block
        for q in range(n_qubits):
            gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
        for q in range(n_qubits):
            gates.append({'gate': 'CNOT', 'wires': [q, (q + 1) % n_qubits], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
#  9. RING OF CNOTS  (dense rotation + ring entanglement)
# ───────────────────────────────────────────────────────────────────
def ring_of_cnots(n_qubits: int, n_layers: int) -> list:
    gates = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'RX', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
        for q in range(n_qubits):
            gates.append({'gate': 'CNOT', 'wires': [q, (q + 1) % n_qubits], 'trainable': False})
        for q in range(n_qubits):
            gates.append({'gate': 'RX', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
    return gates


# ───────────────────────────────────────────────────────────────────
# 10. ALL-TO-ALL ENTANGLING  (all-pairs CZ connectivity)
# ───────────────────────────────────────────────────────────────────
def all_to_all(n_qubits: int, n_layers: int) -> list:
    gates = []
    pairs = list(itertools.combinations(range(n_qubits), 2))
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
        for i, j in pairs:
            gates.append({'gate': 'CZ', 'wires': [i, j], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
# 11. CRISS-CROSS  (butterfly-pattern CNOT)
# ───────────────────────────────────────────────────────────────────
def criss_cross(n_qubits: int, n_layers: int) -> list:
    """Butterfly/criss-cross entanglement pattern — each qubit
    connects to its mirror partner, maximising entanglement spread."""
    gates = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'RX', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
        for q in range(n_qubits // 2):
            partner = n_qubits - 1 - q
            if q != partner:
                gates.append({'gate': 'CNOT', 'wires': [q, partner], 'trainable': False})
        for q in range(n_qubits - 1):
            gates.append({'gate': 'CNOT', 'wires': [q, q + 1], 'trainable': False})
    return gates


# ───────────────────────────────────────────────────────────────────
# 12. CASCADING ENTANGLER  (progressive depth scaling)
# ───────────────────────────────────────────────────────────────────
def cascading(n_qubits: int, n_layers: int) -> list:
    """Progressive cascade — layer k entangles qubit pairs separated
    by distance 2^k, inspired by the QFT structure."""
    gates = []
    for layer in range(n_layers):
        for q in range(n_qubits):
            gates.append({'gate': 'RY', 'wires': [q], 'trainable': True})
            gates.append({'gate': 'RZ', 'wires': [q], 'trainable': True})
        step = min(2 ** layer, n_qubits - 1)
        for q in range(n_qubits):
            target = (q + step) % n_qubits
            if target != q:
                gates.append({'gate': 'CNOT', 'wires': [q, target], 'trainable': False})
    return gates


# ═══════════════════════════════════════════════════════════════════
#  REGISTRY
# ═══════════════════════════════════════════════════════════════════
REGISTRY: Dict[str, Callable] = {
    'strongly_entangling':  strongly_entangling,
    'hardware_efficient':   hardware_efficient,
    'alternating_rx_ry':    alternating_rx_ry,
    'shallow_rx':           shallow_rx,
    'full_rotation':        full_rotation,
    'hadamard_entangling':  hadamard_entangling,
    'simplified_two_design': simplified_two_design,
    'data_reuploading':     data_reuploading,
    'ring_of_cnots':        ring_of_cnots,
    'all_to_all':           all_to_all,
    'criss_cross':          criss_cross,
    'cascading':            cascading,
}


# ═══════════════════════════════════════════════════════════════════
#  CIRCUIT METRICS  (Expressibility & Entangling Capability)
# ═══════════════════════════════════════════════════════════════════

def compute_expressibility(
    arch_fn: Callable,
    n_qubits: int,
    n_layers: int = 1,
    n_samples: int = 200,
    n_bins: int = 75,
) -> float:
    """Compute expressibility of a circuit template via KL divergence
    between its fidelity distribution and the Haar-random distribution.

    Lower KL divergence → more expressible (can reach more of Hilbert space).

    Based on Sim, Johnson & Aspuru-Guzik (2019).
    """
    import pennylane as qml
    from .utils import count_params, apply_gate

    arch = arch_fn(n_qubits, n_layers)
    # filter out ENCODE pseudo-gates for metric computation
    arch_filtered = [g for g in arch if g['gate'] != 'ENCODE']
    n_params = count_params(arch_filtered)

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        p = 0
        for gate in arch_filtered:
            p = apply_gate(gate, params, p)
        return qml.state()

    # Sample fidelities between pairs of random parameter sets
    fidelities = []
    for _ in range(n_samples):
        p1 = np.random.uniform(0, 2 * np.pi, n_params)
        p2 = np.random.uniform(0, 2 * np.pi, n_params)
        s1 = circuit(p1)
        s2 = circuit(p2)
        fid = np.abs(np.dot(np.conj(s1), s2)) ** 2
        fidelities.append(fid)

    # Haar-random distribution for N-qubit system: P(F) = (2^n - 1)(1-F)^(2^n - 2)
    dim = 2 ** n_qubits
    bins = np.linspace(0, 1, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    haar_pdf = (dim - 1) * (1 - centres) ** (dim - 2)
    haar_pdf /= haar_pdf.sum()
    haar_pdf = np.clip(haar_pdf, 1e-10, None)

    circuit_hist, _ = np.histogram(fidelities, bins=bins, density=False)
    circuit_pdf = circuit_hist / circuit_hist.sum()
    circuit_pdf = np.clip(circuit_pdf, 1e-10, None)

    kl = float(np.sum(circuit_pdf * np.log(circuit_pdf / haar_pdf)))
    return kl


def compute_entangling_capability(
    arch_fn: Callable,
    n_qubits: int,
    n_layers: int = 1,
    n_samples: int = 200,
) -> float:
    """Compute entangling capability via the Meyer-Wallach measure Q.

    Q = 0 → product states (no entanglement).
    Q = 1 → maximally entangled (e.g. GHZ states).

    Based on Meyer & Wallach (2002).
    """
    if n_qubits < 2:
        return 0.0

    import pennylane as qml
    from .utils import count_params, apply_gate

    arch = arch_fn(n_qubits, n_layers)
    arch_filtered = [g for g in arch if g['gate'] != 'ENCODE']
    n_params = count_params(arch_filtered)

    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        p = 0
        for gate in arch_filtered:
            p = apply_gate(gate, params, p)
        return qml.state()

    q_values = []
    for _ in range(n_samples):
        params = np.random.uniform(0, 2 * np.pi, n_params)
        state = circuit(params)
        q_values.append(_meyer_wallach(state, n_qubits))

    return float(np.mean(q_values))


def _meyer_wallach(state: np.ndarray, n_qubits: int) -> float:
    """Meyer-Wallach entanglement measure for a pure state vector."""
    dim = 2 ** n_qubits
    state = state.reshape(dim)
    mw = 0.0
    for k in range(n_qubits):
        rho_k = _partial_trace_single(state, k, n_qubits)
        purity = np.real(np.trace(rho_k @ rho_k))
        mw += 1 - purity
    return 2.0 * mw / n_qubits


def _partial_trace_single(state: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Trace out all qubits except `qubit` to get single-qubit density matrix."""
    dim = 2 ** n_qubits
    state = state.reshape([2] * n_qubits)

    # Build full density matrix and partial-trace via einsum
    rho_full = np.outer(state.ravel(), np.conj(state.ravel()))
    rho_full = rho_full.reshape([2] * (2 * n_qubits))

    # Contract bra index i with ket index i for all qubits except `qubit`
    input_indices = list(range(2 * n_qubits))
    for i in range(n_qubits):
        if i != qubit:
            input_indices[i + n_qubits] = input_indices[i]

    # Build einsum string
    chars = 'abcdefghijklmnop'
    label_map = {}
    next_label = 0
    input_labels = []
    for idx in input_indices:
        if idx not in label_map:
            label_map[idx] = chars[next_label]
            next_label += 1
        input_labels.append(label_map[idx])
    output_labels = [label_map[qubit], label_map[qubit + n_qubits]]
    einsum_str = ''.join(input_labels) + '->' + ''.join(output_labels)
    return np.einsum(einsum_str, rho_full)


def rank_ansatze(
    n_qubits: int,
    n_layers: int = 1,
    n_samples: int = 100,
    registry: Optional[Dict[str, Callable]] = None,
) -> list:
    """Rank all registered ansätze by expressibility and entangling capability.

    Returns a list of dicts sorted by expressibility (ascending = more expressible).
    """
    reg = registry or REGISTRY
    results = []
    for name, fn in reg.items():
        if name == 'data_reuploading':
            continue  # skip — needs special handling
        try:
            expr = compute_expressibility(fn, n_qubits, n_layers, n_samples)
            ent = compute_entangling_capability(fn, n_qubits, n_layers, n_samples)
            results.append({
                'name': name,
                'expressibility': expr,
                'entangling_capability': ent,
            })
        except Exception:
            continue
    results.sort(key=lambda r: r['expressibility'])
    return results
