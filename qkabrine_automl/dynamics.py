"""
Training dynamics analysis for variational quantum circuits.

This module provides tools to predict, monitor, and fix common
training pathologies in variational quantum machine learning:

1. **DataQuantumFisherMetric (DQFIM)** — Estimates the effective
   dimension and trainability of a parameterized circuit *before*
   full training, using the quantum Fisher information matrix.
   Circuits with higher effective rank / trainability score are
   more likely to converge successfully.

2. **BarrenPlateauMonitor** — Tracks gradient variance during
   training and detects the onset of barren plateaus (exponentially
   vanishing gradients). Optionally recommends circuit surgery:
   layer removal, re-initialisation, or switching to a local cost.

3. **QuantumNaturalGradient** — A geometry-aware optimiser that
   preconditions the gradient with the (approximate) Fubini-Study
   metric tensor. Follows the steepest descent direction on the
   quantum state manifold rather than in raw parameter space,
   often converging faster than Adam on variational circuits.

References
----------
- Abbas et al. (2021). "The power of quantum neural networks."
  Nature Computational Science 1.
- McClean et al. (2018). "Barren plateaus in quantum neural network
  training landscapes." Nature Communications 9.
- Stokes et al. (2020). "Quantum Natural Gradient."
  Quantum 4, 269.
- Meyer et al. (2021). "Fisher information in noisy intermediate-
  scale quantum applications." Quantum 5, 539.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Dict, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════
#  DATA CONTAINER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DQFIMetrics:
    """Container for Data Quantum Fisher Information diagnostics.

    Attributes
    ----------
    fisher_matrix : np.ndarray
        The (n_params × n_params) data quantum Fisher information
        matrix, averaged over sampled data points.
    eigenvalues : np.ndarray
        Eigenvalues of the Fisher matrix (sorted descending).
    rank : int
        Effective rank — number of eigenvalues above a relative
        threshold (1e-6 × max eigenvalue). A low rank relative to
        n_params signals redundant parameters.
    condition_number : float
        Ratio of largest to smallest *nonzero* eigenvalue. Values
        >> 1 indicate ill-conditioned optimisation landscape.
    trainability_score : float
        Heuristic in [0, 1] combining effective dimension and
        spectral health. Higher is better.
    effective_dimension : float
        Normalised effective rank: rank / n_params.
    spectral_gap : float
        Ratio of the second-largest to the largest eigenvalue.
        Near 1 → flat spectrum → good. Near 0 → dominant direction.
    """
    fisher_matrix: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    trainability_score: float
    effective_dimension: float = 0.0
    spectral_gap: float = 0.0


# ═══════════════════════════════════════════════════════════════════
#  DATA QUANTUM FISHER INFORMATION METRIC
# ═══════════════════════════════════════════════════════════════════

class DataQuantumFisherMetric:
    """Predict circuit trainability via the quantum Fisher information.

    The DQFIM F_{ij} measures how much the circuit output changes
    when parameters θ_i and θ_j are perturbed, averaged over the
    data distribution. It serves as a *pre-training* diagnostic:

    - High effective rank → the circuit can represent many independent
      directions in function space → likely trainable.
    - Low effective rank → many parameters are redundant → likely
      barren plateau or over-parameterisation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuits being analysed.
    n_samples : int
        Number of random parameter sets to sample for Monte Carlo
        estimation of the Fisher matrix.
    seed : int or None
        Random seed for reproducibility.
    epsilon : float
        Finite-difference step size for gradient estimation.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        n_samples: int = 100,
        seed: int = None,
        epsilon: float = 1e-4,
    ):
        self.n_qubits = n_qubits
        self.n_samples = n_samples
        self.seed = seed
        self.epsilon = epsilon
        self._rng = np.random.RandomState(seed)

    def compute_dqfim(
        self,
        circuit_fn: Callable,
        X: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Compute the data quantum Fisher information matrix.

        Estimates F_{ij} = E_x [ ∂f/∂θ_i · ∂f/∂θ_j ] via
        finite differences averaged over data points.

        Parameters
        ----------
        circuit_fn : callable(x, params) → float or array
            The parameterised circuit or model to analyse.
        X : np.ndarray of shape (n_data, n_features)
            Data points to average over.
        params : np.ndarray of shape (n_params,)
            Parameter values at which to compute the DQFIM.

        Returns
        -------
        F : np.ndarray of shape (n_params, n_params)
            Symmetric positive semi-definite Fisher matrix.
        """
        params = np.asarray(params, dtype=float)
        n_params = len(params)
        eps = self.epsilon

        # Sub-sample data if large
        n_data = min(len(X), self.n_samples)
        idx = self._rng.choice(len(X), n_data, replace=False)
        X_sub = X[idx]

        # Compute Jacobian via central finite differences
        # J[d, i] = ∂f(x_d)/∂θ_i
        jacobian = np.zeros((n_data, n_params))
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            for d in range(n_data):
                x = X_sub[d]
                # Pad data if shorter than params (common in quantum ML
                # where n_features < n_params)
                if len(x) < n_params:
                    x_padded = np.zeros(n_params)
                    x_padded[:len(x)] = x
                    x = x_padded
                f_plus = np.atleast_1d(circuit_fn(x, params_plus))
                f_minus = np.atleast_1d(circuit_fn(x, params_minus))
                # Use the mean across output dimensions if vector-valued
                jacobian[d, i] = np.mean(f_plus - f_minus) / (2 * eps)

        # F = (1/n_data) J^T J — the empirical Fisher
        F = jacobian.T @ jacobian / n_data

        # Symmetrise (should already be, but floating-point safety)
        F = (F + F.T) / 2
        return F

    def predict_generalization(
        self,
        circuit_fn: Callable,
        X: np.ndarray,
        n_params: int,
        n_samples: int = None,
    ) -> DQFIMetrics:
        """Predict trainability and generalisation capacity.

        Samples multiple random parameter vectors, computes the
        DQFIM for each, and returns aggregated diagnostics.

        Parameters
        ----------
        circuit_fn : callable(x, params) → float or array
        X : np.ndarray of shape (n_data, n_features)
        n_params : int
            Number of trainable parameters.
        n_samples : int or None
            How many random parameter sets to average over.
            Defaults to self.n_samples.

        Returns
        -------
        DQFIMetrics
        """
        n_samples = n_samples or self.n_samples

        # Average the Fisher matrix over several random param inits
        F_total = np.zeros((n_params, n_params))
        for _ in range(min(n_samples, 20)):  # cap for speed
            params = self._rng.uniform(-np.pi, np.pi, n_params)
            F = self.compute_dqfim(circuit_fn, X, params)
            F_total += F
        F_avg = F_total / min(n_samples, 20)
        F_avg = (F_avg + F_avg.T) / 2

        return self._analyse_fisher(F_avg, n_params)

    def _analyse_fisher(self, F: np.ndarray, n_params: int) -> DQFIMetrics:
        """Extract diagnostics from a Fisher matrix."""
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(F)))[::-1]
        eigenvalues = np.clip(eigenvalues, 0, None)  # PSD enforcement

        # Effective rank: eigenvalues > relative threshold
        max_eig = eigenvalues[0] if eigenvalues[0] > 0 else 1e-30
        threshold = 1e-6 * max_eig
        rank = int(np.sum(eigenvalues > threshold))

        # Condition number: ratio of largest to smallest nonzero
        nonzero = eigenvalues[eigenvalues > threshold]
        if len(nonzero) >= 2:
            condition_number = float(nonzero[0] / nonzero[-1])
        else:
            condition_number = 1.0
        condition_number = max(condition_number, 1.0)

        # Effective dimension (normalised)
        effective_dimension = rank / max(n_params, 1)

        # Spectral gap
        if len(eigenvalues) >= 2 and eigenvalues[0] > 0:
            spectral_gap = float(eigenvalues[1] / eigenvalues[0])
        else:
            spectral_gap = 0.0

        # Trainability score: heuristic combining multiple signals
        # - effective dimension: want high (→ many useful directions)
        # - condition health: want low condition number (→ balanced)
        # - spectral gap: want high (→ no single dominant direction)
        cond_health = 1.0 / (1.0 + np.log1p(condition_number - 1) / 10)
        trainability_score = float(np.clip(
            0.5 * effective_dimension + 0.3 * spectral_gap + 0.2 * cond_health,
            0.0, 1.0,
        ))

        return DQFIMetrics(
            fisher_matrix=F,
            eigenvalues=eigenvalues,
            rank=rank,
            condition_number=condition_number,
            trainability_score=trainability_score,
            effective_dimension=effective_dimension,
            spectral_gap=spectral_gap,
        )


# ═══════════════════════════════════════════════════════════════════
#  BARREN PLATEAU MONITOR
# ═══════════════════════════════════════════════════════════════════

class BarrenPlateauMonitor:
    """Detect and respond to barren plateaus during training.

    Tracks gradient variance across training steps. If variance
    drops below a threshold for several consecutive steps, declares
    a barren plateau and (optionally) recommends circuit surgery.

    The monitor implements the diagnostic from McClean et al. (2018):
    for random circuits with global cost functions, gradient variance
    vanishes as O(2^{-n}) with qubit count n.

    Parameters
    ----------
    threshold : float
        Gradient variance below this value triggers plateau detection.
    window : int
        Number of consecutive low-variance steps before declaring
        a plateau.
    auto_surgery : bool
        If True, enables `.get_surgery_recommendation()`.
    """

    def __init__(
        self,
        threshold: float = 1e-7,
        window: int = 3,
        auto_surgery: bool = False,
    ):
        self.threshold = threshold
        self.window = window
        self.auto_surgery = auto_surgery

        self.gradient_history: List[Dict] = []
        self.variance_history: List[float] = []
        self.plateau_detected: bool = False
        self._consecutive_low: int = 0
        self._layer_sizes: List[int] = []

    def update(
        self,
        gradients: np.ndarray,
        layer_param_counts: List[int] = None,
    ):
        """Record gradient snapshot from one training step.

        Parameters
        ----------
        gradients : np.ndarray
            Flat gradient vector ∂L/∂θ.
        layer_param_counts : list of int, optional
            Number of parameters per circuit layer (used for
            per-layer variance analysis and surgery decisions).
        """
        gradients = np.asarray(gradients, dtype=float)
        variance = float(np.var(gradients))
        mean_abs = float(np.mean(np.abs(gradients)))
        max_abs = float(np.max(np.abs(gradients)))

        # Per-layer variance (if layer structure is known)
        layer_variances = []
        if layer_param_counts:
            self._layer_sizes = layer_param_counts
            offset = 0
            for count in layer_param_counts:
                end = min(offset + count, len(gradients))
                layer_grad = gradients[offset:end]
                layer_variances.append(float(np.var(layer_grad)))
                offset = end

        snapshot = {
            'variance': variance,
            'mean_abs': mean_abs,
            'max_abs': max_abs,
            'layer_variances': layer_variances,
            'n_params': len(gradients),
        }
        self.gradient_history.append(snapshot)
        self.variance_history.append(variance)

        # Plateau detection
        if variance < self.threshold:
            self._consecutive_low += 1
        else:
            self._consecutive_low = 0

        if self._consecutive_low >= self.window:
            self.plateau_detected = True

    def should_trigger_surgery(self) -> bool:
        """Check if circuit surgery should be applied.

        Returns True if a barren plateau is detected and
        auto_surgery is enabled.
        """
        return self.plateau_detected and self.auto_surgery

    def get_surgery_recommendation(self) -> Dict:
        """Get specific circuit surgery recommendations.

        Analyses the gradient history to determine which layers
        are most affected and suggests targeted fixes.

        Returns
        -------
        dict with keys:
            'remove_layers' : list of int
                Layer indices whose gradients are flattest.
            'reinitialize_strategy' : str
                Suggested parameter re-initialisation strategy.
            'suggested_max_layers' : int
                Recommended maximum circuit depth.
            'reason' : str
                Human-readable explanation.
        """
        if not self.gradient_history:
            return {
                'remove_layers': [],
                'reinitialize_strategy': 'small',
                'suggested_max_layers': 1,
                'reason': 'No gradient data available.',
            }

        # Identify layers with vanishing gradients
        last_snapshot = self.gradient_history[-1]
        dead_layers = []
        if last_snapshot['layer_variances']:
            for i, lv in enumerate(last_snapshot['layer_variances']):
                if lv < self.threshold:
                    dead_layers.append(i)

        # If no per-layer data, recommend removing deeper layers
        if not dead_layers and self._layer_sizes:
            n_layers = len(self._layer_sizes)
            # Remove the deeper half
            dead_layers = list(range(n_layers // 2, n_layers))

        # Choose reinit strategy based on severity
        recent_vars = self.variance_history[-min(5, len(self.variance_history)):]
        avg_recent_var = np.mean(recent_vars)
        if avg_recent_var < self.threshold * 0.01:
            reinit = 'zeros'  # Near-identity restart
            reason = 'Severe barren plateau — gradients near machine epsilon.'
        elif avg_recent_var < self.threshold:
            reinit = 'small'  # Small random perturbation
            reason = 'Moderate barren plateau — gradient variance below threshold.'
        else:
            reinit = 'normal'
            reason = 'Mild gradient issues — variance declining.'

        suggested_depth = max(1, len(self._layer_sizes) - len(dead_layers))

        return {
            'remove_layers': dead_layers,
            'reinitialize_strategy': reinit,
            'suggested_max_layers': suggested_depth,
            'reason': reason,
        }

    def summary(self) -> str:
        """Human-readable summary of gradient health."""
        n = len(self.variance_history)
        if n == 0:
            return 'No training data recorded.'

        lines = [
            f'  Steps recorded   : {n}',
            f'  Latest variance  : {self.variance_history[-1]:.2e}',
            f'  Peak variance    : {max(self.variance_history):.2e}',
            f'  Plateau detected : {"YES" if self.plateau_detected else "no"}',
        ]
        if self.plateau_detected:
            lines.append(
                f'  Consecutive low  : {self._consecutive_low} steps '
                f'(threshold={self.threshold:.1e})')
        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
#  QUANTUM NATURAL GRADIENT OPTIMISER
# ═══════════════════════════════════════════════════════════════════

class QuantumNaturalGradient:
    """Geometry-aware optimiser for variational quantum circuits.

    Instead of updating θ ← θ − η·∇L, the quantum natural gradient
    computes the update in the *natural* coordinate system defined
    by the Fubini-Study metric tensor g:

        θ ← θ − η · g⁻¹ · ∇L

    This follows the steepest descent direction on the manifold of
    quantum states rather than in raw parameter space, often
    converging in fewer iterations than Adam.

    The metric tensor is estimated from the circuit function via
    finite differences (no PennyLane dependency required — works
    with any callable circuit).

    Parameters
    ----------
    stepsize : float
        Learning rate η.
    damping : float
        Tikhonov regularisation λ added to the diagonal of g
        before inversion: g⁻¹ → (g + λI)⁻¹. Prevents blow-up
        when g is near-singular.
    adaptive_damping : bool
        If True, automatically adjust λ based on the condition
        number of g at each step.
    epsilon : float
        Finite-difference step size for metric tensor estimation.

    References
    ----------
    Stokes et al. (2020). "Quantum Natural Gradient." Quantum 4, 269.
    """

    def __init__(
        self,
        stepsize: float = 0.01,
        damping: float = 1e-3,
        adaptive_damping: bool = False,
        epsilon: float = 1e-4,
    ):
        self.stepsize = stepsize
        self.damping = damping
        self.adaptive_damping = adaptive_damping
        self.epsilon = epsilon
        self._step_count = 0

    def step(
        self,
        params: np.ndarray,
        grad: np.ndarray,
        circuit_fn: Callable,
        X_batch: np.ndarray,
    ) -> np.ndarray:
        """Perform one natural gradient update.

        Parameters
        ----------
        params : np.ndarray of shape (n_params,)
            Current parameters.
        grad : np.ndarray of shape (n_params,)
            Euclidean gradient ∂L/∂θ at current params.
        circuit_fn : callable(x, params) → float or array
            The circuit function (used to estimate the metric).
        X_batch : np.ndarray of shape (batch_size, n_features)
            Data batch for metric estimation.

        Returns
        -------
        new_params : np.ndarray of shape (n_params,)
        """
        params = np.asarray(params, dtype=float)
        grad = np.asarray(grad, dtype=float)
        n_params = len(params)

        # Estimate the Fubini-Study metric tensor
        g = self._estimate_metric(circuit_fn, X_batch, params)

        # Adaptive damping
        damping = self.damping
        if self.adaptive_damping:
            eigvals = np.linalg.eigvalsh(g)
            max_eig = max(abs(eigvals.max()), 1e-30)
            min_eig = max(abs(eigvals[eigvals > 0].min()), 1e-30) if np.any(eigvals > 0) else 1e-30
            cond = max_eig / min_eig
            # Increase damping for ill-conditioned metric
            damping = self.damping * max(1.0, np.log1p(cond) / 10)

        # Regularised inverse: (g + λI)⁻¹
        g_reg = g + damping * np.eye(n_params)

        try:
            g_inv = np.linalg.solve(g_reg, np.eye(n_params))
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse
            g_inv = np.linalg.pinv(g_reg)

        # Natural gradient: g⁻¹ · ∇L
        natural_grad = g_inv @ grad

        new_params = params - self.stepsize * natural_grad
        self._step_count += 1

        return new_params

    def _estimate_metric(
        self,
        circuit_fn: Callable,
        X: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:
        """Estimate the Fubini-Study metric tensor via finite differences.

        g_{ij} ≈ E_x[ ∂f/∂θ_i · ∂f/∂θ_j ] − E_x[∂f/∂θ_i] · E_x[∂f/∂θ_j]

        This is the covariance of the circuit output Jacobian over data,
        which approximates the quantum geometric tensor for parameterized
        circuits.
        """
        n_params = len(params)
        eps = self.epsilon
        n_data = len(X)

        # Compute Jacobian J[d, i] = ∂f(x_d)/∂θ_i
        jacobian = np.zeros((n_data, n_params))
        for i in range(n_params):
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            for d in range(n_data):
                x = X[d]
                if len(x) < n_params:
                    x_padded = np.zeros(n_params)
                    x_padded[:len(x)] = x
                    x = x_padded
                f_plus = np.atleast_1d(circuit_fn(x, p_plus))
                f_minus = np.atleast_1d(circuit_fn(x, p_minus))
                jacobian[d, i] = np.mean(f_plus - f_minus) / (2 * eps)

        # Covariance-style metric: Cov(J) = E[J^T J] - E[J]^T E[J]
        mean_J = np.mean(jacobian, axis=0, keepdims=True)  # (1, n_params)
        g = (jacobian.T @ jacobian) / n_data - mean_J.T @ mean_J

        # Symmetrise
        g = (g + g.T) / 2
        return g
