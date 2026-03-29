"""
QkabrineAutoML — Automatic Quantum Machine Learning.

A comprehensive framework for automated quantum circuit architecture
search, combining variational circuits and quantum kernel methods with
intelligent search strategies (Bayesian, evolutionary, successive halving)
to find the best quantum model for your data.

Usage
-----
    from qkabrine_automl import QkabrineAutoML

    automl = QkabrineAutoML(task='classification', search_strategy='bayesian')
    automl.fit(X_train, y_train)
    automl.leaderboard()
    preds = automl.predict(X_test)

    # Export best circuit
    print(automl.export_qasm())
"""

from .core import (
    QkabrineAutoML,
    QkabrineError,
    TrainingFailureError,
    SearchExhaustedError,
    InvalidCircuitError,
)
from .ansatz import (
    REGISTRY as ANSATZ_REGISTRY,
    compute_expressibility,
    compute_entangling_capability,
    rank_ansatze,
)
from .kernels import (
    QuantumKernel,
    QuantumKernelClassifier,
    QuantumKernelRegressor,
)
from .search import (
    make_search,
    GridSearch,
    RandomSearch,
    BayesianSearch,
    EvolutionarySearch,
    SuccessiveHalving,
    SEARCH_STRATEGIES,
)
from .utils import (
    DataEncoder,
    prune_circuit,
    simplify_circuit,
    to_qasm,
)
from .dynamics import (
    DataQuantumFisherMetric,
    DQFIMetrics,
    BarrenPlateauMonitor,
    QuantumNaturalGradient,
)

__version__ = '2.1.0'
__author__ = 'Eric Jagwara — Solid Elf Labs'
__all__ = [
    'QkabrineAutoML',
    'QkabrineError',
    'TrainingFailureError',
    'SearchExhaustedError',
    'InvalidCircuitError',
    # Ansätze
    'ANSATZ_REGISTRY',
    'compute_expressibility',
    'compute_entangling_capability',
    'rank_ansatze',
    # Kernels
    'QuantumKernel',
    'QuantumKernelClassifier',
    'QuantumKernelRegressor',
    # Search
    'make_search',
    'GridSearch',
    'RandomSearch',
    'BayesianSearch',
    'EvolutionarySearch',
    'SuccessiveHalving',
    'SEARCH_STRATEGIES',
    # Utilities
    'DataEncoder',
    'prune_circuit',
    'simplify_circuit',
    'to_qasm',
    # Dynamics
    'DataQuantumFisherMetric',
    'DQFIMetrics',
    'BarrenPlateauMonitor',
    'QuantumNaturalGradient',
]
