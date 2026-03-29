"""
Comprehensive tests for Qkabrine AutoML.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def binary_data():
    X, y = make_classification(
        n_samples=60, n_features=4, n_informative=3,
        n_redundant=0, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def multiclass_data():
    X, y = load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=60, n_features=4, n_informative=3, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def high_dim_data():
    """Data with more features than qubits to test dimensionality reduction."""
    X, y = make_classification(
        n_samples=40, n_features=12, n_informative=6,
        n_redundant=2, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ═══════════════════════════════════════════════════════════════════
#  CORE TESTS
# ═══════════════════════════════════════════════════════════════════

class TestBinaryClassification:
    def test_grid_search(self, binary_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = binary_data
        automl = QkabrineAutoML(
            task='classification', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='grid',
            encodings=('angle',), include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        preds = automl.predict(X_te)
        assert len(preds) == len(y_te)

    def test_random_search(self, binary_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = binary_data
        automl = QkabrineAutoML(
            task='classification', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='random',
            include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        assert automl.score(X_te, y_te) >= 0.0

    def test_bayesian_search(self, binary_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = binary_data
        automl = QkabrineAutoML(
            task='classification', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='bayesian',
            include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        preds = automl.predict(X_te)
        assert len(preds) == len(y_te)

    def test_score_returns_valid(self, binary_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = binary_data
        automl = QkabrineAutoML(
            task='classification', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='grid',
            encodings=('angle',), include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        s = automl.score(X_te, y_te)
        assert isinstance(s, float) and 0.0 <= s <= 1.0


class TestMulticlassClassification:
    def test_iris_multiclass(self, multiclass_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = multiclass_data
        automl = QkabrineAutoML(
            task='classification', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='grid',
            encodings=('angle',), include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        preds = automl.predict(X_te)
        assert len(preds) == len(y_te)
        assert set(preds).issubset(set(y_te) | set(y_tr))


class TestRegression:
    def test_regression_fit_predict(self, regression_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = regression_data
        automl = QkabrineAutoML(
            task='regression', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='grid',
            encodings=('angle',), include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        preds = automl.predict(X_te)
        assert len(preds) == len(y_te)


class TestFeatureReduction:
    def test_pca_reduction(self, high_dim_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = high_dim_data
        automl = QkabrineAutoML(
            task='classification', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='grid', feature_reduction='pca',
            encodings=('angle',), include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        preds = automl.predict(X_te)
        assert len(preds) == len(y_te)

    def test_variance_reduction(self, high_dim_data):
        from qkabrine_automl import QkabrineAutoML
        X_tr, X_te, y_tr, y_te = high_dim_data
        automl = QkabrineAutoML(
            task='classification', n_qubits=4, max_layers=1,
            train_steps=3, search_strategy='grid',
            feature_reduction='select_variance',
            encodings=('angle',), include_kernels=False, verbose=False)
        automl.fit(X_tr, y_tr)
        preds = automl.predict(X_te)
        assert len(preds) == len(y_te)


# ═══════════════════════════════════════════════════════════════════
#  KERNEL TESTS
# ═══════════════════════════════════════════════════════════════════

class TestQuantumKernels:
    def test_kernel_classifier(self, binary_data):
        from qkabrine_automl.kernels import QuantumKernelClassifier
        X_tr, X_te, y_tr, y_te = binary_data
        clf = QuantumKernelClassifier(n_qubits=4, embedding='angle')
        clf.fit(X_tr, y_tr)
        score = clf.score(X_te, y_te)
        assert 0.0 <= score <= 1.0

    def test_kernel_matrix_symmetric(self, binary_data):
        from qkabrine_automl.kernels import QuantumKernel
        X_tr, _, _, _ = binary_data
        qk = QuantumKernel(n_qubits=4, embedding='angle')
        K = qk.compute_kernel_matrix(X_tr[:5])
        assert K.shape == (5, 5)
        np.testing.assert_array_almost_equal(K, K.T, decimal=5)

    def test_kernel_target_alignment(self, binary_data):
        from qkabrine_automl.kernels import QuantumKernel
        X_tr, _, y_tr, _ = binary_data
        qk = QuantumKernel(n_qubits=4, embedding='iqp')
        K = qk.compute_kernel_matrix(X_tr[:10])
        kta = qk.kernel_target_alignment(K, y_tr[:10])
        assert isinstance(kta, float)


# ═══════════════════════════════════════════════════════════════════
#  SEARCH STRATEGY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestSearchStrategies:
    def test_grid_generates_candidates(self):
        from qkabrine_automl.search import GridSearch
        gs = GridSearch(max_layers=2, encodings=('angle',))
        cands = gs.generate_candidates()
        assert len(cands) > 0

    def test_bayesian_generates_and_updates(self):
        from qkabrine_automl.search import BayesianSearch
        bs = BayesianSearch(max_layers=1, n_initial=3, n_trials=5,
                            encodings=('angle',))
        cands = bs.generate_candidates()
        assert len(cands) == 3
        for c in cands:
            bs.update(c, np.random.rand(), 1.0)
        nxt = bs.next_candidate()
        assert nxt is not None

    def test_evolutionary_evolves(self):
        from qkabrine_automl.search import EvolutionarySearch
        es = EvolutionarySearch(population_size=6, n_generations=2,
                                max_layers=1, encodings=('angle',))
        pop = es.generate_candidates()
        assert len(pop) == 6
        for c in pop:
            es.update(c, np.random.rand(), 1.0)
        pop2 = es.evolve()
        assert len(pop2) == 6


# ═══════════════════════════════════════════════════════════════════
#  UTILITY TESTS
# ═══════════════════════════════════════════════════════════════════

class TestUtils:
    def test_data_encoder_angle(self):
        from qkabrine_automl.utils import DataEncoder
        import pennylane as qml
        enc = DataEncoder('angle')
        dev = qml.device('default.qubit', wires=2)
        @qml.qnode(dev)
        def circ(x):
            enc.encode(x, 2)
            return qml.state()
        state = circ(np.array([0.5, 1.0]))
        assert len(state) == 4

    def test_prune_circuit(self):
        from qkabrine_automl.utils import prune_circuit
        arch = [
            {'gate': 'RX', 'wires': [0], 'trainable': True},
            {'gate': 'RY', 'wires': [1], 'trainable': True},
            {'gate': 'CNOT', 'wires': [0, 1], 'trainable': False},
        ]
        params = np.array([0.001, 1.5])  # first is near-zero
        pruned_arch, pruned_params = prune_circuit(arch, params, threshold=0.05)
        assert len(pruned_arch) == 2  # RX removed
        assert len(pruned_params) == 1

    def test_simplify_circuit(self):
        from qkabrine_automl.utils import simplify_circuit
        arch = [
            {'gate': 'CNOT', 'wires': [0, 1], 'trainable': False},
            {'gate': 'CNOT', 'wires': [0, 1], 'trainable': False},
            {'gate': 'RX', 'wires': [0], 'trainable': True},
        ]
        simplified = simplify_circuit(arch)
        assert len(simplified) == 1  # CNOT pair removed

    def test_qasm_export(self):
        from qkabrine_automl.utils import to_qasm
        arch = [
            {'gate': 'RY', 'wires': [0], 'trainable': True},
            {'gate': 'CNOT', 'wires': [0, 1], 'trainable': False},
        ]
        params = np.array([1.23])
        qasm = to_qasm(arch, params, 2)
        assert 'OPENQASM 2.0' in qasm
        assert 'ry(1.230000)' in qasm
        assert 'cx' in qasm

    def test_init_strategies(self):
        from qkabrine_automl.utils import init_params
        for strategy in ['uniform', 'small', 'zeros', 'normal', 'block']:
            p = init_params(10, strategy)
            assert len(p) == 10


class TestNotFitted:
    def test_predict_raises(self):
        from qkabrine_automl import QkabrineAutoML
        automl = QkabrineAutoML()
        with pytest.raises(RuntimeError):
            automl.predict(np.zeros((5, 4)))

    def test_leaderboard_raises(self):
        from qkabrine_automl import QkabrineAutoML
        automl = QkabrineAutoML()
        with pytest.raises(RuntimeError):
            automl.leaderboard()

    def test_export_qasm_raises(self):
        from qkabrine_automl import QkabrineAutoML
        automl = QkabrineAutoML()
        with pytest.raises(RuntimeError):
            automl.export_qasm()


# ═══════════════════════════════════════════════════════════════════
#  ANSATZ TESTS
# ═══════════════════════════════════════════════════════════════════

class TestAnsatze:
    def test_all_registry_entries(self):
        from qkabrine_automl.ansatz import REGISTRY
        from qkabrine_automl.utils import count_params
        for name, fn in REGISTRY.items():
            arch = fn(4, 2)
            assert isinstance(arch, list)
            assert len(arch) > 0
            n_p = count_params(arch)
            assert n_p >= 0

    def test_data_reuploading_has_encode(self):
        from qkabrine_automl.ansatz import data_reuploading
        arch = data_reuploading(4, 2)
        encode_gates = [g for g in arch if g['gate'] == 'ENCODE']
        assert len(encode_gates) > 0
