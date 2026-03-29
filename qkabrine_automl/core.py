"""
Core QkabrineAutoML class.

Provides the main entry point for automated quantum machine learning.
Supports multiclass classification via qml.probs() with cross-entropy loss,
model serialization, comprehensive input validation, noise-aware training,
multiple optimizers, and post-search circuit surgery.
"""

from __future__ import annotations

import time
import copy
import json
import pickle
import warnings
import logging
import numpy as np
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from .ansatz import REGISTRY
from .utils import (
    apply_gate, count_params, DataEncoder, init_params,
    prune_circuit, simplify_circuit, to_qasm, reduce_features,
    get_noisy_device, apply_noise_layer,
)
from .search import (
    make_search, BaseSearch, SearchCandidate,
    GridSearch, BayesianSearch, EvolutionarySearch, SuccessiveHalving,
)
from .dynamics import (
    DataQuantumFisherMetric, BarrenPlateauMonitor, QuantumNaturalGradient,
)

logger = logging.getLogger('qkabrine_automl')


# ══════════════════════════════════════════════════════════════
#  CUSTOM EXCEPTIONS
# ══════════════════════════════════════════════════════════════

class QkabrineError(Exception):
    """Base exception for Qkabrine AutoML."""

class TrainingFailureError(QkabrineError):
    """Raised when a candidate circuit fails during training."""

class SearchExhaustedError(QkabrineError):
    """Raised when all search candidates fail and no valid model is found."""

class InvalidCircuitError(QkabrineError):
    """Raised when a circuit configuration is invalid."""


class QkabrineAutoML:
    """
    Automatic Quantum Machine Learning.

    Searches over quantum circuit architectures, data encodings, and
    hyperparameters to find the best quantum model for your data.
    Supports both variational circuits and quantum kernel methods.

    Parameters
    ----------
    task : str
        'classification' or 'regression'.
    n_qubits : int or None
        Number of qubits. Auto-inferred from features if None (max 10).
    max_layers : int
        Maximum circuit depth per ansatz to try.
    train_steps : int
        Gradient steps per candidate.
    time_budget : float or None
        Stop after N seconds. None = no limit.
    search_strategy : str
        'grid', 'random', 'bayesian', 'evolutionary', 'successive_halving'.
    encodings : tuple of str
        Data encodings to search: 'angle', 'angle_yz', 'iqp', 'amplitude'.
    init_strategies : tuple of str
        Parameter init strategies: 'uniform', 'small', 'zeros', 'normal', 'block'.
    optimizer : str
        'adam', 'sgd', 'momentum'.
    include_kernels : bool
        Whether to also evaluate quantum kernel methods.
    cv_folds : int or None
        Cross-validation folds (>= 2). None = single train/val split.
    noise_model : str or None
        Train with noise: 'depolarizing', 'bitflip', 'phaseflip',
        'amplitude_damping'.
    noise_strength : float
        Noise probability (used only if noise_model is set).
    feature_reduction : str or None
        'pca', 'random_projection', 'select_variance', or None.
    use_dqfim_prescreening : bool
        Pre-screen candidates via DQFIM trainability score.
    monitor_barren_plateaus : bool
        Monitor gradient variance and early-stop on barren plateaus.
    verbose : bool
        Print search progress.
    """

    _VALID_TASKS = ('classification', 'regression')
    _VALID_STRATEGIES = ('grid', 'random', 'bayesian', 'evolutionary',
                         'successive_halving')
    _VALID_OPTIMIZERS = ('adam', 'sgd', 'momentum')
    _VALID_ENCODINGS = ('angle', 'angle_yz', 'iqp', 'amplitude')

    def __init__(
        self,
        task: str = 'classification',
        n_qubits: int = None,
        max_layers: int = 3,
        train_steps: int = 40,
        time_budget: float = None,
        search_strategy: str = 'bayesian',
        encodings: tuple = ('angle',),
        init_strategies: tuple = ('uniform',),
        optimizer: str = 'adam',
        include_kernels: bool = True,
        cv_folds: int = None,
        noise_model: str = None,
        noise_strength: float = 0.01,
        feature_reduction: str = 'pca',
        use_dqfim_prescreening: bool = False,
        monitor_barren_plateaus: bool = False,
        random_seed: int = None,
        verbose: bool = True,
    ):
        # ── Input validation ──
        if task not in self._VALID_TASKS:
            raise ValueError(
                f"task must be one of {self._VALID_TASKS}, got '{task}'")
        if search_strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"search_strategy must be one of {self._VALID_STRATEGIES}, "
                f"got '{search_strategy}'")
        if optimizer not in self._VALID_OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {self._VALID_OPTIMIZERS}, "
                f"got '{optimizer}'")
        for enc in encodings:
            if enc not in self._VALID_ENCODINGS:
                raise ValueError(
                    f"Unknown encoding '{enc}'. Valid: {self._VALID_ENCODINGS}")
        if cv_folds is not None and cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2 or None, got {cv_folds}")
        if n_qubits is not None and n_qubits < 1:
            raise ValueError(f"n_qubits must be >= 1, got {n_qubits}")

        self.task = task
        self.n_qubits = n_qubits
        self.max_layers = max_layers
        self.train_steps = train_steps
        self.time_budget = time_budget
        self.search_strategy = search_strategy
        self.encodings = tuple(encodings)
        self.init_strategies = tuple(init_strategies)
        self.optimizer = optimizer
        self.include_kernels = include_kernels
        self.cv_folds = cv_folds
        self.noise_model = noise_model
        self.noise_strength = noise_strength
        self.feature_reduction = feature_reduction
        self.use_dqfim_prescreening = use_dqfim_prescreening
        self.monitor_barren_plateaus = monitor_barren_plateaus
        self.random_seed = random_seed
        self.verbose = verbose

        self._scaler = MinMaxScaler(feature_range=(0, np.pi))
        self._label_enc = LabelEncoder()
        self._feature_reducer = None
        self._results = []
        self._best = None
        self._best_params = None
        self._fitted = False
        self._n_classes = 2
        self._qml = None
        self._eval_cache = {}  # hash(config) → result

        # ── Reproducibility: set global seeds ──
        if random_seed is not None:
            np.random.seed(random_seed)
            import random
            random.seed(random_seed)

        if noise_model and verbose:
            print(f"  Note: noise model '{noise_model}' uses mixed-state "
                  f"simulation which is slower than noiseless.")

    # ══════════════════════════════════════════════════════════════
    #  PUBLIC API
    # ══════════════════════════════════════════════════════════════

    def fit(self, X, y, val_size: float = 0.2):
        """Run architecture search to find the best quantum model."""
        import pennylane as qml
        self._qml = qml
        self._start_time = time.time()

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")
        if len(X) < 5:
            raise ValueError(f"Need at least 5 samples, got {len(X)}")
        if self.task == 'classification' and len(np.unique(y)) < 2:
            raise ValueError("Classification requires >= 2 classes")

        if self.n_qubits is None:
            self.n_qubits = min(X.shape[1], 10)

        if X.shape[1] > self.n_qubits:
            if self.feature_reduction:
                X, self._feature_reducer = reduce_features(
                    X, self.n_qubits, self.feature_reduction)
            else:
                X = X[:, :self.n_qubits]

        if self.task == 'classification':
            y = self._label_enc.fit_transform(y)
            self._n_classes = len(np.unique(y))

        X = self._scaler.fit_transform(X)

        if self.cv_folds and self.cv_folds > 1:
            self._use_cv = True
            self._X_full, self._y_full = X, y
        else:
            self._use_cv = False
            stratify = y if self.task == 'classification' else None
            self._X_tr, self._X_val, self._y_tr, self._y_val = \
                train_test_split(X, y, test_size=val_size, random_state=42,
                                 stratify=stratify)

        # Run search
        runner = {
            'evolutionary': self._run_evolutionary_search,
            'successive_halving': self._run_successive_halving_search,
            'bayesian': self._run_bayesian_search,
        }.get(self.search_strategy, self._run_standard_search)
        runner(X, y)

        if self.include_kernels and self.task == 'classification':
            self._evaluate_kernel_methods(X, y)

        ascending = self.task == 'regression'
        self._results.sort(key=lambda r: r['score'], reverse=not ascending)

        # Filter out failed candidates when selecting best
        valid_results = [r for r in self._results if not r.get('failed', False)]
        if not valid_results:
            n_failed = sum(1 for r in self._results if r.get('failed'))
            self._fitted = True  # allow inspection of results
            raise SearchExhaustedError(
                f"All {n_failed} candidates failed. No valid model found. "
                f"Check your data dimensions, encoding compatibility, and "
                f"qubit count. Use verbose=True for details.")

        self._best = valid_results[0]
        self._best_params = valid_results[0].get('trained_params')

        # Circuit surgery
        if (self._best.get('model_type', 'variational') == 'variational'
                and self._best_params is not None
                and len(self._best_params) > 0):
            pruned_arch, pruned_params = prune_circuit(
                self._best['arch'], self._best_params, threshold=0.05)
            simplified_arch = simplify_circuit(pruned_arch)
            self._best['arch_original'] = self._best['arch']
            self._best['params_original'] = self._best_params
            self._best['arch'] = simplified_arch
            self._best['trained_params'] = pruned_params
            self._best_params = pruned_params
            self._best['n_params_pruned'] = count_params(simplified_arch)
            removed = len(self._best['arch_original']) - len(simplified_arch)
            if removed > 0 and self.verbose:
                print(f'\n  Circuit surgery: removed {removed} redundant gates')

        self._fitted = True
        if self.verbose:
            self._print_footer()
        return self

    def predict(self, X) -> np.ndarray:
        self._check_fitted()
        if self._best.get('model_type') == 'kernel':
            return self._best['model'].predict(self._prep_X(X))
        X = self._prep_X(X)
        if self.task == 'classification' and self._n_classes > 2:
            probs = self._forward_probs(
                X, self._best['arch'],
                self._best.get('trained_params', np.array([])),
                self._best.get('encoding', 'angle'))
            labels = np.argmax(probs, axis=1)
            return self._label_enc.inverse_transform(labels)
        else:
            raw = self._forward(
                X, self._best['arch'],
                self._best.get('trained_params', np.array([])),
                self._best.get('encoding', 'angle'))
            if self.task == 'classification':
                return self._label_enc.inverse_transform((raw >= 0).astype(int))
            else:
                y_min, y_max = self._y_tr.min(), self._y_tr.max()
                return (raw + 1) / 2 * (y_max - y_min) + y_min

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        if self.task != 'classification':
            raise ValueError('predict_proba is only for classification.')
        if self._best.get('model_type') == 'kernel':
            raise NotImplementedError("predict_proba not available for kernel models.")
        X = self._prep_X(X)
        if self._n_classes > 2:
            return self._forward_probs(
                X, self._best['arch'],
                self._best.get('trained_params', np.array([])),
                self._best.get('encoding', 'angle'))
        else:
            raw = self._forward(
                X, self._best['arch'],
                self._best.get('trained_params', np.array([])),
                self._best.get('encoding', 'angle'))
            p1 = np.clip((np.asarray(raw, dtype=float).flatten() + 1) / 2, 0.0, 1.0)
            return np.column_stack([1 - p1, p1])

    def score(self, X, y) -> float:
        preds = self.predict(X)
        return (accuracy_score(y, preds) if self.task == 'classification'
                else r2_score(y, preds))

    # ── Serialization ──

    def save(self, path: str = 'qkabrine_automl_model.pkl'):
        """Save fitted model to disk."""
        self._check_fitted()
        state = {
            'task': self.task, 'n_qubits': self.n_qubits,
            'n_classes': self._n_classes,
            'best': {k: v for k, v in self._best.items() if k != 'model'},
            'best_params': (np.array(self._best_params).tolist()
                            if self._best_params is not None else None),
            'scaler_data_min': self._scaler.data_min_.tolist(),
            'scaler_data_max': self._scaler.data_max_.tolist(),
            'scaler_scale': self._scaler.scale_.tolist(),
            'scaler_min': self._scaler.min_.tolist(),
            'label_classes': (self._label_enc.classes_.tolist()
                              if self.task == 'classification' else None),
            'feature_reduction': self.feature_reduction,
        }
        if self._feature_reducer is not None:
            state['feature_reducer'] = pickle.dumps(self._feature_reducer)
        # Kernel models: serialise the fitted SVM/SVR and training data separately
        if self._best.get('model_type') == 'kernel' and self._best.get('model') is not None:
            km = self._best['model']
            # QuantumKernelClassifier uses _svm; QuantumKernelRegressor uses _svr
            fitted_estimator = getattr(km, '_svm', None) or getattr(km, '_svr', None)
            state['kernel_model'] = pickle.dumps({
                'n_qubits': km.kernel.n_qubits,
                'embedding': km.kernel.embedding,
                'n_layers': km.kernel.n_layers,
                'C': km.C,
                'fitted_estimator': fitted_estimator,
                'X_train': km._X_train,
            })
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        if self.verbose:
            print(f'  Saved → {path}')

    @classmethod
    def load(cls, path: str) -> 'QkabrineAutoML':
        """Load a saved model from disk."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        import pennylane.numpy as pnp

        automl = cls.__new__(cls)
        automl.task = state['task']
        automl.n_qubits = state['n_qubits']
        automl._n_classes = state['n_classes']
        automl._fitted = True
        automl.verbose = False
        automl.noise_model = None
        automl.monitor_barren_plateaus = False
        automl.feature_reduction = state.get('feature_reduction')
        automl.optimizer = 'adam'

        automl._best = state['best']
        if state['best_params'] is not None:
            automl._best_params = pnp.array(state['best_params'],
                                             requires_grad=False)
            automl._best['trained_params'] = automl._best_params
        else:
            automl._best_params = None

        automl._scaler = MinMaxScaler(feature_range=(0, np.pi))
        automl._scaler.data_min_ = np.array(state['scaler_data_min'])
        automl._scaler.data_max_ = np.array(state['scaler_data_max'])
        automl._scaler.scale_ = np.array(state['scaler_scale'])
        automl._scaler.min_ = np.array(state['scaler_min'])
        automl._scaler.n_features_in_ = len(state['scaler_data_min'])

        automl._label_enc = LabelEncoder()
        if state.get('label_classes') is not None:
            automl._label_enc.classes_ = np.array(state['label_classes'])

        automl._feature_reducer = (pickle.loads(state['feature_reducer'])
                                    if state.get('feature_reducer') else None)
        # Reconstitute kernel model so predict() works after load
        if state.get('kernel_model') is not None:
            from .kernels import QuantumKernelClassifier, QuantumKernelRegressor
            kd = pickle.loads(state['kernel_model'])
            Cls = (QuantumKernelClassifier if automl.task == 'classification'
                   else QuantumKernelRegressor)
            model = Cls(kd['n_qubits'], kd['embedding'], kd['n_layers'], kd['C'])
            # Route fitted estimator to the correct attribute name
            if automl.task == 'classification':
                model._svm = kd['fitted_estimator']
            else:
                model._svr = kd['fitted_estimator']
            model._X_train = kd['X_train']
            automl._best['model'] = model
        automl._results = []
        automl._qml = None
        return automl

    # ── Display ──

    def leaderboard(self, top_n: int = 15):
        self._check_fitted()
        metric = 'Accuracy' if self.task == 'classification' else 'MSE'
        arrow = '↑' if self.task == 'classification' else '↓'
        print('\n' + '━' * 78)
        print(f'  ⚛️ Qkabrine AutoML Leaderboard   [{metric} {arrow}]')
        print('━' * 78)
        print(f'{"Rank":<6}{"Model":<26}{"Type":<10}{"Enc":<10}'
              f'{"Params":<8}{metric:<10}{"Time(s)"}')
        print('─' * 78)
        for i, r in enumerate(self._results[:top_n]):
            medal = ['🥇','🥈','🥉'][i] if i < 3 else f'{i+1:>3}.'
            name = r.get('name', '?')
            if r.get('layers'):
                name = f"{name}(L={r['layers']})"
            print(f'{medal:<6}{name:<26}{r.get("model_type","var")[:8]:<10}'
                  f'{r.get("encoding","-")[:8]:<10}{r.get("n_params",0):<8}'
                  f'{r.get("score",0):<10.4f}{r.get("duration",0):.1f}')
        print('━' * 78)
        if self._best:
            b = self._best
            print(f'\n  Best: {b.get("name","?")} ({b.get("model_type","var")})')
            print(f'  Val {metric}: {b["score"]:.4f}')
            if b.get('n_params_pruned') is not None and b.get('n_params'):
                print(f'  Params: {b["n_params"]} → {b["n_params_pruned"]} after surgery')
        print()

    def best_circuit_summary(self):
        self._check_fitted()
        if self._best.get('model_type') == 'kernel':
            print(f'\n  Best: quantum kernel ({self._best["name"]})')
            return
        arch = self._best['arch']
        print(f'\n  Best: {self._best["name"]} '
              f'(L={self._best.get("layers","?")}, '
              f'enc={self._best.get("encoding","angle")})')
        print(f'  {"─"*42}')
        for i, g in enumerate(arch):
            m = '🔵' if g['trainable'] else '⚪'
            print(f'  {i+1:<5}{g["gate"]:<12}{str(g["wires"]):<14}{m}')
        print(f'  {"─"*42}')
        print(f'  Gates: {len(arch)}, Trainable: {count_params(arch)}')

    def export_qasm(self) -> str:
        self._check_fitted()
        if self._best.get('model_type') == 'kernel':
            raise ValueError("QASM export not available for kernel models.")
        return to_qasm(self._best['arch'],
                        self._best.get('trained_params', np.array([])),
                        self.n_qubits,
                        encoding=self._best.get('encoding', 'angle'))

    def export_qasm_to_file(self, path: str = 'best_circuit.qasm'):
        with open(path, 'w') as f:
            f.write(self.export_qasm())
        if self.verbose:
            print(f'  Exported → {path}')

    # ══════════════════════════════════════════════════════════════
    #  SEARCH RUNNERS
    # ══════════════════════════════════════════════════════════════

    def _run_standard_search(self, X, y):
        search = make_search(self.search_strategy, max_layers=self.max_layers,
            time_budget=self.time_budget, encodings=self.encodings,
            init_strategies=self.init_strategies, verbose=self.verbose)
        search._start_time = time.time()
        cands = search.generate_candidates()
        if self.verbose: self._print_header(X, y, len(cands))
        for c in cands:
            if search.should_stop(): break
            self._evaluate_candidate_obj(c)

    def _run_bayesian_search(self, X, y):
        s = BayesianSearch(max_layers=self.max_layers, time_budget=self.time_budget,
            encodings=self.encodings, init_strategies=self.init_strategies, verbose=self.verbose)
        s._start_time = time.time()
        initial = s.generate_candidates()
        if self.verbose:
            self._print_header(X, y, s.n_trials)
            print(f'  Strategy: Bayesian (GP + EI)\n  ' + '─' * 52)
        for c in initial:
            if s.should_stop(): break
            r = self._evaluate_candidate_obj(c)
            s.update(c, r['score'], r['duration'])
        while not s.should_stop():
            nxt = s.next_candidate()
            if nxt is None: break
            r = self._evaluate_candidate_obj(nxt)
            s.update(nxt, r['score'], r['duration'])

    def _run_evolutionary_search(self, X, y):
        s = EvolutionarySearch(max_layers=self.max_layers, time_budget=self.time_budget,
            encodings=self.encodings, init_strategies=self.init_strategies, verbose=self.verbose)
        s._start_time = time.time()
        pop = s.generate_candidates()
        if self.verbose:
            self._print_header(X, y, s.population_size * s.n_generations)
            print(f'  Strategy: Evolutionary\n  ' + '─' * 52)
        for gen in range(s.n_generations):
            if s._budget_exhausted(): break
            if self.verbose: print(f'\n  === Generation {gen+1} ===')
            for c in pop:
                if s._budget_exhausted(): break
                r = self._evaluate_candidate_obj(c)
                s.update(c, r['score'], r['duration'])
            pop = s.evolve()

    def _run_successive_halving_search(self, X, y):
        s = SuccessiveHalving(max_layers=self.max_layers, time_budget=self.time_budget,
            encodings=self.encodings, init_strategies=self.init_strategies, verbose=self.verbose)
        s._start_time = time.time()
        cands = s.generate_candidates()
        if self.verbose:
            self._print_header(X, y, s.n_candidates)
            print(f'  Strategy: Successive Halving\n  ' + '─' * 52)
        rnd = 0
        while not s.should_stop() and len(cands) > 1:
            rnd += 1; budget = s.get_current_budget()
            if self.verbose:
                print(f'\n  === Round {rnd}: {len(cands)} candidates, {budget} steps ===')
            scored = []
            for c in cands:
                if s._budget_exhausted(): break
                r = self._evaluate_candidate_obj(c, train_steps_override=budget)
                scored.append((c, r['score']))
            cands = s.halve(scored)
        if cands and not s._budget_exhausted():
            for c in cands:
                self._evaluate_candidate_obj(c)

    # ══════════════════════════════════════════════════════════════
    #  CANDIDATE EVALUATION
    # ══════════════════════════════════════════════════════════════

    def _evaluate_candidate_obj(self, cand, train_steps_override=None):
        # ── Cache check: skip duplicate configs ──
        cache_key = (cand.name, cand.layers, cand.encoding,
                     cand.init_strategy, cand.learning_rate,
                     train_steps_override or self.train_steps)
        if cache_key in self._eval_cache:
            cached = self._eval_cache[cache_key]
            if self.verbose:
                print(f'  Testing  {cand.name:<22}  L={cand.layers}  '
                      f'enc={cand.encoding:<8}  → CACHED ({cached["score"]:.4f})')
            self._results.append(cached)
            return cached

        t0 = time.time()
        steps = train_steps_override or self.train_steps

        try:
            arch = REGISTRY[cand.name](self.n_qubits, cand.layers)
            n_p = count_params(arch)
        except Exception as e:
            logger.warning(f"Failed to build {cand.name} L={cand.layers}: {e}")
            return self._make_failure_result(cand, time.time() - t0, str(e))

        if self.verbose:
            print(f'  Testing  {cand.name:<22}  L={cand.layers}  '
                  f'enc={cand.encoding:<8}  params={n_p:<4}', end=' ', flush=True)

        try:
            trainability_score = None
            if n_p == 0:
                score = 0.5 if self.task == 'classification' else float('inf')
                trained_params = np.array([])
            elif self.use_dqfim_prescreening:
                dqfim = DataQuantumFisherMetric(n_qubits=self.n_qubits, n_samples=10,
                    seed=self.random_seed)
                def _proxy(x, params):
                    x_pad = np.zeros(n_p); x_pad[:min(len(x),n_p)] = x[:min(len(x),n_p)]
                    return float(np.tanh(np.dot(x_pad, params)))
                metrics = dqfim.predict_generalization(_proxy, self._X_tr[:30], n_params=n_p, n_samples=5)
                trainability_score = metrics.trainability_score
                if trainability_score < 0.05:
                    dt = time.time() - t0
                    if self.verbose: print(f'→ SKIPPED (T={trainability_score:.3f})  ({dt:.1f}s)')
                    result = {'name': cand.name, 'layers': cand.layers, 'arch': arch,
                        'n_params': n_p, 'score': 0.0 if self.task == 'classification' else float('inf'),
                        'duration': dt, 'trained_params': np.array([]),
                        'encoding': cand.encoding, 'init_strategy': cand.init_strategy,
                        'learning_rate': cand.learning_rate, 'model_type': 'variational',
                        'trainability_score': trainability_score, 'failed': False}
                    self._results.append(result)
                    self._eval_cache[cache_key] = result
                    return result
                score, trained_params = self._do_train(arch, n_p, cand, steps)
            else:
                score, trained_params = self._do_train(arch, n_p, cand, steps)

            dt = time.time() - t0
            if self.verbose:
                if self.task == 'classification':
                    bar = '█' * int(score*20) + '░' * (20-int(score*20))
                    extra = f'  T={trainability_score:.2f}' if trainability_score else ''
                    print(f'→ acc={score:.4f}  {bar}{extra}  ({dt:.1f}s)')
                else:
                    print(f'→ mse={score:.4f}  ({dt:.1f}s)')

            result = {'name': cand.name, 'layers': cand.layers, 'arch': arch,
                'n_params': n_p, 'score': score, 'duration': dt,
                'trained_params': trained_params, 'encoding': cand.encoding,
                'init_strategy': cand.init_strategy, 'learning_rate': cand.learning_rate,
                'model_type': 'variational', 'trainability_score': trainability_score,
                'failed': False}
            self._results.append(result)
            self._eval_cache[cache_key] = result
            return result

        except Exception as e:
            dt = time.time() - t0
            logger.warning(f"Training failed for {cand.name} L={cand.layers} "
                           f"enc={cand.encoding}: {e}")
            if self.verbose:
                print(f'→ FAILED ({type(e).__name__})  ({dt:.1f}s)')
            return self._make_failure_result(cand, dt, str(e))

    def _make_failure_result(self, cand, duration, error_msg=''):
        """Create a result dict for a failed candidate."""
        fail_score = 0.0 if self.task == 'classification' else float('inf')
        result = {'name': cand.name, 'layers': cand.layers, 'arch': None,
            'n_params': 0, 'score': fail_score, 'duration': duration,
            'trained_params': np.array([]), 'encoding': cand.encoding,
            'init_strategy': cand.init_strategy, 'learning_rate': cand.learning_rate,
            'model_type': 'variational', 'trainability_score': None,
            'failed': True, 'error': error_msg}
        self._results.append(result)
        return result

    def _do_train(self, arch, n_params, cand, steps):
        if self._use_cv:
            return self._train_and_eval_cv(arch, n_params, cand.encoding,
                cand.init_strategy, cand.learning_rate, steps)
        return self._train_and_eval(arch, n_params, cand.encoding,
            cand.init_strategy, cand.learning_rate, steps)

    # ══════════════════════════════════════════════════════════════
    #  TRAINING — THE CRITICAL METHOD
    # ══════════════════════════════════════════════════════════════

    def _train_and_eval(self, arch, n_params, encoding='angle',
                         init_strategy='uniform', learning_rate=0.05,
                         train_steps=None):
        """Train variational circuit. Multiclass uses qml.probs() + cross-entropy."""
        qml = self._qml
        import pennylane.numpy as pnp
        steps = train_steps or self.train_steps
        encoder = DataEncoder(encoding)

        if self.noise_model:
            dev, _, _ = get_noisy_device(self.n_qubits, self.noise_model, self.noise_strength)
        else:
            dev = qml.device('default.qubit', wires=self.n_qubits)

        has_reuploading = any(g['gate'] == 'ENCODE' for g in arch)

        def _apply_arch(x, params):
            if has_reuploading:
                p = 0
                for gate in arch:
                    if gate['gate'] == 'ENCODE':
                        encoder.encode(x, self.n_qubits)
                    else:
                        p = apply_gate(gate, params, p)
            else:
                encoder.encode(x, self.n_qubits)
                p = 0
                for gate in arch:
                    p = apply_gate(gate, params, p)
            if self.noise_model:
                apply_noise_layer(self.n_qubits, self.noise_model, self.noise_strength)

        # ── Build circuit and cost based on task ──
        if self.task == 'classification' and self._n_classes > 2:
            # MULTICLASS: qml.probs() returns probabilities over basis states
            n_measure = max(1, int(np.ceil(np.log2(self._n_classes))))
            n_measure = min(n_measure, self.n_qubits)
            n_basis = 2 ** n_measure

            @qml.qnode(dev, interface='autograd')
            def circuit_probs(x, params):
                _apply_arch(x, params)
                return qml.probs(wires=range(n_measure))

            # One-hot targets: class k → basis state |k>
            targets = np.zeros((len(self._y_tr), n_basis))
            for i, label in enumerate(self._y_tr):
                targets[i, int(label) % n_basis] = 1.0
            targets_pnp = pnp.array(targets, requires_grad=False)

            def cost(p):
                probs_stack = pnp.stack([circuit_probs(x, p) for x in self._X_tr])
                return -pnp.mean(pnp.sum(targets_pnp * pnp.log(probs_stack + 1e-9), axis=1))

        elif self.task == 'classification':
            # BINARY: PauliZ expectation
            @qml.qnode(dev, interface='autograd')
            def circuit(x, params):
                _apply_arch(x, params)
                return qml.expval(qml.PauliZ(0))

            y_tr_q = pnp.array(2 * self._y_tr.astype(float) - 1, requires_grad=False)
            def cost(p):
                preds = pnp.stack([circuit(x, p) for x in self._X_tr])
                return pnp.mean((preds - y_tr_q) ** 2)
        else:
            # REGRESSION
            @qml.qnode(dev, interface='autograd')
            def circuit(x, params):
                _apply_arch(x, params)
                return qml.expval(qml.PauliZ(0))

            y_min, y_max = self._y_tr.min(), self._y_tr.max()
            y_scaled = pnp.array(2 * (self._y_tr - y_min) / (y_max - y_min + 1e-8) - 1,
                                  requires_grad=False)
            def cost(p):
                preds = pnp.stack([circuit(x, p) for x in self._X_tr])
                return pnp.mean((preds - y_scaled) ** 2)

        # ── Optimizer ──
        params = init_params(n_params, init_strategy)
        if self.optimizer == 'sgd':
            opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
        elif self.optimizer == 'momentum':
            opt = qml.MomentumOptimizer(stepsize=learning_rate)
        else:
            opt = qml.AdamOptimizer(stepsize=learning_rate)

        # ── BP monitor ──
        bp_monitor = (BarrenPlateauMonitor(threshold=1e-7, window=3, auto_surgery=True)
                      if self.monitor_barren_plateaus else None)

        # ── Training loop ──
        for step in range(steps):
            params = opt.step(cost, params)
            if bp_monitor and step > 0 and step % 5 == 0:
                try:
                    c0 = float(cost(params))
                    n_s = min(n_params, 6)
                    grad_est = np.zeros(n_s)
                    for gi in range(n_s):
                        p_np = np.array(params, dtype=float); p_np[gi] += 1e-4
                        grad_est[gi] = (float(cost(pnp.array(p_np, requires_grad=True))) - c0) / 1e-4
                    bp_monitor.update(grad_est)
                    if bp_monitor.plateau_detected:
                        if self.verbose: print(f'[BP@{step}] ', end='', flush=True)
                        break
                except Exception:
                    pass

        # ── Validation ──
        if self.task == 'classification' and self._n_classes > 2:
            val_probs = np.array([np.array(circuit_probs(x, params)) for x in self._X_val])
            val_preds = np.argmax(val_probs[:, :self._n_classes], axis=1)
            val_preds = np.clip(val_preds, 0, self._n_classes - 1)
            score = accuracy_score(self._y_val, val_preds)
        elif self.task == 'classification':
            val_raw = np.array([float(circuit(x, params)) for x in self._X_val])
            score = accuracy_score(self._y_val, (val_raw >= 0).astype(int))
        else:
            val_raw = np.array([float(circuit(x, params)) for x in self._X_val])
            y_min, y_max = self._y_tr.min(), self._y_tr.max()
            val_preds = (val_raw + 1) / 2 * (y_max - y_min) + y_min
            score = mean_squared_error(self._y_val, val_preds)

        return float(score), params

    def _train_and_eval_cv(self, arch, n_params, encoding='angle',
                            init_strategy='uniform', learning_rate=0.05,
                            train_steps=None):
        steps = train_steps or self.train_steps
        X, y = self._X_full, self._y_full
        kf = (StratifiedKFold if self.task == 'classification' else KFold)(
            n_splits=self.cv_folds, shuffle=True, random_state=42)
        fold_scores, best_params, best_score = [], None, (-np.inf if self.task == 'classification' else np.inf)
        for tr_idx, val_idx in kf.split(X, y):
            self._X_tr, self._X_val = X[tr_idx], X[val_idx]
            self._y_tr, self._y_val = y[tr_idx], y[val_idx]
            score, params = self._train_and_eval(arch, n_params, encoding, init_strategy, learning_rate, steps)
            fold_scores.append(score)
            if (score > best_score if self.task == 'classification' else score < best_score):
                best_score, best_params = score, params
        self._X_tr, self._y_tr = X, y
        return float(np.mean(fold_scores)), best_params

    def _evaluate_kernel_methods(self, X, y):
        from .kernels import QuantumKernelClassifier, QuantumKernelRegressor
        if self.verbose: print(f'\n  === Quantum Kernel Methods ===')
        for emb in ['iqp', 'angle', 'hardware_efficient']:
            if self.time_budget and (time.time() - self._start_time) > self.time_budget: break
            if self.verbose: print(f'  Testing  kernel_{emb:<20}', end=' ', flush=True)
            t0 = time.time()
            try:
                Cls = QuantumKernelClassifier if self.task == 'classification' else QuantumKernelRegressor
                if self._use_cv:
                    kf = (StratifiedKFold if self.task == 'classification' else KFold)(
                        n_splits=min(self.cv_folds, 3), shuffle=True, random_state=42)
                    scores = []
                    for tr_i, val_i in kf.split(X, y):
                        m = Cls(self.n_qubits, emb); m.fit(X[tr_i], y[tr_i])
                        scores.append(m.score(X[val_i], y[val_i]))
                    score = float(np.mean(scores))
                    model = Cls(self.n_qubits, emb); model.fit(X, y)
                else:
                    model = Cls(self.n_qubits, emb)
                    model.fit(self._X_tr, self._y_tr)
                    score = model.score(self._X_val, self._y_val)
                dt = time.time() - t0
                if self.verbose:
                    bar = '█' * int(score*20) + '░' * (20-int(score*20))
                    print(f'→ acc={score:.4f}  {bar}  ({dt:.1f}s)')
                self._results.append({'name': f'kernel_{emb}', 'layers': None, 'arch': None,
                    'n_params': 0, 'score': score, 'duration': dt, 'trained_params': None,
                    'encoding': emb, 'model_type': 'kernel', 'model': model})
            except Exception as e:
                if self.verbose: print(f'→ FAILED ({e}) ({time.time()-t0:.1f}s)')

    # ══════════════════════════════════════════════════════════════
    #  FORWARD PASS (inference)
    # ══════════════════════════════════════════════════════════════

    def _forward(self, X, arch, params, encoding='angle'):
        if self._qml is None:
            import pennylane as qml; self._qml = qml
        qml = self._qml
        encoder = DataEncoder(encoding)
        has_re = any(g['gate'] == 'ENCODE' for g in arch)
        dev = qml.device('default.qubit', wires=self.n_qubits)

        @qml.qnode(dev, interface='autograd')
        def circ(x, p):
            if has_re:
                pi = 0
                for gate in arch:
                    if gate['gate'] == 'ENCODE': encoder.encode(x, self.n_qubits)
                    else: pi = apply_gate(gate, p, pi)
            else:
                encoder.encode(x, self.n_qubits)
                pi = 0
                for gate in arch: pi = apply_gate(gate, p, pi)
            return qml.expval(qml.PauliZ(0))
        return np.array([float(circ(x, params)) for x in X])

    def _forward_probs(self, X, arch, params, encoding='angle'):
        if self._qml is None:
            import pennylane as qml; self._qml = qml
        qml = self._qml
        encoder = DataEncoder(encoding)
        has_re = any(g['gate'] == 'ENCODE' for g in arch)
        dev = qml.device('default.qubit', wires=self.n_qubits)
        n_measure = max(1, int(np.ceil(np.log2(self._n_classes))))
        n_measure = min(n_measure, self.n_qubits)

        @qml.qnode(dev, interface='autograd')
        def circ(x, p):
            if has_re:
                pi = 0
                for gate in arch:
                    if gate['gate'] == 'ENCODE': encoder.encode(x, self.n_qubits)
                    else: pi = apply_gate(gate, p, pi)
            else:
                encoder.encode(x, self.n_qubits)
                pi = 0
                for gate in arch: pi = apply_gate(gate, p, pi)
            return qml.probs(wires=range(n_measure))

        results = []
        for x in X:
            p = np.array(circ(x, params))
            results.append(p[:self._n_classes])
        return np.array(results)

    # ══════════════════════════════════════════════════════════════
    #  HELPERS
    # ══════════════════════════════════════════════════════════════

    def _prep_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1: X = X.reshape(1, -1)
        if self._feature_reducer:
            X = self._feature_reducer.transform(X)
        else:
            X = X[:, :self.n_qubits]
        return self._scaler.transform(X)

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Not fitted yet. Call .fit() first.")

    def _print_header(self, X, y, n_candidates):
        print('\n' + '═' * 68)
        print('  ⚛️  Qkabrine AutoML  v2.1')
        print('═' * 68)
        print(f'  Task       : {self.task}')
        print(f'  Samples    : {len(X)}')
        print(f'  Features   : {X.shape[1]} → {self.n_qubits} qubits')
        if self.task == 'classification':
            print(f'  Classes    : {self._n_classes}')
        if self.feature_reduction and self._feature_reducer:
            print(f'  Reduction  : {self.feature_reduction}')
        print(f'  Search     : {self.search_strategy} ({n_candidates} candidates)')
        print(f'  Encodings  : {", ".join(self.encodings)}')
        print(f'  Steps      : {self.train_steps}')
        print(f'  Optimizer  : {self.optimizer}')
        if self.noise_model:
            print(f'  Noise      : {self.noise_model} (p={self.noise_strength})')
        if self.use_dqfim_prescreening:
            print(f'  DQFIM      : ON')
        if self.monitor_barren_plateaus:
            print(f'  BP monitor : ON')
        print('═' * 68)

    def _print_footer(self):
        elapsed = time.time() - self._start_time
        metric = 'Accuracy' if self.task == 'classification' else 'MSE'
        print('\n' + '═' * 68)
        print(f'  Done in {elapsed:.1f}s ({len(self._results)} models)')
        if self._best:
            print(f'  Best: {self._best["name"]} '
                  f'({self._best.get("model_type","var")}), '
                  f'Val {metric}: {self._best["score"]:.4f}')
        print('═' * 68)
