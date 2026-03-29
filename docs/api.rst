API Reference
=============

.. contents:: On this page
   :local:
   :depth: 2

---

QkabrineAutoML
--------------

The main class. All search, training, and inference goes through here.

.. code-block:: python

   from qkabrine_automl import QkabrineAutoML

Constructor Parameters
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 12 54

   * - Parameter
     - Type
     - Default
     - Description
   * - ``task``
     - str
     - ``'classification'``
     - ``'classification'`` or ``'regression'``
   * - ``n_qubits``
     - int or None
     - ``None``
     - Number of qubits. Auto-inferred from features if None (max 10).
   * - ``max_layers``
     - int
     - ``3``
     - Maximum circuit depth (layers) to try per ansatz.
   * - ``train_steps``
     - int
     - ``40``
     - Number of gradient steps per candidate circuit.
   * - ``time_budget``
     - float or None
     - ``None``
     - Stop the entire search after this many seconds. None = no limit.
   * - ``search_strategy``
     - str
     - ``'bayesian'``
     - One of ``'grid'``, ``'random'``, ``'bayesian'``, ``'evolutionary'``, ``'successive_halving'``.
   * - ``encodings``
     - tuple of str
     - ``('angle',)``
     - Encodings to search. Any of ``'angle'``, ``'angle_yz'``, ``'iqp'``, ``'amplitude'``.
   * - ``init_strategies``
     - tuple of str
     - ``('uniform',)``
     - Parameter initialisation strategies: ``'uniform'``, ``'small'``, ``'zeros'``, ``'normal'``, ``'block'``.
   * - ``optimizer``
     - str
     - ``'adam'``
     - ``'adam'``, ``'sgd'``, or ``'momentum'``.
   * - ``include_kernels``
     - bool
     - ``True``
     - Whether to also evaluate quantum kernel + SVM methods.
   * - ``cv_folds``
     - int or None
     - ``None``
     - Cross-validation folds (>= 2). None = single train/val split.
   * - ``noise_model``
     - str or None
     - ``None``
     - Simulate hardware noise: ``'depolarizing'``, ``'bitflip'``, ``'phaseflip'``, ``'amplitude_damping'``.
   * - ``noise_strength``
     - float
     - ``0.01``
     - Noise probability per gate (only used if ``noise_model`` is set).
   * - ``feature_reduction``
     - str or None
     - ``'pca'``
     - Reduce features to fit qubit count: ``'pca'``, ``'random_projection'``, ``'select_variance'``, or None.
   * - ``use_dqfim_prescreening``
     - bool
     - ``False``
     - Pre-screen candidates using DQFIM trainability score. Skips circuits likely to have vanishing gradients.
   * - ``monitor_barren_plateaus``
     - bool
     - ``False``
     - Monitor gradient variance during training and stop early if a barren plateau is detected.
   * - ``random_seed``
     - int or None
     - ``None``
     - Set for reproducible results.
   * - ``verbose``
     - bool
     - ``True``
     - Print search progress.

---

Methods
~~~~~~~

fit
^^^

.. code-block:: python

   automl.fit(X, y, val_size=0.2)

Run the architecture search and train all candidate circuits.

**Parameters:**

- ``X`` — 2D array of shape ``(n_samples, n_features)``
- ``y`` — 1D array of labels (classification) or values (regression)
- ``val_size`` — fraction of data to hold out for validation (default ``0.2``)

**Returns:** ``self``

**Raises:**

- ``ValueError`` — if X is not 2D, lengths mismatch, or fewer than 5 samples
- ``SearchExhaustedError`` — if every candidate circuit fails during training

---

predict
^^^^^^^

.. code-block:: python

   predictions = automl.predict(X)

Predict using the best model found during search.

**Parameters:**

- ``X`` — 2D array of shape ``(n_samples, n_features)``

**Returns:** 1D array of predicted class labels (classification) or values (regression).

---

predict_proba
^^^^^^^^^^^^^

.. code-block:: python

   probs = automl.predict_proba(X)

Return class probabilities. Classification only.

**Parameters:**

- ``X`` — 2D array of shape ``(n_samples, n_features)``

**Returns:** 2D array of shape ``(n_samples, n_classes)`` where each row sums to 1.

**Raises:** ``ValueError`` if called on a regression model. ``NotImplementedError`` if the best model is a quantum kernel.

---

score
^^^^^

.. code-block:: python

   result = automl.score(X, y)

Compute accuracy (classification) or R² (regression) on new data.

**Returns:** float

---

leaderboard
^^^^^^^^^^^

.. code-block:: python

   automl.leaderboard(top_n=15)

Print a ranked table of all evaluated models. Best model is highlighted.

---

best_circuit_summary
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   automl.best_circuit_summary()

Print a gate-by-gate breakdown of the winning circuit, including gate name,
target wires, and whether each gate is trainable.

---

export_qasm
^^^^^^^^^^^

.. code-block:: python

   qasm_str = automl.export_qasm()

Export the best variational circuit as an OpenQASM 2.0 string.

**Returns:** str

**Raises:** ``ValueError`` if the best model is a quantum kernel (kernels have no circuit to export).

---

export_qasm_to_file
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   automl.export_qasm_to_file(path='best_circuit.qasm')

Save the QASM export directly to a file.

---

save
^^^^

.. code-block:: python

   automl.save(path='qkabrine_automl_model.pkl')

Serialise the fitted model to disk. Works for both variational and kernel models.

---

load
^^^^

.. code-block:: python

   loaded = QkabrineAutoML.load(path='qkabrine_automl_model.pkl')

Load a previously saved model. Class method — call on the class, not an instance.

.. code-block:: python

   # Correct usage
   loaded = QkabrineAutoML.load('my_model.pkl')
   preds = loaded.predict(X_test)

---

Search Strategies
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Description
   * - ``'bayesian'``
     - **Default.** Gaussian Process + Expected Improvement. Builds a surrogate model of performance and proposes the most promising candidates. Best sample efficiency.
   * - ``'evolutionary'``
     - Genetic algorithm with crossover and mutation across generations. Good for large search spaces.
   * - ``'successive_halving'``
     - HyperBand-inspired. Starts many candidates with low budget, eliminates the worst half each round, doubles budget for survivors. Efficient when training is expensive.
   * - ``'grid'``
     - Exhaustive enumeration of all combinations. Best for small search spaces.
   * - ``'random'``
     - Random sampling within the budget. Quick baseline.

---

Circuit Architectures (12 Ansätze)
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Description
   * - ``strongly_entangling``
     - High expressibility. CNOT ring with full rotations. Strong entanglement across all qubit pairs.
   * - ``hardware_efficient``
     - Low circuit depth. Designed to run well on real NISQ hardware with limited connectivity.
   * - ``data_reuploading``
     - Data re-encoding between trainable layers. Increases effective expressibility per qubit.
   * - ``simplified_two_design``
     - Near-Haar random coverage. Good baseline expressibility with fewer parameters.
   * - ``all_to_all``
     - Maximum qubit connectivity. Every qubit entangled with every other.
   * - ``cascading``
     - QFT-inspired multi-scale entanglement. Long-range correlations.
   * - ``criss_cross``
     - Butterfly-pattern gate arrangement. Spreads information across qubits efficiently.
   * - ``ring_of_cnots``
     - Dense parameterisation with ring CNOT connectivity.
   * - ``full_rotation``
     - Maximum rotation freedom with RX, RY, RZ on every qubit each layer.
   * - ``alternating_rx_ry``
     - Alternating RX and RY layers. Layer diversity reduces parameter correlation.
   * - ``shallow_rx``
     - Minimal depth. Barren-plateau resistant due to shallow structure.
   * - ``hadamard_entangling``
     - Superposition-first design. Hadamard initialisation before entanglement.

---

Data Encodings
--------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Encoding
     - Description
   * - ``'angle'``
     - Each feature encoded as an RX rotation angle. Simple and fast.
   * - ``'angle_yz'``
     - Features encoded as alternating RY and RZ rotations. More expressive than angle.
   * - ``'iqp'``
     - Instantaneous Quantum Polynomial encoding. Uses Hadamard + RZ + CNOT structure. Strong for kernel methods.
   * - ``'amplitude'``
     - Encodes all features into the amplitude vector of the quantum state. Requires 2^n amplitudes for n qubits.

---

Quantum Kernel Classes
-----------------------

QuantumKernel
~~~~~~~~~~~~~

Computes quantum kernel matrices using circuit fidelity:
``k(x_i, x_j) = |⟨ψ(x_i)|ψ(x_j)⟩|²``

.. code-block:: python

   from qkabrine_automl import QuantumKernel

   kernel = QuantumKernel(n_qubits=4, embedding='iqp', n_layers=1)
   K = kernel.compute_kernel_matrix(X_train)       # train kernel matrix
   K_test = kernel.compute_kernel_matrix(X_test, X_train)  # test kernel matrix

QuantumKernelClassifier
~~~~~~~~~~~~~~~~~~~~~~~

Quantum kernel + classical SVM for classification.

.. code-block:: python

   from qkabrine_automl import QuantumKernelClassifier

   model = QuantumKernelClassifier(n_qubits=4, embedding='iqp', C=1.0)
   model.fit(X_train, y_train)
   preds = model.predict(X_test)
   acc = model.score(X_test, y_test)

QuantumKernelRegressor
~~~~~~~~~~~~~~~~~~~~~~

Quantum kernel + classical SVR for regression.

.. code-block:: python

   from qkabrine_automl import QuantumKernelRegressor

   model = QuantumKernelRegressor(n_qubits=4, embedding='iqp', C=1.0)
   model.fit(X_train, y_train)
   preds = model.predict(X_test)

---

Training Dynamics
-----------------

DataQuantumFisherMetric
~~~~~~~~~~~~~~~~~~~~~~~

Pre-screen circuits for trainability using the Data Quantum Fisher Information Matrix.

.. code-block:: python

   from qkabrine_automl import DataQuantumFisherMetric

   dqfim = DataQuantumFisherMetric(n_qubits=4, n_samples=20, seed=42)
   metrics = dqfim.predict_generalization(circuit_fn, X, n_params=12)
   print(f"Trainability score: {metrics.trainability_score:.3f}")
   # Score < 0.05 indicates likely barren plateau — skip this circuit

BarrenPlateauMonitor
~~~~~~~~~~~~~~~~~~~~

Monitor gradient variance during training. Stops early if gradients vanish.

.. code-block:: python

   from qkabrine_automl import BarrenPlateauMonitor

   monitor = BarrenPlateauMonitor(threshold=1e-7, window=3, auto_surgery=True)
   monitor.update(gradient_estimates)
   if monitor.plateau_detected:
       print("Barren plateau detected — stopping early")

QuantumNaturalGradient
~~~~~~~~~~~~~~~~~~~~~~

Quantum natural gradient optimizer using the quantum geometric tensor.

.. code-block:: python

   from qkabrine_automl import QuantumNaturalGradient

   qng = QuantumNaturalGradient(learning_rate=0.01, regularisation=1e-3)

---

Exceptions
----------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Exception
     - When it is raised
   * - ``QkabrineError``
     - Base exception for all Qkabrine errors.
   * - ``TrainingFailureError``
     - A single candidate circuit failed during training.
   * - ``SearchExhaustedError``
     - Every candidate failed — no valid model was found.
   * - ``InvalidCircuitError``
     - A circuit configuration is invalid.
