Examples & Tutorials
====================

.. contents:: On this page
   :local:
   :depth: 1

---

Example 1 — Binary Classification
-----------------------------------

Classifying cancer data with a Bayesian search over angle and IQP encodings.

.. code-block:: python

   from sklearn.datasets import load_breast_cancer
   from sklearn.model_selection import train_test_split
   from qkabrine_automl import QkabrineAutoML

   # Load data
   X, y = load_breast_cancer(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Create and fit
   automl = QkabrineAutoML(
       task='classification',
       n_qubits=4,               # use 4 qubits
       max_layers=2,             # try up to 2 layers per circuit
       search_strategy='bayesian',
       encodings=('angle', 'iqp'),
       feature_reduction='pca',  # reduce 30 features → 4 qubits via PCA
       train_steps=50,
       verbose=True,
   )
   automl.fit(X_train, y_train)

   # Results
   automl.leaderboard()
   print(f"Test accuracy: {automl.score(X_test, y_test):.4f}")

   # Inspect the winning circuit
   automl.best_circuit_summary()

**What to expect:** The Bayesian search will run ~25 candidates, taking
roughly 3–8 minutes depending on your machine. Typical accuracy on this
dataset is 0.90–0.96.

---

Example 2 — Multi-Class Classification
----------------------------------------

Three-class classification on the Iris dataset using an evolutionary search.

.. code-block:: python

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from qkabrine_automl import QkabrineAutoML

   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   automl = QkabrineAutoML(
       task='classification',
       n_qubits=4,
       search_strategy='evolutionary',
       encodings=('angle', 'angle_yz'),
       max_layers=3,
       train_steps=60,
   )
   automl.fit(X_train, y_train)

   # Predict class labels
   predictions = automl.predict(X_test)
   print(predictions)  # e.g. ['setosa', 'versicolor', 'virginica', ...]

   # Soft class probabilities
   probs = automl.predict_proba(X_test)
   print(probs.shape)  # (n_samples, 3)

**Note:** Multi-class uses ``qml.probs()`` with cross-entropy loss, which
trains all classes simultaneously rather than one-vs-rest.

---

Example 3 — Regression
------------------------

Predicting house prices with a quantum regression model.

.. code-block:: python

   from sklearn.datasets import fetch_california_housing
   from sklearn.model_selection import train_test_split
   from qkabrine_automl import QkabrineAutoML

   X, y = fetch_california_housing(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   automl = QkabrineAutoML(
       task='regression',
       n_qubits=6,
       search_strategy='bayesian',
       encodings=('angle',),
       feature_reduction='pca',
       train_steps=40,
       include_kernels=False,   # skip kernel methods for regression speed
   )
   automl.fit(X_train, y_train)

   print(f"R² score: {automl.score(X_test, y_test):.4f}")

---

Example 4 — Quantum Kernel Methods
------------------------------------

Letting the search compare variational circuits against quantum kernel SVMs.

.. code-block:: python

   from sklearn.datasets import load_wine
   from sklearn.model_selection import train_test_split
   from qkabrine_automl import QkabrineAutoML

   X, y = load_wine(return_X_y=True)
   # Keep it binary for this example
   mask = y < 2
   X, y = X[mask], y[mask]
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   automl = QkabrineAutoML(
       task='classification',
       n_qubits=4,
       search_strategy='grid',
       include_kernels=True,    # compare variational + kernel methods
       feature_reduction='pca',
   )
   automl.fit(X_train, y_train)
   automl.leaderboard()

   # Check which type won
   print(automl._best['model_type'])  # 'variational' or 'kernel'

---

Example 5 — Save, Load, and Deploy
------------------------------------

Train once, save the model, load it later for predictions.

.. code-block:: python

   # --- Train and save ---
   automl = QkabrineAutoML(task='classification', n_qubits=4)
   automl.fit(X_train, y_train)
   automl.save('my_quantum_model.pkl')

   # --- Load later (even in a different script) ---
   from qkabrine_automl import QkabrineAutoML

   loaded = QkabrineAutoML.load('my_quantum_model.pkl')
   predictions = loaded.predict(X_test)
   print(f"Accuracy: {loaded.score(X_test, y_test):.4f}")

---

Example 6 — Export to QASM
----------------------------

Export the best circuit to OpenQASM 2.0 for use on real quantum hardware.

.. code-block:: python

   automl = QkabrineAutoML(task='classification', n_qubits=4)
   automl.fit(X_train, y_train)

   # Print QASM to screen
   qasm_string = automl.export_qasm()
   print(qasm_string)

   # Or save to a file
   automl.export_qasm_to_file('best_circuit.qasm')

The exported QASM file can be submitted to IBM Quantum, Amazon Braket,
or any hardware provider that accepts OpenQASM 2.0.

---

Example 7 — Cross-Validation
------------------------------

More reliable evaluation using k-fold cross-validation.

.. code-block:: python

   automl = QkabrineAutoML(
       task='classification',
       n_qubits=4,
       search_strategy='bayesian',
       cv_folds=5,              # 5-fold stratified cross-validation
       train_steps=30,
   )
   automl.fit(X, y)             # pass the full dataset — no manual split needed

---

Example 8 — Noise-Aware Training
----------------------------------

Train with a noise model to prepare for real NISQ hardware.

.. code-block:: python

   automl = QkabrineAutoML(
       task='classification',
       n_qubits=4,
       noise_model='depolarizing',   # simulate gate errors
       noise_strength=0.01,          # 1% error rate per gate
       train_steps=60,               # more steps to compensate for noise
   )
   automl.fit(X_train, y_train)

Available noise models: ``'depolarizing'``, ``'bitflip'``,
``'phaseflip'``, ``'amplitude_damping'``.

---

Example 9 — Barren Plateau Detection
--------------------------------------

Monitor for vanishing gradients and stop early if detected.

.. code-block:: python

   automl = QkabrineAutoML(
       task='classification',
       n_qubits=6,
       monitor_barren_plateaus=True,   # watch gradient variance
       use_dqfim_prescreening=True,    # skip untrainable circuits early
       train_steps=80,
   )
   automl.fit(X_train, y_train)

When a barren plateau is detected mid-training you will see ``[BP@step]``
printed in the progress output, and training stops early for that candidate.

---

Example 10 — Analysing Circuit Quality
-----------------------------------------

Use the built-in expressibility and entanglement metrics.

.. code-block:: python

   from qkabrine_automl import (
       ANSATZ_REGISTRY,
       compute_expressibility,
       compute_entangling_capability,
       rank_ansatze,
   )

   # Rank all 12 ansätze by expressibility on 4 qubits
   rankings = rank_ansatze(n_qubits=4, n_layers=2)
   for name, expr, ent in rankings:
       print(f"{name:<30} expr={expr:.4f}  ent={ent:.4f}")

   # Check a specific ansatz
   arch = ANSATZ_REGISTRY['strongly_entangling'](n_qubits=4, n_layers=2)
   expr = compute_expressibility(arch, n_qubits=4, n_samples=200)
   print(f"Expressibility (KL divergence): {expr:.4f}")
