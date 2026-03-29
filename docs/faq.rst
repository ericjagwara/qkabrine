FAQ
===

.. contents:: On this page
   :local:
   :depth: 1

---

Installation
------------

**Why do I get an error when running ``pip install qkabrine-automl``?**

Make sure you have Python 3.9 or later:

.. code-block:: bash

   python --version

If you're on an older version, upgrade Python first from `python.org <https://python.org/downloads>`_.

---

**PennyLane fails to install — what do I do?**

Try installing it separately first:

.. code-block:: bash

   pip install pennylane
   pip install qkabrine-automl

If you're on Windows and get a build error, make sure you have the latest pip:

.. code-block:: bash

   python -m pip install --upgrade pip
   pip install qkabrine-automl

---

Running the Search
------------------

**How long does a search take?**

It depends on your settings. A rough guide:

- ``train_steps=20``, ``max_layers=1``, 4 qubits → **2–5 minutes**
- ``train_steps=40``, ``max_layers=3``, 6 qubits → **10–30 minutes**
- ``train_steps=100``, ``max_layers=5``, 8 qubits → **1–3 hours**

To speed things up: reduce ``train_steps``, reduce ``max_layers``, reduce ``n_qubits``,
or set a ``time_budget`` in seconds.

.. code-block:: python

   automl = QkabrineAutoML(
       task='classification',
       train_steps=20,
       max_layers=2,
       time_budget=300,   # stop after 5 minutes no matter what
   )

---

**The search is running but accuracy is very low — what's wrong?**

A few things to check:

1. **Too few training steps** — try increasing ``train_steps`` to 80 or 100
2. **Wrong encoding** — try adding more encodings: ``encodings=('angle', 'iqp', 'angle_yz')``
3. **Data not normalised** — Qkabrine normalises automatically, but check your data has no extreme outliers
4. **Too few qubits** — try increasing ``n_qubits``

---

**I get ``SearchExhaustedError`` — what does that mean?**

It means every single candidate circuit failed during training. This usually happens when:

- Your data has only 1 class (classification requires at least 2)
- Your dataset is very small (need at least 5 samples)
- There is a conflict between your encoding and qubit count (e.g. ``'amplitude'`` encoding requires the number of features to be a power of 2)

Try running with ``verbose=True`` to see exactly which circuits are failing and why.

---

**Can I use this on a real quantum computer?**

Yes. Export your best circuit to QASM and submit it to any hardware provider:

.. code-block:: python

   automl.export_qasm_to_file('my_circuit.qasm')

The file works with IBM Quantum, Amazon Braket, and any provider accepting OpenQASM 2.0.
For IBM specifically, install the IBM backend:

.. code-block:: bash

   pip install qkabrine-automl[ibm]

---

Data and Features
-----------------

**My dataset has 100 features but only 4 qubits — what happens?**

Qkabrine automatically reduces your features to match the qubit count using
the method set by ``feature_reduction``. The default is PCA:

.. code-block:: python

   automl = QkabrineAutoML(
       n_qubits=4,
       feature_reduction='pca',   # reduces 100 features → 4 principal components
   )

Other options: ``'random_projection'``, ``'select_variance'``, or ``None``
(which just takes the first N features — not recommended for large feature sets).

---

**Does Qkabrine handle categorical features?**

No — it expects numerical input only. Encode your categorical features first
using scikit-learn's ``LabelEncoder`` or ``OneHotEncoder`` before passing data
to Qkabrine.

---

**My labels are strings like "cat" and "dog" — is that OK?**

Yes. Qkabrine's ``LabelEncoder`` handles string labels automatically.
``predict()`` will return the original string labels back.

---

Models and Results
------------------

**How do I know which model won — variational or kernel?**

After fitting, check:

.. code-block:: python

   print(automl._best['model_type'])   # 'variational' or 'kernel'
   print(automl._best['name'])         # e.g. 'hardware_efficient' or 'kernel_iqp'

Or just call ``automl.leaderboard()`` — the Type column shows ``var`` or ``kernel``.

---

**Can I save and reload my model?**

Yes — save/load works for both variational and kernel models:

.. code-block:: python

   automl.save('my_model.pkl')

   # Later, in any script:
   loaded = QkabrineAutoML.load('my_model.pkl')
   preds = loaded.predict(X_new)

---

**``predict_proba`` is raising an error — why?**

Two common reasons:

1. You're calling it on a regression model — it's only valid for classification
2. The best model is a quantum kernel — kernel models don't support ``predict_proba``

Check with:

.. code-block:: python

   print(automl.task)                    # must be 'classification'
   print(automl._best['model_type'])     # must not be 'kernel'

---

**The leaderboard shows many models with the same score — is that normal?**

Yes, especially for small datasets where many circuits hit the ceiling
(e.g. 100% accuracy on a simple dataset) or the floor (50% random chance).
The first one listed among ties is selected as best. You can inspect all
tied models in ``automl._results``.

---

Advanced Usage
--------------

**What is a barren plateau and should I worry about it?**

A barren plateau is when gradients become so small that training makes
no progress — the circuit's loss landscape becomes completely flat.
It's more common with deep circuits and many qubits.

Enable monitoring to detect and skip affected circuits automatically:

.. code-block:: python

   automl = QkabrineAutoML(
       monitor_barren_plateaus=True,
       use_dqfim_prescreening=True,
   )

---

**What is DQFIM prescreening?**

DQFIM (Data Quantum Fisher Information Matrix) is a technique that
estimates how trainable a circuit is before spending time actually
training it. Circuits with a trainability score below 0.05 are skipped.

Enable it with ``use_dqfim_prescreening=True``. It adds a small overhead
per candidate but can save significant time on deep circuits.

---

**Can I search over multiple encodings at once?**

Yes — pass a tuple:

.. code-block:: python

   automl = QkabrineAutoML(
       encodings=('angle', 'iqp', 'angle_yz'),
   )

Each circuit architecture will be tested with each encoding combination,
so the search space grows. Consider also setting a ``time_budget``.

---

Getting Help
------------

- **Bug reports:** `github.com/ericjagwara/qkabrine/issues <https://github.com/ericjagwara/qkabrine/issues>`_
- **Author:** `ericjagwara.online <https://ericjagwara.online>`_
- **Lab:** `solidelf.org <https://solidelf.org>`_
