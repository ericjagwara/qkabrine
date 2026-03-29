Getting Started
===============

Installation
------------

Install Qkabrine AutoML from PyPI using pip:

.. code-block:: bash

   pip install qkabrine-automl

**Requirements:**

- Python 3.9 or later
- PennyLane >= 0.35.0
- NumPy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0

**Optional — IBM Quantum backend support:**

.. code-block:: bash

   pip install qkabrine-automl[ibm]

---

Your First Search
-----------------

Here is the simplest possible usage — three lines to train a quantum classifier:

.. code-block:: python

   from qkabrine_automl import QkabrineAutoML
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split

   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

   automl = QkabrineAutoML(task='classification')
   automl.fit(X_train, y_train)
   print(f"Accuracy: {automl.score(X_test, y_test):.4f}")

Qkabrine will automatically:

- Figure out how many qubits to use from your data
- Search through circuit architectures, encodings, and hyperparameters
- Pick the best model
- Print a live progress table as it searches

---

Understanding the Output
------------------------

When you call ``.fit()``, you will see a live search table like this:

.. code-block:: text

   ════════════════════════════════════════════════════════════════════
     ⚛️  Qkabrine AutoML  v2.1
   ════════════════════════════════════════════════════════════════════
     Task       : classification
     Samples    : 120
     Features   : 4 → 4 qubits
     Classes    : 3
     Search     : bayesian (25 candidates)
     Encodings  : angle
     Steps      : 40
     Optimizer  : adam
   ════════════════════════════════════════════════════════════════════
     Testing  strongly_entangling    L=1  enc=angle    → acc=0.8333  ████████████████░░░░  (4.2s)
     Testing  hardware_efficient     L=1  enc=angle    → acc=0.9167  ██████████████████░░  (3.8s)
     ...

Each row shows the circuit name, depth (L=layers), encoding, accuracy score,
a visual progress bar, and how long it took.

After the search finishes, call ``.leaderboard()`` to see all results ranked:

.. code-block:: python

   automl.leaderboard()

.. code-block:: text

   ══════════════════════════════════════════════════════════════════════════════
     ⚛️ Qkabrine AutoML Leaderboard   [Accuracy ↑]
   ══════════════════════════════════════════════════════════════════════════════
   Rank  Model                     Type      Enc       Params  Accuracy  Time(s)
   ──────────────────────────────────────────────────────────────────────────────
   🥇    hardware_efficient(L=2)   var       angle     12      0.9583    6.1
   🥈    strongly_entangling(L=1)  var       angle     8       0.9167    4.2
   🥉    kernel_iqp                kernel    iqp       0       0.9167    12.3
   ...

---

Key Concepts
------------

**Qubits**

A qubit is the quantum equivalent of a bit. Qkabrine maps your data features
onto qubits. By default it uses one qubit per feature, up to a maximum of 10.
If your data has more features than qubits, it automatically reduces dimensions
using PCA.

**Ansatz / Circuit Architecture**

An ansatz is a template for a quantum circuit — a specific arrangement of
quantum gates. Qkabrine searches over 12 different ansätze to find which
works best for your data.

**Data Encoding**

Before a quantum circuit can process your data, the numbers have to be
"loaded" into the quantum state. Different encoding strategies (angle, IQP,
amplitude) can dramatically affect performance. Qkabrine searches over these
automatically.

**Variational vs Kernel**

Qkabrine searches two types of quantum models:

- **Variational circuits** — trainable quantum circuits optimized by gradient descent
- **Quantum kernel methods** — use quantum circuits to compute similarity between data points, then feed into a classical SVM

Both are evaluated and the best one wins.

---

Next Steps
----------

- See :doc:`examples` for detailed worked examples
- See :doc:`api` for the full parameter reference
- See :doc:`faq` if something isn't working as expected
