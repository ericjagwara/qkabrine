⚛️ Qkabrine AutoML Documentation
=================================

**Automatic Quantum Machine Learning** — intelligent search for the best quantum
circuit, encoding, and hyperparameters for your data.

.. code-block:: bash

   pip install qkabrine-automl

.. code-block:: python

   from qkabrine_automl import QkabrineAutoML

   automl = QkabrineAutoML(task='classification', search_strategy='bayesian')
   automl.fit(X_train, y_train)
   automl.leaderboard()
   preds = automl.predict(X_test)

---

Qkabrine AutoML searches over **circuit architectures**, **data encodings**,
**model paradigms** (variational circuits and quantum kernels), and
**hyperparameters** simultaneously — so you don't have to.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 🚀 Getting Started
      :link: getting_started
      :link-type: doc

      Install the package and run your first quantum AutoML search in minutes.

   .. grid-item-card:: 📖 API Reference
      :link: api
      :link-type: doc

      Complete reference for all classes, methods, and parameters.

   .. grid-item-card:: 💡 Examples & Tutorials
      :link: examples
      :link-type: doc

      Step-by-step examples for classification, regression, kernels, and more.

   .. grid-item-card:: ❓ FAQ
      :link: faq
      :link-type: doc

      Answers to the most common questions.

---

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   getting_started
   examples
   api
   faq

.. toctree::
   :maxdepth: 1
   :caption: Project Links
   :hidden:

   PyPI <https://pypi.org/project/qkabrine-automl/>
   GitHub <https://github.com/ericjagwara/qkabrine>
   Author <https://ericjagwara.online>
   Solid Elf Labs <https://solidelf.org>
