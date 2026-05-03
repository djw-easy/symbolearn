# **Transparent class-specific spectral indices via data-driven symbolic learning: Toward interpretable land-cover mapping**

**Authors:** Junwu Dong, Tao Pei, Ci Song, Xi Wang, Yahui Qi, Dayu Cheng, Yanmin Jiang

---

This repository provides the official implementation of the paper:

**"Transparent class-specific spectral indices via data-driven symbolic learning: Toward interpretable land-cover mapping."**

The code implements our data-driven symbolic learning framework for discovering transparent, class-specific spectral indices that support both single-index threshold segmentation and multiclass land-cover classification.

![](./figs/framework.jpg)

![](./figs/index_set_simulated_compare.png)

---

## ⭐ Features

- End-to-end symbolic discovery framework based on genetic programming
- Class-specific spectral indices with full interpretability
- **Dynamic Spectral Aggregation Terminals (DSATs)** — terminal nodes that aggregate adjacent spectral bands on demand
- **Complexity-guided population initialization** — seeds the initial population with expressions spanning a wide range of structural complexity
- **Hinge Loss-based expression-set optimization** — jointly optimizes a set of expressions, a key distinguishing feature of this algorithm that enables coherent multiclass discrimination
- Support for both multispectral and hyperspectral datasets
- Reproduction of all experiments reported in the paper

### 🔧 Evolutionary Engine

The GP engine is inspired by [SymbolicRegression.jl](https://github.com/astroautomata/SymbolicRegression.jl) and shares many of its design characteristics (e.g., typed function sets, hall-of-fame archiving). The key novel contributions over that baseline are:

- **DSATs**: terminals that adaptively aggregate neighboring spectral channels, enabling the discovery of indices that would be impractical to express with raw bands alone
- **Complexity-guided initialization**: seeds the population across the full spectrum of expression complexity, improving exploration without sacrificing diversity

Performance is lower than a compiled Julia implementation, but the NumPy/Numba backend provides reasonable speed for datasets of typical size.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

The core library requires only NumPy, pandas, scipy, scikit-learn, joblib, and tqdm. Numba is optional — if installed, computationally intensive fitness functions are JIT-compiled for a significant speedup; if absent, the pure-NumPy fallback is used transparently.

---

## 🚀 Quick Start

```python
from src.symbolic_estimators import SymbolicClassifier, SymbolicRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SymbolicClassifier(
    n_classes=2,
    feature_names=[f"B{i}" for i in range(X.shape[1])],
    population_size=500,
    n_generations=100,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"Test Accuracy: {model.score(X_test, y_test):.4f}")
print(f"Best expression: {model.best_expression_}")
```

For a full walkthrough, see [`example.ipynb`](./example.ipynb).

---

## 📁 Dataset

### Included datasets

The repository includes example datasets under `./example_data`:

- **Poyang Lake dataset**
- **Multiclass-L8 dataset**

### External hyperspectral datasets

The hyperspectral datasets used in the paper (e.g., **Pavia University**, **Salinas**) can be downloaded from:

👉 [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)

Due to data size limitations, a preview of the Pavia University dataset is included in:

```
./example_data/hyperspectral
```

---

## 📂 Repository Structure

```
.
├── example_data/          # Example multispectral & hyperspectral data
├── figs/                   # Figures and diagrams used in README/paper
├── src/                    # Core implementation
│   ├── base.py             # Base estimator class
│   ├── expression.py        # Expression (chromosome) representation
│   ├── fitness.py          # Fitness evaluation
│   ├── generator.py        # Expression generator (complexity-guided)
│   ├── gpoperator.py       # GP operators (crossover, mutation, selection)
│   ├── halloffame.py       # Hall-of-fame archiving
│   ├── log.py              # Logging utilities
│   ├── metrics/            # Metric functions (classification, regression, transformer)
│   ├── node.py             # Node representation
│   ├── population.py       # Population management
│   ├── symbolic_estimators.py  # High-level Estimator API
│   ├── tree.py             # Tree representation
│   ├── tree_parser.py      # String-to-tree parser
│   └── utils.py            # Utility functions
├── tests/                  # Unit tests
├── example.ipynb           # Main demo notebook
├── train_sc.py             # Training script for hyperspectral classification
├── test.py                 # Lightweight smoke test
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 📧 Contact

If you have questions or encounter issues, feel free to contact:

**Junwu Dong**
Email: *[djw@lreis.ac.cn](mailto:djw@lreis.ac.cn)*

---

## 📚 Citation

If you use this code in your research, please cite our paper:

```
TBD — bibtex entry will be added upon publication
```

---

## 📄 License

This project is for academic use. Please contact the authors for commercial licensing.
