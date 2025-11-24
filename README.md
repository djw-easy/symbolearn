# **Transparent class-specific spectral indices via data-driven symbolic learning: Toward interpretable land-cover mapping**

**Authors:** Junwu Dong, Tao Pei, Ci Song, Xi Wang, Yahui Qi, Dayu Cheng, Yanmin Jiang

---

This repository provides the official implementation of the paper:

**“Transparent class-specific spectral indices via data-driven symbolic learning: Toward interpretable land-cover mapping.”**

The code implements our data-driven symbolic learning framework for discovering transparent, class-specific spectral indices that support both single-index threshold segmentation and multiclass land-cover classification.

![](./figs/framework.jpg)

![](./figs/index_set_simulated_compare.png)

---

## **⭐ Features**

* End-to-end symbolic discovery framework based on genetic programming
* Class-specific spectral indices with full interpretability
* Dynamic spectral aggregation terminals (DSATs)
* Synergistic Softmax–Focal Loss optimization for multiclass expression sets
* Support for both multispectral and hyperspectral datasets
* Reproduction of all experiments reported in the paper

---

## **📦 Environment**

The experiments were conducted with the following environment:

```
python = 3.13.5
scikit-learn = 1.7.1
scipy = 1.16.0
jax = 0.8.0
joblib = 1.5.1
numba = 0.61.2
numpy = 2.2.5
pandas = 2.3.1
```

> We recommend using a virtual environment (conda or venv) to avoid version conflicts.

---

## **📁 Dataset**

### Included datasets

The repository includes example datasets under `./example_data`:

* **Poyang Lake dataset**
* **Multiclass-L8 dataset**

### External hyperspectral datasets

The hyperspectral datasets used in the paper (e.g., **Pavia University**, **Salinas**) can be downloaded from:

👉 [Hyperspectral Remote Sensing Scenes](https://www.ehu.eus/ccwintco/index.php?title=Hyperspectral_Remote_Sensing_Scenes)

Due to data size limitations, a preview of the Pavia University dataset is included in:

```
./example_data/hyperspectral
```

---

## **🚀 Usage**

Main experiments can be reproduced using the interactive notebook:

```
example.ipynb
```

The notebook provides:

* Step-by-step demonstrations
* Visualization of symbolic expressions and classification maps
* Examples of single-index discovery and multiclass index-set evolution
* Clear documentation of parameter settings used in the paper

---

## **📂 Repository Structure**

```
.
├── example_data/        # Example multispectral & hyperspectral data
├── figs/                # Figures and diagrams used in README/paper
├── src/                 # Core symbolic learning and GP implementation
├── example.ipynb        # Main demo notebook
└── README.md            # Project documentation
```

---

## **📧 Contact**

If you have questions or encounter issues, feel free to contact:

**Junwu Dong**
Email: *[djw@lreis.ac.cn](mailto:djw@lreis.ac.cn)*

