# Survival Experiments Module

This folder contains scripts, utilities, and notebooks to perform survival analysis on embeddings derived from clinical text representations. The analysis evaluates various survival models across different experimental subsets using deep learning-based survival models.

---

## 📁 Folder Structure

```
📄 environment_survival.yml        # Conda environment for survival modeling
📄 survival_models.py              # SurvivalModel class with DeepSurv and DeepHit support
📄 survival_tts_ds.ipynb           # Analysis using DeepSeek-based embeddings
📄 survival_tts_l33.ipynb          # Analysis using LLaMA3-based embeddings
📄 utils.py                        # Survival evaluation metrics (concordance, Brier score, AUC)
📁 pycox/                          # Local fork/copy of pycox
📁 torchtuples/                    # Local fork/copy of torchtuples
📁 tts/
    📁 DS_embeddings/              # Embeddings from DeepSeek encoder
    📁 L33_embeddings/             # Embeddings from LLaMA3 encoder
```

---

## 🧠 Deep Learning Survival Models

The `SurvivalModel` class in `survival_models.py` provides a unified interface for training and evaluating the following models:

- **DeepSurv** (via [PyCox](https://github.com/havakv/pycox)):  
  A deep feedforward neural network trained using the Cox partial likelihood.

- **DeepHit** (via [PyCox](https://github.com/havakv/pycox)):  
  A flexible neural network-based approach for direct estimation of survival probabilities via a multinomial likelihood.

Key functionalities:
- Fitting models with configurable hyperparameters
- Predicting survival curves
- Computing median or mean survival time estimates
- Evaluation with concordance index and Brier score

---

## 📘 Notebooks

- **`survival_tts_ds.ipynb`**  
  Applies survival models on DeepSeek-derived textual time series embeddings.

- **`survival_tts_l33.ipynb`**  
  Same pipeline applied to LLaMA3-derived textual time series embeddings.

Each notebook includes:
- Loading of pretrained embeddings
- Model training and evaluation
- Visualization of survival curves

---

## 📊 Evaluation Utilities

`utils.py` provides a set of evaluation functions:

- `get_concordance_score`: Time-dependent concordance index (Antolini)
- `get_integrated_brier_score`: Integrated Brier Score (IBS)
- `compute_cumulative_hazard_vectorized`: Converts survival curves to cumulative hazards
- `CensoringDistributionEstimator`: Custom Kaplan-Meier-based censoring model

These utilities are fully compatible with `pycox`-based neural survival models.

---

## 📦 Environment Setup

Install the conda environment with:

```bash
conda env create -f environment_survival.yml
conda activate pmoa_tts_survival
```

Major dependencies:
- `pycox`
- `torchtuples`
- `lifelines`
- `scikit-survival`
- `torch`, `pandas`, `numpy`

---

## 🧪 Data Inputs

Precomputed embeddings used in analysis are located in:

- `tts/DS_embeddings/`: Embeddings from DeepSeek-based models
- `tts/L33_embeddings/`: Embeddings from LLaMA3-based models

Each subfolder contains a `README.md` with details about the embedding format.

---

## 🚀 Usage

Once the environment is ready, launch the Jupyter notebooks:

```bash
jupyter notebook survival_tts_ds.ipynb
```

Follow the notebook steps to:
- Load embeddings
- Train DeepSurv or DeepHit models
- Evaluate performance and visualize survival curves

---

## 🔍 Notes

- Time-dependent AUC is implemented but optional due to computational cost.
- This modular framework can be extended to other textual embedding types or survival objectives.
