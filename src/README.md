# `src/` — Core Source Code for the `rice2025` Machine Learning Package

This directory contains the full implementation of the `rice2025` machine learning library developed for INDE 577 at Rice University. The package provides clean, modular, from-scratch implementations of machine learning algorithms using only NumPy.

The modules in this directory support all example notebooks in `notebooks/` and are covered by the automated test suite in `tests/`.

-------------------------------------------------------------------------------

## Design Goals

1. Implement algorithms from first principles  
   All logic is written manually using NumPy (no scikit-learn or SciPy).  

2. Maintain a consistent API  
   Models follow scikit-learn-style conventions: fit(), predict(), fit_predict(), score().

3. Keep the codebase readable  
   Each algorithm is structured to be understandable and easy to extend.

4. Provide deterministic behavior  
   When randomness is involved, operations can be made deterministic using fixed seeds.

5. Ensure testability  
   Tests validate behavior, shapes, edge cases, and numerical correctness.

-------------------------------------------------------------------------------

## Package Structure Overview

    rice2025/
    │
    ├── supervised_learning/              # Supervised ML algorithms
    │   ├── linear_regression.py          # Closed-form OLS + GD variant
    │   ├── logistic_regression.py        # Binary logistic regression via GD
    │   ├── knn.py                        # k-nearest neighbors classifier
    │   ├── perceptron.py                 # Online perceptron algorithm
    │   ├── multilayer_perceptron.py      # Feedforward neural network
    │   ├── decision_tree.py              # Classification tree (Gini)
    │   └── regression_tree.py            # Regression tree (MSE-based splits)
    │
    ├── unsupervised_learning/            # Unsupervised ML algorithms
    │   ├── kmeans.py                     # Lloyd’s algorithm + inertia tracking
    │   ├── dbscan.py                     # Density-based clustering
    │   ├── pca.py                        # PCA via covariance eigendecomposition
    │   └── community_detection.py        # Label propagation for graph clustering
    │
    ├── utils/                            # Shared utilities
    │   ├── metrics.py                    # Accuracy, MSE, cross-entropy, silhouette
    │   ├── preprocessing.py              # Encoding + preprocessing helpers
    │   ├── scaling.py                    # StandardScaler, MinMaxScaler
    │   ├── train_test_split.py           # Deterministic data splitting
    │   └── math_utils.py                 # Distance metrics, numerical stability ops
    │
    └── __init__.py                       # Exposes the package namespace

-------------------------------------------------------------------------------

## Module Responsibilities

supervised_learning/  
Implements core predictive models using vectorized math and clear algorithmic structure.  
Tree implementations use recursive splitting.  
MLP includes forward and backward propagation.

unsupervised_learning/  
Implements clustering and dimensionality reduction algorithms.  
KMeans includes inertia minimization; DBSCAN includes core/border/noise labeling.  
PCA uses covariance eigendecomposition; label propagation performs iterative community detection.

utils/  
Utility modules shared across the package.  
Provides metrics, scaling, preprocessing, numerical helpers, and splitting functions.

-------------------------------------------------------------------------------

## Importing the Package

After installation:

    pip install -e .

Example imports:

    from rice2025.supervised_learning.linear_regression import LinearRegression
    from rice2025.unsupervised_learning.kmeans import KMeans
    from rice2025.utils.metrics import accuracy


## Summary

The `src/` directory contains the core implementation of the `rice2025` ML package.  
It is designed to be modular, testable, readable, and easily extensible for future models.
