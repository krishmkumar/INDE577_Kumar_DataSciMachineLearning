# `rice2025` — Machine Learning Package (Source Documentation)

`rice2025` is a modular, from-scratch machine learning package implemented entirely in NumPy.  
It provides core supervised and unsupervised learning algorithms, along with utilities for preprocessing, metrics, scaling, and model evaluation.

This README documents the **Python package itself**, its structure, import paths, and internal design conventions.  
For project-wide documentation, see the repository’s main README.

---

## 🔧 Installation (Editable Mode)

From the root of the repository:

    pip install -e .

This makes `rice2025` available as a standard Python package:

    import rice2025

---

## 📦 Package Structure

The package follows a clean, scikit-learn–inspired modular layout:

    rice2025/
    │
    ├── supervised_learning/              # Supervised ML algorithms
    │   ├── linear_regression.py
    │   ├── logistic_regression.py
    │   ├── knn.py
    │   ├── perceptron.py
    │   ├── multilayer_perceptron.py
    │   ├── decision_tree.py
    │   └── regression_tree.py
    │
    ├── unsupervised_learning/            # Unsupervised ML algorithms
    │   ├── kmeans.py
    │   ├── dbscan.py
    │   ├── pca.py
    │   └── community_detection.py
    │
    ├── utils/                            # Shared utilities
    │   ├── metrics.py
    │   ├── preprocessing.py
    │   ├── scaling.py
    │   ├── train_test_split.py
    │   └── math_utils.py
    │
    ├── basic_functions.py                # Internal helpers (legacy support)
    └── __init__.py                       # Exposes the public namespace

---

## 🧠 Design Philosophy

The package is designed to:

- Provide **fully transparent implementations** of classical ML algorithms  
- Follow **consistent APIs** (`fit`, `predict`, `fit_predict`, `score`)  
- Stay **lightweight** — only NumPy is required  
- Be **modular** and easy to extend  
- Support **testing** and reproducibility  

The implementations prioritize readability and clarity rather than heavy optimization.

---

## 🚀 Usage Examples

### **Linear Regression**
    from rice2025.supervised_learning.linear_regression import LinearRegression

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

### **KMeans Clustering**
    from rice2025.unsupervised_learning.kmeans import KMeans

    km = KMeans(k=3, max_iter=100)
    labels = km.fit_predict(X)

### **Scaling & Metrics**
    from rice2025.utils.scaling import StandardScaler
    from rice2025.utils.metrics import accuracy

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    accuracy(y_true, y_pred)

---

## 📚 Module Responsibilities

### **Supervised Learning**
Predictive models including:
- Linear Regression  
- Logistic Regression  
- KNN Classifier  
- Perceptron  
- Feedforward MLP  
- Decision Trees  
- Regression Trees  

### **Unsupervised Learning**
Clustering & dimensionality reduction:
- KMeans  
- DBSCAN  
- PCA  
- Label Propagation (Community Detection)  

### **Utilities**
Shared tools across algorithms:
- Scaling / normalization  
- Categorical preprocessing  
- Metrics for evaluation  
- Distance functions  
- Train/test splitting  

---

## ➕ Extending the Package

To add your own algorithm:

1. Create a file inside `supervised_learning/` or `unsupervised_learning/`.  
2. Implement `fit()` and `predict()` consistently.  
3. Reuse utilities from `utils/` when possible.  
4. Add a test in the `tests/` directory.  

The structure is intentionally simple to encourage extensibility.

---

## 📌 Summary

This folder contains the **core Python package** supporting the entire project:  
a clean, modular set of machine learning algorithms built from scratch and ready to import.

Use this README as a reference when navigating or extending the `rice2025` codebase.
