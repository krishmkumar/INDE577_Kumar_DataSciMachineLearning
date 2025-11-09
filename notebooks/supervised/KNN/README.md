# K-Nearest Neighbors (KNN)

This notebook provides a hands-on implementation of the **K-Nearest Neighbors (KNN)** algorithm from scratch, demonstrating both its mathematical foundation and practical application using the classic **Iris dataset**.

---

## 📘 Overview

In this notebook (`knn_from_scratch.ipynb`), we explore a simple yet powerful non-parametric model — **K-Nearest Neighbors (KNN)**.

At its core, KNN makes predictions by identifying the *k* most similar observations (neighbors) and using them to infer the outcome for a new data point.  
- For **classification**, it predicts the **majority class** among neighbors.  
- For **regression**, it predicts the **average value**.

Unlike models such as linear or logistic regression, KNN has **no explicit training phase** — it stores the training data and uses a **distance metric** (commonly Euclidean distance) to make predictions at inference time.  
This makes KNN intuitive and flexible, though potentially slower for large datasets.

---

## 🧩 Learning Objectives

By completing this notebook, we aim to:

- **Implement** the KNN algorithm from scratch to understand its inner mechanics.  
- **Apply** KNN to the **Iris flower dataset** to classify species based on sepal and petal measurements.  
- **Experiment** with different values of *k* and distance metrics to observe their effect on accuracy and decision boundaries.  
- **Compare** our custom KNN implementation to `sklearn`’s `KNeighborsClassifier` for validation.  
- **Evaluate** KNN’s strengths and limitations — including interpretability, sensitivity to scaling, and computational efficiency.

---

## ⚙️ Implementation Details

This notebook leverages functions from the custom `rice2025` package, including:

```python
from rice2025.supervised.knn import KNN
from rice2025.metrics import accuracy, confusion_matrix_custom
from rice2025.preprocess import StandardScaler
