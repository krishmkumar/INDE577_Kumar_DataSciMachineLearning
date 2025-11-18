# Supervised Learning Examples

This directory contains example Jupyter notebooks demonstrating supervised learning algorithms implemented from scratch in the `rice2025` package.  
Each notebook includes data loading, preprocessing, model training, evaluation, and interpretation.

The purpose of these examples is to provide a clear, hands-on understanding of classical supervised learning models without relying on external machine learning libraries.

---

## Included Notebooks

### 1. K-Nearest Neighbors (KNN) — From Scratch
**Path:** `knn/knn_classifier_demo.ipynb`  
**Dataset:** `iris.csv`

This notebook implements and explores the KNN classification algorithm using your custom `KNNClassifier`:

- Load and inspect the Iris dataset  
- Perform exploratory data analysis (pairplots, correlations)  
- Standardize features with `StandardScaler`  
- Train and evaluate the KNN model  
- Visualize a confusion matrix  
- Plot decision regions using PCA projection  

The notebook demonstrates why KNN is non-parametric, how distance-based classification works, and how scaling influences performance.

---

### 2. Perceptron — From Scratch
**Path:** `perceptron/perceptron_demo.ipynb`  
**Dataset:** `digits.csv` (binary-class conversion)

This notebook implements the classic perceptron algorithm for binary classification:

- Prepare and visualize the digits dataset  
- Convert multi-class digits into a binary task (e.g., “is the digit 8?”)  
- Train the perceptron using the perceptron update rule  
- Plot training loss over epochs  
- Compute model accuracy on the test set  

The notebook explains linear decision boundaries, misclassification updates, and interprets the perceptron’s limitations and strengths.

---

## Purpose of This Directory

These examples:

- Demonstrate correct usage of the `rice2025` package  
- Provide reproducible end-to-end workflows  
- Reinforce intuition behind core supervised models  
- Serve as reference guides for how to structure experiments, EDA, and evaluations  

---


## Notes

- All datasets used are real datasets stored locally in this directory.  
- Models rely only on implementations from the `rice2025` package—no scikit-learn classifiers are used.  
- Visualizations use Matplotlib and Seaborn for clarity and interpretability.

