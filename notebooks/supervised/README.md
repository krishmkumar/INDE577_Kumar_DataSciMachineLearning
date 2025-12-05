# Supervised Learning Examples

This directory contains example Jupyter notebooks demonstrating **supervised learning algorithms implemented from scratch** using the `rice2025` package.  
Each notebook provides a complete, end-to-end workflow including:

- Data loading and preprocessing  
- Exploratory data analysis (EDA)  
- Model training  
- Visualizations  
- Evaluation and interpretation  

These examples serve as both **tutorials** and **reference implementations** for the algorithms included in the supervised module of the package.

---

## What Is Supervised Learning?

Supervised learning refers to machine learning tasks where a model is trained using **labeled data**—datasets where each example includes both input features and a known output (such as a class label or numeric value).  
The goal is for the model to learn a mapping from inputs to outputs so it can generalize and make predictions on new, unseen data.

Supervised learning methods fall into two main categories:

- **Classification:** predicting discrete categories (e.g., “survived vs. not survived,” “digit 5 vs. not 5”).  
- **Regression:** predicting continuous values (e.g., exam scores, housing prices).

The notebooks in this directory illustrate how these models are implemented **from scratch**, how they behave on real datasets, and how to evaluate their performance.

---
## 📁 Included Methods & Notebooks

Below is a list of all algorithms currently implemented in this directory, organized by folder.

---

## 🌳 Decision Trees  
**Folder:** `decision_trees/`  
**Notebook:** `decision_tree_demo.ipynb`  
**Model:** Custom `DecisionTreeClassifier`

This notebook demonstrates:

- Building a decision tree using information gain / Gini  
- Visualizing decision boundaries  
- Understanding overfitting and depth constraints  
- Evaluating classification performance  
- Interpreting tree splits and feature importance  

---

## 🪶 Ensemble Methods  
**Folder:** `ensemble_methods/`  
**Notebook:** `random_forest_demo.ipynb` (and other ensemble demos)  
**Models:** Custom `RandomForestClassifier`, `BaggingClassifier`

This notebook includes:

- Training an ensemble of decision trees via bagging  
- Understanding variance reduction from ensembling  
- Visualizing prediction stability  
- Comparing single-tree vs. ensemble accuracy  
- Inspecting feature importance across the forest  

---

## 🔢 K-Nearest Neighbors (KNN)  
**Folder:** `knn/`  
**Notebook:** `knn_classifier_demo.ipynb`  
**Model:** `KNNClassifier`

Topics covered:

- Loading and exploring the Iris dataset  
- Scaling features with `MinMaxScaler`  
- Tuning \( k \)  
- Visualizing decision boundaries  
- Reducing dimensions via PCA  
- Evaluating accuracy and confusion matrices  

---

## 📈 Linear Regression  
**Folder:** `linear_regression/`  
**Notebook:** `linear_regression_demo.ipynb`  
**Model:** `LinearRegression` (custom gradient descent implementation)

This notebook demonstrates:

- Predicting continuous student performance metrics  
- Fitting linear regression using gradient descent  
- Visualizing regression fit and residuals  
- Interpreting coefficients  
- Computing RMSE, MAE, and \( R^2 \)  

---

## ✔️ Logistic Regression  
**Folder:** `logistic_regression/`  
**Notebook:** `logistic_regression_demo.ipynb`  
**Model:** `LogisticRegression` (custom implementation)

This notebook covers:

- Preprocessing the Titanic dataset  
- Training logistic regression from scratch  
- Plotting probability curves and decision boundaries  
- Interpreting model coefficients  
- Evaluating performance with accuracy, ROC, AUC  

---

## 🧠 Multilayer Perceptron (Neural Network)  
**Folder:** `multilayer_perceptron/`  
**Notebook:** `mlp_demo.ipynb`  
**Model:** `MLPClassifier`

This notebook explains:

- Implementing a simple feedforward neural network  
- Forward pass, backpropagation, and gradient descent  
- Activation functions (ReLU, sigmoid)  
- Visualizing training loss across epochs  
- Comparing training vs. test accuracy  

---

## ➕ Perceptron (Single-Layer)  
**Folder:** `perceptron/`  
**Notebook:** `perceptron_demo.ipynb`  
**Model:** `Perceptron`

This notebook includes:

- Binary classification on a digits subset  
- The perceptron update rule  
- Visualizing misclassifications  
- Plotting training convergence  
- Understanding linear decision boundaries  

---

## 🎯 Purpose of This Directory

These notebooks are intended to:

- Demonstrate correct usage of the `rice2025` supervised learning module  
- Provide clear, reproducible examples  
- Reinforce intuition behind classical ML models  
- Serve as templates for the INDE 577 project notebooks  
- Support evaluation, debugging, and unit testing  

---

## 📝 Notes

- All models use **only** implementations from the `rice2025` package  
- No scikit-learn classifiers are used  
- Visualizations rely on Matplotlib and Seaborn  
- Datasets are included locally inside this directory  
