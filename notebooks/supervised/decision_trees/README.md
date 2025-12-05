# Decision Tree Classifier Demo — Wine Quality Dataset

This folder contains my decision tree demo for the INDE 577 machine learning project.  
The goal is to provide a clear, end-to-end example of how to use a **Decision Tree classifier implemented from scratch** in the `rice2025` package to perform supervised classification on the **Wine Quality** dataset.

---

## What this notebook is about

A Decision Tree is a highly interpretable supervised learning method that makes predictions by recursively splitting the feature space based on simple threshold rules.  
Each split is chosen to maximize information gain, creating a tree-structured model that is easy to visualize and explain.

In this notebook, I use a decision tree to predict **binary wine quality** (high-quality vs. low-quality) using physicochemical features from the wine dataset.  
The focus is on:

- Understanding and preprocessing the dataset  
- Training a custom-built Decision Tree classifier  
- Evaluating model performance on unseen data  
- Exploring overfitting vs. generalization with tree depth  
- Visualizing decision boundaries using PCA  

Just like the other supervised-learning demos, the goal is clarity and intuition—not just showing a final model, but explaining how we get there.

---

## Dataset: `winequality-red.csv`

The dataset (`winequality-red.csv`) contains various chemical measurements of red wine samples.  
The target variable `quality` ranges from 0–10, so we convert it into a **binary classification task**:

- **1** → quality ≥ 6 (good wine)  
- **0** → quality < 6 (low quality)

Key features include:

- `fixed acidity`  
- `volatile acidity`  
- `citric acid`  
- `residual sugar`  
- `chlorides`  
- `free sulfur dioxide`  
- `total sulfur dioxide`  
- `density`  
- `pH`  
- `sulphates`  
- `alcohol`  

In the notebook, I:

- Inspect the distribution of features  
- Explore correlations with wine quality  
- Normalize / preprocess the features  
- Create a binary target for classification  

---

## Notebook: `decision_tree_demo.ipynb`

The main notebook in this folder is:

- **`decision_tree_demo.ipynb`**

Inside the notebook, I walk through:

### 1. **Exploring the dataset**
- Load `winequality-red.csv`  
- Summaries, histograms, and correlation plots  
- Discussion of target variable imbalance  

### 2. **Preprocessing**
- Convert the multiclass `quality` score into a binary label  
- Scale or normalize features if needed  
- Ensure all features are numeric  

### 3. **Train / test split**
- Split into training and testing sets using a stratified split  
- Inspect class balance in each subset  

### 4. **Training our custom Decision Tree**
- Fit our scratch-built `DecisionTree` from the `rice2025.supervised_learning` module  
- Control tree complexity using `max_depth`  
- Inspect entropy / information gain behavior  
- Evaluate training vs. testing accuracy  

### 5. **Decision Boundary Visualization (PCA)**
- Reduce 11-dimensional features to 2D with our custom PCA class  
- Train a small decision tree in PCA space  
- Plot decision regions, train/test points, and tree boundaries  

### 6. **Model Evaluation**
- Compute accuracy on train and test sets  
- Examine overfitting via depth adjustments  
- Discuss strengths and weaknesses of decision trees  

### 7. **Takeaways**
- Which features appear most relevant  
- How decision trees split data using thresholds  
- Why tree depth matters  
- Visual intuition for decision boundaries  

---

## Purpose of This Directory

This example demonstrates how to:

- Use the custom `DecisionTree` implementation inside the `rice2025` package  
- Work through a complete classification workflow  
- Gain intuition about entropy, information gain, and recursive splitting  
- Visualize high-dimensional decision boundaries with PCA  
- Evaluate model performance and interpret results  

The notebook serves both as a **teaching tool** and a **reference implementation** for the supervised learning portion of the project.

---

## Notes

- All models use **only** custom implementations from `rice2025` — no scikit-learn classifiers.  
- PCA visualizations also use the scratch-built PCA from the unsupervised module.  
- The wine dataset is a standard benchmark for classification and is included locally.  
- Plots rely on Matplotlib and Seaborn for clarity and interpretation.

