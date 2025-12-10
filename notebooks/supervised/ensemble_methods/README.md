# Ensemble Methods Demo — Heart Disease Dataset

This folder contains my ensemble learning demo for the INDE 577 machine learning project.  
The goal here is to show a clean, end-to-end example of how to use **Bagging**, **Random Forests**,  
and **Voting Ensembles** to improve predictive performance over a single decision tree using the  
**Heart Disease** dataset.

---

## What this notebook is about

Ensemble methods are some of the most powerful techniques in machine learning.  
Instead of relying on a single model, they combine multiple learners to produce predictions that are:

- More stable  
- Less sensitive to noise  
- Often more accurate  

In this notebook, I demonstrate how ensemble techniques can significantly improve classification performance compared to a standalone decision tree. The focus is on:

- Understanding why ensemble methods work (variance reduction and decorrelation)  
- Building Bagging and Random Forest classifiers using the custom `rice2025` package  
- Comparing these models against each other in a structured, interpretable way  
- Visualizing differences in decision boundaries using a synthetic dataset  
- Explaining the mathematical intuition behind ensembles  

Just like the linear and logistic regression demos, the goal is to provide a readable, step-by-step workflow rather than just a final model.

---

## Dataset: `heartdisease_R.csv`

The dataset (`heartdisease_R.csv`) is a medical classification dataset commonly used to study risk factors for cardiovascular disease.

Each row represents a patient, and the target variable is the **disease class** (multi-class classification).

Typical features include:

- `age` – age of patient  
- `sex` – gender (1 = male, 0 = female)  
- `cp` – chest pain type  
- `trestbps` – resting blood pressure  
- `chol` – serum cholesterol  
- `fbs` – fasting blood sugar  
- `restecg` – resting ECG results  
- `thalach` – maximum heart rate achieved  
- `exang` – exercise-induced angina  
- `oldpeak` – ST depression  
- `slope`, `ca`, `thal` – additional clinical indicators  
- `class` – target label  

In the notebook, I:

- Explore class distributions  
- Standardize numeric features  
- Split the data into training/testing subsets  
- Train multiple tree-based models  
- Evaluate their performance and compare accuracy  

---

## Notebook: `ensemble_methods_demo.ipynb`

The main notebook included in this folder is:

- **`ensemble_methods_demo.ipynb`**

Inside the notebook, I walk through:

### 1. **Exploring the dataset**  
- Load `heartdisease_R.csv`  
- Examine summary statistics and target distribution  

### 2. **Preprocessing**  
- Standardize the feature matrix  
- Prepare training and testing sets  
- Ensure the target variable is properly encoded  

### 3. **Baseline model: Decision Tree**  
- Train a single decision tree classifier  
- Highlight its limitations (high variance, instability)  

### 4. **Bagging (Bootstrap Aggregating)**  
- Train `BaggingClassifier` from the `rice2025` package  
- Show how bootstrap sampling reduces variance  
- Compare performance to the baseline tree  

### 5. **Random Forest**  
- Train `RandomForestClassifier` (simplified implementation)  
- Explain how feature randomness decorrelates trees  
- Compare accuracy to Bagging  

### 6. **Voting Ensemble**  
- Combine multiple models (tree, bagging, forest)  
- Use majority vote as the final prediction  
- Demonstrate when voting can outperform individual models  

### 7. **Visualization on Synthetic Data**  
- Use `make_moons` to show:  
  - A tree overfits  
  - Bagging smooths the boundary  
  - Random forest smooths it even further  

### 8. **Detailed mathematical explanations**  
- Bias–variance decomposition  
- Ensemble variance formula  
- Why bootstrap sampling works  
- Why trees are ideal base learners  

### 9. **Model evaluation**  
- Accuracy comparisons  
- Confusion matrices  
- Interpretation of ensemble improvements  

### 10. **Takeaways**  
- Ensembles systematically reduce variance  
- Random Forests are strong out-of-the-box classifiers  
- Voting can provide robustness when models disagree  

---

## Why this notebook is useful

This demo showcases practical and theoretical aspects of ensemble learning:

- How bagging stabilizes high-variance learners  
- Why random forests outperform bagging via decorrelation  
- How majority voting works in heterogeneous model combinations  
- How to implement and test ensemble methods using your own custom ML package  

It serves as both a **teaching tool** and a **validation testbed** for the supervised learning algorithms implemented in `rice2025`.

