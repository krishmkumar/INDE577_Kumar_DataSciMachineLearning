# Regression Trees (CART-Style) Demo — Housing Dataset

This folder contains my **Regression Tree** demo for the INDE 577 Machine Learning
project. The goal of this notebook is to present a clean, end-to-end example of
a **tree-based learning model implemented from scratch using NumPy** and applied
to a real-world housing dataset.

Unlike linear models, tree-based methods learn **nonlinear decision rules**
through recursive partitioning of the feature space. This demo highlights how
simple threshold-based splits can capture complex structure in data while
remaining highly interpretable.

---

## What this notebook is about

Tree-based models learn by repeatedly splitting the data according to feature–
threshold rules that increase homogeneity of the target variable within each
region. The result is a hierarchical structure of decisions that can be followed
from root to leaf.

In this notebook, I use a custom-built **CART-style tree** to model housing data.
Although housing prices are continuous, the implementation follows a
**classification-oriented formulation**, and the target is discretized into
price categories prior to training.

The emphasis is on:

- Understanding the intuition behind recursive partitioning  
- Connecting impurity-based splitting to the implemented algorithm  
- Exploring the effect of tree depth on performance  
- Evaluating predictions using classification metrics  

The notebook is designed to be **walk-through–friendly**, focusing on clarity,
interpretability, and alignment between theory and implementation.

---

## Dataset: `housing.csv`

The dataset used in this demo (`housing.csv`) contains housing-related features
used to model property values. Each row corresponds to a geographic region or
housing block.

Typical features include:

- Location-based variables (e.g., latitude, longitude)  
- Demographic summaries  
- Housing density and availability measures  

In this demo:

- The housing price variable is **discretized into categorical labels**
  (low / medium / high price ranges)
- Quantile-based binning is used to balance class frequencies
- All features are treated as numeric inputs to the tree model  

This preprocessing step ensures proper alignment with the classification-based
tree implementation.

---

## Notebook: `regression_trees_demo.ipynb`

The main notebook in this folder is:

- **`regression_tree.ipynb`**

High-level workflow inside the notebook:

1. **Load and explore the data**  
   - Read `housing.csv`  
   - Inspect feature distributions and target values  

2. **Target discretization**  
   - Convert continuous housing prices into categorical labels  
   - Verify class balance  

3. **Train / test split**  
   - Randomly partition data into training and testing sets  

4. **Model construction**  
   - Initialize a CART-style tree with depth and sample constraints  

5. **Model training**  
   - Recursively build the tree using Gini impurity–based splits  

6. **Model evaluation**  
   - Compute classification accuracy  
   - Examine confusion matrices and class-level performance  

7. **Model analysis**  
   - Explore the effect of tree depth on generalization  
   - Interpret misclassifications across price categories  

---

## Key Takeaways

- Tree-based models capture nonlinear structure without requiring parametric assumptions  
- Recursive splitting leads to highly interpretable decision rules  
- Depth control is essential to managing the bias–variance tradeoff  
- Discretization enables classification-oriented trees to be applied to continuous targets  
- The from-scratch implementation highlights the mechanics underlying CART-style models  

This demo emphasizes both the **conceptual intuition** and **algorithmic details**
of tree-based learning in a transparent, educational setting.
