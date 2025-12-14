# Perceptron — From Scratch (Binary Digit Classification)

This folder contains my **Perceptron** demo for the INDE 577 Machine Learning
project. The goal of this notebook is to present a clean, end-to-end example of
a **linear classification algorithm implemented entirely from scratch using
NumPy** and applied to a real-world digits dataset.

The perceptron is one of the most fundamental supervised learning models and
serves as a building block for more advanced algorithms such as neural networks.
This demo emphasizes correctness, interpretability, and alignment with the
classic perceptron learning rule.

---

## What this notebook is about

The perceptron is a **binary linear classifier** that learns a separating
hyperplane by updating its weights whenever it misclassifies an observation.
Unlike probabilistic models such as logistic regression, the perceptron uses a
simple, rule-based update mechanism and does not explicitly optimize a convex
loss function.

In this notebook, I use a custom-built perceptron classifier to solve a **binary
digit classification problem**, predicting whether a given digit image
represents the digit “8”.

The emphasis is on:

- Understanding the intuition behind linear decision boundaries  
- Implementing the perceptron update rule from scratch  
- Visualizing training dynamics through loss curves  
- Evaluating classification performance on held-out data  

The notebook is designed to be **walk-through–friendly**, focusing on clarity
and algorithmic transparency rather than black-box performance.

---

## Dataset: `digits.csv`

The dataset used in this demo (`digits.csv`) contains grayscale digit images
represented as numeric feature vectors. Each row corresponds to a single digit,
with pixel intensities flattened into a fixed-length feature vector and a label
column indicating the true digit.

In this demo:

- The original multi-class problem is converted into **binary classification**  
- The target is defined as:
  - `1` → digit is an “8”  
  - `0` → digit is not an “8”  
- Features are standardized using a custom `StandardScaler`  
- The dataset is split into training and test sets using a custom
  `train_test_split` utility  

This setup ensures compatibility with the perceptron’s binary classification
assumptions and improves training stability.

---

## Notebook: `perceptron_demo.ipynb`

The main notebook in this folder is:

- **`perceptron_demo.ipynb`**

High-level workflow inside the notebook:

1. **Load and explore the data**  
   - Read `digits.csv`  
   - Inspect feature dimensions and label distribution  

2. **Binary target construction**  
   - Convert the multi-class digit labels into a binary task  

3. **Preprocessing**  
   - Standardize numeric features for stable learning  

4. **Train / test split**  
   - Partition the dataset into training and testing subsets  

5. **Model training**  
   - Train the perceptron using the classic update rule  
   - Track training loss over epochs  

6. **Model evaluation**  
   - Compute classification accuracy on test data  
   - Analyze convergence behavior  

7. **Model interpretation**  
   - Visualize the training loss curve  
   - Examine misclassified digit examples  

---

## Key Takeaways

- The perceptron learns a linear decision boundary through mistake-driven updates  
- Feature scaling is critical for stable and meaningful learning  
- The algorithm converges reliably on linearly separable problems  
- Training loss curves provide insight into convergence behavior  
- The from-scratch implementation confirms the correctness of the classic
  perceptron learning rule  

This demo demonstrates the perceptron algorithm end-to-end and validates the
implementation included in the `rice2025` supervised learning module.
