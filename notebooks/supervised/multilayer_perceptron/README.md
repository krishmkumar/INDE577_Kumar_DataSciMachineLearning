# Multilayer Perceptron Demo — Insurance Dataset

This folder contains my **Multilayer Perceptron (MLP)** demo for the INDE 577 Machine Learning project. The goal of this notebook is to present a clean, end-to-end example of how a **feedforward neural network implemented from scratch using NumPy** can be applied to a binary classification problem using the **Insurance** dataset.

Compared to simpler models like logistic regression, this demo highlights how neural networks learn **nonlinear feature interactions** and how training dynamics influence model behavior.

---

## What this notebook is about

A Multilayer Perceptron (MLP) is a supervised learning model that extends logistic regression by introducing **hidden layers** and **nonlinear activation functions**. These additions allow the model to learn more complex decision boundaries than a single linear classifier.

In this notebook, I use a custom-built MLP classifier to predict **whether an individual is a smoker** based on demographic and health-related features. The emphasis is on:

- Understanding the structure of a feedforward neural network  
- Training the model using gradient-based optimization  
- Visualizing and interpreting training behavior  
- Evaluating performance on held-out test data  

The notebook is designed to be **walk-through–friendly**, focusing on clarity and intuition rather than just presenting final results.

---

## Dataset: `insurance.csv`

The dataset used in this demo (`insurance.csv`) contains demographic and health information for individuals used to model insurance-related outcomes. Each row corresponds to one individual.

Key columns include:

- `age` – age of the individual  
- `sex` – biological sex  
- `bmi` – body mass index  
- `children` – number of dependents  
- `smoker` – whether the individual is a smoker (`yes` / `no`)  
- `region` – geographic region  
- `expenses` – medical insurance charges  

In this demo:

- `smoker` is used as the **binary target variable**  
- `expenses` is excluded to avoid target leakage  
- Categorical variables are converted using one-hot encoding  
- Numeric features are standardized prior to training  

---

## Notebook: `multilayer_perceptron_demo.ipynb`

The main notebook in this folder is:

- **`multilayer_perceptron_demo.ipynb`**

High-level workflow inside the notebook:

1. **Load and explore the data**  
   - Read `insurance.csv`  
   - Inspect feature types and distributions  

2. **Preprocessing**  
   - Encode categorical variables  
   - Select relevant features  
   - Apply feature scaling  

3. **Train / test split**  
   - Split the dataset using a custom `train_test_split` utility  

4. **Model construction**  
   - Define a Multilayer Perceptron with hidden layers  
   - Choose activation functions and regularization  

5. **Model training**  
   - Train using full-batch gradient descent  
   - Track binary cross-entropy loss over iterations  

6. **Model evaluation**  
   - Evaluate accuracy on training and test sets  
   - Inspect predicted probabilities  

7. **Interpretation and takeaways**  
   - Analyze convergence behavior  
   - Discuss why the MLP captures nonlinear structure  

---

## Key Takeaways

- Multilayer Perceptrons generalize logistic regression by learning nonlinear feature transformations  
- Hidden layers allow the model to capture interactions between predictors  
- Loss curve visualization provides insight into convergence and optimization stability  
- The custom implementation demonstrates core neural network concepts without relying on external libraries  

This demo emphasizes both the **mathematical foundations** and **practical behavior** of neural networks in a transparent, from-scratch setting.

---
