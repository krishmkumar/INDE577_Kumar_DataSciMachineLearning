# Logistic Regression Demo — Titanic Dataset

This folder contains my logistic regression demo for the INDE 577 machine learning project. The goal here is to show a clean, end-to-end example of how to use logistic regression for a binary classification problem using the **Titanic** dataset.

---

## What this notebook is about

Logistic regression is a simple but powerful supervised learning method for **binary classification**. Instead of predicting a continuous value, it models the probability that an observation belongs to class 1 vs. class 0 (for example, survived vs. did not survive).

In this notebook, I use logistic regression to predict **whether a passenger survived the Titanic** based on their features. The focus is on:

- Getting the data into a usable form  
- Fitting a logistic regression model  
- Evaluating how well it performs  
- Interpreting the coefficients in a way that makes sense

The notebook is meant to be readable and “walk through-able,” not just a final model.

---

## Dataset: `titanic.csv`

The dataset in this folder (`titanic.csv`) is a standard Titanic passenger dataset. Each row corresponds to one passenger.

Key columns I use include (names may vary slightly depending on the version):

- `Survived` – target variable (1 = survived, 0 = did not survive)  
- `Pclass` – passenger class (1st, 2nd, 3rd)  
- `Sex` – passenger gender  
- `Age` – passenger age  
- `SibSp` – number of siblings / spouses aboard  
- `Parch` – number of parents / children aboard  
- `Fare` – ticket fare  
- `Embarked` – port of embarkation  

In the notebook, I:

- Handle missing values (especially Age and Embarked)  
- Convert categorical variables (Sex, Embarked, etc.) into numeric form  
- Optionally scale / standardize numeric features if needed  

---

## Notebook: `logistic_regression_demo.ipynb`

The main notebook in this folder is:

- **`logistic_regression_demo.ipynb`**

Rough outline of what happens inside:

1. **Load and explore the data**  
   - Read `titanic.csv`  
   - Look at basic summaries and distributions  

2. **Preprocessing**  
   - Deal with missing values  
   - Encode categorical variables  
   - Select features for the model  

3. **Train / test split**  
   - Split the data into training and test sets  

4. **Fit logistic regression**  
   - Train a logistic regression classifier  
   - Look at coefficients and what they mean  

5. **Model evaluation**  
   - Compute accuracy on train and test sets  
   - Optionally look at confusion matrix or other metrics  

6. **Takeaways**  
   - What the model seems to learn  
   - Which features matter for survival  

---
