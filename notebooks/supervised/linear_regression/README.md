# Linear Regression Demo — Student Performance Dataset

This folder contains my linear regression demo for the INDE 577 machine learning project. The goal here is to show a clean, end-to-end example of how to use linear regression for predicting a continuous outcome using the **Student Performance** dataset.

---

## What this notebook is about

Linear regression is one of the most fundamental supervised learning methods for **predicting continuous values**. It models the relationship between a target variable and one or more features by fitting a straight line (or hyperplane) that minimizes the squared error.

In this notebook, I use linear regression to predict **students’ final exam scores** based on demographic, behavioral, and academic features. The focus is on:

- Understanding the structure of the dataset  
- Cleaning and preprocessing the features  
- Fitting a linear regression model  
- Evaluating performance using regression metrics  
- Interpreting coefficients in a practical, intuitive way  

Just like the logistic regression notebook, the goal is to make the workflow easy to follow rather than just show a final model.

---

## Dataset: `student_performance.csv`

The dataset (`student_performance.csv`) is a common educational dataset that tracks various factors that may influence academic outcomes.  
Each row corresponds to a student, and the target variable is typically the **final grade**.

Key columns often include (may vary by dataset version):

- `G1` – first-period grade  
- `G2` – second-period grade  
- `G3` – final grade (target variable)  
- `studytime` – weekly study hours  
- `failures` – past class failures  
- `absences` – number of absences  
- `schoolsup` – extra academic support  
- `famsup` – family support  
- `traveltime` – home-to-school travel time  
- `guardian` – mother, father, or other  
- `internet` – internet access at home  

In the notebook, I:

- Handle missing values if any  
- Convert categorical features into numeric encodings  
- Explore correlations and feature importance  
- Build regression models using selected features  

---

## Notebook: `linear_regression_demo.ipynb`

The main notebook included in this folder is:

- **`linear_regression_demo.ipynb`**

Inside the notebook, I walk through:

1. **Exploring the dataset**  
   - Load `student_performance.csv`  
   - Summaries, visualizations, and correlation heatmaps  

2. **Preprocessing**  
   - Handle categorical and numeric features appropriately  
   - Drop or impute missing values  
   - Select features for modeling  

3. **Train / test split**  
   - Create separate training and testing datasets  

4. **Fit linear regression**  
   - Train a linear model on the selected predictors  
   - Inspect coefficients and interpret their meaning  

5. **Model evaluation**  
   - Report R², RMSE, MAE, and residual analysis  
   - Visualize predicted vs. actual performance  

6. **Takeaways**  
   - Which features most strongly predict student performance  
   - Practical implications of model results  

