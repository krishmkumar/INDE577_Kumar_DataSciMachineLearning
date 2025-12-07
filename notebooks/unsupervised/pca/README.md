# PCA Demo — Automobile Dataset

This folder contains the **Principal Component Analysis (PCA)** demonstration for the INDE 577 machine learning project.  
The goal of this notebook is to provide a clear, intuitive, end-to-end example of how PCA works—both mathematically and computationally—using the **Automobile** dataset.

---

## What this notebook is about

Principal Component Analysis (PCA) is a fundamental unsupervised learning method used for:

- Dimensionality reduction  
- Visualization of high-dimensional datasets  
- Identifying dominant patterns and directions of variance  
- Understanding correlations among features  

PCA transforms the original feature space into a new set of orthogonal axes (principal components) that capture the maximum possible variance in the data.  
This notebook walks through both the **intuition** and **math** behind PCA and shows how to apply it using the **custom PCA implementation from `rice2025`**.

The notebook focuses on:

- Understanding why PCA is used  
- Deriving PCA using covariance and SVD formulations  
- Fitting PCA from scratch  
- Interpreting explained variance  
- Visualizing samples in PCA space  
- Examining feature loadings and producing a biplot  

The intention is not just to compute PCA, but to make each step transparent and educational.

---

## Dataset: `Automobile_data.csv`

We use the **Automobile dataset**, which contains numeric attributes such as:

- horsepower  
- width  
- height  
- engine-size  
- curb-weight  
- compression ratio  
- city-mpg / highway-mpg  
- price  
- and many others  

These variables make PCA particularly meaningful, since many are correlated (e.g., engine size and horsepower), and PCA helps uncover underlying structure such as:

- size/power vs. efficiency tradeoffs  
- relationships between fuel economy and physical characteristics  
- redundancy among engineering features  

The dataset is cleaned by:

- Selecting only numeric columns  
- Handling missing values  
- Standardizing features prior to PCA  

---

## Notebook: `pca_demo.ipynb`

This notebook includes:

### 1. Exploratory Data Analysis  
- Inspecting numeric variables  
- Visualizing correlation heatmaps  
- Discussing redundancy among features

### 2. Standardization  
- Scaling features to zero mean and unit variance  
- Why scaling is essential for PCA

### 3. Fit PCA (custom implementation from `rice2025`)  
- Using SVD on centered data  
- Extracting eigenvalues, explained variance, and components  
- Interpreting the meaning of each principal component  

### 4. Explained Variance Analysis  
- Bar plot of variance captured by PC1 and PC2  
- Scree plot for component selection  
- Cumulative explained variance curve  

### 5. PCA Visualization  
- Scatterplot in the PC1–PC2 space  
- Interpretation of clusters and trends  

### 6. PCA Biplot  
- Plotting feature loadings over the 2D PCA projection  
- Understanding which features contribute most to each PC  
- Discussing correlations among variables based on vector directions  

### 7. Conclusions  
- Summary of what PCA revealed about the automobile dataset  
- Insights into underlying structure and feature relationships  
- Discussion of how PCA supports downstream modeling tasks  

---

## Purpose of This Directory

These PCA materials serve to:

- Demonstrate correct usage of the `rice2025` PCA implementation  
- Provide a transparent, mathematical explanation of PCA  
- Give students a reference for dimensionality reduction techniques  
- Offer visual tools (biplot, explained variance charts) for interpreting high-dimensional data  
- Support later unsupervised learning modules and feature engineering decisions  

---

## Notes

- All PCA computations use the **custom PCA class implemented from scratch** in `rice2025.unsupervised_learning.pca`.  
- No scikit-learn PCA is used except optionally for comparison.  
- Visualizations rely on Matplotlib and Seaborn.  
- The dataset is included locally within this directory.

