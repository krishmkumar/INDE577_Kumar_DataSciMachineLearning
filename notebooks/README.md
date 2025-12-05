# Example Notebooks

This directory contains all example Jupyter notebooks demonstrating how to use the **`rice2025`** machine learning package.  
The notebooks are organized into two major categories:

---

## 📘 Supervised Learning  
**Folder:** `supervised/`  

These notebooks show end-to-end implementations of **supervised learning algorithms**, including:

- Classification models  
- Regression models  
- Neural network–based models  

Each example includes:

- Data loading and preprocessing  
- Exploratory data analysis  
- Model training using the custom algorithms in `rice2025`  
- Visualizations and evaluation metrics  
- Clear explanations of how each algorithm works under the hood  

See the README inside `supervised/` for detailed descriptions of every model included.

---

## 🔍 Unsupervised Learning  
**Folder:** `unsupervised/`  

These notebooks demonstrate **unsupervised machine learning algorithms**, including:

- Clustering methods (K-Means, DBSCAN)  
- Dimensionality reduction (PCA)  
- Graph-based community detection  

Each notebook provides:

- Intuition behind the algorithm  
- Mathematical or geometric interpretation  
- Step-by-step implementation using `rice2025`  
- Visualizations of clusters, embeddings, or communities  

See the README inside `unsupervised/` for full details on each method included.

---

## 🎯 Purpose of This Directory

The notebooks in this folder are designed to:

- Serve as tutorials for students using the `rice2025` package  
- Demonstrate the correct usage of each algorithm  
- Provide clean, reproducible examples for INDE 577 coursework  
- Help users understand the **theory**, **implementation**, and **practical behavior** of classical ML methods  

All code in the notebooks calls functions implemented from scratch in the `rice2025` package—**no scikit-learn models are used**.

---

## 📝 Notes

- Datasets used in the notebooks are stored locally in each subfolder.  
- Visualizations rely on Matplotlib and Seaborn.  
- These examples complement the unit tests in the `tests/` directory and help verify correct algorithm behavior.  
