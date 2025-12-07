# K-Means Clustering Demo — Mall Customers Dataset

This folder contains my K-Means clustering demo for the INDE 577 machine learning project. The goal here is to show a clean, end-to-end example of how to use K-Means for an **unsupervised learning** task using the popular **Mall Customers** dataset.

---

## What this notebook is about

K-Means is one of the most widely used algorithms for **unsupervised clustering**. Instead of predicting a label, the goal is to uncover natural groupings in the data based on similarity. Each cluster is represented by a centroid, and points are assigned to the cluster with the closest centroid.

In this notebook, I use K-Means to identify **customer segments** based on:

- Annual income  
- Spending score  

The focus is on:

- Exploring the feature space  
- Running the elbow method to select an appropriate number of clusters  
- Fitting K-Means using the implementation in the `rice2025` package  
- Visualizing clusters and centroids  
- Interpreting the structure of the resulting segments  

The notebook is intended to be clear and walk-through–oriented, highlighting each step of the clustering process rather than just presenting a final result.

---

## Dataset: `Mall_Customers.csv`

The dataset in this folder (`Mall_Customers.csv`) contains records of customers at a shopping mall. Each row represents one customer.

Important columns used in the notebook include:

- `CustomerID` – customer identifier  
- `Gender` – male/female  
- `Age` – customer age  
- `Annual Income (k$)` – annual income  
- `Spending Score (1–100)` – a behavior-based score assigned by the mall  

In this demo, I focus on:

- **Annual Income (k$)**  
- **Spending Score (1–100)**  

These two features form a natural 2D space where customer clusters are easy to visualize and interpret.

---

## Notebook: `kmeans_demo.ipynb`

The main notebook in this folder is:

- **`kmeans_demo.ipynb`**

Rough outline of what happens inside:

1. **Load and briefly explore the dataset**  
   - Read `Mall_Customers.csv`  
   - Visualize the income vs. spending score relationship  

2. **Feature selection**  
   - Extract the two key numeric features  
   - Optionally scale or standardize the data  

3. **Elbow method**  
   - Compute within-cluster sum of squares for different values of \( k \)  
   - Plot the elbow curve to choose an appropriate number of clusters  

4. **Fit K-Means**  
   - Train the K-Means model using the `rice2025` implementation  
   - Retrieve cluster assignments and centroid locations  

5. **Cluster visualization**  
   - Plot customer points colored by cluster  
   - Overlay centroid positions  
   - Optionally visualize cluster regions or convergence behavior  

6. **Interpretation**  
   - Describe the patterns in the clusters  
   - Relate income and spending behavior to meaningful customer segments  

7. **Takeaways**  
   - What the clustering reveals about shopping behavior  
   - How segmentation like this can inform marketing or business decisions  

---
