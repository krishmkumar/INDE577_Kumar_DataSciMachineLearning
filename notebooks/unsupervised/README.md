# Unsupervised Learning Examples

This directory contains example Jupyter notebooks demonstrating **unsupervised learning algorithms implemented from scratch** using the `rice2025` package.

Each notebook includes:

- Data loading and preprocessing  
- Visualization and exploratory data analysis (EDA)  
- Algorithm implementation and step-by-step explanation  
- Cluster interpretation and evaluation  
- Comparisons across methods when relevant  

These examples are intended to build strong intuition for how unsupervised learning works without relying on external libraries like scikit-learn.

---

## 📁 Included Methods & Notebooks

Below are all unsupervised methods implemented in this directory.

---

## 🔍 PCA (Principal Component Analysis)  
**Folder:** `pca/`  
**Notebook:** `pca_demo.ipynb`  
**Model:** `PCA` (custom implementation using eigen-decomposition)

This notebook demonstrates:

- Standardizing features  
- Computing the covariance matrix  
- Deriving principal components via eigenvalues & eigenvectors  
- Projecting data into lower-dimensional space  
- Variance explained by each component  
- Visualizing clusters or structure revealed by PCA  

The example highlights how PCA helps with dimensionality reduction and visualization.

---

## 🔢 K-Means Clustering  
**Folder:** `kmeans/`  
**Notebook:** `kmeans_demo.ipynb`  
**Model:** `KMeans` (custom implementation)

Topics covered:

- Random initialization vs. k-means++  
- Cluster assignment and centroid update steps  
- Tracking inertia and convergence  
- Choosing \( k \) using the elbow method  
- Visualizing cluster boundaries and centroids  
- Comparing ground-truth labels (when available)

This notebook explains why K-Means works well for spherical clusters and where it fails.

---

## 🌀 DBSCAN (Density-Based Clustering)  
**Folder:** `dbscan/`  
**Notebook:** `dbscan_demo.ipynb`  
**Model:** `DBSCAN` (custom implementation using ε-neighborhoods and minPts)

This notebook includes:

- Intuition behind density-based clustering  
- Identifying core, border, and noise points  
- Demonstrating DBSCAN’s robustness to irregular cluster shapes  
- Showing why DBSCAN outperforms K-Means on non-convex datasets  
- Visualizing clusters and noise points

The example focuses on density connectivity rather than centroid assignment.

---

## 🕸️ Community Detection (Graph Clustering)  
**Folder:** `community_detection/`  
**Notebook:** `community_detection_demo.ipynb`  
**Model:** Label Propagation (custom implementation)

This notebook demonstrates:

- Loading or creating graph data  
- Running label propagation for community formation  
- Visualizing graph clusters with network plots  
- Understanding modularity and community structure  
- Comparing community detection to traditional clustering

This example shows how clustering works when data is represented as a graph instead of points in space.

---

## 🎯 Purpose of This Directory

These notebooks are designed to:

- Demonstrate correct usage of the `rice2025` unsupervised learning module  
- Provide clean, reproducible examples for each algorithm  
- Help students visualize how unsupervised learning separates structure in data  
- Act as templates for your INDE 577 project deliverables  
- Support testing, debugging, and model comparison  

---

## 📝 Notes

- All algorithms use **only** implementations from the `rice2025` package  
- No clustering functions from scikit-learn are used  
- Visualizations rely on Matplotlib and Seaborn  
- Graph visualizations used in community detection rely on NetworkX  

---
