# DBSCAN Clustering Demo — Two-Moons Dataset

This folder contains my DBSCAN clustering demo for the INDE 577 machine learning project. The goal is to show how density-based clustering differs from centroid-based methods like K-Means, and to demonstrate the ability of DBSCAN to recover arbitrarily shaped clusters using the custom implementation included in the `rice2025` package.

---

## What this notebook is about

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is an unsupervised learning algorithm that groups points based on *density* rather than distance to a centroid. Unlike K-Means, DBSCAN:

- Identifies clusters of arbitrary shape  
- Automatically detects noise points  
- Does not require specifying the number of clusters  
- Performs well when clusters are curved or non-spherical  

In this notebook, I apply DBSCAN to the **two-moons dataset**, a classic benchmark that highlights the limitations of K-Means and the strengths of density-based clustering. The focus is on:

- Visualizing the raw data  
- Understanding DBSCAN’s parameters (`eps` and `min_samples`)  
- Fitting the custom DBSCAN model from the `rice2025` package  
- Plotting core points, border points, and noise  
- Exploring how parameter choices affect the number and shape of clusters  
- Interpreting results in the context of density-based clustering  

The notebook is designed as an intuitive walkthrough rather than a presentation of final results.

---

## Dataset: `cluster_moons.csv`

The dataset used in this demo (`cluster_moons.csv`) contains simulated two-dimensional points arranged in two interlocking “moon-shaped” clusters. This structure is specifically chosen because:

- K-Means fails to recover the nonlinear boundaries  
- DBSCAN naturally identifies the curved clusters  
- Noise points may appear depending on `eps`  

The dataset includes two columns:

- `X1` — first coordinate  
- `X2` — second coordinate  

These features allow for clear visualization of density clusters in 2D space.

---

## Notebook: `dbscan_demo.ipynb`

The main notebook in this folder is:

- **`dbscan_demo.ipynb`**

It covers the full DBSCAN workflow:

1. **Load and visualize the dataset**  
   - Scatter plot of raw points  
   - Observe curvature and cluster structure  

2. **Run the DBSCAN algorithm**  
   - Fit the custom implementation (`rice2025.unsupervised_learning.DBSCAN`)  
   - Retrieve cluster labels and identify noise  

3. **Cluster visualization**  
   - Color clusters by label  
   - Highlight noise points  
   - Distinguish core vs. border nodes  

4. **Parameter exploration**  
   - Compare different `eps` values  
   - Observe under-clustering and over-clustering  
   - Show sensitivity to density thresholds  

5. **k-distance graph (optional)**  
   - Plot sorted distances to the k-th nearest neighbor  
   - Use the “elbow” to estimate a reasonable `eps` value  

6. **Interpretation of results**  
   - Discuss density separation  
   - Compare against limitations of K-Means  
   - Understand noise, reachability, and density connectivity  

---

## Key Takeaways

- DBSCAN is highly effective for datasets with **nonlinear, curved, or irregular cluster shapes**.  
- The two-moons dataset clearly demonstrates why centroid-based clustering is limited.  
- Parameter selection (`eps` and `min_samples`) strongly influences the clustering outcome.  
- The custom `rice2025` DBSCAN implementation reproduces the core mechanics of the standard algorithm while keeping the code transparent and instructional.  
- Noise detection is a natural part of the algorithm, providing useful outlier information.  

---

## Files in This Folder

- `dbscan_demo.ipynb` — complete notebook demo  
- `cluster_moons.csv` — dataset used for clustering  
- `README.md` — this file  

---

If additional datasets or experiments are added later, they can be documented here.
