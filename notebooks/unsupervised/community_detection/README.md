# Community Detection Demo — Label Propagation (Facebook Page Graph)

This folder contains my community detection demo for the INDE 577 machine learning project. The goal is to explore how **Label Propagation (LPA)** can uncover community structure in graph data using the custom implementation included in the `rice2025` package.

---

## What this notebook is about

Label propagation is a simple but effective unsupervised algorithm for discovering communities in complex networks. Each node begins with a unique label, and through iterative updates based on its neighbors' labels, clusters emerge that represent densely connected groups of nodes.

In this notebook, I apply label propagation to a **subgraph of the Facebook Page–Page Network**, a real-world social graph where edges represent relationships between pages. The focus is on:

- Constructing a graph from an edge list  
- Selecting a 500-node high-degree subgraph for analysis  
- Running the label propagation algorithm from the `rice2025` package  
- Visualizing communities using a spectral embedding of the graph Laplacian  
- Summarizing community sizes  
- Exploring the effect of randomness and algorithm parameters  

The notebook is designed to be a clear, step-by-step demonstration of community detection using label propagation.

---

## Dataset Overview

This demo uses three files from the Facebook Page–Page Network dataset. Each file contributes different information:

### **1. `musae_facebook_edges.csv`**  
This is the primary file used for community detection.

- Contains an **undirected edge list**  
- Each row represents a connection between two Facebook pages  
- Columns:  
  - `id_1` — first node (integer, zero-based)  
  - `id_2` — second node (integer, zero-based)  
- Node IDs are consecutive integers starting from 0  

This file defines the graph structure on which label propagation is applied.

---

### **2. `musae_facebook_features.csv`**  
This file contains **node-level feature vectors** for each page.

- Each row corresponds to one node in the graph  
- Columns include numerous binary or numeric attributes describing the page  
- Features may represent:  
  - Page metadata  
  - Category indicators  
  - Topic affinities  
  - Behavioral signals  

These features are **not used directly** in the label propagation notebook, since LPA relies only on graph structure. However, they can be used in extended analyses such as:

- Node classification  
- Community characterization  
- Feature-driven clustering  

---

### **3. `musae_facebook_target.csv`**  
This file provides **ground-truth labels or metadata** for each node.

- Contains one row per node  
- Includes columns such as:  
  - `page_id` — original node identifier  
  - `category` — page’s assigned category or class  
  - Additional metadata  

These labels allow for:

- Evaluating how well community detection aligns with real-world categories  
- Coloring communities by known attributes  
- Comparing supervised vs. unsupervised structure  

Again, this file is **not required** in the main LPA demo, but it is extremely useful for interpretation and downstream evaluation.

---

## Notebook: `community_detection_demo.ipynb`

The main notebook in this folder is:

- **`community_detection_demo.ipynb`**

It walks through the following workflow:

1. **Load the edge list**  
   - Read the raw graph data  
   - Compute node degrees  
   - Select the top 500 high-degree nodes  

2. **Construct the subgraph**  
   - Filter edges to the selected node set  
   - Remap node IDs to a contiguous range  
   - Build a dense adjacency matrix for label propagation  

3. **Run Label Propagation**  
   - Apply the `LabelPropagation` class from the `rice2025` package  
   - Extract final labels  
   - Summarize community sizes  

4. **Visualization**  
   - Compute a spectral embedding using the graph Laplacian  
   - Plot the communities in 2D  
   - Highlight structure revealed by the algorithm  

5. **Stability Analysis**  
   - Repeat label propagation with different random seeds  
   - Compare the number of communities detected  
   - Discuss the effect of damping and randomness  

---

## Key Takeaways

- Label propagation provides a fast, intuitive method for detecting communities in large graphs.  
- Even a simple implementation can reveal meaningful structure in real-world networks.  
- Spectral embeddings offer a powerful way to visualize and interpret community assignments.  
- Additional dataset files (features and targets) enable richer downstream analysis.  

---

## Files in This Folder

- `community_detection_demo.ipynb` — full notebook demo  
- `README.md` — this file  
- `musae_facebook_edges.csv` — edge list defining the graph  
- `musae_facebook_features.csv` — feature vectors for each node  
- `musae_facebook_target.csv` — target labels / metadata for each node  

---

If additional datasets or analyses are added later, they can be documented here.
