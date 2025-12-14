# K-Nearest Neighbors (KNN) — From Scratch (Iris Dataset)

This folder contains my **K-Nearest Neighbors (KNN)** demo for the INDE 577 Machine
Learning project. The goal of this notebook is to present a clean, end-to-end
example of an **instance-based learning algorithm implemented entirely from
scratch using NumPy** and applied to the classic Iris dataset.

Unlike parametric models, KNN makes predictions directly from the training data
by comparing distances between observations. This demo highlights how KNN
learns nonlinear decision boundaries without explicitly fitting a model.

---

## What this notebook is about

K-Nearest Neighbors is a **non-parametric, instance-based** supervised learning
algorithm. Instead of learning explicit parameters during training, KNN stores
the training data and makes predictions by comparing new observations to their
nearest neighbors in feature space.

For a new data point, the algorithm:

- Computes distances to all training samples  
- Identifies the \( k \) closest neighbors  
- Predicts the label via majority vote (classification)  

In this notebook, I use a custom-built KNN classifier to perform **multi-class
classification** on the Iris dataset.

The emphasis is on:

- Understanding the geometric intuition behind distance-based learning  
- Implementing KNN from scratch without external ML libraries  
- Visualizing decision boundaries and classification behavior  
- Evaluating performance on held-out test data  

The notebook is designed to be **walk-through–friendly**, focusing on clarity,
intuition, and transparency rather than optimized performance.

---

## Dataset: `iris.csv`

The dataset used in this demo (`iris.csv`) is the classic Iris flower dataset.
Each row corresponds to a flower sample with four numeric features and a known
species label.

Features include:

- `sepal.length`  
- `sepal.width`  
- `petal.length`  
- `petal.width`  

In this demo:

- Flower species are encoded as integer class labels  
- Features are standardized using a custom `StandardScaler`  
- The dataset is split into training and test sets using a custom
  `train_test_split` utility  

Feature scaling is essential for KNN because distance computations are sensitive
to differences in feature magnitude.

---

## Notebook: `knn_classifier_demo.ipynb`

The main notebook in this folder is:

- **`knn_classifier_demo.ipynb`**

High-level workflow inside the notebook:

1. **Load and explore the data**  
   - Read `iris.csv`  
   - Inspect feature distributions and class balance  

2. **Exploratory data analysis (EDA)**  
   - Pairwise feature plots  
   - Correlation heatmaps  

3. **Preprocessing**  
   - Standardize numeric features  

4. **Train / test split**  
   - Partition the dataset into training and testing subsets  

5. **Model training**  
   - Train a KNN classifier with a chosen value of \( k \)  

6. **Model evaluation**  
   - Compute classification accuracy  
   - Analyze confusion matrices  

7. **Visualization**  
   - Project data to two dimensions using a custom PCA implementation  
   - Visualize decision regions in the reduced feature space  

---

## Key Takeaways

- KNN is a simple yet powerful non-parametric classification algorithm  
- Distance-based learning naturally captures nonlinear decision boundaries  
- Feature scaling is critical for meaningful distance computations  
- Smaller values of \( k \) increase variance, while larger values smooth
  decision boundaries  
- The from-scratch implementation demonstrates the core mechanics of KNN
  without reliance on external machine learning libraries  

This demo validates the correctness of the `KNNClassifier` included in the
`rice2025` supervised learning module and illustrates its behavior on a
well-known multi-class dataset.
