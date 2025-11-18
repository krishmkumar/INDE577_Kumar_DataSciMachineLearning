# K-Nearest Neighbors (KNN)

In the notebook `knn_classifier_demo.ipynb`, we explore a simple but powerful non-parametric method: K-Nearest Neighbors (KNN).  
Rather than learning coefficients or parameters, KNN makes predictions by identifying the most similar observations in the training set and taking a majority vote (classification) or an average (regression).

KNN is considered non-parametric because it does not impose a fixed functional form. There are no learned weights or model structure; instead, the "model" consists entirely of the training data. The algorithm’s behavior is driven by:

- the choice of distance metric (we use Euclidean distance),
- the number of neighbors k,
- and proper feature scaling.

In this example, we implement KNN **from scratch** using the `rice2025` package and apply it to the Iris dataset to classify flower species.

## In this project, we:

- Implement a full KNN classifier from scratch inside the `rice2025.supervised_learning` module.
- Use our custom `StandardScaler` and `train_test_split` utilities to prepare the data.
- Apply KNN to the Iris dataset, predicting among the three species.
- Explore the math and intuition behind the algorithm, including Euclidean distance and majority voting.
- Demonstrate the impact of scaling on distance-based methods.
- Evaluate our model using accuracy, confusion matrices, and a PCA-projected decision boundary plot.

This notebook is intended as a clean, hands-on demonstration of KNN using a real dataset, while validating that our custom implementation behaves as expected.
