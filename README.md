# Machine Learning Algorithms and Applications


![Machine Learning Visualization](machine_learning_banner.jpg)

This repository contains a custom machine learning package developed for **INDE 577 — Machine Learning Package Development** at **Rice University**.

The project implements classic supervised and unsupervised learning algorithms **from scratch using NumPy**, wrapped in a clean Python package (`rice2025`) and demonstrated through structured example notebooks.

The repo showcases:

- Fully custom implementations of core ML algorithms  
- Modular Python package design  
- Educational Jupyter notebooks with real datasets  
- Utility modules for preprocessing, metrics, and evaluation  
- A complete `pytest` test suite for reliability  

---

## 🚀 Capabilities

### Supervised Learning

- Linear Regression  
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Perceptron  
- Multilayer Perceptron (neural network)  
- Decision Trees  
- Regression Trees  
- Basic ensemble utilities  

### Unsupervised Learning

- KMeans clustering  
- DBSCAN  
- PCA (dimensionality reduction)  
- Community detection / Label propagation  

### Utility Tools

- Scaling and normalization  
- General preprocessing utilities  
- Train/test splitting  
- Metric functions (accuracy, MSE, cross-entropy, etc.)  
- Postprocessing helpers  
- Core mathematical helper functions  

---

## 📁 Repository Structure 

A high-level view of the important pieces:

    INDE577_Kumar_DataSciMachineLearning/
    │
    ├── .github/                     # Issue templates, workflows
    │
    ├── notebooks/
    │   ├── supervised/
    │   │   ├── decision_trees/         # Decision tree demos + Wine Quality data
    │   │   ├── ensemble_methods/       # Notebooks for ensemble approaches
    │   │   ├── knn/                    # KNN classifier demo on Iris data
    │   │   ├── linear_regression/      # Student Performance regression demo
    │   │   ├── logistic_regression/    # Titanic survival classification demo
    │   │   ├── multilayer_perceptron/  # MLP examples
    │   │   ├── perceptron/             # Perceptron demo on digits
    │   │   └── regression_trees/       # Regression tree demonstrations
    │   │
    │   └── unsupervised/
    │       ├── community_detection/    # Label propagation / community detection
    │       ├── dbscan/                 # DBSCAN clustering on synthetic data
    │       ├── kmeans/                 # KMeans on Mall Customers data
    │       └── pca/                    # PCA on Automobile dataset
    │
    ├── src/
    │   └── rice2025/
    │       ├── supervised_learning/    # All supervised algorithms
    │       ├── unsupervised_learning/  # KMeans, DBSCAN, PCA, community detection
    │       └── utils/                  # Metrics, preprocessing, scaling, split, etc.
    │
    ├── tests/                          # Pytest suite for every algorithm/module
    │
    ├── requirements.txt                # Python dependencies
    ├── pyproject.toml                  # Package configuration (build + metadata)
    ├── LICENSE                         # MIT license
    └── README.md

This structure reflects a complete ML toolkit built from first principles, fully tested and demo-ready.

---

## 📘 Demo Notebooks

Each algorithm has an associated notebook under `notebooks/` that walks through:

- Loading and exploring a real or synthetic dataset  
- Preprocessing (scaling, encoding, splitting)  
- Training and evaluating the custom implementation from `rice2025`  
- Visualizing predictions, clusters, decision boundaries, or PCA components  
- Brief commentary on algorithm behavior and limitations  

Examples:

- **Linear Regression** – predicting student performance  
- **Logistic Regression** – classifying Titanic survival  
- **KNN** – classifying the Iris dataset  
- **Decision/Regression Trees** – modeling wine quality and regression tasks  
- **PCA** – reducing dimensionality of automobile data  
- **KMeans / DBSCAN** – clustering mall customers and synthetic moon-shaped clusters  
- **Perceptron / MLP** – classification on digit-style data  

These notebooks are meant to be **teaching resources** as much as demos.

---

## 🧪 Testing

All major algorithms and utilities are tested via `pytest` under the `tests/` directory.  
The tests cover:

- Mathematical / numerical correctness  
- Input validation and shapes  
- Edge-case behavior  
- Consistency of metrics and outputs  
- Clustering and classification performance on small, known datasets  

To run the tests:

    pytest -q

---

## 🔧 Installation

Clone the repository:

    git clone https://github.com/krishmkumar/INDE577_Kumar_DataSciMachineLearning.git
    cd INDE577_Kumar_DataSciMachineLearning

Install the package in editable mode:

    pip install -e .

You can then import and use the custom algorithms like this:

    from rice2025.supervised_learning import linear_regression, logistic_regression, knn
    from rice2025.unsupervised_learning import kmeans, pca, dbscan
    from rice2025.utils import metrics, train_test_split

---

## 🎯 Project Goals

This project was built to:

- Deepen understanding of ML algorithms by implementing them **from first principles**  
- Practice professional-quality Python package development  
- Integrate testing, documentation, and examples into one coherent codebase  
- Provide reusable code and educational notebooks for future students  
- Cover the full ML workflow: preprocessing → modeling → evaluation → visualization  

---

## 📜 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for full details.

---

## 👤 Author

**Krish Kumar**  
kmk10@rice.edu
Rice University — INDE 577  
GitHub: https://github.com/krishmkumar
