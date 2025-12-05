"""
rice2025 - A lightweight educational machine learning package.
"""

# Package-wide basic utility
from .basic_functions import add

# Distances (canonical versions from supervised_learning)
from .supervised_learning.distances import (
    euclidean_distance,
    manhattan_distance,
)

# Preprocessing / postprocessing
from .utils.preprocess import (
    normalize,
    train_test_split,
)
from .utils.postprocess import (
    majority_label,
    average_label,
)

# Subpackages
from . import supervised_learning
from . import unsupervised_learning

# Supervised learning models
from .supervised_learning.knn import KNNClassifier, KNNRegressor
from .supervised_learning.perceptron import Perceptron
from .supervised_learning.decision_tree import DecisionTree

# Unsupervised learning models
from .unsupervised_learning.kmeans import KMeans

# Utilities
from .utils.scaling import StandardScaler

__all__ = [
    # basic utilities
    "add",
    # distances
    "euclidean_distance",
    "manhattan_distance",
    # preprocessing / postprocessing
    "normalize",
    "train_test_split",
    "majority_label",
    "average_label",
    # models
    "KNNClassifier",
    "KNNRegressor",
    "Perceptron",
    "DecisionTree",
    "KMeans",
    # utilities
    "StandardScaler",
    # subpackages
    "supervised_learning",
    "unsupervised_learning",
]
