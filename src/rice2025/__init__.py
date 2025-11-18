"""
rice2025 - A lightweight educational machine learning package.
"""

# Expose simple package-wide utilities
from .basic_functions import add

# Correct imports for distances (they live under supervised_learning now!)
from .supervised_learning.distances import (
    euclidean_distance,
    manhattan_distance,
)

# Preprocessing / Postprocessing
from .supervised_learning.preprocess import (
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

__all__ = [
    "add",
    "euclidean_distance",
    "manhattan_distance",
    "normalize",
    "train_test_split",
    "majority_label",
    "average_label",
    "supervised_learning",
    "unsupervised_learning",
]


# Supervised learning
from .supervised_learning.knn import KNNClassifier, KNNRegressor
from .supervised_learning.perceptron import Perceptron
from .supervised_learning.decision_tree import DecisionTree

# Unsupervised learning
from .unsupervised_learning.kmeans import KMeans

# Utilities
from .utils.metrics import euclidean_distance, manhattan_distance
from .utils.scaling import StandardScaler
from .utils.train_test_split import train_test_split

