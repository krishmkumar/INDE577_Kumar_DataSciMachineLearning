__all__ = ["preprocess", "metrics", "knn", "postprocess"]

"""
rice2025 - A lightweight educational machine learning package.
"""

from .basic_functions import add
from .metrics import euclidean_distance, manhattan_distance
from .preprocess import normalize, train_test_split
from .postprocess import majority_label, average_label

# Expose subpackages
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
