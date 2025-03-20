import numpy as np
from typing import Dict, Tuple

class DistributionToClassificationConverter:
    """Converts probability distributions into class labels and vice versa."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.label_to_dist = self._create_label_to_dist_dict()

    def _create_label_to_dist_dict(self) -> Dict[int, np.ndarray]:
        """Creates a lookup dictionary mapping class labels to one-hot vectors."""
        label_dict = {i: np.eye(self.num_classes)[i] for i in range(self.num_classes)}
        return label_dict

    def dist_to_class(self, y_dist: np.ndarray) -> np.ndarray:
        """Converts a probability distribution to class labels using argmax."""
        return np.argmax(y_dist, axis=1)

    def class_to_dist(self, y_class: np.ndarray) -> np.ndarray:
        """Converts class labels into one-hot probability distributions."""
        return np.array([self.label_to_dist[label] for label in y_class])

    def convert_training_data(self, y_dist: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Converts probability distributions into class labels for training."""
        y_class = self.dist_to_class(y_dist)
        return y_class, self.class_to_dist(y_class)
