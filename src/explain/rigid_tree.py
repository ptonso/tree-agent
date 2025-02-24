import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split

from src.run.config import RigidConfig


class RigidDecisionTree:
    """A rigid decision tree with fixed structure, thresholds, and feature selection."""

    def __init__(self, config: RigidConfig):
        self.config = config
        self.depth = config.depth
        self.n_nodes = 2**self.depth - 1
        self.n_leaves = 2**self.depth
        self.rng = np.random.RandomState(config.seed)

        self.feature_indices = None
        self.thresholds = None
        self.leaf_values = None

    def _initialize_tree_structure(self, n_features: int):
        """Randomly assign feature indices for decision nodes and initialize parameters."""
        self.feature_indices = self.rng.randint(0, n_features, size=self.n_nodes)
        self.thresholds = np.zeros(self.n_nodes)
        self.leaf_values = np.zeros(self.n_leaves)

    def _get_node_indices(self, X: np.ndarray) -> np.ndarray:
        """Traverse the tree to determine which leaf each sample belongs to."""
        n_samples = X.shape[0]
        node_indices = np.zeros(n_samples, dtype=np.int32)

        for level in range(self.depth):
            level_offset = 2**level - 1
            level_size = 2**level

            for node in range(level_size):
                current_node = level_offset + node
                mask = node_indices == current_node

                if not np.any(mask):
                    continue

                feature_idx = self.feature_indices[current_node]
                feature_values = X[mask, feature_idx]
                left_mask = feature_values <= self.thresholds[current_node]

                node_indices[mask] = np.where(
                    left_mask,
                    2 * current_node + 1,
                    2 * current_node + 2
                )

        return node_indices - (2**self.depth - 1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the tree by setting thresholds and assigning leaf values."""
        if self.feature_indices is None:
            self._initialize_tree_structure(X.shape[1])

        for node in range(self.n_nodes):
            feature_idx = self.feature_indices[node]
            feature_values = X[:, feature_idx]
            self.thresholds[node] = np.median(feature_values)

        leaf_indices = self._get_node_indices(X)

        for leaf in range(self.n_leaves):
            mask = leaf_indices == leaf
            if np.any(mask):
                self.leaf_values[leaf] = np.argmax(np.bincount(y[mask]))
            else:
                self.leaf_values[leaf] = 0

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        leaf_indices = self._get_node_indices(X)
        return self.leaf_values[leaf_indices]

    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the model and evaluate on a validation set."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.seed, stratify=y
        )

        self.fit(X_train, y_train)
        y_pred = self.predict(X_val)

        kl_loss = None
        mse_loss = None
        accuracy = np.mean(y_pred == y_val)

        return {
            "actual_loss": None,
            "kl_loss": kl_loss,
            "mse_loss": mse_loss,
            "accuracy": accuracy
        }


if __name__ == "__main__":

    np.random.seed(42)
    latent_dim = 8
    action_dim = 3
    num_samples = 1000

    X_train = np.random.randn(num_samples, latent_dim).astype(np.float32)
    y_train = np.random.randint(0, action_dim, size=(num_samples,))

    config = RigidConfig(depth=4, test_size=0.2)
    rigid_tree = RigidDecisionTree(config)
    metrics = rigid_tree.train_step(X_train, y_train)

    print("\nFinal Training Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
