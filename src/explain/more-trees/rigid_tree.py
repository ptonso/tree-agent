import numpy as np
import logging
from typing import Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


from src.run.config import RigidConfig
from src.explain.dist2class import DistributionToClassificationConverter


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

        self.converter: Optional[DistributionToClassificationConverter] = None
        self.num_classes: Optional[int] = None
        self.input_dim: Optional[int] = None

    def _initialize_tree_structure(self, n_features: int):
        """Randomly assign feature indices for decision nodes and initialize parameters."""
        self.feature_indices = self.rng.randint(0, n_features, size=self.n_nodes)
        self.thresholds = np.zeros(self.n_nodes)
        self.leaf_values = np.zeros((self.n_leaves, self.num_classes))

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

    def fit_class(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the tree by setting thresholds and assigning leaf values."""
        if self.feature_indices is None:
            self._initialize_tree_structure(X_train.shape[1])

        for node in range(self.n_nodes):
            feature_idx = self.feature_indices[node]
            feature_values = X_train[:, feature_idx]
            self.thresholds[node] = np.median(feature_values)

        leaf_indices = self._get_node_indices(X_train)

        for leaf in range(self.n_leaves):
            mask = leaf_indices == leaf
            if np.any(mask):
                self.leaf_values[leaf] = np.mean(y_train[mask], axis=0)
            else:
                self.leaf_values[leaf] = np.ones(self.num_classes) / self.num_classes


        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for input samples."""
        leaf_indices = self._get_node_indices(X)
        return self.leaf_values[leaf_indices]


    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluates the model on a test set. Given y_test distributions."""
        y_pred_dist = self.predict(X_test)
        
        y_test_class = self.converter.dist_to_class(y_test)
        y_pred_class = self.converter.dist_to_class(y_pred_dist)
        
        eps = 1e-8
        kl_loss = np.sum(y_test * np.log((y_test + eps) / (y_pred_dist + eps))) / len(y_test)
        mse_loss = mean_squared_error(y_test, y_pred_dist)
        accuracy = accuracy_score(y_test_class, y_pred_class)

        return {
            "actual_loss": accuracy,
            "kl_loss": kl_loss,
            "mse_loss": mse_loss,
            "argmax_acc_loss": accuracy
        }

    def train_step(self, X: np.ndarray, y_dist: np.ndarray, dones: np.ndarray, values: np.ndarray) -> Dict[str, float]:
        """Trains the model by converting distribution labels into class labels."""

        if self.converter is None:
            self.num_classes = y_dist.shape[1]
            self.input_dim = X.shape[1]
            self.converter = DistributionToClassificationConverter(num_classes=self.num_classes)

        weights = np.abs(values - np.roll(values, shift=-1))
        weights[dones.astype(bool)] = 0
        weights = weights / (weights.sum() + 1e-8)

        X_train, X_val, y_train, y_val, _, _ = train_test_split(
            X, y_dist, weights, test_size=self.config.test_size, random_state=self.config.seed
        )

        y_train_class, _ = self.converter.convert_training_data(y_train)

        self.fit_class(X_train, y_train_class)
        return self.evaluate(X_val, y_val)


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
