import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from typing import Optional, Dict

from src.run.config import SKLearnConfig
from src.explain.dist2class import DistributionToClassificationConverter

class SKLearnDecisionTree:
    """Decision tree classifier for action prediction, initialized lazily."""

    def __init__(self, config: SKLearnConfig):
        self.seed = config.seed
        self.config = config
        self.device = config.device
        self.converter: Optional[DistributionToClassificationConverter] = None
        self.num_classes: Optional[int] = None
        self.input_dim: Optional[int] = None

        self.model = None  # Initialized lazily

    def _initialize_model(self):
        """Lazily initializes the decision tree model based on training data dimensions."""
        self.model = DecisionTreeClassifier(
            random_state=self.seed, 
            criterion=self.config.criterion,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            min_weight_fraction_leaf=self.config.min_weight_fraction_leaf,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_impurity_decrease=self.config.min_impurity_decrease,
            ccp_alpha=self.config.ccp_alpha,
        )

    def _compute_weights(self, values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Computes weights based on value differences between consecutive steps."""
        next_values = np.roll(values, shift=-1)
        next_values = np.copy(next_values)
        values = np.copy(values)
        next_values[dones.astype(bool)] = 0
        values[dones.astype(bool)] = 0
        weights = np.abs(values - next_values)
        weights = weights / (weights.sum() + 1e-8)
        return weights

    def fit_class(self, X_train: np.ndarray, y_train: np.ndarray, weights: Optional[np.ndarray]):
        """Trains the decision tree classifier."""
        self.model.fit(X_train, y_train, sample_weight=weights)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluates the model on a test set. Given y_test distributions."""
        y_pred_class = self.model.predict(X_test)

        y_test_class = self.converter.dist_to_class(y_test)
        y_pred_dist = self.converter.class_to_dist(y_pred_class)

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
    

    def train_step(self, X: np.ndarray, y_dist: np.ndarray, dones: np.ndarray, values: np.ndarray) -> Dict[str, np.ndarray]:
        """Trains the model by converting distribution labels into class labels."""
        
        if self.converter is None:
            self.num_classes = y_dist.shape[1]
            self.input_dim = X.shape[1]
            self.converter = DistributionToClassificationConverter(num_classes=self.num_classes)
        
        if self.model is None:
            self._initialize_model()

        weights = self._compute_weights(values, dones)

        X_train, X_val, y_train, y_val, weights_train, _ = train_test_split(
            X, y_dist, weights, test_size=0.2, random_state=self.seed
        )

        y_train_class, _ = self.converter.convert_training_data(y_train)

        self.fit_class(X_train, y_train_class, weights_train)
        return self.evaluate(X_val, y_val)


if __name__ == "__main__":
    np.random.seed(42)    
    latent_dim = 8
    action_dim = 3
    num_samples = 1000

    X_train = np.random.randn(num_samples, latent_dim).astype(np.float32)
    y_train = np.random.rand(num_samples, action_dim).astype(np.float32)
    y_train /= y_train.sum(axis=1, keepdims=True)  # Ensure valid distributions
    dones = np.random.choice([0, 1], size=(num_samples,), p=[0.8, 0.2])
    values = np.random.randn(num_samples).astype(np.float32)

    config = SKLearnConfig()
    dtree = SKLearnDecisionTree(seed=42, config=config)

    metrics = dtree.train_step(X_train, y_train, dones, values)

    print("\nFinal Training Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

