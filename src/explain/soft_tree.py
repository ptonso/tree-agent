import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, List, Dict

from src.run.config import SoftConfig


class InnerNode:
    """Represents an internal node in the decision tree with a sigmoid split function."""

    def __init__(
        self, depth: int, input_dim: int, output_dim: int, max_depth: int, lmbda: float, device: str = 'cpu'
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.device = device

        self.fc = nn.Linear(self.input_dim, 1).to(self.device)
        self.beta = nn.Parameter(torch.randn(1, device=self.device))
        self.lmbda = lmbda * 2 ** (-depth)

        self.build_child(depth)

    def build_child(self, depth: int):
        """Recursively build left and right child nodes."""
        if depth < self.max_depth:
            self.left = InnerNode(depth + 1, self.input_dim, self.output_dim, self.max_depth, self.lmbda, self.device)
            self.right = InnerNode(depth + 1, self.input_dim, self.output_dim, self.max_depth, self.lmbda, self.device)
        else:
            self.left = LeafNode(self.output_dim, self.device)
            self.right = LeafNode(self.output_dim, self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the probability of selecting the right child node."""
        return torch.sigmoid(self.beta * self.fc(x))

    def cal_prob(self, x: torch.Tensor, path_prob: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Recursively compute the probability of reaching each leaf node."""
        prob = self.forward(x)
        left_accumulator = self.left.cal_prob(x, path_prob * (1 - prob))
        right_accumulator = self.right.cal_prob(x, path_prob * prob)
        return left_accumulator + right_accumulator


class LeafNode:
    """Represents a leaf node in the decision tree, predicting a probability distribution."""

    def __init__(self, output_dim: int, device: str = 'cpu'):
        self.output_dim = output_dim
        self.device = device
        self.param = nn.Parameter(torch.randn(self.output_dim, device=self.device))
        self.softmax = nn.Softmax(dim=1)

    def forward(self) -> torch.Tensor:
        """Return the probability distribution for the actions."""
        return self.softmax(self.param.view(1, -1))

    def cal_prob(self, x: torch.Tensor, path_prob: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return path probability along with leaf action probability distribution."""
        Q = self.forward().expand((path_prob.size(0), self.output_dim))
        return [(path_prob, Q)]


class SoftDecisionTree(nn.Module):
    """A differentiable decision tree for predicting stochastic action distributions."""

    def __init__(self, config: SoftConfig):
        super(SoftDecisionTree, self).__init__()
        self.config = config
        self.device = config.device
        self.max_depth = config.depth
        self.lr = config.lr
        self.momentum = config.momentum
        self.lmbda = config.lmbda
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.test_size = config.test_size

        self.root: Optional[InnerNode] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        self.optimizer: Optional[optim.Optimizer] = None

    def setup_network(self, X: torch.Tensor, y: torch.Tensor):
        """Dynamically initializes the tree structure based on input and output dimensions."""
        if self.root is not None:
            return

        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        self.root = InnerNode(1, self.input_dim, self.output_dim, self.max_depth, self.lmbda, self.device)
        self.collect_parameters()
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        self.to(self.device)

    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Trains the model for multiple epochs and returns a dictionary of KL and MSE losses."""
        if self.root is None:
            self.setup_network(X, y)

        B = X.shape[0]
        split_idx = int(B * (1 - self.test_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        best_kl_loss, best_mse_loss = float('inf'), float('inf')

        for _ in range(self.num_epochs):
            train_kl = self._train_epoch(X_train, y_train)
            val_kl, val_mse = self._evaluate(X_val, y_val)

            if val_kl < best_kl_loss:
                best_kl_loss, best_mse_loss = val_kl, val_mse

        return {
            "actual_loss": best_kl_loss, 
            "kl_loss": best_kl_loss, 
            "mse_loss": best_mse_loss
            }

    def _train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Trains for one epoch and returns KL loss."""
        self.train()
        indices = torch.randperm(len(X))
        total_kl = 0

        for start in range(0, len(X), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_X, batch_y = X[batch_indices].to(self.device), y[batch_indices].to(self.device)

            self.optimizer.zero_grad()
            kl_loss, _ = self.cal_loss(batch_X, batch_y)
            kl_loss.backward()
            self.optimizer.step()

            total_kl += kl_loss.item()

        return total_kl / (len(X) / self.batch_size)

    def _evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Evaluates KL and MSE loss on validation set."""
        self.eval()
        total_kl, total_mse = 0, 0

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch_X, batch_y = X[start:start + self.batch_size].to(self.device), y[start:start + self.batch_size].to(self.device)
                kl_loss, mse_loss = self.cal_loss(batch_X, batch_y)
                total_kl += kl_loss.item()
                total_mse += mse_loss.item()

        return total_kl / (len(X) / self.batch_size), total_mse / (len(X) / self.batch_size)

    def cal_loss(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute KL and MSE loss between predicted and target distributions."""
        batch_size = y.size(0)
        leaf_accumulator = self.root.cal_prob(X, torch.ones(batch_size, 1, device=self.device))

        kl_loss, mse_loss = 0, 0
        for path_prob, Q in leaf_accumulator:
            kl_div = F.kl_div(Q.log(), y, reduction='batchmean')
            mse = F.mse_loss(Q, y)

            kl_loss += (path_prob * kl_div).sum()
            mse_loss += (path_prob * mse).sum()

        return kl_loss / batch_size, mse_loss / batch_size

    def collect_parameters(self):
        """Register parameters from all nodes in the tree."""
        nodes = [self.root]
        self.module_list = nn.ModuleList()
        self.param_list = nn.ParameterList()

        while nodes:
            node = nodes.pop(0)
            if isinstance(node, LeafNode):
                self.param_list.append(node.param)
            else:
                nodes.append(node.left)
                nodes.append(node.right)
                self.param_list.append(node.beta)
                self.module_list.append(node.fc)


if __name__ == "__main__":
    torch.manual_seed(42)
    input_dim = 5
    output_dim = 3
    num_samples = 500

    X_train = torch.randn(num_samples, input_dim)
    y_train = F.softmax(torch.randn(num_samples, output_dim), dim=1)

    config = SoftConfig()
    ddt = SoftDecisionTree(config)
    metrics = ddt.train_step(X_train, y_train)

    print("\nFinal Training Metrics:", metrics)
