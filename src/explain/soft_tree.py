
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, List, Dict

from src.run.config import SoftConfig
from src.agent.batch import Batch


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
        if depth < self.max_depth-1:
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
        left_accumulator  = self.left.cal_prob (x, path_prob * prob)
        right_accumulator = self.right.cal_prob(x, path_prob * (1 - prob))
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

    def __init__(self, config: SoftConfig, logger: Optional[logging.Logger] = None):
        super(SoftDecisionTree, self).__init__()
        self.config = config
        self.device = config.device
        self.logger = logger
        self.max_depth = config.depth
        self.lr = config.lr
        # self.momentum = config.momentum
        self.lmbda = config.lmbda
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.test_size = config.test_size

        self.root: Optional[InnerNode] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.metric: Optional[float] = None

        self.sigma_init = 10.0

    def setup_network(self, X: torch.Tensor, y: torch.Tensor):
        """Dynamically initializes the tree structure based on input and output dimensions."""
        if self.root is not None:
            return
        self.input_dim = X.shape[1]
        self.output_dim = y.shape[1]

        self.root = InnerNode(0, self.input_dim, self.output_dim, self.max_depth, self.lmbda, self.device)
        self.collect_parameters()

        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999)
        )
        # self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        self.to(self.device)


    def train_step(self, trajectories: List["Trajectory"], keep_tree: bool = False) -> Dict[str, float]:
        """
        if tree already exists, use depth-based warm reinitialization and retrain
        if tree does not exist, initialize with default values.
        if keep_tree, simply proceed to train existing tree with gradient.

        """
        batch = Batch(trajectories, self.device)
        batch_data = batch.prepare_tensors()
        X = batch_data.states
        y = batch_data.actions_prob

        # full rebuild
        self.root = None
        self.setup_network(X, y)
        self._initialize_leaf_nodes(y)

        mu = X.mean(dim=0)
        std = X.std(dim=0) + 1e-6

        self.register_buffer('x_mean', mu)
        self.register_buffer('x_std', std)

        metrics = self._train_step(X, y)
        self.log_metrics(metrics)
        self.metric = metrics["argmax_acc_loss"]
        return metrics


    def _reinitialize_tree(self, X: torch.Tensor, y: torch.Tensor):
        """
        If no tree exists, 
            initialize it using the current batch and initialize leaves smartly.
        """
        if self.root is None:
            self.setup_network(X, y)
            self._initialize_leaf_nodes(y)
        else:
            self._reinitialize_node(self.root)

    def _reinitialize_node(self, node):
        """Normal reinit: sample each parameter from N(old, sigma^2)."""
        if isinstance(node, InnerNode):
            with torch.no_grad():
                node.fc.weight.data.copy_(
                    torch.normal(mean=node.fc.weight.data, std=self.sigma_init)
                )
                node.fc.bias.data.copy_(
                    torch.normal(mean=node.fc.bias.data, std=self.sigma_init)
                )
                node.beta.data.copy_(
                    torch.normal(mean=node.beta.data, std=self.sigma_init)
                )
            # Recurse
            self._reinitialize_node(node.left)
            self._reinitialize_node(node.right)

        elif isinstance(node, LeafNode):
            with torch.no_grad():
                node.param.data.copy_(
                    torch.normal(mean=node.param.data, std=self.sigma_init)
                )


    def _initialize_leaf_nodes(self, y: torch.Tensor):
        """Initialize leaf node parameters smartly based on the distribution over action probabilities."""
        avg = y.mean(dim=0) + 1e-6  # add epsilon to avoid log(0)
        base_probs = avg / avg.sum()


        def _set_leaf(node, depth: int):
            if isinstance(node, LeafNode):

                sample_probs = torch.distributions.Dirichlet(
                    base_probs * self.config.concentration).sample()
                
                sharpened = sample_probs ** self.config.leaf_sharpen
                sample_probs = sharpened / sharpened.sum()
                node.param.data.copy_(torch.log(sample_probs))
            else:
                _set_leaf(node.left, depth + 1)
                _set_leaf(node.right, depth + 1)
        _set_leaf(self.root, 0)


    def _train_step(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Trains the model for multiple epochs and returns a dictionary of KL and MSE losses."""
        if self.root is None:
            self.setup_network(X, y)

        B = X.shape[0]
        split_idx = int(B * (1 - self.test_size))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        best_total_loss, best_kl_loss, best_mse_loss, best_argmax_acc_loss = float('inf'), float('inf'), float('inf'), float('inf')

        patience_ctr = 0
        for epoch in range(self.num_epochs):
            train_kl = self._train_epoch(X_train, y_train)
            val_total, val_kl, val_mse, val_acc = self._evaluate(X_val, y_val)

            if val_total < best_total_loss:
                best_total_loss, best_kl_loss, best_mse_loss, best_argmax_acc_loss = val_total, val_kl, val_mse, val_acc
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.config.patience:
                    break


        return {
            "actual_loss": best_total_loss, 
            "kl_loss": best_kl_loss, 
            "mse_loss": best_mse_loss,
            "argmax_acc_loss" : best_argmax_acc_loss
            }

    def _train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Trains for one epoch and returns KL loss."""
        self.train()
        indices = torch.randperm(len(X))
        total = 0

        for start in range(0, len(X), self.batch_size):
            batch_indices = indices[start:start + self.batch_size]
            batch_X, batch_y = X[batch_indices].to(self.device), y[batch_indices].to(self.device)

            self.optimizer.zero_grad()
            total_loss, _, _, _ = self.cal_loss(batch_X, batch_y)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
            self.optimizer.step()

            total += total_loss.item()

        return total / (len(X) / self.batch_size)

    def _evaluate(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[float, float]:
        """Evaluates KL and MSE loss on validation set."""
        self.eval()
        total, total_kl, total_mse, total_acc = 0, 0, 0, 0

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch_X, batch_y = X[start:start + self.batch_size].to(self.device), y[start:start + self.batch_size].to(self.device)
                total_loss, kl_loss, mse_loss, acc_loss = self.cal_loss(batch_X, batch_y)
                total += total_loss.item()
                total_kl += kl_loss.item()
                total_mse += mse_loss.item()
                total_acc += acc_loss.item()

        k = (len(X) / self.batch_size)
        return total / k, total_kl / k, total_mse / k, total_acc / k
    

    def cal_loss(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute KL and MSE loss between predicted and target distributions, and argmax accuracy."""
        B = y.size(0)

        if hasattr(self, 'x_mean') and hasattr(self, 'x_std'):
            X = (X - self.x_mean) / self.x_std

        pred = torch.zeros_like(y)
        leaf_acc = self.root.cal_prob(
            X,
            torch.ones(B, 1, device=self.device)
        )

        for path_prob, Q in leaf_acc:
            pred += path_prob * Q
        
        pred = pred / (pred.sum(dim=1, keepdim=True) + 1e-8)

        T = self.config.temp_teacher
        if T != 1.0:
            teacher_logits = torch.log(y + 1e-8) / T
            y_soft         = torch.softmax(teacher_logits, dim=1)
            kl_scale       = T * T
        else:
            y_soft, kl_scale = y, 1.0

        kl_loss   = F.kl_div(torch.log(pred + 1e-8), y_soft, reduction='batchmean') * kl_scale
        mse_loss  = F.mse_loss(pred, y, reduction='mean')
        total     = kl_loss + self.config.beta_mse * mse_loss

        argmax_acc = (pred.argmax(dim=1) == y.argmax(dim=1)).float().mean()

        return total, kl_loss, mse_loss, argmax_acc

        # T = self.config.temp_teacher
        # if T != 1.0:
        #     eps = 1e-8
        #     teacher_logits   = torch.log(y + eps)
        #     teacher_logits_T = teacher_logits / T
        #     y_sharp = torch.softmax(teacher_logits_T, dim=1)
        #     kl_scale = T * T
        # else:
        #     y_sharp = y
        #     kl_scale = 1.0
        
        # leaf_acc = self.root.cal_prob(
        #     X,
        #     torch.ones(batch_size, 1, device=self.device)
        # )
    
        # kl_loss, mse_loss = 0., 0.
        # correct, total_weight = 0, 0

        # for path_prob, Q in leaf_acc:
            
        #     # KL divergence between predicted and true label
        #     kl_div = F.kl_div(Q.log(), y_sharp, reduction='batchmean') * kl_scale
            
        #     # MSE loss
        #     mse = F.mse_loss(Q, y)

        #     kl_loss += (path_prob * kl_div).sum()
        #     mse_loss += (path_prob * mse).sum()

        #     pred_class = Q.argmax(dim=1)
        #     true_class = y.argmax(dim=1)
            
        #     weight = path_prob.squeeze() 
        #     correct += (weight * (pred_class == true_class).float()).sum().item()
        #     total_weight += weight.sum()

        # ratio = float(correct / total_weight) if total_weight > 0 else 0.0
        # argmax_accuracy = torch.tensor(ratio, device=self.device)

        # total_loss = kl_loss + self.config.beta_mse * mse_loss
        
        # return total_loss / batch_size, kl_loss / batch_size, mse_loss / batch_size, argmax_accuracy


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



    def log_metrics(
            self,
            metrics: Dict[str, float]
        ):
        """Log decision tree loss values"""
        if self.logger is None:
            return
        self.logger.info(f"DecisionTree Loss:   {metrics['actual_loss']:.4f}")
        self.logger.info(f"    KL Loss:         {metrics['kl_loss']:.4f}")
        self.logger.info(f"    MSE Loss:        {metrics['mse_loss']:.4f}")
        self.logger.info(f"    Argmax Acc Loss: {metrics['argmax_acc_loss']:.4f}")

    def set_seed(self, seed):
        """Ensure all randomness is controlled for reproductibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        


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
