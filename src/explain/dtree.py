import torch
import numpy as np
from typing import List, Dict, Literal, Optional
import logging

from src.agent.trajectory import Trajectory
from src.agent.batch import Batch, BatchTensors, BatchNumpys
from src.run.config import Config, SoftConfig, RigidConfig, SKLearnConfig

from src.explain.sklearn_tree import SKLearnDecisionTree, SKLearnConfig
from src.explain.rigid_tree import RigidDecisionTree, RigidConfig
from src.explain.soft_tree import SoftDecisionTree, SoftConfig



class DecisionTreeWrapper:
    def __init__(
            self,
            tree_type: Literal["ddt", "rigid", "sklearn"],
            config: Config,
            logger: logging.Logger
        ):
        self.tree_type = tree_type
        self.config = config
        self.device = config.device
        self.logger = logger
        self.input_dim = config.agent.world_model.latent_dim
        self.output_dim = config.agent.action_dim
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initiize appropriate decision tree."""

        if self.tree_type == "sklearn":
            return SKLearnDecisionTree(config=SKLearnConfig())
        elif self.tree_type == "ddt" or self.tree_type == "soft":
            return SoftDecisionTree(config=SoftConfig())
        elif self.tree_type == "rigid":
            return RigidDecisionTree(config=RigidConfig())
        else:
            raise ValueError(f"Unsupported decision tree type: {self.tree_type}")
        
    def train_step(self, trajectories: List[Trajectory]) -> Dict[str, float]:

        batch = Batch(trajectories, self.device)

        if self.tree_type == "sklearn":
            batch_data: BatchNumpys = batch.prepare_numpys()
            metrics = self.model.train_step(
                batch_data.states,
                batch_data.actions_prob,
                batch_data.dones
            )
        
        elif self.tree_type in ["ddt", "soft"]:
            batch_data: BatchTensors = batch.prepare_tensors()
            metrics = self.model.train_step(
                batch_data.states,
                batch_data.actions_prob
            )

        else:
            return None

        self.log_metrics(metrics)

    def log_metrics(
            self,
            metrics: Dict[str, float]
        ):
        """Log decision tree loss values"""
        self.logger.info(f"DecisionTree Loss:   {metrics['actual_loss']:.4f}")
        self.logger.info(f"    KL Loss:         {metrics['kl_loss']:.4f}")
        self.logger.info(f"    MSE Loss:        {metrics['mse_loss']:.4f}")
        if "accuracy" in metrics.keys():
            self.logger.info(f"    Accuracy:    {metrics['accuracy']:.4f}")


    def set_seed(self, seed):
        """Ensure all randomness is controlled for reproductibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        