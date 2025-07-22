import torch
import numpy as np
from typing import List, Dict, Literal
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
            tree_types: List[Literal["soft", "rigid", "sklearn"]],
            config: Config,
            logger: logging.Logger
        ):
        self.tree_types = tree_types
        self.config = config
        self.device = config.device
        self.logger = logger
        self.input_dim = config.agent.world_model.latent_dim
        self.output_dim = config.agent.action_dim
        self.models = self._initialize_models()

    def _initialize_models(self):
        """Initiize appropriate decision tree."""
        models = {}
        for tree_type in self.tree_types:
            if tree_type == "sklearn":
                models[tree_type] = SKLearnDecisionTree(config=SKLearnConfig())
            elif tree_type == "soft":
                models[tree_type] = SoftDecisionTree(config=SoftConfig())
            elif tree_type == "rigid":
                models[tree_type] = RigidDecisionTree(config=RigidConfig())
            else:
                raise ValueError(f"Unsupported decision tree type: {tree_type}")
        return models

    def train_step(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        batch = Batch(trajectories, self.device)
        all_metrics = {}

        for tree_type, model in self.models.items():            

            if tree_type == "sklearn" or tree_type == "rigid":
                batch_data: BatchNumpys = batch.prepare_numpys()

                metrics = model.train_step(
                    batch_data.states,
                    batch_data.actions_prob,
                    batch_data.dones,
                    batch_data.returns # It should be values, but meh
                )
            elif tree_type == "soft":
                batch_data: BatchTensors = batch.prepare_tensors()
                metrics = model.train_step(
                    batch_data.states,
                    batch_data.actions_prob
                )
            else:
                continue
        
            prefixed_metrics = {f"{tree_type}_{key}": value for key, value in metrics.items()}

            self.log_metrics(tree_type, metrics)
            all_metrics.update(prefixed_metrics)

        return all_metrics

    def log_metrics(
            self,
            tree_type: str,
            metrics: Dict[str, float]
        ):
        """Log decision tree loss values"""
        self.logger.info(f"[{tree_type.upper()+']':<10} DecisionTree Loss:  {metrics['actual_loss']:.4f}")
        self.logger.info(f"[{tree_type.upper()+']':<10}    KL Loss:         {metrics['kl_loss']:.4f}")
        self.logger.info(f"[{tree_type.upper()+']':<10}    MSE Loss:        {metrics['mse_loss']:.4f}")
        self.logger.info(f"[{tree_type.upper()+']':<10}    Argmax Acc Loss: {metrics['argmax_acc_loss']:.4f}")

    def set_seed(self, seed):
        """Ensure all randomness is controlled for reproductibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        