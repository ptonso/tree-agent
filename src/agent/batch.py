import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BatchTensors:
    """Container for batched tensors ready for training
    Shape: [B, T, *] with padding."""
    trajectory_lengths: List[int]
    states: torch.Tensor            # Shape: [B*T, *state_shape]
    actions: torch.Tensor           # Shape: [B*T]
    actions_prob: torch.Tensor      # Shape: [B*T, *action_dim]
    rewards: torch.Tensor           # Shape: [B*T]
    next_states:torch.Tensor        # Shape: [B*T, *state_shape]
    dones: torch.Tensor             # Shape: [B*T]
    returns: torch.Tensor           # Shape: [B*T]
    masks: torch.Tensor             # Shape: [B*T]
    observations: Optional[torch.Tensor] = None      # Shape: [B*T, C, H, W]
    next_observations: Optional[torch.Tensor] = None # Shape: [B*T, C, H, W]


@dataclass
class BatchNumpys:
    """Container for batched NumPy arrays, ready for storage or processing.
    Shape: [B*T, *] with no padding."""
    trajectory_lengths: List[int]
    states: np.ndarray        # Shape: [B*T, *state_shape]
    actions: np.ndarray       # Shape: [B*T]
    actions_prob: np.ndarray  # Shape: [B*T, *action_dim]
    rewards: np.ndarray       # Shape: [B*T]
    next_states: np.ndarray   # Shape: [B*T, *state_shape]
    dones: np.ndarray         # Shape: [B*T]
    returns: np.ndarray       # Shape: [B*T]
    masks: np.ndarray         # Shape: [B*T]
    observations: Optional[np.ndarray] = None      # Shape: [B*T, C, H, W]
    next_observations: Optional[np.ndarray] = None # Shape: [B*T, C, H, W]



class Batch:
    """Convert list of trajectories into batch tensors for training"""
    def __init__(self, trajectories: List['Trajectory'], device: str):
        self.trajectories = trajectories
        self.device = device
        self.trajectory_lengths = [len(traj) for traj in self.trajectories]

    def _process_batch(self, index, tensor_mode=True):
        if tensor_mode:
            return torch.cat([traj[index] for traj in self.trajectory_tensors], dim=0)
        else:
            return np.concatenate([traj[index] for traj in self.trajectory_numpys], axis=0)


    def prepare_tensors(self) -> BatchTensors:
        self.trajectory_tensors = [traj.get_tensors() for traj in self.trajectories]
        
        batch_states        = self._process_batch(0, tensor_mode=True)  # [B*T, *]
        batch_actions       = self._process_batch(1, tensor_mode=True)  # [B*T]
        batch_actions_prob  = self._process_batch(2, tensor_mode=True).squeeze(1) # [B*T, *]
        batch_rewards       = self._process_batch(3, tensor_mode=True)  # [B*T]
        batch_next_states   = self._process_batch(4, tensor_mode=True)  # [B*T, *] 
        batch_dones         = self._process_batch(5, tensor_mode=True)  # [B*T]
        batch_returns       = self._process_batch(6, tensor_mode=True)  # [B*T]
        batch_masks = torch.ones_like(batch_rewards, dtype=torch.float32)

        has_observations = (self.trajectory_tensors[0][7] is not None)
        batch_obs, batch_next_obs = None, None
        if has_observations:
            batch_obs       = self._process_batch(7, tensor_mode=True)  # [B*T, C, H, W]
            batch_next_obs  = self._process_batch(8, tensor_mode=True)  # [B*T, C, H, W]

        return BatchTensors(
            trajectory_lengths=self.trajectory_lengths,
            states=batch_states,
            actions=batch_actions,
            actions_prob=batch_actions_prob,
            rewards=batch_rewards,
            next_states=batch_next_states,
            dones=batch_dones,
            returns=batch_returns,
            masks=batch_masks,
            observations=batch_obs,
            next_observations=batch_next_obs,
        )
    
    def prepare_numpys(self) -> BatchNumpys:
        self.trajectory_numpys = [traj.get_numpys() for traj in self.trajectories]
        
        batch_states        = self._process_batch(0, tensor_mode=False)  # [B*T, *]
        batch_actions       = self._process_batch(1, tensor_mode=False)  # [B*T]
        batch_actions_prob  = self._process_batch(2, tensor_mode=False)  # [B*T]
        batch_rewards       = self._process_batch(3, tensor_mode=False)  # [B*T]
        batch_next_states   = self._process_batch(4, tensor_mode=False)  # [B*T, *]
        batch_dones         = self._process_batch(5, tensor_mode=False)  # [B*T]
        batch_returns       = self._process_batch(6, tensor_mode=False)  # [B*T]
        batch_masks = np.ones_like(batch_rewards, dtype=np.float32)

        has_observations = (self.trajectory_numpys[0][7] is not None)
        batch_obs, batch_next_obs = None, None
        if has_observations:
            batch_obs       = self._process_batch(7, tensor_mode=False)  # [B*T, C, H, W]
            batch_next_obs  = self._process_batch(8, tensor_mode=False)  # [B*T, C, H, W]

        return BatchNumpys(
            trajectory_lengths=self.trajectory_lengths,
            states=batch_states,
            actions=batch_actions,
            actions_prob=batch_actions_prob,
            rewards=batch_rewards,
            next_states=batch_next_states,
            dones=batch_dones,
            returns=batch_returns,
            masks=batch_masks,
            observations=batch_obs,
            next_observations=batch_next_obs,
        )



    def __len__(self) -> int:
        return len(self.trajectories)
    
    