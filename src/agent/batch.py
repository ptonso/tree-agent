import torch
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BatchTensors:
    """Container for batched tensors ready for training
    Shape: [B, T, *] with padding."""
    trajectory_lengths: List[int]
    states: torch.Tensor            # Shape: [B*T, *state_shape]
    actions: torch.Tensor           # Shape: [B*T]
    rewards: torch.Tensor           # Shape: [B*T]
    next_states:torch.Tensor        # Shape: [B*T, *state_shape]
    dones: torch.Tensor             # Shape: [B*T]
    returns: torch.Tensor           # Shape: [B*T]
    masks: torch.Tensor             # Shape: [B*T]
    observations: Optional[torch.Tensor] = None      # Shape: [B*T, C, H, W]
    next_observations: Optional[torch.Tensor] = None # Shape: [B*T, C, H, W]


class Batch:
    """Convert list of trajectories into batch tensors for training"""
    def __init__(self, trajectories: List['Trajectory'], device: str):
        self.trajectories = trajectories
        self.device = device
        self.tensors = self._prepare_batch()

    def _prepare_batch(self) -> BatchTensors:
        trajectory_tensors = [traj.get_tensors() for traj in self.trajectories]
        trajectory_lengths = [len(traj) for traj in self.trajectories]
        max_length = max(trajectory_lengths)

        def pad_tensor(tensor, pad_value=0):
            """Pads tensor to shape (B, T, *)"""
            pad_size = max_length - tensor.shape[0]
            if pad_size > 0:
                return F.pad(tensor, (0, 0) * (tensor.dim() - 1) + (0, pad_size), value=pad_value)
            return tensor
        
        def process_batch_tensors(index):
            """COllects all tensors at a given index from trajectories and optionally pads them."""
            # [B*T, *]
            tensor_list = [traj[index] for traj in trajectory_tensors]
            return torch.cat(tensor_list, dim=0)


        batch_states        = process_batch_tensors(0)  # [B*T, *] or [B, T, *]
        batch_actions       = process_batch_tensors(1)  # [B*T] or [B, T]
        batch_rewards       = process_batch_tensors(2)  # [B*T] or [B, T]
        batch_next_states   = process_batch_tensors(3)  # [B*T, *] or [B, T, *] 
        batch_dones         = process_batch_tensors(4)  # [B*T] or [B, T]
        batch_returns       = process_batch_tensors(5)  # [B*T] or [B, T]

        batch_masks = torch.ones_like(batch_rewards, dtype=torch.float32)

        has_observations = (trajectory_tensors[0][6] is not None)
        batch_obs, batch_next_obs = None, None
        if has_observations:
            batch_obs       = process_batch_tensors(6)  # [B*T, C, H, W] or [B, T, C, H, W]
            batch_next_obs  = process_batch_tensors(7)  # [B*T, C, H, W] or [B, T, C, H, W]


        return BatchTensors(
            trajectory_lengths=trajectory_lengths,
            states=batch_states,
            actions=batch_actions,
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
    
    