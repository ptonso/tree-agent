import torch
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BatchTensors:
    """Container for batched tensors ready for training"""
    trajectory_lengths: List[int]
    states: torch.Tensor            # Shape: [B*T, *state_shape]
    actions: torch.Tensor           # Shape: [B*T, action_dim]
    rewards: torch.Tensor           # Shape: [B*T]
    next_states:torch.Tensor        # Shape: [B*T, *state_shape]
    dones: torch.Tensor             # Shape: [B*T]
    returns: torch.Tensor           # Shape: [B*T]
    masks: torch.Tensor             # Shape: [B*T], used to mask padding
    observations: Optional[torch.Tensor] = None      # Shape: [B*T, H, W, C]
    next_observations: Optional[torch.Tensor] = None # Shape: [B*T, H, W, C]


class Batch:
    """Convert list of trajectories into batch tensors for training"""
    def __init__(self, trajectories: List['Trajectory'], device: str):
        self.trajectories = trajectories
        self.device = device
        self.tensors = self._prepare_batch()

    def _prepare_batch(self) -> BatchTensors:
        """Convert list of trajectories into batched tensors."""
        trajectory_tensors = [traj.get_tensors() for traj in self.trajectories]
        trajectory_lengths = [len(traj) for traj in self.trajectories]
        max_length = max(trajectory_lengths)
        
        batch_size = len(self.trajectories)
        first_traj_tensors = trajectory_tensors[0]

        # Get shapes for initialization
        state_shape = first_traj_tensors[0].shape[1:]
        action_shape = first_traj_tensors[1].shape[1:]

        padded_states = torch.zeros( 
            (batch_size, max_length, *state_shape), 
            device=self.device)
        padded_actions = torch.zeros( 
            (batch_size, max_length), 
            device=self.device, dtype=torch.long)
        padded_rewards = torch.zeros( 
            (batch_size, max_length), 
            device=self.device)
        padded_next_states = torch.zeros( 
            (batch_size, max_length, *state_shape), 
            device=self.device)
        padded_dones = torch.zeros( 
            (batch_size, max_length), 
            device=self.device)
        padded_returns = torch.zeros(
            (batch_size, max_length),
            device=self.device)
        masks = torch.zeros(
            (batch_size, max_length),
            device=self.device)

        has_observations = self.trajectories[0].observations is not None
        if has_observations:
            obs_shape = first_traj_tensors[-1].shape[1:]
            padded_observations = torch.zeros(
                (batch_size, max_length, *obs_shape), 
                device=self.device)
            padded_next_observations = torch.zeros(
                (batch_size, max_length, *obs_shape), 
                device=self.device)
        else:
            padded_observations = None
            padded_next_observations = None

        for i, (states, actions, rewards, next_states, dones, returns, *obs_data) in enumerate(trajectory_tensors):
            length = trajectory_lengths[i]
            padded_states[i, :length] = states
            padded_actions[i, :length] = actions
            padded_rewards[i, :length] = rewards
            padded_next_states[i, :length] = next_states
            padded_dones[i, :length] = dones
            padded_returns[i, :length] = returns
            masks[i, :length] = 1 # Mark valid entries

            if has_observations and len(obs_data) == 2:
                observations, next_observations = obs_data
                padded_observations[i, :length] = observations
                padded_next_observations[i, :length] = next_observations

        # Reshape to [B*T, ...]
        B, T = batch_size, max_length
        return BatchTensors(
            trajectory_lengths=trajectory_lengths,
            states=padded_states.reshape(B * T, *state_shape),
            actions=padded_actions.reshape(B * T, *action_shape),
            rewards=padded_rewards.reshape(B * T),
            next_states=padded_next_states.reshape(B * T, *state_shape),
            dones=padded_dones.reshape(B * T),
            returns=padded_returns.reshape(B * T),
            masks=masks.reshape(B * T),
            observations=padded_observations.reshape(B * T, *obs_shape) if has_observations else None,
            next_observations=padded_next_observations.reshape(B * T, *obs_shape) if has_observations else None
        )
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    