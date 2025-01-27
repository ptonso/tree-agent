
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

from src.agent.structures import State, Action

@dataclass
class TrajectoryData:
    """Static container for trajectory data, optimized for storage in replay buffer.
    Uses numpy arrays for effiecient storage and serialization."""
    states: np.ndarray      # Shape: [T, *state_shape]
    actions: np.ndarray     # Shape: [T]
    rewards: np.ndarray     # Shape: [T]
    dones: np.ndarray       # Shape: [T]
    next_states: np.ndarray # Shape: [T, *state_shape]
    returns: Optional[np.ndarray] = None    # Shape: [T]
    advantages: Optional[np.ndarray] = None # Shape: [T]


class Trajectory:
    """Active trajectory being collected, with methods for adding steps and computing returns.
    Can be initialized empty for collection or from TrajectoryData for modification"""
    def __init__(self, device: str, trajectory_id: int = 0, gamma: float = 0.997):
        self.states:      List[np.ndarray] = []
        self.actions:     List[int] = []
        self.rewards:     List[float] = []
        self.dones:       List[bool] = []
        self.next_states: List[np.ndarray] = []
    
        self.trajectory_id = trajectory_id
        self.device = device
        self.gamma = gamma

    @classmethod
    def from_trajectory_data(cls, trajectory_data: TrajectoryData, gamma: float = 0.997) -> 'Trajectory':
        trajectory = cls(gamma=gamma)
        trajectory.states = [state for state in trajectory_data.states]
        trajectory.actions = [action for action in trajectory_data.actions]
        trajectory.rewards = trajectory_data.rewards.tolist()
        trajectory.dones = trajectory_data.dones.tolist()
        trajectory.next_states = [state for state in trajectory_data.next_states]
        return trajectory

    def append(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            done: bool,
            next_state: np.ndarray
        ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)

    def reset(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()

    def compute_returns(self) -> np.ndarray:
        """Compute discounted returns for the trajectory."""
        returns = np.zeros(len(self.rewards))
        next_return = 0
        for t in reversed(range(len(self.rewards))):
            returns[t] = self.rewards[t] + self.gamma * next_return * (1 - self.dones[t])
            next_return = returns[t]
        return returns
        
    def to_storage(self) -> TrajectoryData:
        """Convert to static TrajectoryData for storage"""
        trajectory_data = TrajectoryData(
            states=np.stack(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards),
            dones=np.array(self.dones),
            next_states=np.stack(self.next_states)
        )
        # trajectory_data.returns = self.compute_returns()
        return trajectory_data

    def __len__(self) -> int:
        return len(self.states)

    def get_instances(self):
        states = [State(st, device=self.device) for st in self.states]
        next_states = [State(st, device=self.device) for st in self.next_states]
        actions = [Action(sampled_action=act, device=self.device) for act in self.actions]
        return states, actions, self.rewards, next_states, self.dones, self.compute_returns() 

    def get_tensors(self):
        states, actions, rewards, next_states, dones, returns = self.get_instances()

        states_tensor = torch.cat([s.as_flattened_tensor for s in states], dim=0)
        actions_tensor = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.cat([s.as_flattened_tensor for s in next_states], dim=0)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(self.device)

        return states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor, returns_tensor
    