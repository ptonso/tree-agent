
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

from src.agent.structures import State, Action, Observation

@dataclass
class TrajectoryData:
    """Static container for trajectory data, optimized for storage in replay buffer.
    Uses numpy arrays for effiecient storage and serialization."""
    states: np.ndarray       # Shape: [T, *state_shape]
    actions: np.ndarray      # Shape: [T]
    actions_prob: np.ndarray # Shape: [T, *action_dim]
    rewards: np.ndarray      # Shape: [T]
    dones: np.ndarray        # Shape: [T]
    next_states: np.ndarray  # Shape: [T, *state_shape]
    returns: Optional[np.ndarray] = None       # Shape: [T]
    advantages: Optional[np.ndarray] = None    # Shape: [T]
    observations: Optional[np.ndarray] = None       # Shape: [T, H, W, C]
    next_observations: Optional[np.ndarray] = None  # Shape: [T, H, W, C]


class Trajectory:
    """Active trajectory being collected, with methods for adding steps and computing returns.
    Can be initialized empty for collection or from TrajectoryData for modification"""
    def __init__(self, device: str, trajectory_id: int = 0, gamma: float = 0.997, n_steps: int = 3):
        self.states:           List[np.ndarray] = []
        self.actions:          List[int] = []
        self.actions_prob:     List[np.ndarray] = []
        self.rewards:          List[float] = []
        self.dones:            List[bool] = []
        self.next_states:      List[np.ndarray] = []
        self.observations:      List[np.ndarray] = []
        self.next_observations: List[np.ndarray] = []
    
        self.trajectory_id = trajectory_id
        self.device = device
        self.gamma = gamma
        self.n_steps = n_steps

    @classmethod
    def from_trajectory_data(cls, trajectory_data: TrajectoryData, gamma: float = 0.997) -> 'Trajectory':
        trajectory = cls(gamma=gamma)
        trajectory.states = [state for state in trajectory_data.states]
        trajectory.actions = [action for action in trajectory_data.actions]
        trajectory.actions_prob = [action for action in trajectory_data.actions]
        trajectory.rewards = trajectory_data.rewards.tolist()
        trajectory.dones = trajectory_data.dones.tolist()
        trajectory.next_states = [state for state in trajectory_data.next_states]
        trajectory.observations = [obs for obs in trajectory_data.observations]
        trajectory.next_observations = [obs for obs in trajectory_data.observations]
        return trajectory

    def append(
            self,
            state: np.ndarray,
            action: int,
            action_prob: np.ndarray,
            reward: float,
            done: bool,
            next_state: np.ndarray,
            observation: np.ndarray,
            next_observation: np.ndarray,
        ):
        self.states.append(state)
        self.actions.append(action)
        self.actions_prob.append(action_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.observations.append(observation)
        self.next_observations.append(next_observation)

    def reset(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.actions_prob.clear()
        self.rewards.clear()
        self.dones.clear()
        self.next_states.clear()
        self.observations.clear()
        self.next_observations.clear()

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
            actions_prob=np.array(self.actions_prob),
            rewards=np.array(self.rewards),
            dones=np.array(self.dones),
            next_states=np.stack(self.next_states),
            observations=np.stack(self.observations),
            next_observations=np.stack(self.next_observations),
        )
        # trajectory_data.returns = self.compute_returns()
        return trajectory_data

    def __len__(self) -> int:
        return len(self.states)

    def get_instances(self):
<<<<<<< HEAD
        states = [State.from_numpy(st, device=self.device) for st in self.states]
        next_states = [State.from_numpy(st, device=self.device) for st in self.next_states]
        actions = [Action(sampled_action=act, device=self.device) for act in self.actions]
=======
        states = [State(st, device=self.device) for st in self.states]
        next_states = [State(st, device=self.device) for st in self.next_states]
        actions = [Action(sampled_action=act, action_probs=prob, device=self.device) for act, prob in zip(self.actions, self.actions_prob)]
>>>>>>> temp-work
        
        observations = None
        next_observations = None
        if self.observations is not None:
            observations = [Observation(obs, device=self.device) for obs in self.observations]
            next_observations = [Observation(obs, device=self.device) for obs in self.next_observations]
        return states, actions, self.rewards, next_states, self.dones, self.compute_returns(), observations, next_observations

    def get_numpys(self):
        """Convert trajectory data into NumPy arrays."""
        states, actions, rewards, next_states, dones, returns, observations, next_observations = self.get_instances()

        states_numpy = np.stack([s.state_data.squeeze() for s in states])
        actions_numpy = np.array([a.sampled_action for a in actions], dtype=np.int32)
        actions_prob_numpy = np.stack([a.as_numpy.squeeze() for a in actions])
        rewards_numpy = np.array(rewards, dtype=np.float32)
        next_states_numpy = np.stack([s.state_data.squeeze() for s in next_states])
        dones_numpy = np.array(dones, dtype=np.bool_)
        returns_numpy = np.array(returns, dtype=np.float32)

        if observations is not None:
            observations_numpy = np.stack([obs.obs_data for obs in observations])
            next_observations_numpy = np.stack([obs.obs_data for obs in next_observations])
        else:
            observations_numpy = None
            next_observations_numpy = None

        return (states_numpy, actions_numpy, actions_prob_numpy, rewards_numpy, 
                next_states_numpy, dones_numpy, returns_numpy, 
                observations_numpy, next_observations_numpy)


    def get_tensors(self):
        states, actions, rewards, next_states, dones, returns, observations, next_observations = self.get_instances()

        states_tensor = torch.cat([s.as_tensor for s in states], dim=0) 
        actions_tensor = torch.tensor([a.sampled_action for a in actions], dtype=torch.long, device=self.device)
        actions_prob_tensor = torch.cat([a.as_tensor for a in actions], dim=0)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.cat([s.as_tensor for s in next_states], dim=0)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        

        if observations is not None:
            observations_tensor = torch.cat([obs.as_tensor for obs in observations], dim=0)
            next_observations_tensor = torch.cat([obs.as_tensor for obs in next_observations], dim=0)
        else:
            observations_tensor = None
            next_observations_tensor = None

        return (states_tensor, actions_tensor, actions_prob_tensor, rewards_tensor, 
                next_states_tensor, dones_tensor, returns_tensor, 
                observations_tensor, next_observations_tensor)
    