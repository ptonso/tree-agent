import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict

from .actor import Actor
from .critic import Critic

from .structures import State, Action
from .trajectory import Trajectory
from .batch import Batch

class Agent:
    def __init__(
            self, 
            state_dim: Tuple[int, ...], # (H, W, C)
            action_dim: int, 
            config: object
            ):
        self.device = config.device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.gamma = config.agent.gamma
        self.setup()

    def setup(self):
        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=3,
            hidden_layers=self.config.agent.actor_layers,
            device=self.device
        ).to(self.device)

        self.critic = Critic(
            state_dim=self.state_dim,
            hidden_layers=self.config.agent.critic_layers,
            device=self.device
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.config.agent.actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.config.agent.critic_lr
        )

    def policy(self, state: State) -> Action:
        """Select action based on current policy"""
        with torch.no_grad():
            action_probs = self.actor(state.as_flattened_tensor) # (7,)
            action_dist = torch.distributions.Categorical(action_probs)
            discrete_action = action_dist.sample().item()

        return Action(
            action_probs=action_probs,        # (3,)
            sampled_action = discrete_action, # int
            device=self.device
        )
        
    def compute_actor_loss(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            advantages: torch.Tensor
        ) -> torch.Tensor:
        """Compute actor loss using policy gradient with normalized advantages"""
        action_probs = self.actor(states) # [T, action_dim]
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze() # [T]
        actor_loss = -(advantages.detach() * log_probs).mean()

        entropy = -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=1).mean()
        actor_loss -= self.config.agent.entropy_coef * entropy
        return actor_loss
    
    def compute_critic_loss(
            self,
            state_values: torch.Tensor,
            returns: torch.Tensor
        ) -> torch.Tensor:
        """Compute critic loss using MSE between predicted values and returns"""
        critic_loss = F.mse_loss(state_values, returns.unsqueeze(-1))
        return critic_loss

    def train(
            self,
            trajectories: List[Trajectory],
            verbose: bool = False
        ) -> Dict[str, float]:
        """Train agent using list of trajectory"""

        batch = Batch(trajectories, self.device)
        tensors = batch.tensors

        state_values = self.critic(tensors.states)
        next_state_values = self.critic(tensors.next_states)

        advantages = tensors.returns - state_values.detach()
        advantages = advantages * tensors.masks

        # standardize advantages for stability
        if advantages.numel() > 1:
            valid_advantages = advantages[tensors.masks > 0]
            advantages = (advantages - valid_advantages.mean()) / (valid_advantages.std() + 1e-8)

        actor_loss = self.compute_actor_loss(
            tensors.states,
            tensors.actions,
            advantages
        ) * tensors.masks.mean() # Scale by average mask value

        critic_loss = self.compute_critic_loss(
            state_values,
            tensors.returns
        ) * tensors.masks.mean() # Scale by average mask value

        total_loss = actor_loss + critic_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        total_loss.backward()

        # Clip gradient if configured

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        valid_values = state_values[tensors.masks > 0]
        valid_returns = tensors.returns[tensors.masks > 0]
        valid_advantages = advantages[tensors.masks > 0]

        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'total_loss': total_loss.item(),
            'mean_value': state_values.mean().item(),
            'mean_return': valid_returns.mean().item(),
            'mean_advantage': valid_advantages.mean().item(),
            'num_trajectories': len(batch),
            'max_trajectory_length': max(tensors.trajectory_lengths),
            'min_trajectory_length': min(tensors.trajectory_lengths)
        }

        if verbose:
            print(f"Actor Loss: {metrics['actor_loss']:.4f}, "
                f"Critic Loss: {metrics['critic_loss']:.4f}")
        
        return metrics
        