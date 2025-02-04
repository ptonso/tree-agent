import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Dict, Optional

from src.agent.actor import Actor
from src.agent.critic import Critic
from src.agent.world_model import WorldModel
from src.agent.structures import State, Action
from src.agent.trajectory import Trajectory
from src.agent.batch import Batch, BatchTensors

class Agent:
    def __init__(
            self, 
            action_dim: int, 
            config: object
        ):
        self.device = config.device
        self.action_dim = action_dim
        self.state_dim = config.agent.world_model.latent_dim
        self.config = config
        self.gamma = config.agent.gamma

        self.actor = Actor(
            state_dim=self.state_dim,
            action_dim=3,
            config=self.config
        ).to(self.device)

        self.critic = Critic(
            state_dim=self.state_dim,
            config=self.config
        ).to(self.device)

        self.world_model = WorldModel(
            config=self.config,
            critic=self.critic
        ).to(self.device)



    def policy(self, state: State) -> Action:
        """
        Select action based on current policy.
        State: encoded state [B, E*2]
        """
        with torch.no_grad():
            action_probs = self.actor(state.as_tensor) # (B, 7)
            action_dist = torch.distributions.Categorical(action_probs)
            discrete_action = action_dist.sample().item()

        return Action(
            action_probs=action_probs,        # (B, 3)
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
        action_probs = self.actor(states) # [B*T, action_dim]
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze() # [B*T]
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
    
    def adaptive_gradient_clipping(
            self,
            parameters, 
            clip_factor=0.3, 
            eps=1e-3
        ):
        for param in parameters:
            if param.grad is not None:
                param_norm = param.norm()
                grad_norm = param.grad.norm()
                clip_value = clip_factor * param_norm + eps
                if grad_norm > clip_value:
                    param.grad *= clip_value / (grad_norm + eps)


    def train_step(
            self,
            trajectories: List[Trajectory],
            logger: Optional[object] = None,
        ) -> Dict[str, float]:
        """Train agent using list of trajectory"""

        batch = Batch(trajectories, self.device)
        tensors = batch.tensors
        N = tensors.states.size(0)
        self.mb_size = self.config.agent.mb_size


        state_values_full, next_state_values_full,advantages_full = \
            self._setup_values_and_advantage(tensors)
        
        indices = torch.randperm(N, device=self.device)

        sum_actor_loss, sum_critic_loss, sum_total_loss = 0.0, 0.0, 0.0
        count_samples = 0

        for start in range(0, N, self.mb_size):
            end = start + self.mb_size
            mb_idx = indices[start:end]

            mb_states, mb_actions, mb_returns, mb_adv, mb_masks = \
                self._setup_minibatch(tensors, advantages_full, mb_idx)
            
            mb_state_values = self.critic(mb_states)

            actor_loss = self.compute_actor_loss(mb_states, mb_actions, mb_adv.squeeze(-1))
            actor_loss = actor_loss * mb_masks.mean()
            critic_loss = self.compute_critic_loss(mb_state_values, mb_returns)
            critic_loss = critic_loss * mb_masks.mean()
            total_loss = actor_loss + critic_loss

            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()

            self.adaptive_gradient_clipping(
                list(self.actor.parameters()) + 
                list(self.critic.parameters())
                )
        
            self.actor.optimizer.step()
            self.critic.optimizer.step()

            batch_size_this = (end - start)
            sum_actor_loss  += actor_loss.item()  * batch_size_this
            sum_critic_loss += critic_loss.item() * batch_size_this
            sum_total_loss  += total_loss.item()  * batch_size_this
            count_samples  += batch_size_this

        metrics = self._setup_metrics(
            state_values_full, next_state_values_full,
            advantages_full, tensors,
            sum_actor_loss, sum_critic_loss, sum_total_loss,
            count_samples
        )

        if logger is not None:
            logger.info(f"Actor Loss: {metrics['actor_loss']:.4f}")
            logger.info(f"Critic Loss: {metrics['critic_loss']:.4f}")

        return metrics


    def _setup_values_and_advantage(self, tensors: BatchTensors):
        """
        1) Compute state_values and next_state_values in smaller chunks if needed.
        2) Compute advantages = returns - state_values (masking out invalid).
        3) Standardize advantages (only on valid entries).
        4) Return the full-size tensors for state_values, next_state_values, advantages.
        """
        N = tensors.states.size(0)

        state_values_full = torch.zeros(N, 1, device=self.device)
        next_state_values_full = torch.zeros(N, 1, device=self.device)

        mb_size = self.config.agent.mb_size
        
        for start in range(0, N, mb_size):
            end = start + mb_size
            mb_states = tensors.states[start:end]
            mb_next_states = tensors.next_states[start:end]
            with torch.no_grad():
                sv  = self.critic(mb_states)
                nsv = self.critic(mb_next_states)
            state_values_full[start:end]      = sv
            next_state_values_full[start:end] = nsv

        advantages_full = (tensors.returns.unsqueeze(-1) - state_values_full)
        advantages_full = advantages_full * tensors.masks.unsqueeze(-1)

        valid_mask = (tensors.masks > 0).unsqueeze(-1)
        valid_advantages = advantages_full[valid_mask]
        if valid_advantages.numel() > 1:
            mean_adv = valid_advantages.mean()
            std_adv  = valid_advantages.std() + 1e-8
            advantages_full[valid_mask] = (valid_advantages - mean_adv) / std_adv

        return state_values_full, next_state_values_full, advantages_full
    

    # def _setup_values_and_advantage(self, tensors: BatchTensors):
    #     """
    #     1) Compute state_values and next_state_values in smaller chunks if needed.
    #     2) Compute advantages = returns - state_values (masking out invalid).
    #     3) Standardize advantages (only on valid entries).
    #     4) Return the full-size tensors for state_values, next_state_values, advantages.
    #     """
    #     N = tensors.states.size(0)

    #     state_values_full = torch.zeros(N, 1, device=self.device)
    #     next_state_values_full = torch.zeros(N, 1, device=self.device)

    #     mini_batch_size = self.config.agent.mini_batch_size
        
    #     for start in range(0, N, mini_batch_size):
    #         end = start + mini_batch_size
    #         mb_states = tensors.states[start:end]
    #         mb_next_states = tensors.next_states[start:end]
    #         with torch.no_grad():
    #             sv  = self.critic(mb_states)
    #             nsv = self.critic(mb_next_states)
    #         state_values_full[start:end]      = sv
    #         next_state_values_full[start:end] = nsv

    #     normalized_returns = self._normalize_returns(tensors)
    #     advantages_full = (normalized_returns - state_values_full) * tensors.masks.unsqueeze(-1)
    #     return state_values_full, next_state_values_full, advantages_full


    def _normalize_returns(
        self,
        tensors: BatchTensors
    ) -> torch.tensor:
        valid_mask = (tensors.masks > 0).unsqueeze(-1)
        valid_returns = tensors.returns.unsqueeze(-1)[valid_mask]
        if valid_returns.numel() > 1:
            min_return = torch.quantile(valid_returns, 0.05)
            max_return = torch.quantile(valid_returns, 0.95)
            return_range = max_return - min_return
            return_range = torch.clamp(return_range, min=1.0) # Prevent zero division
            normalized_returns = (tensors.returns.unsqueeze(-1) - min_return) / return_range
        else:
            normalized_returns = tensors.returns.unsqueeze(-1)
        return normalized_returns


    def _setup_minibatch(
        self,
        tensors: BatchTensors,
        advantages_full: torch.Tensor,
        mb_idx: torch.Tensor
    ):
        """
        Given a set of indices, gather the relevant slices from the batch:
        - states, actions, returns, advantages, masks
        Returns them on the correct device and shape.
        """
        mb_states    = tensors.states[mb_idx]
        mb_actions   = tensors.actions[mb_idx]
        mb_returns   = tensors.returns[mb_idx]
        mb_masks     = tensors.masks[mb_idx]
        mb_adv       = advantages_full[mb_idx]

        return mb_states, mb_actions, mb_returns, mb_adv, mb_masks


    def _setup_metrics(
        self,
        state_values_full: torch.Tensor,
        next_state_values_full: torch.Tensor,
        advantages_full: torch.Tensor,
        tensors: BatchTensors,
        sum_actor_loss: float,
        sum_critic_loss: float,
        sum_total_loss: float,
        count_samples: int
    ) -> Dict[str, float]:
        """
        Compute final metrics after the mini-batch loop.
        E.g. average losses, mean value, mean returns, etc.
        """
        if count_samples > 0:
            actor_loss_avg  = sum_actor_loss  / count_samples
            critic_loss_avg = sum_critic_loss / count_samples
            total_loss_avg  = sum_total_loss  / count_samples
        else:
            actor_loss_avg  = 0.0
            critic_loss_avg = 0.0
            total_loss_avg  = 0.0

        valid_mask_1d = (tensors.masks > 0)
        valid_values  = state_values_full[valid_mask_1d]
        valid_returns = tensors.returns[valid_mask_1d]
        valid_adv     = advantages_full[valid_mask_1d]

        metrics = {
            'actor_loss': actor_loss_avg,
            'critic_loss': critic_loss_avg,
            'total_loss': total_loss_avg,
            'mean_value': valid_values.mean().item() if valid_values.numel() > 0 else 0.0,
            'mean_return': valid_returns.mean().item() if valid_returns.numel() > 0 else 0.0,
            'mean_advantage': valid_adv.mean().item() if valid_adv.numel() > 0 else 0.0,
            'num_trajectories': len(tensors.trajectory_lengths),
            'max_trajectory_length': max(tensors.trajectory_lengths),
            'min_trajectory_length': min(tensors.trajectory_lengths)
        }
        return metrics



