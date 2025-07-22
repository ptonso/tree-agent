import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from src.agent.mlp import MLP
from src.agent.cnn import CNN
from src.agent.structures import State, Action

class Actor(nn.Module):
    def __init__(
            self,
            state_dim: Union[Tuple[int,...], int], # d*K
            action_dim: int,
            config: object
            ):
        super().__init__()
        self.config = config
        self.device = config.device
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.in_type = "latent"

        if isinstance(state_dim, Tuple):
            self.in_type = "obs"
            self.cnn = CNN(
                input_channels=state_dim[0]
            )
            with torch.no_grad():
                dummy_input = torch.zeros(1, *state_dim)
                cnn_output_dim = self.cnn(dummy_input).shape[1]
            state_dim = cnn_output_dim

        self.policy_network = MLP(
            input_dim=state_dim,
            output_dim=action_dim, # 3 options
            hidden_layers=self.config.agent.actor.layers
        )
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.agent.actor.lr
        )

        self._initialize_weights()


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, *state_dim)
        Returns:
            action_probs: (batch_size, action_dim)
        """
        if self.in_type == "obs":
            flattened_features = self.cnn(state)
            logits = self.policy_network(flattened_features)
        else:
            logits = self.policy_network(state) # (batch_size, action_dim)
        
        return F.softmax(logits, dim=-1)
    
    def policy(self, state: State) -> Action:
        """
        Select action based on current policy.
        State: encoded state [B, E*2]
        """
        with torch.no_grad():
            action_probs = self.forward(state.as_tensor) # (B, 7)
            action_dist = torch.distributions.Categorical(action_probs)
            discrete_action = action_dist.sample()

        return Action(
            action_probs=action_probs,        # (B, 3)
            sampled_action = discrete_action, # int
            device=self.device
        )        
    
    def _initialize_weights(self) -> None:
        for m in self.policy_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)