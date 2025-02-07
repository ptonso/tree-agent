import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from src.agent.mlp import MLP
from src.agent.cnn import CNN

class Actor(nn.Module):
    def __init__(
            self,
            state_dim: Union[Tuple[int,...], int], # E*2
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
            action_probs: (batch_size, 7) with only `action_dim` used
        """
        if self.in_type == "obs":
            flattened_features = self.cnn(state)
            logits = self.policy_network(flattened_features)
        else:
            logits = self.policy_network(state) # (batch_size, action_dim)
        
        probs_3d = F.softmax(logits, dim=-1)
        batch_size = state.size(0)
        probs_7d = torch.zeros(batch_size, 7, device=self.device) # (batch_size, 7)
        probs_7d[:, :3] = probs_3d
        return probs_7d
    
    def _initialize_weights(self) -> None:
        for m in self.policy_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)