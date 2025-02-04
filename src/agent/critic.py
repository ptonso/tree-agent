import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from src.agent.mlp import MLP
from src.agent.cnn import CNN


class Critic(nn.Module):
    def __init__(
            self,
            state_dim: int, # E*2
            config: object
            ):
        super().__init__()
        self.config = config
        self.device = config.device
        self.state_dim = state_dim

        self.value_network = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_layers=self.config.agent.critic.layers
        )
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.agent.critic.lr
        )

        self._initialize_weights()



    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, *state_dim)
        Returns:
            value: (batch_size, 1)
        """
        return self.value_network(state)
    

    def _initialize_weights(self) -> None:
        for m in self.value_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)