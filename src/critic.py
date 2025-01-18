import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .mlp import MLP

class Critic(nn.Module):
    def __init__(
            self,
            state_dim: Tuple[int, ...],
            hidden_layers: List[int],
            device: str):
        super().__init__()
        self.device = device
        self.state_dim = state_dim

        self.value_network = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_layers=hidden_layers
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
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)