import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from src.agent.mlp import MLP
from src.agent.cnn import CNN


class Critic(nn.Module):
    def __init__(
            self,
            state_dim: Tuple[int, ...],
            config: object
            ):
        super().__init__()
        self.config = config
        self.device = config.device
        self.state_dim = state_dim


        self.cnn = CNN(
            input_channels=state_dim[0],
            config=self.config
            )   

        with torch.no_grad():
            dummy_input = torch.zeros(1, *state_dim) # Batch size 1
            cnn_output_dim = self.cnn(dummy_input).shape[1]

        self.value_network = MLP(
            input_dim=cnn_output_dim,
            output_dim=1,
            hidden_layers=self.config.agent.critic.layers
        )
        
        self._initialize_weights()


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, *state_dim)
        Returns:
            value: (batch_size, 1)
        """
        flattened_features = self.cnn(state)
        return self.value_network(flattened_features)
    

    def _initialize_weights(self) -> None:
        for m in self.value_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)