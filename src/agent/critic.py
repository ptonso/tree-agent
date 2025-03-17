import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple

from src.agent.mlp import MLP
from src.agent.cnn import CNN

class Critic(nn.Module):
    def __init__(
            self,
            state_dim: Union[Tuple[int,...], int], 
            config: object
            ):
        super().__init__()
        self.config = config
        self.device = config.device
        self.state_dim = state_dim
        self.in_type = "latent"

        if isinstance(state_dim, Tuple):
            self.in_type = "obs"
            self.cnn = CNN(
                input_channels=state_dim[0],
            )
            with torch.no_grad():
                dummy_input = torch.zeros(1, *state_dim)
                cnn_output_dim = self.cnn(dummy_input).shape[1]
            state_dim = cnn_output_dim

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
        if self.in_type == "obs":
            flattened_features = self.cnn(state)
            return self.value_network(flattened_features)
        return self.value_network(state)
    

    def compute_saliency(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Compute saliency map"""
        state = state.clone().detach().requires_grad_(True)
        next_state = next_state.clone().detach()
        value = self.forward(state)
        next_value = self.forward(next_state)
        (next_value - value).backward(torch.ones_like(value))
        saliency = state.grad.clone()
        saliency[saliency < 0] = 0
        return saliency.detach().cpu().numpy()


    def _initialize_weights(self) -> None:
        for m in self.value_network.modules():
            if isinstance(m, nn.Linear):
                # nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)