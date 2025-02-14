import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agent.mlp import MLP
from src.run.config import TransitionConfig

class TransitionModel(nn.Module):
    def __init__(
            self, 
            latent_dim: int, 
            action_dim: int, 
            cfg:"TransitionConfig",
            device: str):
        super().__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.device = device
        
        self.transition_network = MLP(
            input_dim=latent_dim + action_dim, 
            output_dim=latent_dim*2, 
            hidden_layers=cfg.layers)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=cfg.lr
        )

        self._initialize_weights()

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        action = action.unsqueeze(-1)  # (B,) -> (B,1) 
        state_action = torch.cat([z, action], dim=-1)
        mu, logvar = self.transition_network(state_action).chunk(2, dim=1)
        return mu, logvar


    def _initialize_weights(self) -> None:
        for m in self.transition_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
