import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from src.agent.mlp import MLP

class Actor(nn.Module):
    def __init__(
            self,
            state_dim: Tuple[int, ...],
            action_dim: int,
            hidden_layers: List[int],
            device: str):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.policy_network = MLP(
            input_dim=state_dim,
            output_dim=action_dim, # 3 values
            hidden_layers=hidden_layers
        )
        
        self._initialize_weights()


    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch_size, *state_dim)
        Returns:
            action_probs: (batch_size, 7) with only `action_dim` used
        """
        logits = self.policy_network(state)
        probs_3d = F.softmax(logits, dim=-1)
        batch_size = state.size(0)
        probs_7d = torch.zeros(batch_size, 7, device=self.device)
        probs_7d[:, :3] = probs_3d
        return probs_7d
    
    def _initialize_weights(self) -> None:
        for m in self.policy_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)