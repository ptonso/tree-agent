import torch
import numpy as np
from typing import Optional


class State:
    def __init__(self, state_data: np.ndarray, device: str):
        """Handle state data transformations
        Args:
            state_data: RGB fraome of shape (H, W, C) or (B, H, W, C)
            device: torch device to use
        """
        if state_data.ndim == 3:
            state_data = np.expand_dims(state_data, axis=0)

        # uint8 [0..255] -> float32 [0..1]
        state_data = state_data.astype(np.float32) / 255.0

        self.state_data = state_data # (B, H, W, C)
        self.shape = state_data.shape
        self.device = device

        self._as_numpy: Optional[np.ndarray] = None
        self._as_tensor: Optional[torch.Tensor] = None
        self._as_tensor_with_grad: Optional[torch.Tensor] = None
        self._as_flattened_tensor: Optional[torch.Tensor] = None

    @property
    def as_numpy(self) -> np.ndarray:
        if self._as_numpy is None:
            self._as_numpy = self.state_data
        return self._as_numpy
    
    @property
    def as_tensor(self) -> torch.Tensor:
        if self._as_tensor is None:
            tensor = torch.tensor(self.state_data, dtype=torch.float32)
            self._as_tensor = tensor.permute(0, 3, 1, 2).to(self.device) # (B, C, H, W)
        return self._as_tensor
    
    @property
    def as_tensor_with_grad(self) -> torch.Tensor:
        if self._as_tensor_with_grad is None:
            self._as_tensor_with_grad = self.as_tensor.clone().detach().requires_grad_(True)
        return self._as_tensor_with_grad
    
    @property
    def as_flattened_tensor(self) -> torch.Tensor:
        if self._as_flattened_tensor is None:
            tensor = self.as_tensor.contiguous()
            batch_size = tensor.shape[0]
            self._as_flattened_tensor = tensor.view(batch_size, -1) # (B, C*H*W)
        return self._as_flattened_tensor



class Action:
    def __init__(self, action_probs: Optional[torch.Tensor] = None,
                 sampled_action: int = None,
                 device: str = "cpu"):
        """Handle action data transformations
        Structures:
            action_probs: Policy network output probs for each action dimension
            sampled_action: sampled action int in {0,1,2}
            lab_action: (7,) sampled numpy array ready for Lab Enviornment interaction
            device: torch device to use
        """
        self.action_probs = action_probs # (B, action_dim)
        self.sampled_action = sampled_action # int
        self.device = device

        self._as_numpy: Optional[np.ndarray] = None       # (action_dim,) probability
        self._as_tensor: Optional[torch.Tensor] = None    # (action_dim,) probability
        self._as_lab: Optional[np.ndarray] = None         # (7,) lab integers

    @property
    def as_tensor(self) -> Optional[torch.Tensor]:
        if self._as_tensor is None and self.action_probs is not None:
            self._as_tensor = self.action_probs.to(self.device)
        return self._as_tensor
    
    @property
    def as_numpy(self) -> Optional[np.ndarray]:
        if self._as_numpy is None and self.action_probs is not None:
            self._as_numpy = self.action_probs.cpu().detach().numpy()
        return self._as_numpy
    
    @property
    def as_lab(self) -> Optional[np.ndarray]:
        if self._as_lab is None and self.sampled_action is not None:
            self._as_lab = self._convert_discrete_to_lab_action(self.sampled_action)
        return self._as_lab

    def _convert_discrete_to_lab_action(self, discrete_action: int) -> np.ndarray:
        """Convert discrete action {0,1,2} to DeepMind Lab action array"""
        lab_action = np.zeros(7, dtype=np.intc)
        if discrete_action == 0:  # Move forward
            lab_action[3] = 1
        elif discrete_action == 1: # Rotate left
            lab_action[0] = -50
        elif discrete_action == 2: # Rotate right
            lab_action[0] = 50
        return lab_action



# class FeatureMap:
#     def __init__(self, feature_maps: torch.Tensor, device: str):
#         """Handle CNN feature map transformations
#         Args:
#             feature_maps: CNN output of shape (B, n_feature_maps, H', W')
#             device: torch device to use
#         """
#         self.feature_maps = feature_maps # (B, n_feature_maps, H', W')
#         self.device = device
#         self.shape = feature_maps.shape

#         self._as_tensor: Optional[torch.Tensor] = None
#         self._as_numpy: Optional[np.ndarray] = None
#         self._flattened: Optional[torch.Tensor] = None

#     @property
#     def as_tensor(self) -> torch.Tensor:
#         if self._as_tensor is None:
#             self._as_tensor = self.feature_maps.to(self.device)
#         return self._as_tensor
    
#     @property
#     def as_numpy(self) -> np.ndarray:
#         if self._as_numpy is None:
#             self._as_numpy = self.feature_maps.cpu().detach().numpy()
#         return self._as_numpy
    
#     @property
#     def flattened(self) -> torch.Tensor:
#         if self._flattened is None:
#             self._flattened = self.as_tensor.contiguous().view(self.as_tensor.size(0), -1)
#         return self._flattened
    