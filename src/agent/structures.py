import torch
import numpy as np
from typing import Optional, Tuple, List


class State:
    """Handle (B, E) latent representations."""
    def __init__(self, state_data: np.ndarray, device: str):
        """
        Args:
            state_data: encoded output in (B, E) format, float32, scaled to [0..1]
        """
        if state_data.ndim < 2:
            state_data = np.expand_dims(state_data, axis=0)
            print(f"state_data.shape: {state_data.shape}")


        self.state_data = state_data 
        self.shape = state_data.shape # (B, E)
        self.device = device

        self._as_tensor: Optional[torch.Tensor] = None
        self._for_render: Optional[np.ndarray] = None # (E,)
        self._mu_logvar : Optional[Tuple[np.ndarray, np.ndarray]] = None # (mu, logvar) each is (B, latent_dim)
    
    @property
    def as_tensor(self) -> torch.Tensor:
        if self._as_tensor is None:
            tensor = torch.tensor(self.state_data, dtype=torch.float32)
            self._as_tensor = tensor.to(self.device) # (B, E)
        return self._as_tensor
        
    @property
    def for_render(self) -> np.ndarray:
        if self._for_render is None:
            self._for_render = self.state_data[0]
        return self._for_render
    
    @classmethod
    def from_encoder(cls, state_data: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, device: str) -> "State":
        """
        initializes from encoder output
        """
        instance = cls(state_data, device)
        instance._mu_logvar = (
            mu.cpu().detach().numpy(), 
            logvar.cpu().detach().numpy()
            )
        return instance




class Observation:
    """Handle (B, C, H, W) observation."""
    def __init__(self, obs_data: np.ndarray, device: str):
        """
        Args:
            obs_data: (B, C, H, W), float32[0..1]
        """
        self.device = device
        self.obs_data = obs_data
        self.shape = obs_data.shape # (B, C, H, W)
        self._as_tensor: Optional[torch.Tensor] = None
        self._for_render: Optional[np.ndarray] = None # (H, W, C) raw format


    @classmethod
    def from_env(cls, obs_data: np.ndarray, device: str) -> "Observation":
        """
        Handle initialize from raw deepmind lab class
        Params:
         - observation : (H, W, C)"""
        raw_obs = obs_data.copy()

        obs_data = np.expand_dims(obs_data, axis=0)

        # uint8 [0..255] -> float32[-0.5..0.5]
        obs_data = (obs_data.astype(np.float32) / 255.0)
        obs_data = np.clip( (obs_data - 0.5), -0.5, 0.5)

        # from (B, H, W, C) to (B, C, H, W)
        obs_data = np.transpose(obs_data, (0, 3, 1, 2))

        instance = cls(obs_data, device)
        instance._for_render = raw_obs
        return instance

    @classmethod
    def from_decoder(cls, obs_data: torch.Tensor, device: str) -> "Observation":
        """
        initializes from decoder (B, H, W, C) float32[-0.5..0.5]
        """
        obs_data = obs_data.cpu().detach().numpy()
        instance = cls(obs_data, device)
        return instance

    @property
    def as_tensor(self) -> torch.Tensor:
        if self._as_tensor is None:
            self._as_tensor = torch.tensor(self.obs_data, dtype=torch.float32).to(self.device)
        return self._as_tensor
    
    @property
    def for_render(self) -> List[np.ndarray]:
        """
        generate (H, W, C) uint8[0..255] image or
        (B, H, W, C) uint8[0..255]
        """
        if self._for_render is None:
            self._for_render = []
            for i in range(self.obs_data.shape[0]):
                img = np.transpose(self.obs_data[i], (1, 2, 0))
                img = (img + 0.5) * 255. 
                img = np.clip(img, 0, 255).astype(np.uint8)
                self._for_render.append(img)

        if len(self._for_render) == 1:
            self._for_render = self._for_render[0]
        return self._for_render


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
        self.sampled_action = (sampled_action.item() if isinstance(sampled_action, torch.Tensor)
                               else sampled_action) # int
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

