import torch
import cv2
import numpy as np
from typing import Optional, Tuple, List


class Latent:
    """Handle latent representations:
    parameter logits
    parameter probabilities
    sampled onehots
    sampled indices
    """
    def __init__(
            self, 
            logits: torch.Tensor,
            device: str, 
        ):
        """
        Args:
            logits: (B, d, K) - d variables of K categories.
        """
<<<<<<< HEAD
        self.logits = logits # (B, d, K)
        self.shape = logits.shape
        self.B = logits.shape[0]
        self.d = logits.shape[1]
        self.K = logits.shape[2]
=======
        if state_data.ndim < 2:
            state_data = np.expand_dims(state_data, axis=0)
            print(f"state_data.shape: {state_data.shape}")


        self.state_data = state_data 
        self.shape = state_data.shape # (B, E)
>>>>>>> temp-work
        self.device = device

        self._as_probs: Optional[torch.Tensor] = None # (B, d, K) - parameter
        self._as_onehot: Optional[torch.Tensor] = None   # (B, d, K) - sampled
        self._as_indices: Optional[np.ndarray] = None    # (B, d)    - sampled
        self._as_tensor: Optional[torch.Tensor] = None # (B, d*K)   - sampled - batch ready


    @classmethod
    def from_encoder(cls, logits: torch.Tensor, device: str) -> "Latent":
        return cls(logits, device)
    
    @classmethod
    def from_numpy(cls, logits: np.ndarray, device: str) -> "Latent":
        tensor_logits = torch.tensor(logits, device=device)
        return cls(tensor_logits, device)
    
    @property
    def as_probs(self) -> torch.Tensor:
        if self._as_probs is None:
            self._as_probs = torch.nn.functional.softmax(self.logits, dim=-1)
        return self._as_probs
    
    @property
    def as_onehot(self) -> torch.Tensor:
        """Samples one-hot representation fo categorical latent."""
        if self._as_onehot is None:
            sample = torch.distributions.Categorical(logits=self.logits).sample() # (B, d)
            self._onehot = torch.nn.functional.one_hot(sample, num_classes=self.K).float() # (B, d, K)
        return self._onehot
    
    @property
    def as_tensor(self) -> torch.Tensor:
        """Flatten onehots to batch processing."""
        if self._as_tensor is None:
            self._as_tensor = self.as_onehot.flatten(start_dim=1) # [B, d, K] -> [B, d*K]
        return self._as_tensor

    @property
    def as_indices(self) -> np.ndarray:
        """Returns the selected class index for each latent dimension."""
        if self._as_indices is None:
            self._as_indices = torch.argmax(self.logits, dim=-1).cpu().numpy() # (B, d)
        return self._as_indices

class Hidden:
    def __init__(self):
        pass



class Observation:
    """Handle (B, C, H, W) observation."""
    def __init__(
            self, 
            obs_data: np.ndarray, 
            device: str,
        ):
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
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)        
                self._for_render.append(img)

        if len(self._for_render) == 1:
            self._for_render = self._for_render[0]
        return self._for_render


class State:
    """Hub class for agent input.
    Can handdle latent input, (future: latent+hidden), raw_observation"""
    def __init__(
            self,
            device: str,
            latent: Latent,
        ):
        self.device = device
        self.latent = latent

        self._as_tensor: Optional[torch.Tensor] = None # (B, K*d,)
        self._for_render: Optional[np.ndarray] = None  # (d,)
        self._state_data: Optional[np.ndarray] = None  # (B, K*d,)


    @classmethod
    def from_encoder(cls, logits: torch.Tensor, device: str) -> "State":
        latent = Latent(logits, device=device)
        return cls(latent=latent, device=device)

    @classmethod
    def from_numpy(cls, logits: np.ndarray, device: str) -> "State":
        latent = Latent.from_numpy(logits, device=device)
        return cls(latent=latent, device=device)

    @property    
    def as_tensor(self) -> torch.Tensor:
        if self._as_tensor is None:
            self._as_tensor = self.latent.as_tensor
        return self._as_tensor

    @property
    def for_render(self) -> np.ndarray:
        if self._for_render is None and self.latent is not None:
            self._for_render = self.latent.as_indices[0]
        return self._for_render


    @property
    def state_data(self) -> np.ndarray:
        if self._state_data is None:
            self._state_data = self.latent.logits.detach().cpu().numpy()
        return self._state_data
    


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

