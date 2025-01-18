import numpy as np
import cv2
import deepmind_lab
from typing import Optional, Tuple, Dict

from .config import Config

class LabEnvironment:
    def __init__(self, config: Config):
        self.config = config
        self._env = None
        self.action_space = np.zeros(7)

        self._setup_env()

    def _setup_env(self) -> None:
        self._env = deepmind_lab.Lab(
            level=self.config.env.level,
            observations=self.config.env.observations,
            config={
                "width": str(self.config.env.width),
                "height": str(self.config.env.height),
                "fps": str(self.config.env.fps)
            },
            renderer="hardware"
        )

    def reset(self, seed: Optional[int] = 123) -> Tuple[np.ndarray]:
        self._env.reset(seed=seed) if seed is not None else self._env.reset()
        obs = self._env.observations()
        return obs["RGB_INTERLEAVED"]
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Execute action in environment
        Args:
            action: (7,) containing
                [LOOK_LEFT_RIGHT, LOOK_DOWN_UP, STRAFE, MOVE_BACK_FORWARD,
                FIRE, JUMP, CROUCH]
        
        Returns:
            frame: (H,W,C)
            reward: scalar value
            terminated: episode ended naturally
        """
        reward = self._env.step(action, num_steps=self.config.env.num_steps)

        if self._env.is_running():
            obs = self._env.observations()
            frame = obs["RGB_INTERLEAVED"]
            done = False
        else:
            frame = np.zeros((self.config.env.height, self.config.env.width, 3))
            done = True
        
        return frame, reward, done
    

    def render(self, observation: Tuple[np.ndarray]) -> None:
        frame = cv2.cvtColor(observation.astype(np.uint8), cv2.COLOR_RGB2BGR)
        frame = cv2.resize(frame, (self.config.env.render_width, self.config.env.render_height), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("DeepMind Lab", frame)
        cv2.waitKey(1)


    def close(self) -> None:
        if self._env:
            self._env.close()

    