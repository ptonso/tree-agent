from typing import List
from dataclasses import dataclass, field
import torch

@dataclass
class EnvConfig:
    width: int = 80
    height: int = 60
    fps: int = 60
    level: str = "nav_maze_random_goal_01"
    observations: List[str] = field(default_factory=lambda: ["RGB_INTERLEAVED"])
    num_steps: int = 4
    render_width: int = 780
    render_height: int = 480


@dataclass
class AgentConfig:
    actor_lr: float = 0.0005
    critic_lr: float = 0.001
    cnn_lr: float = 0.0005
    gamma: float = 0.99
    entropy_coef: float = 0.01
    actor_layers: List[int] = field(default_factory=lambda: [64, 64])
    critic_layers: List[int] = field(default_factory=lambda: [128])
   
@dataclass
class SessionConfig:
    name: str = "baseline"
    type: str = "train"
    n_episodes: int = 100
    n_steps: int = 1000
    seed: int = 42
    model_dir: str = "data/model"
    render: bool = True

@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"