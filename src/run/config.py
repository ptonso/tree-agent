import torch
from typing import List
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    # 16:12
    width: int = 80    # 16*5
    height: int = 60   # 12*5
    fps: int = 60
    level: str = "seekavoid_arena_01"
    observations: List[str] = field(default_factory=lambda: ["RGB_INTERLEAVED"])
    num_steps: int = 4
    render_width: int = 780
    render_height: int = 480

@dataclass
class AgentConfig:
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    cnn_lr: float = 4e-4
    gamma: float = 0.99
    entropy_coef: float = 0.05
    actor_layers: List[int] = field(default_factory=lambda: [64, 64])
    critic_layers: List[int] = field(default_factory=lambda: [128])
    verbose_train: bool = True


@dataclass
class SessionConfig:
    type: str = "train"
    n_episodes: int = 100
    n_steps: int = 1000
    seed: int = 42
    render: bool = True

@dataclass
class ExpConfig:
    name: str = "baseline"
    console_log: bool = True
    seed: int = 42
    n_runs: int = 3


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


