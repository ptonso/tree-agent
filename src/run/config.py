import torch
from typing import List
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    # 16:12
    width: int  = 80    # 16*5
    height: int = 60   # 12*5
    fps: int    = 60
    level: str  = "seekavoid_arena_01"
    observations: List[str] = field(default_factory=lambda: ["RGB_INTERLEAVED"])
    num_steps: int = 4
    render_width: int = 780
    render_height: int = 480


@dataclass
class WorldModelConfig:
    cnn_lr: float = 1e-4
    hidden_channels: List[int] = field(default_factory=lambda: [16, 32])
    kernel_sizes:    List[int] = field(default_factory=lambda: [ 5,  4])
    strides:         List[int] = field(default_factory=lambda: [ 2,  2])
    paddings:        List[int] = field(default_factory=lambda: [ 1,  1])
    use_batchnorm: bool = False,
    activation_fn: torch.nn.Module = torch.nn.ReLU()

@dataclass
class ActorConfig:
    lr: float = 2e-5
    layers: List[int] = field(default_factory=lambda: [32, 32])

@dataclass
class CriticConfig:
    lr: float = 1e-5
    layers: List[int] = field(default_factory=lambda: [64])

@dataclass
class AgentConfig:
    gamma: float = 0.99
    entropy_coef: float = 0.05
    verbose_train: bool = True
    mini_batch_size: int = 256
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)


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


