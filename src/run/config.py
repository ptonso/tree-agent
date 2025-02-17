import torch
from typing import List, Tuple
from dataclasses import dataclass, field

@dataclass
class EnvConfig:
    # 16:12
    fps:            int  = 60
    width:          int  = 64    # 16*4
    height:         int  = 48    # 12*4
    channels:       int  = 3
    level:          str  = "seekavoid_arena_01"
    num_steps:      int  = 4
    render_width:   int  = 780
    render_height:  int  = 480
    action_dim:     int  = 7


@dataclass
class EncoderConfig:
    channels: List[int]  = field(default_factory=lambda: [ 32,  64, 128])
    kernels:  List[int]  = field(default_factory=lambda: [  4,   4,   3])
    strides:  List[int]  = field(default_factory=lambda: [  2,   2,   1])
    paddings: List[int]  = field(default_factory=lambda: [  1,   1,   1])
    fc_layers:  int      = 2
    fc_units:   int      = 256
    activation: str      = "silu"
    norm_type:  str      = "layer"

@dataclass
class DecoderConfig:
    channels: List[int]  = field(default_factory=lambda: [128,  64, 32])
    kernels:  List[int]  = field(default_factory=lambda: [  4,   4,  4])
    strides:  List[int]  = field(default_factory=lambda: [  2,   2,  1])
    paddings: List[int]  = field(default_factory=lambda: [  1,   1,  1])
    fc_layers:  int      = 2
    fc_units:   int      = 256
    activation: str      = "silu"
    norm_type:  str      = "layer"

@dataclass
class TransitionConfig:
    lr: float = 1e-4
    layers: List[int] = field(default_factory=lambda: [64, 64])

    
@dataclass
class WorldModelConfig:
    lr:              float  = 2e-4
    latent_dim:      int    = 32
    num_classes:     int    = 32
    n_epochs:        int    = 3
    mb_size:         int    = 64
    beta_pred:       float  = 3.0
    beta_dym:        float  = 1.0
    beta_rep:        float  = 0.1
    horizon:         int    = 2
    free_nats:    float  = 1.00
    gradient_clipping: float = 0.5
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    transition: TransitionConfig = TransitionConfig()


@dataclass
class ActorConfig:
    lr: float = 3e-5
    layers: List[int] = field(default_factory=lambda: [128, 128])

@dataclass
class CriticConfig:
    lr: float = 2e-5
    layers: List[int] = field(default_factory=lambda: [256])

@dataclass
class AgentConfig:
    gamma:        float = 0.99
    entropy_coef: float = 0.05
    verbose_train: bool = True
    mb_size:       int  = 64
    actor: ActorConfig  = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)

@dataclass
class SessionConfig:
    type: str = "train"
    n_episodes: int = 100
    n_steps: int    = 1000
    seed: int       = 42
    render: bool    = True
    vae_vis: bool   = False

@dataclass
class ExpConfig:
    name: str         = "baseline"
    console_log: bool = True
    seed: int         = 42
    n_runs: int       = 3


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


