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


@dataclass
class EncoderConfig:
    channels: List[int]  = field(default_factory=lambda: [32, 64, 128])
    kernels:  List[int]  = field(default_factory=lambda: [4, 4, 4])
    strides:  List[int]  = field(default_factory=lambda: [2, 2, 1])
    paddings: List[int]  = field(default_factory=lambda: [1, 1, 1])
    fc_layers:  int      = 3
    fc_units:   int      = 256
    activation: str      = "silu"
    norm_type:  str      = "layer"

@dataclass
class DecoderConfig:
    channels: List[int]  = field(default_factory=lambda: [128, 64, 32])
    kernels:  List[int]  = field(default_factory=lambda: [4, 4, 4])
    strides:  List[int]  = field(default_factory=lambda: [1, 1, 2])
    paddings: List[int]  = field(default_factory=lambda: [1, 1, 1])
    fc_layers:  int      = 3
    fc_units:   int      = 256
    activation: str      = "silu"
    norm_type:  str      = "layer"
    
@dataclass
class WorldModelConfig:
    cnn_lr:       float  = 1e-3
    latent_dim:   int    = 128
    n_epochs:     int    = 5
    mb_size:      int    = 32
    beta_pred:    float  = 1.0
    beta_triplet: float  = 0.0
    beta_kl:      float  = 0.0
    beta_reward:  float  = 0.0
    beta_value:   float  = 0.0
    free_nats:    float  = 1.0
    gradient_clipping: float = 0.5
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()


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
    gamma:        float = 0.99
    entropy_coef: float = 0.05
    verbose_train: bool = True
    mb_size:       int  = 512
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


