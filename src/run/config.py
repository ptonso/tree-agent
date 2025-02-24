import torch
from typing import List, Literal, Union, Optional
from dataclasses import dataclass, field

@dataclass
class BaseConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed:   int = 42

@dataclass
class EnvConfig(BaseConfig):
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
class EncoderConfig(BaseConfig):
    channels: List[int]  = field(default_factory=lambda: [ 32,  64, 128])
    kernels:  List[int]  = field(default_factory=lambda: [  4,   4,   3])
    strides:  List[int]  = field(default_factory=lambda: [  2,   2,   1])
    paddings: List[int]  = field(default_factory=lambda: [  1,   1,   1])
    fc_layers:  int      = 2
    fc_units:   int      = 256
    activation: str      = "silu"
    norm_type:  str      = "layer"

@dataclass
class DecoderConfig(BaseConfig):
    channels: List[int]  = field(default_factory=lambda: [128,  64, 32])
    kernels:  List[int]  = field(default_factory=lambda: [  4,   4,  4])
    strides:  List[int]  = field(default_factory=lambda: [  2,   2,  1])
    paddings: List[int]  = field(default_factory=lambda: [  1,   1,  1])
    fc_layers:  int      = 2
    fc_units:   int      = 256
    activation: str      = "silu"
    norm_type:  str      = "layer"

@dataclass
class TransitionConfig(BaseConfig):
    lr: float = 1e-4
    layers: List[int] = field(default_factory=lambda: [64, 64])

    
@dataclass
class WorldModelConfig(BaseConfig):
    lr:              float  = 2e-4
    latent_dim:      int    = 64
    n_epochs:        int    = 3
    mb_size:         int    = 64
    beta_pred:       float  = 3.0
    beta_dym:        float  = 0.001
    beta_rep:        float  = 0.0001
    horizon:         int    = 2
    free_nats:    float  = 1.00
    gradient_clipping: float = 0.5
    encoder: EncoderConfig = EncoderConfig()
    decoder: DecoderConfig = DecoderConfig()
    transition: TransitionConfig = TransitionConfig()


@dataclass
class ActorConfig(BaseConfig):
    lr: float = 3e-5
    layers: List[int] = field(default_factory=lambda: [128, 128])

@dataclass
class CriticConfig(BaseConfig):
    lr: float = 2e-5
    layers: List[int] = field(default_factory=lambda: [256])

@dataclass
class AgentConfig(BaseConfig):
    gamma:        float = 0.99
    entropy_coef: float = 0.05
    verbose_train: bool = True
    mb_size:       int  = 64
    action_dim:    int  = 3
    actor: ActorConfig  = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)


@dataclass
class SoftConfig(BaseConfig):
    depth:       int   = 4
    lr:          float = 0.01
    momentum:    float = 0.5
    lmbda:       float = 0.1
    num_epochs:  int   = 10
    batch_size:  int   = 64
    test_size:   float = 0.2


@dataclass
class RigidConfig(BaseConfig):
    depth:         int = 4
    test_size:   float = 0.2


@dataclass
class SKLearnConfig(BaseConfig):
    criterion: Literal['gini', 'entropy', 'log_loss'] = "gini"
    max_depth: Optional[int] = None
    min_samples_split: Union[float, int] = 2
    min_samples_leaf: Union[float, int] = 1
    min_weight_fraction_leaf: float = 0
    max_leaf_nodes: Optional[int] = None
    min_impurity_decrease: float = 0
    ccp_alpha: float = 0

@dataclass
class ExplainConfig(BaseConfig):
    soft: SoftConfig = field(default_factory=SoftConfig)
    rigid: RigidConfig = field(default_factory=RigidConfig)
    sklearn: SKLearnConfig = field(default_factory=SKLearnConfig)
    

@dataclass
class SessionConfig(BaseConfig):
    type: str = "train"
    n_episodes: int = 100
    n_steps: int    = 1000
    seed: int       = 42
    render: bool    = True
    vae_vis: bool   = False

@dataclass
class ExpConfig(BaseConfig):
    name: str         = "baseline"
    console_log: bool = True
    n_runs: int       = 3


@dataclass
class Config(BaseConfig):
    env: EnvConfig = field(default_factory=EnvConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    exp: ExpConfig = field(default_factory=ExpConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)


