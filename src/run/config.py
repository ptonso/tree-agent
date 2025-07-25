import torch
from typing import List, Literal, Union, Optional, Tuple
from dataclasses import dataclass, field


# BASE CONFIG

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


# MODELS

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
    channels: List[int]  = field(default_factory=lambda:  [ 128,  64, 32])
    kernels:  List[int]  = field(default_factory=lambda:  [   4,   4,  4])
    strides:  List[int]  = field(default_factory=lambda:  [   2,   2,  1])
    paddings: List[int]  = field(default_factory=lambda:  [   1,   1,  1])
    fc_layers:  int      = 2
    fc_units:   int      = 256
    activation: str      = "silu"
    norm_type:  str      = "layer"

@dataclass
class TransitionConfig(BaseConfig):
    lr: float = 1e-4
    layers: List[int] = field(default_factory=lambda: [64, 64])

    
@dataclass
<<<<<<< HEAD
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
=======
class WorldModelConfig(BaseConfig):
    lr:                float = 2e-4
    latent_dim:        int   = 64
    n_epochs:          int   = 4
    mb_size:           int   = 64
    beta_pred:         float = 3.0
    beta_dym:          float = 0.001
    beta_rep:          float = 0.0001
    horizon:           int   = 0
    saliency_bonus:    float = 3.0
    blur_kernel:       int   = 3
    free_nats:         float = 1.00
>>>>>>> temp-work
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
    actor:       ActorConfig      = field(default_factory=ActorConfig)
    critic:      CriticConfig     = field(default_factory=CriticConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)


@dataclass
class SoftConfig(BaseConfig):
    depth:           int   = 3
    lr:              float = 2e-4
    # momentum:        float = 0.9
    beta_uniform:    float = 0.0
    beta_mse:        float = 0.05
    lmbda:           float = 0.1
    num_epochs:      int   = 40
    patience:        int   = 5
    batch_size:      int   = 64
    test_size:       float = 0.2
    concentration:   float = 20.0
    leaf_sharpen:    float = 1.0
    temp_teacher:    float = 1.2



@dataclass
class BaseVisConfig(BaseConfig):
    bgc:  Tuple[int, int, int] = (255, 255, 255)
    blue: Tuple[int, int, int] = (255, 0, 0)
    red:  Tuple[int, int, int] = (0, 0, 255)
    embedding_width:       int = 100
    top_margin:            int = 20
    lateral_margin:        int = 20


@dataclass
class VAEVisualizerConfig(BaseVisConfig):
    window_width:  int = 720
    window_height: int = 480
    main_height:   int = 200
    window_name:   str = "Autoencoder"
    mode: Literal["full", "actual"] = "actual"
    saliency_mode: bool = True

@dataclass
class TreeVisualizerConfig(BaseVisConfig):
    window_width:      int   = 720
    window_height:     int   = 680
    font_scale:        float = 0.5
    font_thickness:    int   = 1
    window_name:       str   = "Soft Decision Tree"
    show_embed:        bool  = False
    show_legend:       bool  = False
    show_prob_text:    bool  = True
    show_label:        bool  = True
    img_size:          int   = 120 # [16..256]
    separation_factor: float = 3.0

@dataclass
class OverallVisualizerConfig(BaseVisConfig):
    window_width: int = 1280
    window_height: int = 950
    window_name: str = "Visualizer"

@dataclass
class VisConfig(BaseVisConfig):
    vae: VAEVisualizerConfig = field(default_factory=VAEVisualizerConfig)
    tree: TreeVisualizerConfig = field(default_factory=TreeVisualizerConfig)
    overall: OverallVisualizerConfig = field(default_factory=OverallVisualizerConfig)


@dataclass
class SessionConfig(BaseConfig):
    type:                  str  = "train"
    n_episodes:            int  = 100
    n_steps:               int  = 1000
    seed:                  int  = 42
    render:                bool = True
    vae_vis:               bool = False
    online_buffer:         int  = 4
    replay_buffer:         int  = 12
    vae_warmup_episodes:   int  =  0# 200 # 100
    dtree_warmup_episodes: int  =  0# 200 # 700
    vis_prints_path: Optional[str] = None # "reports/vae-saleincy-test"


@dataclass
class ExpConfig(BaseConfig):
    name: str         = "baseline"
    console_log: bool = True
    n_runs: int       = 3


@dataclass
class Config(BaseConfig):
    env:     EnvConfig     = field(default_factory=EnvConfig)
    agent:   AgentConfig   = field(default_factory=AgentConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    exp:     ExpConfig     = field(default_factory=ExpConfig)
    soft:    SoftConfig    = field(default_factory=SoftConfig)


