
from src.agent.cnn import CNN

class WorldModel:
    def __init__(
            self,
            config
        ):
        self.config = config
        self.device = config.device

        self.cnn = CNN(
            input_channels=config.worldmodel.input_channels,
            hidden_channels=config.worldmodel.hidden_channels,
            kernel_sizes=config.worldmodel.kernel_sizes,
            strides=config.worldmodel.strides,
            paddings=config.worldmodel.paddings,
            use_batchnorm=config.worldmodel.paddings,
            activation_fn=config.worldmodel.activation_fn
        )


# add to config
@dataclass
class WorldModel:
    input_channels: int = 3,
    hidden_channels: List[int] = [32, 64,  3],
    kernel_sizes:    List[int] = [ 3,  3,  3],
    strides:         List[int] = [ 1,  1,  1],
    paddings:        List[int] = [ 1,  1,  1],
    use_batchnorm: bool = True,
    activation_fn: torch.nn.Module = torch.nn.ReLU()
