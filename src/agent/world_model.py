
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


