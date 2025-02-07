import torch
import torch.nn as nn
from typing import List, Optional

class CNN(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: List[int] = [ 32,  64,  96],
            kernel_sizes:    List[int] = [  4,   4,   4],
            strides:         List[int] = [  2,   2,   2],
            paddings:        List[int] = [  1,   1,   1],
            use_batchnorm: bool = True,
            activation_fn: nn.Module = nn.ReLU()
            ):
        """
        Parameters:
        - input_channels: e.g. 3 for RGB
        - hidden_channels: last layer is n of output feature_maps (size n)
        - kernel_sizes: (size n)
        - strides:      (size n)
        - paddings:     (size n)
        - use_batchnorm: if true, add BatchNorm2d
        - activation_fn: activation function after each conv
        """
        super().__init__()

        assert len(hidden_channels) == len(kernel_sizes), "kernel_sizes must match hidden_channels in size"
        assert len(hidden_channels) == len(strides),      "strides must match hidden_channels in size"
        assert len(hidden_channels) == len(paddings),     "paddings must match hidden_channels in size"

        layers = []
        in_channels = input_channels
        for (out_channels, kernel_size, stride, padding) in zip(hidden_channels, kernel_sizes, strides, paddings):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn)
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.flatten(x)
        return x
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = CNN(input_channels=3)
    input_tensor = torch.rand(1, 3, 64, 64) # B x C x W x H

    # hidden layers: [32, 64, 3]
    # kernel sizes : [8, 4, 3]
    # strides :      [4, 3, 1]

    output_tensor = model(input_tensor)
    print("Input shape:", input_tensor.shape)
    print("Output shape:", output_tensor.shape)