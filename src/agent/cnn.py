
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import torch
import torch.nn as nn
from typing import List


class RMSNorm(nn.Module):
    """Root Mean Square Normalization"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        var = x.pow(2).mean(dim=(1,2,3), keepdim=True) # Channel-wise RMS
        return x / torch.sqrt(var + self.eps) * self.weight.view(1, -1, 1, 1)

class UpsampleDecoderBlock(nn.Module):
    """
    1) Interpolate to the target size
    2) COnvolution + Norm + Activation
    """
    def __init__(self, in_ch, out_ch, target_size=None,
                 norm_layer=None, activation_fn=nn.ReLU()):
        super().__init__()
        self.target_size = target_size
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = norm_layer(out_ch) if norm_layer else nn.Identity()
        self.act = activation_fn

    def forward(self, x):
        if self.target_size is not None:
            H_out, W_out = self.target_size
            x = F.interpolate(x, size=(H_out, W_out), mode='bilinear', alrign_corners=False)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x



class CNN(nn.Module):
    def __init__(
        self,
        config: object,
        transposed: bool = False,
        latent_multiplier: int = 1, 
    ):
        """
        Symmetric CNN that acts as either:
          - Encoder (if transposed=False)
          - Decoder (if transposed=True)

        For a β-VAE encoder (`vae_encoder=True`), we output 2*latent_dim
        so that we can split into mu and logvar. For a standard AE or a
        β-VAE decoder (`vae_encoder=False`), we keep output_dim = latent_dim.
        
        :param config: configuration object with .env.{channels,height,width}
                       and .agent.world_model attributes.
        :param transposed: if True, interpret as a decoder
        :param vae_encoder: if True, output = 2 * latent_dim
                            if False, output = latent_dim
        """
        super().__init__()

        self.config = config
        self.transposed = transposed
        self.latent_multiplier = latent_multiplier

        cnn_config = config.agent.world_model
        self.input_channels = config.env.channels
        self.input_height = config.env.height   # e.g. 60
        self.input_width = config.env.width     # e.g. 80

        self.hidden_channels = cnn_config.hidden_channels.copy()
        self.kernel_sizes = cnn_config.kernel_sizes.copy()
        self.strides = cnn_config.strides.copy()
        self.paddings = cnn_config.paddings.copy()
        self.use_batchnorm = cnn_config.use_batchnorm
        self.activation_fn = cnn_config.activation_fn

        self.latent_dim = cnn_config.latent_dim
        self.output_dim = self.latent_multiplier * cnn_config.latent_dim

        # chose normalization
        norm_type = getattr(cnn_config, "norm_type", "batch")  # Default: BatchNorm
        self.norm_layer = {
            "batch": lambda c: nn.BatchNorm2d(c),
            "group": lambda c: nn.GroupNorm(num_groups=32, num_channels=c),
            "rms": lambda c: RMSNorm(c),
        }.get(norm_type, lambda c: nn.Identity())


        self._initialize_dimensions()
        self._initialize_layers()
        self._initialize_weights()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.transposed:
            x = self.conv_layers(x)
            x = self.latent_projection(x)
        else:
            x = self.latent_projection(x)
            x = self.conv_layers(x)
        return x

    def _initialize_dimensions(self):
        """Precompute feature map shapes"""
        self.shapes = self._compute_all_shapes()
        self.final_shape = self.shapes[-1] if not self.transposed else self.shapes[0]
        # e.g. (C, H, W) => total C*H*W
        self.final_conv_dim = self.final_shape[0] * self.final_shape[1] * self.final_shape[2]

    def _compute_all_shapes(self) -> List[tuple]:
        """Compute shape (C, H, W) at each layer"""
        shapes = []
        C, H, W = self.input_channels, self.input_height, self.input_width
        shapes.append((C, H, W))

        for out_ch, k, s, p in zip(
            self.hidden_channels, self.kernel_sizes, self.strides, self.paddings
        ):
            H, W = self._conv_output_shape(H, W, k, s, p)
            C = out_ch
            shapes.append((C, H, W))
        
        if self.transposed:
            shapes.reverse()
        return shapes

    @staticmethod
    def _conv_output_shape(h_in, w_in, kernel, stride, padding):
        """COmpute H_out, W_out fiven Conv parameters."""
        h_out = (h_in + 2 * padding - kernel) // stride + 1
        w_out = (w_in + 2 * padding - kernel) // stride + 1
        return h_out, w_out

    def _initialize_layers(self):
        if not self.transposed:
            # ENCODER
            preproc_layer = nn.Identity()
            postproc_layer = nn.Flatten()
            channels_in = [shape[0] for shape in self.shapes[:-1]]
            channels_out = [shape[0] for shape in self.shapes[1:]]
            conv_layers = []
        else:
            conv_class = UpsampleDecoderBlock
            preproc_layer = nn.Unflatten(dim=1, unflattened_size=self.final_shape)
            postproc_layer = nn.Sigmoid()
            channels_in = [shape[0] for shape in self.shapes[:-1]]
            channels_out = [shape[0] for shape in self.shapes[1:]]

        layers: List[nn.Module] = [preproc_layer]


        for i, (in_ch, out_ch, k, s, p) in enumerate(zip(channels_in, channels_out, self.kernel_sizes, self.strides, self.paddings)):
            if not self.transposed:
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p))
            else:
                layers.append(UpsampleDecoderBlock(in_ch, out_ch, up_scale=s, norm_layer=self.norm_layer, activation_fn=self.activation_fn))

            layers.append(self.norm_layer(out_ch))

            # activation (except final layer)
            if i < len(channels_in) - 1:
                layers.append(self.activation_fn)

        layers.append(postproc_layer)
        self.conv_layers = nn.Sequential(*layers)

        if not self.transposed:
            self.latent_projection = nn.Linear(self.final_conv_dim, self.output_dim)
        else:
            self.latent_projection = nn.Linear(self.latent_dim, self.final_conv_dim)



    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    from src.run.config import Config

    config = Config()
    encoder = CNN(config=config, transposed=False)
    decoder = CNN(config=config, transposed=True)

    print("Layer shapes in encoder:", encoder.shapes)
    print("Layer shapes in decoder:", decoder.shapes)
    print(f"Encoder embedding dimension: {encoder.latent_dim}")

    # Note: Input tensor should have shape [batch, channels, height, width]
    batch_size = 2
    input_tensor = torch.randn(batch_size, config.env.channels, config.env.height, config.env.width)
    print(f"Input shape: {input_tensor.shape}")

    embedding = encoder(input_tensor)
    print(f"Embedding shape: {embedding.shape}")

    reconstructed = decoder(embedding)
    print(f"Reconstructed shape: {reconstructed.shape}")

    assert reconstructed.shape == input_tensor.shape, "Reconstructed shape does not match input shape."
    print("Reconstruction successful and matches input shape.")