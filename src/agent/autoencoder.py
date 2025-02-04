import torch
import torch.nn as nn
import torch.nn.functional as F

from src.run.config import EncoderConfig, DecoderConfig, WorldModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies RMS normalization over the channel dimension."""
        if x.dim() == 4:    # (B, C, H, W)
            var = x.pow(2).mean(dim=(1,), keepdim=True)  # Normalize per channel
            return x / torch.sqrt(var + self.eps) * self.weight.view(1, -1, 1, 1)
        elif x.dim() == 2:  # (B, Features)
            var = x.pow(2).mean(dim=-1, keepdim=True)  # Normalize per feature
            return x / torch.sqrt(var + self.eps) * self.weight
        else:
            raise ValueError(f"Unexpected input shape {x.shape} for RMSNorm.")
        
class Norm(nn.Module):
    def __init__(self, norm_type: str, num_channels: int):
        super().__init__()
        self.norm_type = norm_type.lower()
        if self.norm_type == "layer":
            self.norm = nn.LayerNorm(num_channels)
        elif self.norm_type == "batch":
            self.norm = nn.BatchNorm2d(num_channels)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(32, num_channels)
        elif self.norm_type == "rms":
            self.norm = RMSNorm(num_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_type == "layer" and x.dim() == 4:
            b,c,h,w = x.shape
            x = x.permute(0,2,3,1) # (B, C, H, W) -> (B, H, W, C)
            x = self.norm(x)
            x = x.permute(0,3,1,2) # Back to (B, C, H, W)
            return x
        else:
            return self.norm(x)

def truncated_normal_(tensor: torch.Tensor, mean=0.0, std=0.02):
    with torch.no_grad():
        tensor.normal_(mean, std)
        while True:
            invalid = (tensor < mean - 2*std) | (tensor > mean + 2*std)
            if not invalid.any():
                break
            tensor[invalid] = torch.randn_like(tensor[invalid]) * std + mean


class BaseVAE(nn.Module):
    def __init__(self):
        super().__init__()
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
            truncated_normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

def make_activation(name: str) -> nn.Module:
    if name.lower() == "silu":
        return nn.SiLU()
    elif name.lower() == "relu":
        return nn.ReLU()
    return nn.Identity()


class Encoder(BaseVAE):
    def __init__(
            self, in_channels: int, in_height: int, in_width: int, 
            latent_dim: int, cfg: "EncoderConfig"):
        super().__init__()
        self.cfg = cfg
        act = make_activation(cfg.activation)
        conv_layers = []
        c_in = in_channels
        for c_out, k, s, p in zip(cfg.channels, cfg.kernels, cfg.strides, cfg.paddings):
            conv_layers.append(nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p))
            conv_layers.append(Norm(cfg.norm_type, c_out))
            conv_layers.append(act)
            conv_layers.append(nn.MaxPool2d(2))
            c_in = c_out
        self.conv = nn.Sequential(*conv_layers)
        with torch.no_grad():
            test_in = torch.zeros(1, in_channels, in_height, in_width)
            test_out = self.conv(test_in)
            self.conv_out_dim = test_out.numel()
        fc_layers = []
        in_dim = self.conv_out_dim
        for _ in range(cfg.fc_layers):
            fc_layers.append(nn.Linear(in_dim, cfg.fc_units))
            fc_layers.append(Norm(cfg.norm_type, cfg.fc_units))
            fc_layers.append(act)
            in_dim = cfg.fc_units
        fc_layers.append(nn.Linear(in_dim, latent_dim*2))
        self.mlp = nn.Sequential(*fc_layers)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.mlp(x)


class Decoder(BaseVAE):
    def __init__(
            self, out_channels: int, out_height: int, out_width: int, 
            latent_dim: int, cfg: "DecoderConfig"):
        super().__init__()
        self.cfg = cfg
        self.out_height = out_height
        self.out_width = out_width
        act = make_activation(cfg.activation)

        # FC stack
        fc_layers = []
        in_dim = latent_dim
        for _ in range(cfg.fc_layers):
            fc_layers.append(nn.Linear(in_dim, cfg.fc_units))
            fc_layers.append(Norm(cfg.norm_type, cfg.fc_units))
            fc_layers.append(act)
            in_dim = cfg.fc_units

        bottleneck_ch = cfg.channels[0]
        bottleneck_h = 8
        bottleneck_w = 8
        fc_layers.append(nn.Linear(in_dim, bottleneck_ch * bottleneck_h * bottleneck_w))
        self.fc = nn.Sequential(*fc_layers)

        deconvs = []
        c_in = bottleneck_ch
        shape_h, shape_w = bottleneck_h, bottleneck_w
        for c_out, k, s, p in zip(cfg.channels[1:], cfg.kernels[1:], cfg.strides[1:], cfg.paddings[1:]):
            output_pad = 1 if s > 1 else 0
            deconvs.append(nn.ConvTranspose2d(c_in, c_out, kernel_size=k, stride=s, padding=p, output_padding=output_pad))
            deconvs.append(Norm(cfg.norm_type, c_out))
            deconvs.append(act)
            c_in = c_out
            shape_h *= 2
            shape_w *= 2

        output_pad = 1 if cfg.strides[0] > 1 else 0
        deconvs.append(nn.ConvTranspose2d(c_in, out_channels, kernel_size=cfg.kernels[0], stride=cfg.strides[0], padding=cfg.paddings[0], output_padding=output_pad))
        self.deconv = nn.Sequential(*deconvs)
        self.apply(self._init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        c0 = self.cfg.channels[0]
        x = x.view(x.size(0), c0, 8, 8)
        x = self.deconv(x)
        x = F.interpolate(x, size=(self.out_height, self.out_width), mode='bilinear', align_corners=False)
        return x


if __name__ == "__main__":
    cfg = WorldModelConfig(latent_dim=32)
    enc = Encoder(3, 64, 64, cfg.latent_dim, cfg.encoder)
    dec = Decoder(3, 64, 64, cfg.latent_dim, cfg.decoder)
    inp = torch.randn(2, 3, 64, 64)
    z = enc(inp)
    mu, logvar = z.chunk(2, dim=1)
    z_sample = mu + torch.exp(0.5*logvar)*torch.randn_like(mu)
    out = dec(z_sample)
    print("Input:", inp.shape, "Latent:", z.shape, "Output:", out.shape)