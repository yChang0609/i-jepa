import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast
from src.models.VAE.base import *

class Encoder(nn.Module):
    def __init__(self, z_dim, in_channels, in_feature_width ,stem_channels=256):
        super().__init__()
        self.in_feature_width = int(in_feature_width)
        # -- stem
        backbone = []
        backbone.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
        backbone.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
        backbone.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        feature_width = self.in_feature_width//2
        channels = stem_channels
        backbone.append(nn.BatchNorm2d(stem_channels))
        backbone.append(nn.ReLU(inplace=True))
        
        # layers
        while True:
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            backbone.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels *= 2
            feature_width //= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))

            if feature_width <= 4:
                break
        
        mlp = []
        mlp.append(nn.Linear(int(channels*feature_width*feature_width), stem_channels))
        mlp.append(nn.BatchNorm1d(stem_channels))

        self.backbone = nn.Sequential(*backbone)
        self.mlp = nn.Sequential(*mlp)
        self.fc_z_mu = nn.Linear(stem_channels, z_dim)
        self.fc_z_logvar = nn.Linear(stem_channels, z_dim)

        self.last_channels = channels
        self.final_feature_width = feature_width

    def forward(self, x):
        batch = x.shape[0]
        x = rearrange(x, "B (H W) C  -> B C H W",H=self.in_feature_width)
        x = self.backbone(x)
        x = rearrange(x, "B C H W  -> B (H W C) ",B=batch)
        x = self.mlp(x)
        mu = self.fc_z_mu(x)
        z_logvar = self.fc_z_logvar(x)
        return mu, z_logvar

class Decoder(nn.Module):
    def __init__(self, z_dim, last_channels, final_feature_width, recover_channels, stem_channels=256):
        super().__init__()
        backbone = []
        # stem
        #1024 * 2 * 2
        backbone.append(nn.Linear(z_dim, int(last_channels*final_feature_width*final_feature_width), bias=False))
        backbone.append(Rearrange('B (C H W) -> B C H W', C=last_channels, H=final_feature_width, W=final_feature_width))
        backbone.append(nn.BatchNorm2d(last_channels))
        backbone.append(nn.ReLU(inplace=True))
        # residual_layer
        # backbone.append(ResidualStack(last_channels, 1, last_channels//4))
        # layers
        channels = last_channels
        feat_width = final_feature_width
        while True:
            if channels == stem_channels:
                break
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels//2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            channels //= 2
            feat_width *= 2
            backbone.append(nn.BatchNorm2d(channels))
            backbone.append(nn.ReLU(inplace=True))
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            backbone.append(
                nn.ConvTranspose2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                )
            )
            
        # recover layer
        backbone.append(
            nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=recover_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        backbone.append(nn.BatchNorm2d(recover_channels))
        backbone.append(nn.ReLU(inplace=True))
        backbone.append(
            nn.ConvTranspose2d(
                in_channels=recover_channels,
                out_channels=recover_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )
        backbone.append(
            nn.ConvTranspose2d(
                in_channels=recover_channels,
                out_channels=recover_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "B C H W -> B (H W) C ", B=batch_size)
        return obs_hat

class VAE(BaseVAE):
    def __init__(self, z_dim, in_channels, in_feature_width, use_amp):
        super().__init__()
        self.use_amp = use_amp
        self.encoder = Encoder(z_dim=z_dim, 
                               in_channels=in_channels, 
                               in_feature_width=in_feature_width)
        
        self.decoder = Decoder(z_dim=z_dim,
                               last_channels=self.encoder.last_channels, 
                               final_feature_width=self.encoder.final_feature_width, 
                               recover_channels=in_channels)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    # -- VAE interface
    def encode(self, input: Tensor) -> List[Tensor]:
        z_mu, z_logvar = self.encoder(input)
        return [z_mu, z_logvar]
    
    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)
    
    def sample(self, params:List[Tensor], **kwargs) -> Tensor:
        return self.reparameterize(params[0],params[1])
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    
