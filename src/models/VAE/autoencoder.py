import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from src.models.VAE.base import *
import src.models.VAE.encoder_decoder as model
from math import sqrt

class AutoEncoderHead(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim=0):
        super().__init__()
        in_dim = input_dim
        self.hidden_dim = hidden_dim
        if not hidden_dim == 0:
            self.hidden_mlp = nn.Linear(input_dim, hidden_dim)
            in_dim = hidden_dim
        self.lanten_mlp = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        if  not self.hidden_dim == 0:
            x = self.hidden_mlp(x)
        return self.lanten_mlp(x)

class AutoEncoder(BaseVAE):
    def __init__(self, z_dim:int, in_channels:int, in_feature_width:int, use_amp:bool):
        super().__init__()
        final_feature_width = 2
        self.use_amp = use_amp
        self.in_channels=int(in_channels)
        self.in_feature_width=int(in_feature_width)

        stem_channels=32
        encoder_in_channels = 3
        r = int(sqrt(in_channels//encoder_in_channels))
        
        self.pixel_shuffle = nn.PixelShuffle(r)
        self.pixel_unshuffle = nn.PixelUnshuffle(r)
        self.encoder = model.Encoder(
            in_channels=encoder_in_channels,
            in_feature_width=in_feature_width*r, 
            final_feature_width=final_feature_width,
            stem_channels=stem_channels,
            num_repeat=2
            )

        self.ae_head = AutoEncoderHead(
            input_dim=self.encoder.last_channels*self.encoder.final_feature_width*self.encoder.final_feature_width,
            z_dim=z_dim
        )
        
        self.decoder = model.Decoder(
            in_dim=z_dim, 
            last_channels=self.encoder.last_channels, 
            final_feature_width=self.encoder.final_feature_width, 
            recover_channels=encoder_in_channels,
            recover_width=in_feature_width*r,
            stem_channels=stem_channels,
            num_repeat=self.encoder.num_repeat
        )

    # -- VAE interface
    def encode(self, input: Tensor) -> List[Tensor]:
        x = rearrange(input, "B (H W) C -> B C H W", C=self.in_channels, H=self.in_feature_width)
        x = self.pixel_shuffle(x)
        x = self.encoder(x)
        x = rearrange(x, "B C H W  -> B (C H W)", C=self.encoder.last_channels, H=self.encoder.final_feature_width)
        z= self.ae_head(x)
        return [z]
    
    def decode(self, z: Tensor) -> Tensor:
        x = self.decoder(z)
        x = self.pixel_unshuffle(x)
        x = rearrange(x, "B C H W  -> B (H W) C", C=self.in_channels, H=self.in_feature_width)
        return x
    
    def sample(self, params:List[Tensor], **kwargs) -> Tensor:
        return params[0]
    
    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        z = self.encode(input)[0]
        return  [self.decode(z), input, z]

    
