import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, Normal
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.cuda.amp import autocast

class Encoder(nn.Module):
    def __init__(self, in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()
        self.in_feature_width = 16
        backbone = []
        # stem
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

            if feature_width == final_feature_width:
                break

        self.backbone = nn.Sequential(*backbone)
        self.last_channels = channels

    def forward(self, x):
        batch_size = x.shape[0]
        x = rearrange(x, "B (H W) C  -> B C H W",H=self.in_feature_width)
        x = self.backbone(x)
        x = rearrange(x, "B C H W -> B (C H W)", B=batch_size)
        return x
    
class DistHead(nn.Module):
    '''
    Dist: abbreviation of distribution
    '''
    def __init__(self, image_feat_dim, dyanmic_hidden_dim, stoch_dim) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.post_head = nn.Linear(image_feat_dim, stoch_dim*stoch_dim)
        self.prior_head = nn.Linear(dyanmic_hidden_dim, stoch_dim*stoch_dim)

    def unimix(self, logits, mixing_ratio=0.01):
        # uniform noise mixing
        probs = F.softmax(logits, dim=-1)
        mixed_probs = mixing_ratio * torch.ones_like(probs) / self.stoch_dim + (1-mixing_ratio) * probs
        logits = torch.log(mixed_probs)
        return logits

    def forward_post(self, x):
        logits = self.post_head(x)
        logits = rearrange(logits, "B (K C) -> B K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits

    def forward_prior(self, x):
        logits = self.prior_head(x)
        logits = rearrange(logits, "B (K C) -> B K C", K=self.stoch_dim)
        logits = self.unimix(logits)
        return logits
    
class Decoder(nn.Module):
    def __init__(self, stoch_dim, last_channels, original_in_channels, stem_channels, final_feature_width) -> None:
        super().__init__()

        backbone = []
        # stem
        backbone.append(nn.Linear(stoch_dim, last_channels*final_feature_width*final_feature_width, bias=False))
        backbone.append(Rearrange('B (C H W) -> B C H W', C=last_channels, H=final_feature_width))
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
                out_channels=original_in_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )
        )
        self.backbone = nn.Sequential(*backbone)

    def forward(self, sample):
        batch_size = sample.shape[0]
        obs_hat = self.backbone(sample)
        obs_hat = rearrange(obs_hat, "B C H W -> B (H W) C ", B=batch_size)
        return obs_hat
    
class CategoricalVAE(nn.Module):
    '''Categorical Variational Auto Encoder'''
    def __init__(self, in_channels, dyanmic_hidden_dim, use_amp):
        super().__init__()
        stem_channels = 256
        self.use_amp = use_amp
        self.final_feature_width = 4
        self.stoch_dim = 32
        self.stoch_flattened_dim = self.stoch_dim*self.stoch_dim
        self.encoder = Encoder(
            in_channels=in_channels,
            stem_channels=stem_channels,
            final_feature_width=self.final_feature_width
        )
        self.dist_head = DistHead(
            image_feat_dim=self.encoder.last_channels*self.final_feature_width*self.final_feature_width,
            dyanmic_hidden_dim=dyanmic_hidden_dim,
            stoch_dim=self.stoch_dim
        )
        self.decoder = Decoder(
            stoch_dim=self.stoch_flattened_dim,
            last_channels=self.encoder.last_channels,
            original_in_channels=in_channels,
            stem_channels=stem_channels,
            final_feature_width=self.final_feature_width
        )

    def stright_throught_gradient(self, logits, sample_mode="random_sample"):
        dist = OneHotCategorical(logits=logits)
        if sample_mode == "random_sample":
            sample = dist.sample() + dist.probs - dist.probs.detach()
        elif sample_mode == "mode":
            sample = dist.mode
        elif sample_mode == "probs":
            sample = dist.probs
        return sample
    
    def encode(self, input):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            embedding = self.encoder(input)
            post_logits = self.dist_head.forward_post(embedding)
            sample = self.stright_throught_gradient(post_logits, sample_mode="random_sample")
            flattened_sample = self.flatten_sample(sample)
        return post_logits, flattened_sample
    
    def flatten_sample(self, sample):
        return rearrange(sample, "B K C -> B (K C)")
    
    def decode(self, flattened_sample):
        return self.decoder(flattened_sample)