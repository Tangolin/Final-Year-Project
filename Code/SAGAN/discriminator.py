import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ops import AttentionBlock
from torch.nn.utils.parametrizations import spectral_norm


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.out_channels = out_channels

        self.main_bloc = nn.Sequential(
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ),
        )

        self.skip_conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        self.initialise_weights()

    def initialise_weights(self):
        for layer in self.main_bloc.children():
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)

        init.xavier_uniform_(self.skip_conv.weight)
        init.constant_(self.skip_conv.bias, 0)

    def forward(self, x):
        x_0 = x
        x = self.main_bloc(x)

        if self.downsample:
            x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)

        if self.downsample or x_0.shape[1] != self.out_channels:
            x_0 = self.skip_conv(x_0)
            if self.downsample:
                x_0 = F.avg_pool2d(x_0, kernel_size=2, stride=2, padding=0)

        return x_0 + x


class DiscriminatorOptimBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.main_bloc = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ),
        )

        self.skip_conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

        self.initialise_weights()

    def initialise_weights(self):
        for layer in self.main_bloc.children():
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)

        init.xavier_uniform_(self.skip_conv.weight)
        init.constant_(self.skip_conv.bias, 0)

    def forward(self, x):
        x_0 = x

        x = self.main_bloc(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=0)
        x_0 = F.avg_pool2d(x_0, kernel_size=2, stride=2, padding=0)
        x_0 = self.skip_conv(x_0)

        return x + x_0


class Discriminator(nn.Module):
    def __init__(self, in_channels, d_feature_dim, num_classes):
        super().__init__()

        self.sequential = nn.Sequential(
            # Img size 64 * 64
            DiscriminatorOptimBlock(in_channels, d_feature_dim),
            DiscriminatorBlock(d_feature_dim, d_feature_dim * 2),
            # Img size 32 * 32
            AttentionBlock(d_feature_dim * 2),
            DiscriminatorBlock(d_feature_dim * 2, d_feature_dim * 4),
            DiscriminatorBlock(d_feature_dim * 4, d_feature_dim * 8),
            # Img size 4 * 4
            DiscriminatorBlock(d_feature_dim * 8, d_feature_dim * 16),
            DiscriminatorBlock(
                d_feature_dim * 16, d_feature_dim * 16, downsample=False
            ),
            nn.ReLU(),
        )

        self.sn_linear = spectral_norm(nn.Linear(d_feature_dim * 16, 1))
        self.sn_embedding = spectral_norm(nn.Embedding(num_classes, d_feature_dim * 16))

    def initialise_weights(self):
        init.xavier_uniform_(self.sn_linear.weight)
        init.constant_(self.sn_linear.bias, 0)
        init.xavier_uniform_(self.sn_embedding.weight)

    def forward(self, x, y):
        raw_out = self.sequential(x)

        # Output from the discriminator is first summed to n * num_features(d_feature_dim * 32)
        d_out = torch.sum(raw_out, [2, 3])

        # Projection from the linear head of d_out
        linear_head_out = self.sn_linear(d_out)

        # Create embeddings and find the inner prod
        label_emb = self.sn_embedding(y)
        inner_prod = torch.sum(d_out * label_emb, dim=1, keepdim=True)

        output = linear_head_out + inner_prod

        return output
