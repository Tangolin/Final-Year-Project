import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from numpy import isin
from ops import AttentionBlock, ConditionalBatchNorm
from torch.nn.utils.parametrizations import spectral_norm


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        # Paddings have to be 1 to ensure the output is the same size

        self.main_bloc = nn.Sequential(
            spectral_norm(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            ),
            ConditionalBatchNorm(num_classes, out_channels),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            ),
        )

        self.bn1 = ConditionalBatchNorm(num_classes, in_channels)
        self.skip_conv = spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        self.act = nn.ReLU()

        self.initialise_weights()

    def forward(self, x, y):
        x_0 = x
        x = self.act(self.bn1(x, y))
        x = F.interpolate(x, scale_factor=2)

        for layer in self.main_bloc:
            if isinstance(layer, ConditionalBatchNorm):
                x = layer(x, y)
            else:
                x = layer(x)

        x_0 = F.interpolate(x_0, scale_factor=2)
        x_0 = self.skip_conv(x_0)

        return x_0 + x

    def initialise_weights(self):
        for layer in self.main_bloc.children():
            if isinstance(layer, nn.Conv2d):
                init.xavier_uniform_(layer.weight)
                init.constant_(layer.bias, 0)

        init.xavier_uniform_(self.skip_conv.weight)
        init.constant_(self.skip_conv.bias, 0)


class Generator(nn.Module):
    def __init__(self, in_features, g_feature_dim, num_classes):
        super().__init__()
        self.g_feature_dim = g_feature_dim
        self.sn_linear = spectral_norm(
            nn.Linear(in_features, g_feature_dim * 32 * 4 * 4)
        )

        self.sequential = nn.Sequential(
            GeneratorBlock(g_feature_dim * 32, g_feature_dim * 32, num_classes),
            GeneratorBlock(g_feature_dim * 32, g_feature_dim * 16, num_classes),
            GeneratorBlock(g_feature_dim * 16, g_feature_dim * 8, num_classes),
            AttentionBlock(g_feature_dim * 8),
            GeneratorBlock(g_feature_dim * 8, g_feature_dim * 4, num_classes),
            GeneratorBlock(g_feature_dim * 4, g_feature_dim * 2, num_classes),
            GeneratorBlock(g_feature_dim * 2, g_feature_dim, num_classes),
            nn.BatchNorm2d(g_feature_dim),
            nn.ReLU(),
            spectral_norm(
                nn.Conv2d(g_feature_dim, out_channels=1, kernel_size=3, padding=1)
            ),
            nn.Tanh(),
        )

    def forward(self, x, y):
        print(x.shape)
        print(y.shape)
        out = self.sn_linear(x)
        print(out.shape)
        out = out.view(-1, self.g_feature_dim * 32, 4, 4)
        print(out.shape)
        for layer in self.sequential.children():
            print(layer)
            if isinstance(layer, GeneratorBlock):
                out = layer(out, y)
                print(out.shape)
            else:
                out = layer(out)
                print(out.shape)

        return out


generator = Generator(1024, 64, 5).cuda()
labels = torch.randint(0, 5, (16,)).cuda()
x = torch.randn((16, 1024)).cuda()
generator(x, labels)
