import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.parametrizations import spectral_norm
from torchvision.transforms import v2


class ConditionalBatchNorm(nn.Module):
    """Taken and modified from https://github.com/pytorch/pytorch/issues/8985"""

    def __init__(self, num_classes, num_features, decay_rate=0.999):
        super().__init__()
        self.num_features = num_features
        self.weights = spectral_norm(nn.Embedding(num_classes, num_features))
        self.biases = spectral_norm(nn.Embedding(num_classes, num_features))
        self.bn = nn.BatchNorm2d(
            num_features, momentum=decay_rate, affine=False, eps=1e-5
        )

        self.initialise_weights()

    def initialise_weights(self):
        # Same initialization as the original code
        init.ones_(self.weights.weight)
        init.zeros_(self.biases.weight)

    def forward(self, x, y):
        gamma, beta = self.weights(y), self.biases(y)
        out = self.bn(x)

        gamma_reshaped, beta_reshaped = gamma.view(
            -1, self.num_features, 1, 1
        ), beta.view(-1, self.num_features, 1, 1)

        out = gamma_reshaped * out + beta_reshaped

        return out


class AttentionBlockNoPool(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_q = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        )
        self.conv_k = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=4, stride=2, padding=1)
        )
        self.conv_v = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=4, stride=2, padding=1)
        )
        self.conv_o = spectral_norm(
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Learnable parameter sigma
        self.sigma = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)

        self.initialise_weights()

    def initialise_weights(self):
        for layer in [self.conv_q, self.conv_k, self.conv_v, self.conv_o]:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        n, _, h, w = x.shape

        q = self.conv_q(x).view(n, -1, h * w)
        k = self.conv_k(x).view(n, -1, (h * w) // 4)
        v = self.conv_v(x).view(n, -1, (h * w) // 4)

        attn = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        attd_value = torch.bmm(v, attn.permute(0, 2, 1)).view(n, -1, h, w)

        out = self.conv_o(attd_value)

        return x + self.sigma * out


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_q = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        )
        self.conv_k = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        )
        self.conv_v = spectral_norm(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        )
        self.conv_o = spectral_norm(
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Learnable parameter sigma
        self.sigma = nn.parameter.Parameter(torch.zeros(1), requires_grad=True)

        self.initialise_weights()

    def initialise_weights(self):
        for layer in [self.conv_q, self.conv_k, self.conv_v, self.conv_o]:
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

    def forward(self, x):
        n, _, h, w = x.shape

        q = self.conv_q(x).view(n, -1, h * w)
        k = self.pool(self.conv_k(x)).view(n, -1, (h * w) // 4)
        v = self.pool(self.conv_v(x)).view(n, -1, (h * w) // 4)

        attn = F.softmax(torch.bmm(q.permute(0, 2, 1), k), dim=-1)
        attd_value = torch.bmm(v, attn.permute(0, 2, 1)).view(n, -1, h, w)

        out = self.conv_o(attd_value)

        return x + self.sigma * out


class ToGray(v2.Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return torch.mean(img, dim=0, keepdim=True)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)
