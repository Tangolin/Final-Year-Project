import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from scipy import linalg
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


def calc_fid_score(outputs, feature_path, eps=1e-6):
    # Load the feature statistics
    loaded_data = np.load(feature_path)

    # Access the arrays by their keys
    feature_mu = loaded_data["mu"]
    feature_sigma = loaded_data["sigma"]

    # Calculate the generated image statistics
    gan_mu = np.mean(outputs, axis=0)
    gan_sigma = np.cov(outputs, rowvar=False)

    # Code below from:
    # https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    diff = gan_mu - feature_mu

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(gan_sigma.dot(feature_sigma), disp=False)
    if not np.isfinite(covmean).all():
        print(
            "FID calculation produces singular product, "
            + f"adding {eps} to diagonal of cov estimates"
        )
        offset = np.eye(gan_sigma.shape[0]) * eps
        covmean = linalg.sqrtm((gan_sigma + offset).dot(feature_sigma + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (
        diff.dot(diff) + np.trace(gan_sigma) + np.trace(feature_sigma) - 2 * tr_covmean
    )


def calc_is_score(predictor_logits, eps=1e-10):
    activation = nn.Softmax(dim=1)
    pred_prob = activation(predictor_logits)

    marginal_prob = torch.mean(pred_prob, dim=0)
    kl_div = pred_prob * (torch.log(pred_prob + eps) - torch.log(marginal_prob + eps))
    kl_div = torch.sum(kl_div, dim=1)

    avg_div = torch.mean(kl_div)

    return torch.exp(avg_div).item()
