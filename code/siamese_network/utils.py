import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, scaling, margin):
        super().__init__()
        self.distance = nn.PairwiseDistance(2)
        self.margin = margin
        self.scaling = scaling

    def forward(self, x1, x2, targets):
        dist = self.distance(x1, x2)
        output = 0.5 * (
            targets * torch.pow(dist, 2)
            + (1 - targets) * torch.pow(torch.clamp(self.margin - dist, min=0), 2)
        )

        return self.scaling * torch.mean(output)
