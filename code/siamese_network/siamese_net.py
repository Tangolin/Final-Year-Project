import torch
import torch.nn as nn
from torchvision import models


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load a pretrained resnet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.fc_in_features = self.resnet.fc.in_features

        # Remove the last fc layer
        self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

        # Final layer to predict if is same image, i.e. learn similarity score
        self.fc = nn.Sequential(
            nn.Linear(self.fc_in_features * 2, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(256, 1),
        )

        self.act = nn.Sigmoid()

        # Initialize the weights in linear layer
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)

    def extract_features(self, x):
        output = self.resnet(x)
        output = torch.flatten(output, 1)

        return output

    def forward(self, main_input, sub_input):
        main_feature = self.extract_features(main_input)
        sub_feature = self.extract_features(sub_input)
        output = torch.cat((main_feature, sub_feature), 1)

        output = self.fc(output)
        output = self.act(output)

        return main_feature, sub_feature, output.squeeze()
