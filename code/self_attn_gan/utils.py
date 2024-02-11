import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision import models
from torchvision.transforms import v2


class ToGray(v2.Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return torch.mean(img, dim=0, keepdim=True)


class ToRGB(v2.Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return torch.tile(img, (1, 3, 1, 1))


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
    np.savez("fid_resources/gan_feature_stats.npz", mu=gan_mu, sigma=gan_sigma)

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


def get_resnet_model(num_neurons, num_classes, ckpt=None, split=False):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Customise model by replacing head and reduce in_channels
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, num_neurons),
        nn.LeakyReLU(0.1, inplace=True),
        nn.Linear(num_neurons, num_classes),
    )

    if ckpt is not None:
        # Start from a trained model
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    if split:
        # Break it up to extract the features directly
        label_predictor = model.fc[-1]
        model.fc = model.fc[:-1]

        return model, label_predictor

    return model


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


def get_siamese_model(ckpt):
    model = SiameseNetwork()
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model
