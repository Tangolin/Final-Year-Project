import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from torchvision import models, transforms
from torchvision.transforms import v2


class ToGray(v2.Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return torch.mean(img, dim=0, keepdim=True)


# Prepocessing steps before feeding in to evaluator
preprocess = transforms.Compose(
    [
        transforms.Resize(128),  # Resize to 128 to emulate actual process
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def calc_fid_score(outputs1, outputs2, eps=1e-6):
    # Access the arrays by their keys
    feature_mu = np.mean(outputs1, axis=0)
    feature_sigma = np.cov(outputs1, rowvar=False)

    # Calculate the generated image statistics
    gan_mu = np.mean(outputs2, axis=0)
    gan_sigma = np.cov(outputs2, rowvar=False)

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
