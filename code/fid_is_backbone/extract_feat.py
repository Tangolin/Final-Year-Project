import os

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchvision import datasets
from utils import calc_fid_score, get_resnet_model, preprocess

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.deterministic = True
cudnn.benchmark = False

data_dir = "../data/sample/"
output_dir = "../output"
num_classes = 5

for i in [8, 16, 32, 64, 128, 256, 512]:
    print("=" * 60)
    num_neurons = i
    model_dir = f"../models/gait_resnet_{num_neurons}.pt"
    print(f"Trial with {num_neurons} neurons.")

    # Load the relevant model
    model, label_predictor = get_resnet_model(
        num_neurons=num_neurons, num_classes=num_classes, ckpt=model_dir, split=True
    )
    model.to(device)
    label_predictor.to(device)

    # Set up the dataset
    sample_dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
    sample_loader = DataLoader(
        sample_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # 'diplegic': 0, 'hemiplegic': 1, 'neuropathic': 2, 'normal': 3, 'parkinsonian': 4
    print("The mapping is", sample_dataset.class_to_idx)

    features = []
    labels = []
    accuracy = MulticlassAccuracy()

    with torch.no_grad():
        for idx, (img, label) in enumerate(sample_loader):
            img, label = img.to(device), label.to(device)
            out_features = model(img)
            features.append(out_features)
            labels.append(label)

            logits = label_predictor(out_features)
            _, pred = torch.max(logits, 1)
            accuracy.update(pred, label)

    # Checking the accuracy on the samples for fun
    print(
        f"The accuracy of the model is {accuracy.compute().item()}.",
    )

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Save the tensors in case of future needs
    features = torch.cat(features, dim=0).cpu().numpy()
    labels = torch.cat(labels, dim=0).cpu().numpy()

    # Get statistics for FID
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    np.savez(
        f"../output/feature_stats_{num_neurons}.npz",
        mu=mu,
        sigma=sigma,
        features=features,
    )
    print(
        f"The features have mean shape {mu.shape} and covariance shape {sigma.shape}."
    )

    # Calculate the matrix of FID scores
    matrix = [[0] * 5 for _ in range(5)]
    for i in range(5):
        for j in range(5):
            out1 = features[np.where(labels == i)]
            out2 = features[np.where(labels == j)]
            matrix[i][j] = calc_fid_score(out1, out2)

    print(*matrix, sep="\n")
