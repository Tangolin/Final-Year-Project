import numpy as np
import torch
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy
from torchvision import datasets, models, transforms

model_dir = "./models/best.pt"
data_dir = "./data/sample/"
num_classes = 5

# Set device (CPU or GPU) and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
checkpoint = torch.load(model_dir)
model.load_state_dict(checkpoint["model_state_dict"])
model_feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
model_label_predictor = list(model.children())[-1]
model_feature_extractor.to(device)
model_label_predictor.to(device)
del model

preprocess = transforms.Compose(
    [
        transforms.Resize(128),  # Resize to 128 to emulate actual process
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

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

features = []
accuracy = MulticlassAccuracy()

with torch.no_grad():
    for idx, (imgs, labels) in enumerate(sample_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        out = model_feature_extractor(imgs)
        output = torch.flatten(out, 1)
        features.append(output)

        logits = model_label_predictor(output)
        _, pred = torch.max(logits, 1)
        accuracy.update(pred, labels)

# Checking the accuracy on the samples for fun
print("Accuracy", accuracy.compute().item())

# Save the tensors in case of future needs
features = torch.cat(features, dim=0).to(torch.device("cpu"))
torch.save(features, "features.pt")

# Get statistics for FID
features = features.numpy()
mu = np.mean(features, axis=0)
sigma = np.cov(features, rowvar=False)
np.savez("feature_stats.npz", mu=mu, sigma=sigma)
print(f"The features have mean shape {mu.shape} and covariance shape {sigma.shape}.")
