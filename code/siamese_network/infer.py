import numpy as np
import torch
from siamese_net import SiameseNetwork
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

torch.manual_seed(42)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ToRGB(v2.Transform):
    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return torch.tile(img, (1, 3, 1, 1))


preprocess = transforms.Compose(
    [
        transforms.Resize(128),  # Resize to 128 to emulate actual process
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = SiameseNetwork()
checkpoint = torch.load("../models/siamese_net.pt", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

dataset = ImageFolder("../data/sample", transform=preprocess)
print(f" The mapping is {dataset.class_to_idx}.")
dataloader = DataLoader(
    dataset, batch_size=128, num_workers=2, shuffle=False, pin_memory=True
)

features = []
labels = []
all_preds = []

sim_matrix = np.zeros((len(dataset), len(dataset)))

with torch.no_grad():
    for data, label in dataloader:
        data, label = data.to(device), label.to(device)
        out = model.extract_features(data)
        features.append(out)
        labels.append(label)

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)

    for i, f in enumerate(features):
        f = torch.tile(f, (len(features), 1))
        in_f = torch.cat((f, features), 1)
        preds = model.act(model.fc(in_f)).squeeze()
        all_preds.append(preds)

    sim_matrix = torch.stack(all_preds, dim=0).cpu().numpy()
    print(sim_matrix.shape)

# Each image's similarity score to itself
average_sim_to_itself = np.trace(sim_matrix) / len(sim_matrix)
print(f"The average similarity is {average_sim_to_itself:.6f}.")

# Calculate the average patho gait similarity
patho_sim_mat = np.zeros((5, 5))
counts = torch.bincount(labels)
cum_counts = torch.cumsum(counts, 0).cpu()
cum_counts = torch.cat([torch.tensor([0]), cum_counts], dim=0)

for i in range(5):
    for j in range(i, 5):
        rel_mat = sim_matrix[
            cum_counts[i] : cum_counts[i + 1], cum_counts[j] : cum_counts[j + 1]
        ]
        rel_mat = np.triu(rel_mat)
        sum_sim = np.sum(rel_mat) - np.trace(rel_mat)
        avg_sim = sum_sim / (
            (counts[i] * (counts[i] + 1) / 2) - counts[i]
        )  # divide it by the number of real elements
        patho_sim_mat[i][j] = avg_sim

print(patho_sim_mat)
