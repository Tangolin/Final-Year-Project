import os

import numpy as np
import torch
from PIL import Image
from siamese_net import SiameseNetwork
from torch.backends import cudnn
from torchvision import transforms
from torchvision.transforms import v2

if not os.path.exists("sim_mat.npz"):
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
            ToRGB(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = SiameseNetwork()
    checkpoint = torch.load(
        "./models/SIAMESE-2024-02-06-16-37/best.pt", map_location=device
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    filenames = []

    for patho in os.listdir("./data/sample"):
        for name in os.listdir(os.path.join("./data/sample", patho)):
            filenames.append(os.path.join("./data/sample", patho, name))

    sim_matrix = np.zeros((1000, 1000))

    for i in range(len(filenames)):
        img_1 = preprocess(Image.open(filenames[i])).to(device)

        for j in range(i, len(filenames)):
            img_2 = preprocess(Image.open(filenames[j])).to(device)

            with torch.no_grad():
                _, _, output = model(img_1, img_2)

            sim_matrix[i][j] = output

    np.savez("sim_mat.npz", sim_mat=sim_matrix)

else:
    loaded_data = np.load("sim_mat.npz")
    sim_matrix = loaded_data["sim_mat"]

# Each image's similarity score to itself
average_sim_to_itself = np.trace(sim_matrix) / 1000
print(average_sim_to_itself)

# Calculate the average patho gait similarity
patho_sim_mat = np.zeros((5, 5))

for i in range(5):
    for j in range(i, 5):
        rel_mat = sim_matrix[i * 200 : (i + 1) * 200, j * 200 : (j + 1) * 200]
        sum_sim = np.sum(rel_mat) - np.trace(rel_mat)
        avg_sim = sum_sim / 19900  # 19900 is the number of elements
        patho_sim_mat[i][j] = avg_sim

print(patho_sim_mat)
