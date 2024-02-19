import numpy as np
import torch
from generator import Generator
from torch.backends import cudnn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from utils import (
    ToRGB,
    calc_fid_score,
    calc_is_score,
    denorm,
    get_resnet_model,
    get_siamese_model,
)

torch.manual_seed(42)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters for the script
num_neurons = 64
batch_size = 64
num_classes = 5
resnet_path = f"../models/gait_resnet_{num_neurons}.pt"
siamese_model_path = "../models/siamese_net.pt"
feature_path = f"../output/feature_stats_{num_neurons}.npz"

# Load the gan model
gan_generator = Generator(
    in_features=128,
    g_feature_dim=64,
    num_classes=5,
    out_channels=1,
).to(device)
gan_checkpoint = torch.load("../models/sagan.pt")
gan_generator.load_state_dict(gan_checkpoint["gan_generator_state_dict"])
print("Generator model loaded.")

# Load the gait resnet model for fid/is score
gait_feature_extractor, gait_label_predictor = get_resnet_model(
    num_neurons,
    num_classes,
    ckpt=resnet_path,
    split=True,
)
gait_feature_extractor.to(device)
gait_label_predictor.to(device)
print("Gait model loaded.")

# Load the siamese network for siamese similarity
gait_similarity_net = get_siamese_model(siamese_model_path)
gait_similarity_net.to(device)

# Prepocessing steps before feeding in to evaluator
preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        ToRGB(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create 1000 image samples
noise = torch.randn(1000, 128)
label = torch.arange(5).repeat(200).sort()[0]

z_dataset = TensorDataset(noise, label)
z_dataloader = DataLoader(
    z_dataset, batch_size=batch_size, num_workers=2, pin_memory=True
)
outputs = []
preds = []

features = []
labels = []
siam_preds = []

with torch.no_grad():
    for noise, label in z_dataloader:
        noise, label = noise.to(device), label.to(device)
        gen_imgs = gan_generator(noise, label)
        gen_imgs = denorm(gen_imgs)  # Restore the pixel values to [0, 1]
        gen_imgs = preprocess(gen_imgs)  # Usual preprocessing step to model

        # Append the feature output to one array
        out_features = torch.flatten(gait_feature_extractor(gen_imgs), 1)
        outputs.append(out_features)

        # Save the actual predictions to another array
        logits = gait_label_predictor(out_features)
        preds.append(logits)

        # TODO: Implement siamese network testing loop
        siam_feat = gait_similarity_net.extract_features(gen_imgs)
        features.append(siam_feat)
        labels.append(label)

outputs = torch.cat(outputs, dim=0).cpu().numpy()
preds = torch.cat(preds, dim=0).cpu()

IS_score = calc_is_score(preds)

FID_score = calc_fid_score(outputs, feature_path)

print(f"IS Score: {IS_score}, FID Score: {FID_score}")

features = torch.cat(features, dim=0)
labels = torch.cat(labels, dim=0)

with torch.no_grad():
    for i, f in enumerate(features):
        f = torch.tile(f, (len(features), 1))
        in_f = torch.cat((f, features), 1)
        preds = gait_similarity_net.act(gait_similarity_net.fc(in_f)).squeeze()
        siam_preds.append(preds)

sim_matrix = torch.stack(siam_preds, dim=0).cpu().numpy()

patho_sim_mat = np.zeros((5, 5))
counts = torch.bincount(labels).cpu()
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
