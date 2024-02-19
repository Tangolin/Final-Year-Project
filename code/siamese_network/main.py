import argparse
import os
from datetime import datetime

import torch
from dataset import SiameseDataset
from siamese_net import SiameseNetwork
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from train import train, validate


def get_parameters():
    parser = argparse.ArgumentParser(
        description="Siamese Network for Gait Similarity Scoring."
    )

    # Training setting
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--optim_beta_1", type=float, default=0.8)
    parser.add_argument("--optim_beta_2", type=float, default=0.999)
    parser.add_argument("--dry_run", action="store_true")

    # Resume training settings
    parser.add_argument("--resume_training", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default=None)

    # Path settings
    parser.add_argument("--train_data_dir", type=str, default="../data/train")
    parser.add_argument("--val_data_dir", type=str, default="../data/val")
    parser.add_argument("--model_save_path", type=str, default="../models")

    # Saving and logging settings
    parser.add_argument("--log_n_step", type=int, default=10)

    return parser.parse_args()


def main(config):
    torch.manual_seed(42)

    cudnn.deterministic = True
    cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_preprocess = transforms.Compose(
        [
            transforms.Resize(128),  # Resize to 128 to emulate actual process
            transforms.RandomResizedCrop((224, 224), (0.65, 0.85)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_preprocess = transforms.Compose(
        [
            transforms.Resize(128),  # Resize to 128 to emulate actual process
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dataset = SiameseDataset(config.train_data_dir, transform=train_preprocess)
    val_dataset = SiameseDataset(  # TODO: implement a test process
        config.val_data_dir, transform=val_preprocess
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model = SiameseNetwork()
    best_val_loss = float("inf")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        config.lr,
        betas=[config.optim_beta_1, config.optim_beta_2],
    )

    # Continue training a model
    if config.resume_training:
        assert config.ckpt_path is not None
        print(f"Resuming training from {config.ckpt_path}.", flush=True)
        checkpoint = torch.load(config.ckpt_path)
        best_val_loss = checkpoint["best_val_loss"]
        epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    model.to(device)

    for epoch in range(0, config.num_epochs):
        train(config, model, device, train_loader, optimizer, epoch)
        val_loss = validate(model, device, val_loader)

        if val_loss < best_val_loss:
            print("Saving model...")
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(config.model_save_path, "siamese_net.pt"),
            )


if __name__ == "__main__":
    configs = get_parameters()
    main(configs)
