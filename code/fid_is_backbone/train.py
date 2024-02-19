import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score
from torchvision import datasets, transforms
from utils import get_resnet_model

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

# Declare the parameters for training
num_classes = 5
batch_size = 128
learning_rate = 0.1
num_epochs = 50

# Set device (CPU or GPU) and create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_preprocess = transforms.Compose(
    [
        transforms.Resize(128),  # Resize to 128 to emulate actual process
        transforms.RandomResizedCrop((224, 224), (0.65, 1)),
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

# Create the train and test dataset
train_dataset = datasets.ImageFolder(root="../data/train", transform=train_preprocess)
val_dataset = datasets.ImageFolder(root="../data/val", transform=val_preprocess)

for num_neurons in [8, 16, 32, 64, 128, 256, 512]:
    print("*" * 100)
    print(f"Begin training for Resnet with {num_neurons}.")

    # Begin a new training loop
    best_eval = float("inf")
    best_acc = float("inf")
    best_f1 = float("inf")

    model = get_resnet_model(num_neurons, num_classes)
    model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    # Declare the helpers for the training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * int(len(train_dataset) / batch_size)
    )

    for epoch in range(num_epochs):
        # Train
        print("=====================Training=====================")
        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()

            data, target = data.to(device), target.to(device)

            if batch_idx == 0 and epoch == 0:
                print("Data has shape", data.shape)
                print("Target has shape", target.shape)

            # Forward pass
            output = model(data)
            loss = criterion(output, target)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx + 1) % 5 == 0 or batch_idx == len(train_loader) - 1:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Step {batch_idx+1}/{len(train_loader)},"
                    + f" Loss: {loss.item()}."
                )

        # Test
        print("====================Validation====================")
        model.eval()  # Set the model to evaluation mode
        accuracy = MulticlassAccuracy()
        f1_score = MulticlassF1Score(num_classes=num_classes, average="macro")
        val_loss = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                output = model(images)
                val_loss += criterion(output, labels)
                _, pred = torch.max(output, 1)
                accuracy.update(pred, labels)
                f1_score.update(pred, labels)

        acc_value = accuracy.compute().item()
        f1_value = f1_score.compute().item()

        if val_loss < best_eval:
            # Save the model if f1 is best
            print(
                f"Saving model to gait_resnet_{num_neurons}.pt, current val-loss: {val_loss}, prev best: {best_eval}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "best_acc": f1_value,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"../models/gait_resnet_{num_neurons}.pt",
            )
            best_eval = val_loss
            best_acc = acc_value
            best_f1 = f1_value

        print(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}, Accuracy: {acc_value}, F1: {f1_value}."
        )

    print("Training completed.")
