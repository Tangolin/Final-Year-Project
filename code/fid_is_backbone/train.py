import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score
from torchvision import datasets
from utils import get_resnet_model, preprocess

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)

# Declare the parameters for training
num_classes = 5
batch_size = 128
learning_rate = 0.1
num_epochs = 50
num_neurons = 8
best_eval = float("inf")
best_acc = float("inf")
best_f1 = float("inf")

# Set device (CPU or GPU) and create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_resnet_model(num_neurons, num_classes)
model.to(device)
print("Model initialised.")
print(model)

# Create the train and test dataset
train_dataset = datasets.ImageFolder(root="./data/train", transform=preprocess)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)
val_dataset = datasets.ImageFolder(root="./data/val", transform=preprocess)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
print("Training and validation datasets created.")


# Creating an early stopping mechanism
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# Declare the helpers for the training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.8, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs * int(len(train_dataset) / batch_size)
)
# early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

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
            f"./models/gait_resnet_{num_neurons}.pt",
        )
        best_eval = val_loss
        best_acc = acc_value
        best_f1 = f1_value

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss}, Accuracy: {acc_value}, F1: {f1_value}."
    )

    # if early_stopper.early_stop(val_loss):
    #     print(f"Early stopping. Final epoch: {epoch}.")
    #     print(f"Loss: {best_eval}, Accuracy: {best_acc}, F1: {best_f1}.")
    #     break

print("Training completed.")
