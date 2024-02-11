import torch
import torch.nn as nn
from torcheval.metrics import BinaryAccuracy, BinaryF1Score
from utils import ContrastiveLoss


def train(args, model, device, train_loader, optimizer, epoch):
    print("=" * 60)
    model.train()

    # Contrastive for detecting same class, BCE for detecting same image
    class_criterion = ContrastiveLoss(scaling=1 / 100, margin=50)
    img_criterion = nn.BCELoss()

    # Declare metrics to keep track of performance
    accuracy = BinaryAccuracy(threshold=0.5)
    f1_score = BinaryF1Score(threshold=0.5)

    for batch_idx, (main_img, sub_img, pic_target, cls_target) in enumerate(
        train_loader
    ):
        main_img, sub_img, pic_target, cls_target = (
            main_img.to(device),
            sub_img.to(device),
            pic_target.float().to(device),
            cls_target.float().to(device),
        )

        optimizer.zero_grad()
        main_feat, sub_feat, preds = model(main_img, sub_img)

        # Update loss
        cls_loss = class_criterion(main_feat, sub_feat, cls_target)
        img_loss = img_criterion(preds, pic_target)
        loss = cls_loss + img_loss

        loss.backward()
        optimizer.step()

        # Update metrics
        accuracy.update(preds, pic_target)
        f1_score.update(preds, pic_target)

        if (batch_idx + 1) % args.log_n_step == 0:
            print(
                f"Epoch: [{epoch+1}], Batch: [{batch_idx + 1}/{len(train_loader)}], "
                + f"Cls loss: {cls_loss.item():.4f} Img loss: {img_loss.item():.4f}, "
                # + f"Img loss: {img_loss.item():.4f}, "
                + f"Accuracy: {accuracy.compute().item():.4f}"
            )
            if args.dry_run:
                break


def validate(model, device, test_loader):
    model.eval()
    test_loss = 0

    # Contrastive for detecting same class, BCE for detecting same image
    class_criterion = ContrastiveLoss(scaling=1 / 100, margin=50)
    img_criterion = nn.BCELoss()

    # Declare metrics to keep track of performance
    accuracy = BinaryAccuracy(threshold=0.5)
    f1_score = BinaryF1Score(threshold=0.5)

    with torch.no_grad():
        for main_img, sub_img, pic_target, cls_target in test_loader:
            main_img, sub_img, pic_target, cls_target = (
                main_img.to(device),
                sub_img.to(device),
                pic_target.float().to(device),
                cls_target.float().to(device),
            )

            main_feat, sub_feat, preds = model(main_img, sub_img)
            cls_loss = class_criterion(main_feat, sub_feat, cls_target)
            img_loss = img_criterion(preds, pic_target)
            test_loss += (cls_loss) + (img_loss)

            accuracy.update(preds, pic_target)
            f1_score.update(preds, pic_target)

    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy.compute().item():.4f}")

    return test_loss
