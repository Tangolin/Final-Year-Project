import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
from discriminator import Discriminator
from generator import Generator
from ops import ToGray, calc_fid_score, calc_is_score, denorm, get_resnet_model
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image


def train(config, device):
    # Define a dataset
    t = transforms.Compose(
        [
            transforms.Resize(config.imsize),
            transforms.ToTensor(),
            ToGray(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    dataset = ImageFolder(config.train_data_dir, transform=t)

    print(
        f"Dataset created from {config.train_data_dir}. {len(dataset)} images in total."
        + f" The mapping is {dataset.class_to_idx}.",
        flush=True,
    )

    # Define a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    data_iter = iter(dataloader)

    # Create the validation vector
    val_z = torch.randn(config.batch_size, config.z_dim).to(device)
    val_lab = torch.randint(0, 5, size=(config.batch_size,)).to(device)
    print(
        f"Validation noise vector created with shape {tuple(val_z.shape)}.", flush=True
    )

    model_path = config.eval_model_path.format(config.gait_network_neurons)
    # Create the persistent pretrained model for evaluation of outputs
    gait_feature_extractor, gait_label_predictor = get_resnet_model(
        config.gait_network_neurons,
        config.num_classes,
        ckpt=model_path,
        split=True,
    )
    gait_feature_extractor.to(device)
    gait_label_predictor.to(device)

    # Create the model and the optimizer
    gan_generator = Generator(
        in_features=config.z_dim,
        g_feature_dim=config.feature_dim,
        num_classes=config.num_classes,
        out_channels=config.num_channels,
    ).to(device)
    gan_discriminator = Discriminator(
        in_channels=config.num_channels,
        d_feature_dim=config.feature_dim,  # Should be same as config.g_feature_dim
        num_classes=config.num_classes,
    ).to(device)
    print(gan_generator)
    print(gan_discriminator)
    print("GAN model initialised.", flush=True)

    g_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gan_generator.parameters()),
        config.g_lr,
        betas=[config.optim_beta_1, config.optim_beta_2],
    )
    d_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, gan_discriminator.parameters()),
        config.d_lr,
        betas=[config.optim_beta_1, config.optim_beta_2],
    )
    print("Optimizers initialised.", flush=True)

    # Define the current step
    cur_step = 0
    best_g_loss = np.inf
    best_d_loss = np.inf

    # Continue training a model
    if config.resume_training:
        assert config.ckpt_path is not None
        print(f"Resuming training from {config.ckpt_path}.", flush=True)
        checkpoint = torch.load(config.ckpt_path)
        best_d_loss = checkpoint["best_d_loss"]
        best_g_loss = checkpoint["best_g_loss"]
        cur_step = checkpoint["step"] + 1
        gan_generator.load_state_dict(checkpoint["gan_generator_state_dict"])
        gan_discriminator.load_state_dict(checkpoint["gan_discriminator_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])

    print(f"Starting training from step {cur_step}.", flush=True)
    gan_generator.train()
    gan_discriminator.train()

    start_time = time.time()
    for step in range(cur_step, config.total_steps):
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        gan_discriminator.zero_grad()

        try:
            real_images, labels = next(data_iter)
        except:
            data_iter = iter(dataloader)
            real_images, labels = next(data_iter)

        # Prepare the real and fake images
        real_images, labels = real_images.to(device), labels.to(device)
        if step == 0:
            print(f"Images has shape: {tuple(real_images.shape)}.", flush=True)
            print(f"Labels has shape: {tuple(labels.shape)}.", flush=True)

        with torch.no_grad():
            noise_vector = torch.randn((config.batch_size, config.z_dim)).to(device)
            fake_images = gan_generator(
                # Take in the fake dimensions and labels
                noise_vector,
                labels,
            )

        d_real_out = gan_discriminator(real_images, labels)
        d_fake_out = gan_discriminator(fake_images, labels)

        # Calculate the 2 losses separately
        d_real_loss = nn.ReLU()(1 - d_real_out).mean()
        d_fake_loss = nn.ReLU()(1 + d_fake_out).mean()
        d_loss = d_real_loss + d_fake_loss

        # Do a backward pass on generator first
        d_loss.backward()
        d_optimizer.step()

        noise_vector = torch.randn((config.batch_size, config.z_dim)).to(device)
        fake_images = gan_generator(noise_vector, labels)

        # Discriminator accumulates gradient here but do not step d_optimizer
        g_out = gan_discriminator(fake_images, labels)
        g_loss = -g_out.mean()

        g_loss.backward()
        g_optimizer.step()

        # Print out log info
        if (step + 1) % config.log_n_step == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(
                f"Elapsed [{elapsed}]"
                + f" G_step [{step + 1}/{config.total_steps}], D_step[{(step + 1)}/{config.total_steps}],"
                + f" d_real_loss: {d_real_loss.item():.4f}, d_fake_loss: {d_fake_loss.item():.4f},"
                + f" g_loss: {g_loss.item():.4f}",
                flush=True,
            )

        # Sample images
        if (step + 1) % config.sample_step == 0:
            with torch.no_grad():
                fake_images = gan_generator(val_z, val_lab)
            save_image(
                denorm(real_images.detach()),
                os.path.join(config.sample_img_path, f"{step + 1}_real.png"),
            )
            save_image(
                denorm(fake_images.detach()),
                os.path.join(config.sample_img_path, f"{step + 1}_fake.png"),
            )

        # Save checkpoint if model performs well or every n steps
        if (d_loss < best_d_loss and g_loss < best_g_loss) or (
            (step + 1) % config.save_step == 0
        ):
            if (step + 1) % config.save_step == 0:
                filename = f"step_{step + 1}.pt"
            if d_loss < best_d_loss and g_loss < best_g_loss:
                filename = "best.pt"
                best_d_loss = d_loss
                best_g_loss = g_loss
            torch.save(
                {
                    "step": cur_step,
                    "best_d_loss": d_loss,
                    "best_d_loss": g_loss,
                    "gan_generator_state_dict": gan_generator.state_dict(),
                    "gan_discriminator_state_dict": gan_discriminator.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                },
                os.path.join(config.model_save_path, filename),
            )

        # Evaluate the model output with FID and IS
        if (step + 1) % config.eval_step == 0:
            # Prepocessing steps before feeding in to evaluator
            preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            # Create 1000 image samples
            noise = torch.randn(1000, config.z_dim)
            label = torch.randint(0, 5, size=(1000,))
            z_dataset = TensorDataset(noise, label)
            z_dataloader = DataLoader(
                z_dataset, batch_size=config.batch_size, num_workers=4, pin_memory=True
            )
            outputs = []
            preds = []

            with torch.no_grad():
                for noise, label in z_dataloader:
                    noise, label = noise.to(device), label.to(device)
                    gen_imgs = gan_generator(noise, label)
                    gen_imgs = denorm(gen_imgs)  # Restore the pixel values to [0, 1]
                    gen_imgs = torch.tile(gen_imgs, (1, 3, 1, 1))  # Create 3 channels
                    gen_imgs = preprocess(gen_imgs)  # Usual preprocessing step to model

                    # Append the feature output to one array
                    out_features = torch.flatten(gait_feature_extractor(gen_imgs), 1)
                    outputs.append(out_features)

                    # Save the actual predictions to another array
                    logits = gait_label_predictor(out_features)
                    preds.append(logits)

            outputs = torch.cat(outputs, dim=0).cpu().numpy()
            preds = torch.cat(preds, dim=0).cpu()

            IS_score = calc_is_score(preds)
            feature_path = config.feature_path.format(config.gait_network_neurons)
            FID_score = calc_fid_score(outputs, feature_path)
            print("=====================Score Details=====================")
            print(
                f"Elapsed [{elapsed}] G_step [{step + 1}/{config.total_steps}]"
                + f" , IS_score [{IS_score}], FID_score [{FID_score}]"
            )
            print("=======================================================")
