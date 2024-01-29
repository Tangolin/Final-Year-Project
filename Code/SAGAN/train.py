import os

import torch
from discriminator import Discriminator
from generator import Generator
from ops import CustomTransform
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image


def train(config, device):
    # Define a dataset
    dataset = ImageFolder(config.train_data_dir, transform=CustomTransform())
    print(
        f"Dataset created from {config.train_data_dir}. {len(dataset)} images in total."
        + f" The mapping is {dataset.class_to_idx}."
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
    print(f"Validation noise vector created with shape {tuple(val_z.shape)}.")

    # Create the model and the optimizer
    gan_generator = Generator(
        in_features=config.z_dim,
        g_feature_dim=config.feature_dim,
        num_classes=config.num_classes,
    ).to(device)
    gan_discriminator = Discriminator(
        config.in_channels,
        d_feature_dim=config.feature_dim,  # Should be same as config.g_feature_dim
        num_classes=config.num_classes,
    ).to(device)
    print("GAN model initialised.")

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
    print("Optimizers initialised.")

    # Define the current step and loss
    cur_step = 0

    # Continue training a model
    if config.resume_training:
        assert config.ckpt_path is not None
        print(f"Resuming training from {config.ckpt_path}.")
        checkpoint = torch.load(config.ckpt_path)
        gan_generator.load_state_dict(checkpoint["gan_generator_state_dict"])
        gan_discriminator.load_state_dict(checkpoint["gan_discriminator_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        cur_step = checkpoint["step"] + 1

    print(f"Starting training from step {cur_step}.")
    gan_generator.train()
    gan_discriminator.train()

    for step in range(cur_step, config.total_steps):
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        try:
            real_images, labels = next(data_iter)
        except:
            data_iter = iter(dataloader)
            real_images, labels = next(data_iter)

        # Prepare the real and fake images
        real_images, labels = real_images.to(device), labels.to(device)

        with torch.no_grad():
            noise_vector = torch.randn((config.batch_size, config.z_dim)).to(device)
            fake_images = gan_generator(
                # Take in the fake dimensions and labels
                noise_vector,
                labels,
            )

        d_real_out, d_fake_out = gan_discriminator(
            real_images, labels
        ), gan_discriminator(fake_images, labels)

        # Calculate the 2 losses separately
        d_real_loss = torch.nn.ReLU()(1 - d_real_out).mean()
        d_fake_loss = torch.nn.ReLU()(1 + d_fake_out).mean()
        d_loss = d_real_loss + d_fake_loss

        # Do a backward pass on generator first
        d_loss.backward()
        d_optimizer.step()

        noise_vector = torch.randn((config.batch_size, config.z_dim)).to(device)
        fake_images = gan_generator(
            noise_vector,
            labels,
        )

        # Discriminator accumulates gradient here but do not step d_optimizer
        g_out = gan_discriminator(fake_images, labels)
        g_loss = -g_out.mean()

        g_loss.backward()
        g_optimizer.step()

        # Print out log info
        if (step + 1) % config.log_n_step == 0:
            print(
                f"G_step [{step + 1}/{config.total_steps}], D_step[{(step + 1)}/{config.total_steps}],"
                + f" d_real_loss: {d_real_loss.item():.4f}, d_fake_loss: {d_fake_loss.item():.4f},"
                + f" g_loss: {g_loss.item():.4f}"
            )

        # Sample images
        if (step + 1) % config.sample_step == 0:
            with torch.no_grad():
                fake_images = gan_generator(val_z, val_lab)
            save_image(
                real_images.to("cpu") * torch.Tensor([0.5]) + torch.Tensor([0.5]),
                os.path.join(config.sample_img_path, f"{step + 1}_real.png"),
            )
            save_image(
                fake_images.to("cpu") * torch.Tensor([0.5]) + torch.Tensor([0.5]),
                os.path.join(config.sample_img_path, f"{step + 1}_fake.png"),
            )

        # Save checkpoint
        if (step + 1) % config.save_step == 0:
            torch.save(
                {
                    "step": cur_step,
                    "gan_generator_state_dict": gan_generator.state_dict(),
                    "gan_discriminator_state_dict": gan_discriminator.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                },
                os.path.join(config.model_save_path, f"step_{step + 1}.pt"),
            )
