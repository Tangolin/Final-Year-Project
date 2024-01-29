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
    dataset = ImageFolder(config.train_data_dir, transform=CustomTransform)

    # Define a dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    steps_per_epoch = len(dataloader)
    save_step = int(config.save_step * steps_per_epoch)
    data_iter = iter(dataloader)

    # Create the validation vector
    val_z = torch.randn(config.batch_size, config.z_dim).to(device)

    # Create the model and the optimizer
    gan_generator = Generator(
        in_features=config.z_dim,
        g_feature_dim=config.feature_dim,
        num_classes=config.num_classes,
    )
    gan_discriminator = Discriminator(
        config.in_channels,
        d_feature_dim=config.feature_dim,  # Should be same as config.g_feature_dim
        num_classes=config.num_classes,
    )

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

    # Define the current step and loss
    cur_step = 0

    # Continue training a model
    if config.resume_training:
        assert config.ckpt_path is not None
        checkpoint = torch.load(config.ckpt_path)
        gan_generator.load_state_dict(checkpoint["gan_generator_state_dict"])
        gan_discriminator.load_state_dict(checkpoint["gan_discriminator_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        cur_step = checkpoint["step"] + 1

    for step in range(cur_step, config.total_steps):
        gan_generator.train()
        gan_discriminator.train()
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

        with torch.no_grad():
            g_out = gan_discriminator(fake_images, labels)
        g_loss = -g_out.mean()

        g_loss.backward()
        g_optimizer.step()

        # Print out log info
        if (step + 1) % config.log_step == 0:
            print(
                f"G_step [{step + 1}/{config.total_step}], D_step[{(step + 1)}/{config.total_step}],"
                + f"d_out_real: {d_real_loss.item():.4f}, d_out_real: {g_loss.item():.4f}"
            )

        # Sample images
        if (step + 1) % config.sample_step == 0:
            fake_images = gan_generator(val_z)
            save_image(
                fake_images.detach() * torch.Tensor(0.5) + torch.Tensor(0.5),
                os.path.join(config.sample_img_path, "{}_fake.png".format(step + 1)),
            )

        # Save checkpoint
        if (step + 1) % save_step == 0:
            torch.save(
                {
                    "step": cur_step,
                    "gan_generator_state_dict": gan_generator.state_dict(),
                    "gan_discriminator_state_dict": gan_discriminator.state_dict(),
                    "g_optimizer_state_dict": g_optimizer.state_dict(),
                    "d_optimizer_state_dict": d_optimizer.state_dict(),
                },
                config.model_save_path,
            )
