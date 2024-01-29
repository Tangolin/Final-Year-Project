import argparse
import os
from datetime import datetime

import torch
from infer import infer
from torch.backends import cudnn
from train import train


def get_parameters():
    parser = argparse.ArgumentParser(
        description="A faithful replication of SAGAN code published in Tensorflow."
    )

    parser.add_argument("--train", action="store_true")
    # parser.add_argument("--parallel", type=str2bool, default=False)

    # GAN settings
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument(
        "--z_dim", type=int, default=128, help="Dimensions of the input noise vector."
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        default=64,
        help="Dimensions of the feature matrix in both the discriminator and generator.",
    )

    # Training setting
    parser.add_argument(
        "--total_steps",
        type=int,
        default=10000,
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--g_lr", type=float, default=0.0001)
    parser.add_argument("--d_lr", type=float, default=0.0004)
    parser.add_argument("--optim_beta_1", type=float, default=0.0)
    parser.add_argument("--optim_beta_2", type=float, default=0.999)

    # Resume training settings
    parser.add_argument(
        "--resume_training", action="store_true", help="Description of the flag"
    )
    parser.add_argument("--ckpt_path", type=str, default=None)

    # Path settings
    parser.add_argument("--train_data_dir", type=str, default="./data/train")
    parser.add_argument("--model_save_path", type=str, default="./models")
    parser.add_argument("--sample_img_path", type=str, default="./samples")

    # Saving and logging settings
    parser.add_argument("--log_n_step", type=int, default=10)
    parser.add_argument("--sample_step", type=int, default=10)
    parser.add_argument("--save_step", type=float, default=20)

    return parser.parse_args()


def main(config):
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    exp_name = "SAGAN-" + current_time

    if config.train:
        config.model_save_path = os.path.join(config.model_save_path, exp_name)
        config.sample_img_path = os.path.join(config.sample_img_path, exp_name)
        os.makedirs(config.model_save_path, exist_ok=True)
        os.makedirs(config.sample_img_path, exist_ok=True)

        print(config)
        train(config, device)

    else:
        infer(config, device)


if __name__ == "__main__":
    config = get_parameters()
    main(config)
