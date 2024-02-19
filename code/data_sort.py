import argparse
import os
import random
import re
import shutil


def split_gei_data(root_dir, split_sample=False):
    data_dir = "data/GEI_data"
    train_dir = "data/train"
    val_dir = "data/val"
    if split_sample:
        sample_dir = "data/sample"

    for patho in os.listdir(os.path.join(root_dir, data_dir)):
        patho_train = os.path.join(root_dir, train_dir, patho)
        patho_val = os.path.join(root_dir, val_dir, patho)
        os.makedirs(patho_train, exist_ok=True)
        os.makedirs(patho_val, exist_ok=True)

        if split_sample:
            patho_sample = os.path.join(root_dir, sample_dir, patho)
            os.makedirs(patho_sample, exist_ok=True)

        patho_dir = os.path.join(root_dir, data_dir, patho)

        # First choose for sampling for IS and FID
        if split_sample:
            sample = random.sample(
                os.listdir(patho_dir), int(0.1 * len(os.listdir(patho_dir)))
            )
            for item in sample:
                shutil.copy(os.path.join(patho_dir, item), patho_sample)
        else:
            sample = []

        train_val = [i for i in os.listdir(patho_dir) if i not in sample]
        train = []
        val = []

        for item in train_val:
            pattern = re.compile(r"s(ub)*(\d+)")
            match = pattern.search(item)
            sub_num = int("".join([i for i in match.group() if i.isnumeric()]))

            if "GAIT_IST" in item:
                if sub_num <= 8:
                    train.append(item)
                else:
                    val.append(item)
            else:
                if sub_num <= 18:
                    train.append(item)
                else:
                    val.append(item)

        for item in train:
            shutil.copy(os.path.join(patho_dir, item), patho_train)
        for item in val:
            shutil.copy(os.path.join(patho_dir, item), patho_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split the GEI data into train, val and optionally sample."
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root folder where the data folder is stored.",
    )

    parser.add_argument(
        "--split_sample",
        action="store_true",
        help="If a sample folder is required. (Only for GAN fid backbone)",
    )

    args = parser.parse_args()
    split_gei_data(args.root_dir, args.split_sample)
