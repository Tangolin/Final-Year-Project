import os
import random
import shutil

root_dir = "./data/GEI_data"

os.makedirs(os.path.join("./data/train"))
for patho in os.listdir(root_dir):
    patho_train = os.path.join("./data/train", patho)
    patho_val = os.path.join("./data/val", patho)
    patho_sample = os.path.join("./data/sample", patho)
    os.makedirs(patho_train)
    os.makedirs(patho_val)
    os.makedirs(patho_sample)

    patho_dir = os.path.join(root_dir, patho)

    # First distribute into train and val
    train_val = random.sample(
        os.listdir(patho_dir), int(0.8 * len(os.listdir(patho_dir)))
    )
    cutoff = int(len(train_val) * 0.8)
    train, val = train_val[:cutoff], train_val[cutoff:]
    for item in train:
        shutil.copy(os.path.join(patho_dir, item), patho_train)
    for item in val:
        shutil.copy(os.path.join(patho_dir, item), patho_val)

    # Then distribute to sample
    for i in os.listdir(patho_dir):
        if i not in train_val:
            shutil.copy(os.path.join(patho_dir, i), patho_sample)
