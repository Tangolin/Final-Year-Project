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

    # First choose for sampling for IS and FID
    sample = random.sample(os.listdir(patho_dir), 200)
    for item in sample:
        shutil.copy(os.path.join(patho_dir, item), patho_sample)

    train_val = [i for i in os.listdir(patho_dir) if i not in sample]
    train = random.sample(train_val, int(len(train_val) * 0.8))
    val = [i for i in train_val if i not in train]
    for item in train:
        shutil.copy(os.path.join(patho_dir, item), patho_train)
    for item in val:
        shutil.copy(os.path.join(patho_dir, item), patho_val)
