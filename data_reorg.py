"""This file reorganises the files in GAIT_IST and GAIT_IT dataset to a single
usable folder for model training."""

import os
import shutil

root_dir = "./data"
destination_root_dir = "./data/GEI_data"
datasets = ["GAIT_IST", "GAIT_IT"]

# Create the destination folder
if not os.path.exists(destination_root_dir):
    os.makedirs(destination_root_dir)

for dataset in datasets:
    dataset_path = os.path.join(root_dir, dataset)

    for pathology in os.listdir(dataset_path):
        if pathology.lower() not in [
            "diplegic",
            "hemiplegic",
            "neuropathic",
            "parkinson",
            "normal",
            "parkinsonian",
        ]:
            continue

        # There is a inner folder with the same name on both datasets
        pathology_path = os.path.join(dataset_path, pathology, pathology)

        if pathology.lower() == "parkinson":
            pathology = "parkinsonian"

        # Create the corresponding destination directory
        destination_dir = os.path.join(destination_root_dir, pathology.lower())
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        for subject in os.listdir(pathology_path):
            if subject == ".DS_Store":
                continue
            subject_GEI_path = os.path.join(pathology_path, subject, "GEIs")

            # GAIT_IT has one more layer of folder to clear
            if dataset == "GAIT_IT":
                subject_GEI_path = os.path.join(subject_GEI_path, "side_view")

            for view in os.listdir(subject_GEI_path):
                # GAIT_IT folders has .DS_Store files
                if view == ".DS_Store":
                    continue
                view_path = os.path.join(subject_GEI_path, view)

                for img in os.listdir(view_path):
                    if img == ".DS_Store":
                        continue

                    file_path = os.path.join(view_path, img)
                    img_name = dataset + "_" + pathology + "_" + view + "_" + img

                    shutil.copy2(file_path, os.path.join(destination_dir, img_name))
                    print(f"{file_path} copied to {destination_dir}")
