import random

import torch
from torchvision.datasets import ImageFolder


class SiameseDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        # 3 separate cases - same image, diff image same class, diff img diff class
        path, target = self.samples[index]
        main_sample = self.loader(path)

        p = torch.rand(1).item()

        if p < 1 / 3:  # Same image
            sub_sample = main_sample

            main_sample = self.transform(main_sample)
            sub_sample = self.transform(sub_sample)

            # Third value dictates if same image, fourth dictates if same class
            return main_sample, sub_sample, 1, 1

        # Different image pair
        sub_index = index

        if 1 / 3 <= p <= 2 / 3:  # Same class
            while sub_index == index:
                sub_index = random.sample(
                    [i for i, t in enumerate(self.targets) if t == target], 1
                )[0]
        elif p > 2 / 3:  # Different class
            while sub_index == index:
                sub_index = random.sample(
                    [i for i, t in enumerate(self.targets) if t != target], 1
                )[0]

        sub_path, sub_target = self.samples[sub_index]
        sub_sample = self.loader(sub_path)

        main_sample = self.transform(main_sample)
        sub_sample = self.transform(sub_sample)

        return main_sample, sub_sample, 0, target == sub_target
