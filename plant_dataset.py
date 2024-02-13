import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from torchvision import transforms
from PIL import Image


class PlantDataset(Dataset):
    def __init__(self, is_train= True):
        self.is_train = 0
        if is_train:
            self.is_train = 1
        df = pd.read_csv("data/info_modified.csv")
        df = df[df["is_train"] == self.is_train]
        self.image_files = list(df["File.Name"])
        self.group = list(df["Group"])
        self.label_mapping = {"S":0, "T":1}

    def __len__(self):
        return len(self.group)

    def __getitem__(self, idx):
        group = self.group[idx]
        group = self.label_mapping[group]
        group = torch.tensor(group)
        image_file = self.image_files[idx]
        image_file = image_file + ".png"

        image_path = f"data/images/{image_file}"
        transform = transforms.Compose([
            transforms.Resize((125, 75)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        image = Image.open(image_path)
        tensor_image = transform(image)
        return tensor_image, group


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    ds = PlantDataset(is_train=True)
    dl = DataLoader(ds, batch_size=20, shuffle=True)

    for image, group in dl:
        print(image.shape)
        print(group.shape)

        image = image[0]
        tensor_image_display = image.squeeze().numpy()
        tensor_image_display = (tensor_image_display + 1) / 2.0

        plt.imshow(tensor_image_display, cmap='gray')
        plt.axis('off')
        plt.show()

        break