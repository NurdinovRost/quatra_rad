import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict


encode = {
    "bottle": 0,
    "packet": 1,
    "glass": 2,
}


class ContainerDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str, fold: int = 0, transform=None):
        self.transform = transform
        self.df = df

        if mode in ["train", "val"]:
            self.df = df[df.fold != fold] if mode == "train" else df[df.fold == fold]
        elif mode == "test":
            self.df = df
        else:
            raise RuntimeError(f"Unsupported mode: {mode}. The mode can only be train, val or test")

        self.path = list(self.df.image_path)
        self.targets = list(self.df.container)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, index: int) -> Dict:
        image_path = self.path[index]
        pil_image = Image.open(image_path).convert("RGB")
        image = np.array(pil_image)

        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        out = {
            "features": image,
            "targets": encode[target],
            "path": str(image_path)
        }
        return out
