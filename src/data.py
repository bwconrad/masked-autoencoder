import os
from glob import glob
from typing import Callable, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (Compose, Normalize, RandomHorizontalFlip,
                                    RandomResizedCrop, ToTensor)
from torchvision.transforms.functional import InterpolationMode


class SimpleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        size: int = 224,
        min_scale: float = 0.2,
        max_scale: float = 1.0,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        num_val_samples: int = 1000,
        batch_size: int = 32,
        workers: int = 4,
    ):
        """Simple data module

        Args:
            root: Path to image directory
            size: Size of image crop
            min_scale: Minimum random crop scale ratio
            max_scale: Maximum random crop scale ratio
            mean: Normalization channel means
            std: Normalization channel standard deviations
            num_val_samples: Number of validation samples
            batch_size: Number of batch samples
            workers: Number of data workers
        """
        super().__init__()
        self.save_hyperparameters()
        self.root = root
        self.size = size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.mean = mean
        self.std = std
        self.num_val_samples = num_val_samples
        self.batch_size = batch_size
        self.workers = workers

        self.transforms = Compose(
            [
                RandomResizedCrop(
                    self.size,
                    scale=(self.min_scale, self.max_scale),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )

    def setup(self, stage="fit"):
        if stage == "fit":
            dataset = SimpleDataset(self.root, self.transforms)

            # Randomly take num_val_samples images for a validation set
            self.train_dataset, self.val_dataset = data.random_split(
                dataset,
                [len(dataset) - self.num_val_samples, self.num_val_samples],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )


class SimpleDataset(data.Dataset):
    def __init__(self, root: str, transforms: Callable):
        """Image dataset from nested directory

        Args:
            root: Path to directory
            transforms: Image augmentations
        """
        super().__init__()
        self.root = root
        self.paths = [
            f for f in glob(f"{root}/**/*", recursive=True) if os.path.isfile(f)
        ]
        self.transforms = transforms

        print(f"Loaded {len(self.paths)} images from {root}")

    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)
