import os
from glob import glob
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import (ColorJitter, Compose, Normalize,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    ToTensor)
from torchvision.transforms.functional import InterpolationMode

from src.mask_generator import MaskingGenerator


class MIMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        size: int = 224,
        patch_size: int = 0,
        mask_ratio: float = 0.75,
        mask_min_block_patches: int = 1,
        mask_max_block_patches: Optional[int] = 1,
        min_scale: float = 0.08,
        max_scale: float = 1.0,
        flip_prob: float = 0.5,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        num_val_samples: int = 1000,
        batch_size: int = 32,
        workers: int = 4,
    ) -> None:
        """Data module for Masked Image Modeling

        Args:
            root: Path to image directory
            size: Size of image crop
            patch_size: Model patch size
            mask_ratio: Ratio of input image patches to mask
            mask_min_block_patches: Min number of patches within a masking block
                (when mask_min_block_patches = mask_max_block_patches = 1 then it is random masking)
            mask_max_block_patches: Max number of patches within a masking block
                (when mask_min_block_patches = mask_max_block_patches = 1 then it is random masking)
            min_scale: Minimum random crop scale ratio
            max_scale: Maximum random crop scale ratio
            flip_prob: Probability of applying horizontal flip
            brightness: Brightness jitter intensity
            contrast: Contrast jitter intensity
            saturation: Saturation jitter intensity
            hue: Hue jitter intensity
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
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_min_block_patches = mask_min_block_patches
        self.mask_max_block_patches = mask_max_block_patches
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.flip_prob = flip_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
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
                RandomHorizontalFlip(p=self.flip_prob),
                ColorJitter(
                    brightness=self.brightness,  # type:ignore
                    contrast=self.contrast,  # type:ignore
                    saturation=self.saturation,  # type:ignore
                    hue=self.hue,  # type:ignore
                ),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )

        # Initialize mask generator
        window_size = self.size // self.patch_size
        num_masking_patches = int(window_size**2 * self.mask_ratio)
        self.mask_generator = MaskingGenerator(
            input_size=window_size,
            num_masking_patches=num_masking_patches,
            min_num_patches=self.mask_min_block_patches,
            max_num_patches=self.mask_max_block_patches,
        )

    def setup(self, stage="fit") -> None:
        if stage == "fit":
            dataset = ImageMaskDataset(self.root, self.transforms, self.mask_generator)

            # Randomly take num_val_samples images for a validation set
            self.train_dataset, self.val_dataset = data.random_split(
                dataset,
                [len(dataset) - self.num_val_samples, self.num_val_samples],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )


class ImageMaskDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        transforms: Compose,
        mask_generator: MaskingGenerator,
        extensions: Tuple[str, ...] = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
            ".webp",
        ),
    ) -> None:
        """Image dataset from nested directory which also generates a random mask

        Args:
            root: Path to image directory
            transforms: Image augmentations pipeline
            mask_generator: Mask generator object
            extensions: File extensions of image files to load
        """
        super().__init__()
        self.root = root
        self.paths = [
            f
            for f in glob(f"{root}/**/*", recursive=True)
            if os.path.isfile(f) and f.lower().endswith(extensions)
        ]
        self.transforms = transforms
        self.mask_generator = mask_generator

        if len(self.paths) == 0:
            raise Exception(f"No files found in image root directory '{root}'")

        print(f"Loaded {len(self.paths)} images from {root}")

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Load image
        img = Image.open(self.paths[index]).convert("RGB")
        img = self.transforms(img)

        # Generate a binary mask
        mask = torch.BoolTensor(self.mask_generator()).flatten(0)

        # Calculate the indices to restore the patch order
        idx_restore = torch.argsort(torch.argsort(mask, dim=0, stable=True), dim=0)

        return img, mask, idx_restore  # type:ignore

    def __len__(self) -> int:
        return len(self.paths)
