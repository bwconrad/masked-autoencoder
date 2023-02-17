import os
from typing import Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange
from timm.optim.optim_factory import param_groups_weight_decay
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid, save_image
from transformers.optimization import get_cosine_schedule_with_warmup

from .network.decoder import LinearDecoder, VisionTransformerDecoder
from .network.encoder import build_encoder


class MaskedAutoencoderModel(pl.LightningModule):
    def __init__(
        self,
        encoder_arch: str = "vit_base_patch16",
        decoder_type: str = "vit",
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        norm_pixel_loss: bool = True,
        image_size: int = 224,
        lr: float = 1.5e-4,
        optimizer: str = "adamw",
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.05,
        momentum: float = 0.9,
        scheduler: str = "cosine",
        warmup_epochs: int = 0,
    ):
        """Masked Autoencoder Pretraining Model

        Args:
            encoder_arch: Name of encoder network architecture. One of [vit_tiny_patch16, vit_small_patch16,
                 vit_base_patch16, vit_large_patch16, vit_huge_14]
            decoder_type: Decoder type. One of [vit, linear]
            decoder_embed_dim: Embed dim of ViT decoder
            decoder_depth: Number of transformer blocks in ViT decoder
            decoder_num_heads: Number of attention heads in ViT decoder
            norm_pixel_loss: Calculate loss using normalized pixel value targets
            image_size: Size of input image
            lr: Learning rate (should be linearly scaled with batch size)
            optimizer: Name of optimizer (adam | adamw | sgd)
            betas: Adam beta parameters
            weight_decay: Optimizer weight decay
            momentum: SGD momentum parameter
            scheduler: Name of learning rate scheduler [cosine, none]
            warmup_epochs: Number of warmup epochs
        """
        super().__init__()
        self.save_hyperparameters()
        self.encoder_arch = encoder_arch
        self.decoder_type = decoder_type
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.norm_pixel_loss = norm_pixel_loss
        self.image_size = image_size
        self.lr = lr
        self.optimizer = optimizer
        self.betas = betas
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs

        # Initialize networks
        self.encoder, self.patch_size = build_encoder(
            encoder_arch, img_size=self.image_size
        )
        if self.decoder_type == "vit":
            self.decoder = VisionTransformerDecoder(
                patch_size=self.patch_size,
                num_patches=self.encoder.patch_embed.num_patches,
                in_dim=self.encoder.embed_dim,
                embed_dim=self.decoder_embed_dim,
                depth=self.decoder_depth,
                num_heads=self.decoder_num_heads,
            )
        else:
            self.decoder = LinearDecoder(
                in_dim=self.encoder.embed_dim,
                patch_size=self.patch_size,
            )

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Rearrange image into patches
        (b, 3, h, w) -> (b, l, patch_size^2 * 3)
        """
        assert x.shape[2] == x.shape[3] and x.shape[2] % self.patch_size == 0

        return rearrange(
            x,
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=self.patch_size,
            p2=self.patch_size,
        )

    def unpatchify(self, x):
        """Rearrange patches back to an image
        (b, l, patch_size^2 * 3) -> (b, 3, h, w)
        """
        h = w = int(x.shape[1] ** 0.5)
        return rearrange(
            x,
            " b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            h=h,
            w=w,
        )

    def log_samples(
        self, inp: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> None:
        """Log sample outputs"""
        # Only log up to 16 images
        inp, pred, mask = inp[:16], pred[:16], mask[:16]

        # Patchify the input image
        inp = self.patchify(inp)

        # Merge original and predicted patches
        pred = pred * mask[:, :, None]
        inp = inp * (~mask[:, :, None])
        res = self.unpatchify(inp) + self.unpatchify(pred)

        # Log result
        if "CSVLogger" in str(self.logger.__class__):
            path = os.path.join(self.logger.log_dir, "samples")  # type:ignore
            if not os.path.exists(path):
                os.makedirs(path)
            filename = os.path.join(path, str(self.current_epoch) + "ep.png")
            save_image(res, filename, nrow=4, normalize=True)
        elif "WandbLogger" in str(self.logger.__class__):
            grid = make_grid(res, nrow=4, normalize=True)
            self.logger.log_image(key="sample", images=[grid])  # type:ignore

    def loss(
        self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """MSE loss on masked patches"""
        target = self.patchify(target)

        # Normalize pixel values
        if self.norm_pixel_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        # Calculate MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Per patch loss
        loss = (loss * mask).sum() / mask.sum()  # Mean of masked patches

        return loss

    def shared_step(self, batch, mode="train", idx=None):
        x, mask, idx_restore = batch

        # Pass through auto-encoder
        z = self.encoder(x, mask)
        pred = self.decoder(z, idx_restore)

        # Calculate loss
        loss = self.loss(pred, x, mask)

        # Log
        self.log(f"{mode}_loss", loss)
        if mode == "val" and idx == 0:
            self.log_samples(x, pred, mask)

        return loss

    def training_step(self, x, _):
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return self.shared_step(x, mode="train")

    def validation_step(self, x, batch_idx):
        return self.shared_step(x, mode="val", idx=batch_idx)

    def configure_optimizers(self):
        """Initialize optimizer and learning rate schedule"""

        # Set weight decay to 0 for bias and norm layers
        params = param_groups_weight_decay(
            self.encoder, self.weight_decay
        ) + param_groups_weight_decay(self.decoder, self.weight_decay)

        # Initialize optimizer
        if self.optimizer == "adam":
            optimizer = Adam(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adamw":
            optimizer = AdamW(
                params,
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "sgd":
            optimizer = SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(
                f"{self.optimizer} is not an available optimizer. Should be one of ['adam', 'adamw', 'sgd']"
            )

        # Initialize learning rate schedule
        if self.scheduler == "cosine":
            epoch_steps = (
                self.trainer.estimated_stepping_batches
                // self.trainer.max_epochs  # type:ignore
            )
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=self.trainer.estimated_stepping_batches,  # type:ignore
                num_warmup_steps=epoch_steps * self.warmup_epochs,
            )
        elif self.scheduler == "none":
            scheduler = LambdaLR(optimizer, lambda _: 1)
        else:
            raise ValueError(
                f"{self.scheduler} is not an available optimizer. Should be one of ['cosine', 'none']"
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
