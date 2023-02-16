from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
from einops import repeat
from timm.models.vision_transformer import VisionTransformer

from src.network.utils import get_2d_sincos_pos_embed


class VisionTransformerMAE(VisionTransformer):
    def __init__(self, **kwargs) -> None:
        super(VisionTransformerMAE, self).__init__(**kwargs)
        assert self.num_prefix_tokens == 1  # Must have cls token

        # Re-initialize with fixed sin-cos position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(self.pos_embed.shape), requires_grad=False
        )
        self.init_pos_embed()

    def init_pos_embed(self) -> None:
        # Initialize to sin-cos position embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Patch embed image
        x = self.patch_embed(x)
        b, n, d = x.size()

        # Add position embedding
        x = x + self.pos_embed[:, 1:, :]  # Skip cls token

        # Collect unmasked patches: [b, n, d] -> [b, n * (1-mask_ratio), d]
        mask = ~mask[:, :, None]  # Change to 1 = keep patch
        x = torch.masked_select(x, mask).reshape(b, -1, d)

        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = repeat(cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat([cls_token, x], dim=1)

        # Apply transformer layers
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)

        return x


def build_encoder(model, **kwargs) -> Tuple[VisionTransformerMAE, int]:
    try:
        model_fn, patch_size = MODEL_DICT[model]
    except:
        raise ValueError(
            f"{model} is not an available encoder. Should be one of {[k for k in MODEL_DICT.keys()]}"
        )

    return model_fn(**kwargs), patch_size


def vit_tiny_patch16(**kwargs) -> VisionTransformerMAE:
    return VisionTransformerMAE(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_small_patch16(**kwargs) -> VisionTransformerMAE:
    return VisionTransformerMAE(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base_patch16(**kwargs) -> VisionTransformerMAE:
    return VisionTransformerMAE(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large_patch16(**kwargs) -> VisionTransformerMAE:
    return VisionTransformerMAE(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_huge_patch14(**kwargs) -> VisionTransformerMAE:
    return VisionTransformerMAE(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


MODEL_DICT = {
    "vit_tiny_patch16": (vit_tiny_patch16, 16),
    "vit_small_patch16": (vit_small_patch16, 16),
    "vit_base_patch16": (vit_base_patch16, 16),
    "vit_large_patch16": (vit_large_patch16, 16),
    "vit_huge_patch14": (vit_huge_patch14, 14),
}
