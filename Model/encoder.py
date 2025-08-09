"""
vit.py

This file contains a modified implementation of vit.py from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .util import ColorEmbedding
from .attention import  JointAttention
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)



class ViTEncoderBlock(nn.Module):
    def __init__(self, dim, dim_head, heads, dropout = 0.):
        super().__init__()
        self.join_attn = JointAttention(dim, heads = heads, dim_head=dim_head, dropout = dropout)
        self.ff_x = FeedForward(dim, dim_head, dropout = dropout)
        self.ff_y = FeedForward(dim, dim_head, dropout = dropout)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
    def forward(self, x,y, mask):
        x_out,y_out = self.join_attn(x, y, mask)
        x = x + x_out
        y = y + y_out
        x = self.ff_x(x)
        y = self.ff_y(y)
        x = self.norm_x(x)
        y = self.norm_y(y)
        return x,y



class ViTEncoder(nn.Module):
    def __init__(self, *, image_size, patch_size,dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.to_color_embedding = ColorEmbedding(num_patches, dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding_image = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embedding_palette= nn.Parameter(torch.randn(1, num_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(
                 ViTEncoderBlock(dim, dim_head, heads, dropout = emb_dropout),
            )

    def forward(self, img, palette, mask = None):
        x = self.to_patch_embedding(img)
        c = self.to_color_embedding(palette)

        b, n, _ = x.shape
        c += self.pos_embedding_image
        x += self.pos_embedding_palette

        x = self.dropout(x)
        for block in self.blocks:
            x, c = block(x, c, mask )

        return c