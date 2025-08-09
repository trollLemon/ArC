from torch import nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .util import ColorEmbedding
from .attention import Attention, EncoderDecoderAttention

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



class PredictionHead(nn.Module):
    def __init__(self, num_patches, hidden_dim, channels = 3, classes=256,sequence_length=64):
        super().__init__()
        self.channels = channels
        self.classes = classes
        self.num_patches = num_patches
        self.sequence_length = sequence_length
        self.mlp = nn.Sequential(
            nn.Linear(num_patches * hidden_dim, sequence_length * channels * classes),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.mlp(x)
        x = x.view(B, self.sequence_length, self.channels, self.classes)
        return x



class DecoderBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()

        self.self_attention = Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)
        self.enc_dec_attention = EncoderDecoderAttention(dim, heads, dim_head, dropout = dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout = dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, encoding, mask):
        x += self.self_attention(x, mask)
        x = self.norm1(x)
        x += self.enc_dec_attention(x,encoding, mask)
        x = self.norm2(x)
        x = self.ff(x)
        return x

class PaletteDecoder(nn.Module):
    def __init__( self, image_size, patch_size,dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.to_color_embedding = ColorEmbedding(num_patches, dim)
        self.dropout = nn.Dropout(emb_dropout)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.self_attention = Attention(dim, heads = heads, dim_head=dim_head, dropout = dropout)
        self.prediction_head = PredictionHead(num_patches, dim, channels, classes=256)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append( DecoderBlock(dim, depth, heads, dim_head, mlp_dim, dropout = emb_dropout) )

    def forward(self, x, encoding, mask = None):
        x = self.to_color_embedding(x)
        x += self.pos_embedding
        for block in self.blocks:
            x = block(x, encoding, mask)

        x = self.prediction_head(x)
        return x