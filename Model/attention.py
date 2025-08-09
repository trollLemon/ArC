from torch import nn
import torch.nn.functional as F
import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def create_causal_mask(batch_size,sequence_len):
    mask = torch.triu(torch.full((batch_size,sequence_len, sequence_len), -1e9), diagonal=2)
    bool_mask = mask != 0
    return bool_mask


def create_padding_mask(batch):
   pad_flags = (batch.sum(dim=-1) ==  0)
   padded = pad_flags.float().masked_fill(pad_flags, -1e9)
   B, seq_len = batch.shape[0], batch.shape[1]
   mask = torch.zeros(B, seq_len, seq_len, device=batch.device) + padded[:, :, None] + padded[:, None, :]
   bool_mask = mask != 0
   return bool_mask


class JointAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv_x = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_qkv_y = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out_x = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.to_out_y = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y, mask = None):
        x = self.norm_x(x)
        y = self.norm_y(y)

        qkv_x = self.to_qkv_x(x).chunk(3, dim = -1)
        qkv_y = self.to_qkv_y(y).chunk(3, dim = -1)

        q_x, k_x, v_x = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_x)
        q_y, k_y, v_y = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv_y)

        q = torch.cat((q_x, q_y), dim = 1)
        k = torch.cat((k_x, k_y), dim = 1)
        v = torch.cat((v_x, v_y), dim = 1)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # apply mask only to the color palette scores,
            # we don't want to mask the image scores.
            palette_dots = dots[:, x.shape[1]:, : ]
            palette_dots = palette_dots.masked_fill(mask=mask, value=-1e9)
            dots[:, x.shape[1]:, : ] = palette_dots

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        # split back into x and y
        x, y = out.chunk(2, dim = 1)

        out_x = rearrange(x, 'b h n d -> b n (h d)')
        out_y = rearrange(x, 'b h n d -> b n (h d)')
        return self.to_out_x(out_x), self.to_out_y(out_y)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            dots = dots.masked_fill(mask=mask, value=-1e9)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        #print(y.shape)
        return self.to_out(out)


class EncoderDecoderAttention(nn.Module):
        def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
            super().__init__()
            inner_dim = dim_head * heads
            project_out = not (heads == 1 and dim_head == dim)

            self.num_heads = heads
            self.scale = dim_head ** -0.5

            self.norm = nn.LayerNorm(dim)

            self.attend = nn.Softmax(dim=-1)
            self.dropout = nn.Dropout(dropout)

            self.q_proj = nn.Linear(dim, inner_dim * 3, bias = False)
            self.k_proj = nn.Linear(dim, inner_dim * 3, bias = False)
            self.v_proj = nn.Linear(dim, inner_dim * 3, bias = False)

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()

        def forward(self, x, y, mask = None):
            batch_size = x.shape[0]

            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

            inner_dim = k.shape[-1]
            head_dim = inner_dim // self.num_heads
            q = q.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)

            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(batch_size, -1, self.num_heads * head_dim)

            return out
