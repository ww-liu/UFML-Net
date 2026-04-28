import torch
import torch.nn as nn
import torch.nn.functional as F


class DSwin1DAttention(nn.Module):
    """
    Simplified DSwin-style 1D attention for ECG
    Sliding window self-attention
    Input:  (B, C, T)
    Output: (B, C, T)
    """

    def __init__(self, dim, window_size=16, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        B, T, C = x.shape

        # pad to window size
        pad_len = (self.window_size - T % self.window_size) % self.window_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))  # pad time dimension

        Tp = x.shape[1]
        num_windows = Tp // self.window_size

        # (B, num_win, win, C)
        x = x.view(B, num_windows, self.window_size, C)

        # QKV
        qkv = self.qkv(x)  # (B, num_win, win, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for multi-head
        q = q.view(B, num_windows, self.window_size, self.num_heads, self.head_dim)
        k = k.view(B, num_windows, self.window_size, self.num_heads, self.head_dim)
        v = v.view(B, num_windows, self.window_size, self.num_heads, self.head_dim)

        # attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v  # (B, num_win, win, heads, head_dim)
        out = out.reshape(B, Tp, C)

        out = self.proj(out)

        # remove padding
        out = out[:, :T, :]
        return out.permute(0, 2, 1)


class DSwinBlock(nn.Module):
    """
    DSwin Block with residual connection
    """

    def __init__(self, dim, window_size=16, num_heads=4):
        super().__init__()
        self.norm = nn.BatchNorm1d(dim)
        self.attn = DSwin1DAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads
        )

    def forward(self, x):
        # x: (B, C, T)
        return x + self.attn(self.norm(x))