from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin
import math
import torch
from einops import rearrange

from src.models.motion_module import zero_module
from src.models.resnet import InflatedConv3d
from torch.cuda.amp import autocast


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
    
    
class TemporalAttention(nn.Module):
    def __init__(self, channels, num_heads, max_len):
        super().__init__()
        self.attention = nn.MultiheadAttention(channels, num_heads)
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Linear(channels, channels)
        self.max_len = max_len
        self.pe = self.generate_positional_encoding(max_len, channels)

    def generate_positional_encoding(self, seq_len, channels):
        pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-math.log(10000.0) / channels))
        pe = torch.zeros(seq_len, channels, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(1)  # (seq_len, 1, channels)

    def forward(self, x):
        b, h, w, c, f = x.shape
        x = rearrange(x, "b h w c f -> f (b h w) c")

        seq_len = x.size(0)
        pos_encoding = self.pe.to(x.device)[:seq_len, :, :]
        hidden = x + pos_encoding

        with autocast():
            hidden = self.norm(hidden)
            hidden, _ = self.attention(hidden, hidden, hidden)
        hidden = self.proj(hidden)
        x = x + hidden
        x = rearrange(x, "f (b h w) c -> b h w c f", b=b, h=h, w=w)
        return x


class PoseGuiderWithTemporal(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
        max_len: int = 24,
    ):
        super().__init__()
        self.max_len = max_len
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])
        self.attn_blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            #self.attn_blocks.append(TemporalAttention(channel_in, 4, max_len))
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
            self.attn_blocks.append(TemporalAttention(channel_out, 4, max_len))

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)
        assert len(self.blocks) // 2 == len(self.attn_blocks)
        for i in range(len(self.blocks) // 2):
            embedding = self.blocks[2*i](embedding)
            embedding = F.silu(embedding)
            embedding = self.blocks[2*i+1](embedding)
            embedding = F.silu(embedding)
            embedding = rearrange(embedding, "b c f h w -> b h w c f")
            embedding = self.attn_blocks[i](embedding)
            embedding = rearrange(embedding, "b h w c f -> b c f h w")
        embedding = self.conv_out(embedding)
        return embedding
