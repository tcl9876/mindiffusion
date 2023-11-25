"""
Simple Unet Structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

try:
    from flash_attn import flash_attn_func
except Exception as e:
    flash_attn_func = None


def attn_func(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    if flash_attn_func is not None:
        return flash_attn_func(
            query, key, value, dropout_p=0.0, softmax_scale=None, causal=False
        )
    else:
        attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(channels, channels * 2)
        self.o_proj = nn.Linear(channels, channels)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        q = self.q_proj(x)
        if context is None:
            context = x
        k, v = torch.chunk(self.kv_proj(x), 2, dim=-1)
        attn_out = attn_func(q, k, v)
        return self.o_proj(attn_out)


class Conv3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, is_res: bool = False
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )

        self.is_res = is_res

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)


class UnetDown(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_attn: bool = False
    ) -> None:
        super(UnetDown, self).__init__()

        self.conv = Conv3(in_channels, out_channels)
        self.use_attn = use_attn
        if use_attn is not None:
            self.self_attn = MultiHeadAttention(out_channels)
            self.cross_attn = MultiHeadAttention(out_channels)

        self.pool = nn.MaxPool2d(2)

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.conv(x)
        if self.use_attn:
            x = self.self_attn(x)
            x = self.cross_attn(x, context)
        x = self.pool(x)
        return x


class UnetUp(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_attn: bool = False
    ) -> None:
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        ]
        self.model = nn.Sequential(*layers)
        if use_attn is not None:
            self.self_attn = MultiHeadAttention(out_channels)
            self.cross_attn = MultiHeadAttention(out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        if self.use_attn:
            x = self.self_attn(x)
            x = self.cross_attn(x, context)
        return x


class TimeSiren(nn.Module):
    def __init__(self, emb_dim: int) -> None:
        super(TimeSiren, self).__init__()

        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x


class NaiveUnet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_feat: int = 256) -> None:
        super(NaiveUnet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat, use_attn=False)
        self.down2 = UnetDown(n_feat, 2 * n_feat, use_attn=True)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat, use_attn=True)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat, use_attn=True)
        self.up2 = UnetUp(4 * n_feat, n_feat, use_attn=True)
        self.up3 = UnetUp(2 * n_feat, n_feat, use_attn=False)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1, context)
        down3 = self.down3(down2, context)

        thro = self.to_vec(down3)
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)

        thro = self.up0(thro + temb)

        up1 = self.up1(thro, down3, context) + temb
        up2 = self.up2(up1, down2, context)
        up3 = self.up3(up2, down1)

        out = self.out(torch.cat((up3, x), 1))

        return out
