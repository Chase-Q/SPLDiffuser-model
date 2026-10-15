import torch

import torch.nn as nn

import torch.nn.functional as F





class MultiScaleInvolution2d(nn.Module):

    """Lightweight 2-D multi-scale involution.

    A position-specific kernel is generated from the local feature at each (y, x).
    Two kernel sizes are used in parallel and fused by a learned gate.
    """

    def __init__(self, channels, kernel_sizes=(3, 5), reduction=4, groups=4):

        super().__init__()

        assert channels % groups == 0, "channels must be divisible by groups"

        self.channels = channels

        self.groups = groups

        self.group_channels = channels // groups

        self.kernel_sizes = kernel_sizes



        hidden = max(channels // reduction, 16)

        self.kernel_generators = nn.ModuleList([

            nn.Sequential(

                nn.Conv2d(channels, hidden, 1, bias=False),

                nn.GroupNorm(1, hidden),

                nn.GELU(),

                nn.Conv2d(hidden, groups * k * k, 1)

            ) for k in kernel_sizes

        ])

        self.mix_gate = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(channels, hidden, 1),

            nn.GELU(),

            nn.Conv2d(hidden, len(kernel_sizes), 1)

        )



    def _apply_involution(self, x, kernel_gen, k):

        b, c, h, w = x.shape

        kernel = kernel_gen(x).view(b, self.groups, k * k, h, w)

        patches = F.unfold(x, kernel_size=k, padding=k // 2)

        patches = patches.view(b, self.groups, self.group_channels, k * k, h, w)

        out = (patches * kernel.unsqueeze(2)).sum(dim=3)

        return out.view(b, c, h, w)



    def forward(self, x):

        branch_outs = [

            self._apply_involution(x, gen, k)

            for gen, k in zip(self.kernel_generators, self.kernel_sizes)

        ]

        weights = torch.softmax(self.mix_gate(x).flatten(1), dim=1)

        out = 0.0

        for i, branch in enumerate(branch_outs):

            out = out + branch * weights[:, i].view(-1, 1, 1, 1)

        return out





class ChannelSpatialAttention(nn.Module):

    """Lightweight channel + spatial attention for flow features."""

    def __init__(self, channels, reduction=8):

        super().__init__()

        hidden = max(channels // reduction, 8)

        self.channel = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(channels, hidden, 1),

            nn.GELU(),

            nn.Conv2d(hidden, channels, 1),

            nn.Sigmoid(),

        )

        self.spatial = nn.Sequential(

            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),

            nn.Sigmoid(),

        )



    def forward(self, x):

        x = x * self.channel(x)

        avg_map = x.mean(dim=1, keepdim=True)

        max_map = x.amax(dim=1, keepdim=True)

        x = x * self.spatial(torch.cat([avg_map, max_map], dim=1))

        return x





class ResidualInvolutionBlock(nn.Module):

    def __init__(self, channels, kernel_sizes=(3, 5), groups=4):

        super().__init__()

        self.pre = nn.Sequential(

            nn.GroupNorm(1, channels),

            nn.GELU(),

            nn.Conv2d(channels, channels, 1, bias=False),

        )

        self.inv = MultiScaleInvolution2d(

            channels, kernel_sizes=kernel_sizes, groups=groups

        )

        self.post = nn.Sequential(

            nn.GroupNorm(1, channels),

            nn.GELU(),

            nn.Conv2d(channels, channels, 1, bias=False),

        )

        self.attn = ChannelSpatialAttention(channels)



    def forward(self, x):

        y = self.pre(x)

        y = self.inv(y)

        y = self.post(y)

        y = self.attn(y)

        return x + y





class BottleneckSelfAttention(nn.Module):

    """Global self-attention only at the compact 8x32 bottleneck."""

    def __init__(self, channels, num_heads=4):

        super().__init__()

        assert channels % num_heads == 0

        self.norm1 = nn.LayerNorm(channels)

        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(channels)

        self.ff = nn.Sequential(

            nn.Linear(channels, channels * 2),

            nn.GELU(),

            nn.Linear(channels * 2, channels),

        )



    def forward(self, x):

        b, c, h, w = x.shape

        q = x.flatten(2).transpose(1, 2)  # [B, HW, C]

        qn = self.norm1(q)

        a, _ = self.mha(qn, qn, qn, need_weights=False)

        q = q + a

        q = q + self.ff(self.norm2(q))

        return q.transpose(1, 2).view(b, c, h, w)





class Downsample(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),

            nn.GroupNorm(1, out_ch),

            nn.GELU(),

        )



    def forward(self, x):

        return self.net(x)





class Upsample(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),

            nn.GroupNorm(1, out_ch),

            nn.GELU(),

        )



    def forward(self, x):

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        return self.conv(x)





class PMIAAEncoder(nn.Module):

    """Physics-aware Multi-scale Involution-Attention encoder.

    Input  : [B, 5, 64, 256] = [u, v, p, u_t, v_t]
    Output : [B, latent_ch, 8, 32]
    """

    def __init__(self, latent_ch=3):

        super().__init__()

        # Physics-aware dual branch: spatial state vs local temporal dynamics.

        self.spatial_stem = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1, bias=False),

            nn.GroupNorm(1, 32),

            nn.GELU(),

        )

        self.dynamic_stem = nn.Sequential(

            nn.Conv2d(2, 32, 3, padding=1, bias=False),

            nn.GroupNorm(1, 32),

            nn.GELU(),

        )

        self.spatial_block = ResidualInvolutionBlock(32, groups=4)

        self.dynamic_block = ResidualInvolutionBlock(32, groups=4)



        self.fuse = nn.Sequential(

            nn.Conv2d(64, 64, 1, bias=False),

            nn.GroupNorm(1, 64),

            nn.GELU(),

            ChannelSpatialAttention(64),

        )



        self.down1 = Downsample(64, 96)    # 64x256 -> 32x128

        self.block1 = ResidualInvolutionBlock(96, groups=4)

        self.down2 = Downsample(96, 128)   # 32x128 -> 16x64

        self.block2 = ResidualInvolutionBlock(128, groups=4)

        self.down3 = Downsample(128, 192)  # 16x64 -> 8x32

        self.block3 = ResidualInvolutionBlock(192, groups=4)

        self.global_attn = BottleneckSelfAttention(192, num_heads=4)

        self.to_latent = nn.Sequential(

            nn.Conv2d(192, latent_ch, 1),

            nn.GroupNorm(1, latent_ch),

        )



    def forward(self, x):

        if x.shape[1] != 5:

            raise ValueError(f"PMIAAEncoder expects 5 channels, got {x.shape[1]}")

        xs = self.spatial_block(self.spatial_stem(x[:, :3]))

        xd = self.dynamic_block(self.dynamic_stem(x[:, 3:5]))

        x = self.fuse(torch.cat([xs, xd], dim=1))

        x = self.block1(self.down1(x))

        x = self.block2(self.down2(x))

        x = self.block3(self.down3(x))

        x = self.global_attn(x)

        return self.to_latent(x)





class PMIAADecoder(nn.Module):

    """Decoder mapping latent [B,C,8,32] back to 5x64x256 flow fields."""

    def __init__(self, latent_ch=3):

        super().__init__()

        self.from_latent = nn.Sequential(

            nn.Conv2d(latent_ch, 192, 3, padding=1, bias=False),

            nn.GroupNorm(1, 192),

            nn.GELU(),

        )

        self.global_attn = BottleneckSelfAttention(192, num_heads=4)

        self.block3 = ResidualInvolutionBlock(192, groups=4)

        self.up1 = Upsample(192, 128)       # 8x32 -> 16x64

        self.block2 = ResidualInvolutionBlock(128, groups=4)

        self.up2 = Upsample(128, 96)        # 16x64 -> 32x128

        self.block1 = ResidualInvolutionBlock(96, groups=4)

        self.up3 = Upsample(96, 64)         # 32x128 -> 64x256

        self.head = nn.Sequential(

            ChannelSpatialAttention(64),

            nn.Conv2d(64, 32, 3, padding=1),

            nn.GELU(),

            nn.Conv2d(32, 5, 1),

        )



    def forward(self, z):

        x = self.from_latent(z)

        x = self.global_attn(x)

        x = self.block3(x)

        x = self.block2(self.up1(x))

        x = self.block1(self.up2(x))

        x = self.up3(x)

        return self.head(x)





class PMIAA(nn.Module):

    """Compress [B,5,64,256] to [B,3,8,32] (about 106.67x)."""

    def __init__(self, latent_ch=3):

        super().__init__()

        self.encoder = PMIAAEncoder(latent_ch=latent_ch)

        self.decoder = PMIAADecoder(latent_ch=latent_ch)



    def encode(self, x):

        return self.encoder(x)



    def decode(self, z):

        return self.decoder(z)



    def forward(self, x):

        z = self.encode(x)

        x_hat = self.decode(z)

        return x_hat, z





