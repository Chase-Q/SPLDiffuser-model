import torch

import torch.nn as nn

import torch.nn.functional as F



class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.double_conv = nn.Sequential(

            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),

            nn.GroupNorm(1, out_channels),

            nn.GELU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),

            nn.GroupNorm(1, out_channels),

        )

    def forward(self, x): return self.double_conv(x)



class Down(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):

        super().__init__()

        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, t):

        x = self.maxpool_conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb



class Up(nn.Module):

    def __init__(self, in_channels, out_channels, emb_dim=256):

        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv = DoubleConv(in_channels, out_channels)

        self.emb_layer = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))

    def forward(self, x, skip_x, t):

        x = self.up(x)

        x = torch.cat([skip_x, x], dim=1)

        x = self.conv(x)

        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + emb





class SpatialCrossAttention(nn.Module):

    def __init__(self, channels, cond_dim=256, num_heads=4):

        super().__init__()

        self.channels = channels

        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)

        self.ln_q = nn.LayerNorm([channels])

        self.ln_kv = nn.LayerNorm([channels])

        self.proj_kv = nn.Linear(cond_dim, channels)

        

        self.ff = nn.Sequential(

            nn.LayerNorm([channels]), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)

        )



    def forward(self, x, cond_emb):

        # x: [B, C, H, W], cond_emb: [B, 1, cond_dim]

        B, C, H, W = x.shape

        q = x.view(B, C, -1).swapaxes(1, 2) # [B, H*W, C]

        

        kv = self.proj_kv(cond_emb)         # [B, 1, C]

        q_ln, kv_ln = self.ln_q(q), self.ln_kv(kv)



        

        attn_out, _ = self.mha(query=q_ln, key=kv_ln, value=kv_ln)

        

        out = q + attn_out

        out = self.ff(out) + out

        return out.swapaxes(2, 1).view(B, C, H, W)



class UNet_conditional_1D(nn.Module):

    def __init__(self, c_in=3, c_out=3, cond_in=3, time_dim=256, device="cuda"):

        super().__init__()

        self.device = device

        self.time_dim = time_dim

        

        

        self.cond_encoder = nn.Sequential(

            nn.Linear(cond_in, 128), nn.SiLU(), nn.Linear(128, 256)

        )

        

        self.inc = DoubleConv(c_in, 64)

        self.down1 = Down(64, 128) 

        self.down2 = Down(128, 256) 

        

        

        self.cross_attn2 = SpatialCrossAttention(256, cond_dim=256)

        self.down3 = Down(256, 256) 

        self.cross_attn3 = SpatialCrossAttention(256, cond_dim=256)



        self.bot1 = DoubleConv(256, 512)

        self.bot2 = DoubleConv(512, 512)

        self.bot3 = DoubleConv(512, 256)



        self.up1 = Up(512, 128) 

        self.up2 = Up(256, 64) 

        self.up3 = Up(128, 64) 

        self.outc = nn.Conv2d(64, c_out, kernel_size=1)



    def pos_encoding(self, t, channels):

        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))

        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)

        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)

        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)



    def forward(self, x, t, cond):

        t = self.pos_encoding(t.unsqueeze(-1).type(torch.float), self.time_dim)



        # cond shape: [B, 3] -> cond_emb shape: [B, 1, 256]

        cond_emb = self.cond_encoder(cond).unsqueeze(1)



        x1 = self.inc(x)

        x2 = self.down1(x1, t) 

        

        x3 = self.down2(x2, t)

        x3 = self.cross_attn2(x3, cond_emb) 

        

        x4 = self.down3(x3, t)

        x4 = self.cross_attn3(x4, cond_emb) 



        x4 = self.bot1(x4)

        x4 = self.bot2(x4)

        x4 = self.bot3(x4)



        x = self.up1(x4, x3, t)

        x = self.up2(x, x2, t)

        x = self.up3(x, x1, t)



        return self.outc(x)



class EMA:

    def __init__(self, beta):

        self.beta = beta

        self.step = 0

    def update_model_average(self, ma_model, current_model):

        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):

            old_weight, up_weight = ma_params.data, current_params.data

            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):

        if old is None: return new

        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):

        if self.step < step_start_ema:

            self.reset_parameters(ema_model, model)

            self.step += 1

            return

        self.update_model_average(ema_model, model)

        self.step += 1

    def reset_parameters(self, ema_model, model):

        ema_model.load_state_dict(model.state_dict())
