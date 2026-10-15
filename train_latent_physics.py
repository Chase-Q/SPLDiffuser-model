import os

import copy

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import optim

from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt



from jiadataset_latent import FluidPhysicsDatasetLatent

from model_physics import UNet_conditional_1D, EMA

from diffusion import Diffusion

from pmiaa import PMIAA

from physics_losses import ns_physics_loss



BASE_DIR = "/dataset/"

FEATURE_FILES = {

    "/": os.path.join(BASE_DIR, "1.txt"),

    "/": os.path.join(BASE_DIR, "2.txt"),


}

NPY_DIR = os.path.join(BASE_DIR, "")

STATS_PATH = os.path.join(BASE_DIR, "")

COND_STATS_PATH = os.path.join(BASE_DIR, "")



# First train PMIAA, then point this to the selected checkpoint.

PMIAA_CKPT =

CHECKPOINT_DIR = "

VIS_DIR =



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




# Loss weights. Noise MSE remains the primary diffusion objective.







def load_pmiaa():

    ckpt = torch.load(PMIAA_CKPT, map_location=DEVICE)

    latent_ch = ckpt.get("latent_ch", LATENT_CH)

    if latent_ch != LATENT_CH:

        raise ValueError(f"PMIAA checkpoint latent_ch={latent_ch}, config LATENT_CH={LATENT_CH}")

    ae = PMIAA(latent_ch=LATENT_CH).to(DEVICE)

    ae.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    ae.eval()

    for p in ae.parameters():

        p.requires_grad_(False)

    return ae





@torch.no_grad()

def sample_latent(model, diffusion, cond, shape):

    b = cond.shape[0]

    z = torch.randn((b, *shape), device=DEVICE)

    for i in reversed(range(diffusion.noise_steps)):

        t = torch.full((b,), i, device=DEVICE, dtype=torch.long)

        predicted_noise = model(z, t, cond)

        alpha = diffusion.alpha[t][:, None, None, None]

        alpha_hat = diffusion.alpha_hat[t][:, None, None, None]

        beta = diffusion.beta[t][:, None, None, None]

        noise = torch.randn_like(z) if i > 1 else torch.zeros_like(z)

        z = 1 / torch.sqrt(alpha) * (

            z - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise

        ) + torch.sqrt(beta) * noise

    return z





def save_visualization(ema_model, ae, diffusion, cond, gt_img, epoch, mean_tensor, std_tensor):

    ema_model.eval()

    with torch.no_grad():

        z_hat = sample_latent(ema_model, diffusion, cond, (LATENT_CH, LATENT_H, LATENT_W))

        x_hat = ae.decode(z_hat)

        gt_phys = gt_img * std_tensor + mean_tensor

        pr_phys = x_hat * std_tensor + mean_tensor



    names = ["u", "v", "p"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 9))

    for c in range(3):

        g = gt_phys[0, c].cpu().numpy()

        p = pr_phys[0, c].cpu().numpy()

        e = p - g

        vmin, vmax = min(g.min(), p.min()), max(g.max(), p.max())

        em = max(abs(e.min()), abs(e.max()), 1e-12)

        ims = [

            axes[c,0].imshow(g, origin="lower", cmap="jet", vmin=vmin, vmax=vmax),

            axes[c,1].imshow(p, origin="lower", cmap="jet", vmin=vmin, vmax=vmax),

            axes[c,2].imshow(e, origin="lower", cmap="bwr", vmin=-em, vmax=em),

        ]

        axes[c,0].set_title(f"GT {names[c]}")

        axes[c,1].set_title(f"Latent diffusion {names[c]}")

        axes[c,2].set_title(f"Error {names[c]}")

        for j, im in enumerate(ims):

            axes[c,j].axis("off")

            fig.colorbar(im, ax=axes[c,j], fraction=0.046, pad=0.04)

    fig.suptitle(f"Conditional latent diffusion epoch {epoch}")

    plt.tight_layout()

    plt.savefig(os.path.join(VIS_DIR, f"latent_diff_ep_{epoch}.png"), dpi=180, bbox_inches="tight")

    plt.close(fig)

    ema_model.train()





def train():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    os.makedirs(VIS_DIR, exist_ok=True)



    dataset = FluidPhysicsDatasetLatent(FEATURE_FILES, NPY_DIR, STATS_PATH, COND_STATS_PATH, save_cond_stats=True)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    mean_tensor = torch.tensor(dataset.mean_img, dtype=torch.float32, device=DEVICE).view(1, 5, 1, 1)

    std_tensor = torch.tensor(dataset.std_img, dtype=torch.float32, device=DEVICE).view(1, 5, 1, 1)

    cond_dim = dataset.cond_dim



    ae = load_pmiaa()

    model = UNet_conditional_1D(

        c_in=LATENT_CH,

        c_out=LATENT_CH,

        cond_in=cond_dim,

        device=DEVICE,

    ).to(DEVICE)

    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(DEVICE)

    ema = EMA(beta=0.995)

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    mse = nn.MSELoss()

    diffusion = Diffusion(img_size=None, device=DEVICE)



    fixed_cond, fixed_gt = next(iter(loader))

    fixed_cond = fixed_cond[:1].to(DEVICE)

    fixed_gt = fixed_gt[:1].to(DEVICE)



    # Shape guard: fail early if the selected PMIAA checkpoint has different geometry.

    with torch.no_grad():

        z_check = ae.encode(fixed_gt)

    if tuple(z_check.shape[1:]) != (LATENT_CH, LATENT_H, LATENT_W):

        raise ValueError(f"Expected latent {(LATENT_CH,LATENT_H,LATENT_W)}, got {tuple(z_check.shape[1:])}")



    for epoch in range(EPOCHS):

        pbar = tqdm(loader, desc=f"Latent diffusion epoch {epoch+1}")

        for cond, images in pbar:

            images = images.to(DEVICE, non_blocking=True)

            cond = cond.to(DEVICE, non_blocking=True)



            with torch.no_grad():

                z0 = ae.encode(images)



            t = diffusion.sample_timesteps(images.shape[0]).to(DEVICE)

            z_t, noise = diffusion.noise_images(z0, t)

            predicted_noise = model(z_t, t, cond)

            loss_diff = mse(predicted_noise, noise)



            # Estimate clean latent directly from the predicted noise.

            alpha_hat_t = diffusion.alpha_hat[t][:, None, None, None]

            z0_hat = (z_t - torch.sqrt(1 - alpha_hat_t) * predicted_noise) / torch.sqrt(alpha_hat_t)



            # Auxiliary latent-space matching.

            loss_latent = F.l1_loss(z0_hat, z0)



            # Decode back to physical-field representation so field/NS constraints remain meaningful.

            x0_hat = ae.decode(z0_hat)

            loss_field = F.l1_loss(x0_hat, images)

            loss_cont, loss_mom = ns_physics_loss(

                x0_hat, mean_tensor, std_tensor, DX, DY, NU

            )



            aux_on = epoch >= AUX_WARMUP

            wz = LAMBDA_LATENT if aux_on else 0.0

            wf = LAMBDA_FIELD if aux_on else 0.0

            wp = LAMBDA_PHYS if aux_on else 0.0

            loss = loss_diff + wz * loss_latent + wf * loss_field + wp * (loss_cont + loss_mom)



            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            ema.step_ema(ema_model, model)



            pbar.set_postfix(

                Diff=f"{loss_diff.item():.4f}",

                Z=f"{(wz*loss_latent).item():.4f}",

                Field=f"{(wf*loss_field).item():.4f}",

                Phys=f"{(wp*(loss_cont+loss_mom)).item():.3e}",

            )



        if (epoch + 1) % VIS_INTERVAL == 0:

            save_visualization(

                ema_model, ae, diffusion, fixed_cond, fixed_gt,

                epoch + 1, mean_tensor, std_tensor

            )

        if (epoch + 1) % 50 == 0:

            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_latent_ep_{epoch+1}.pt"))

            torch.save(ema_model.state_dict(), os.path.join(CHECKPOINT_DIR, f"ema_latent_ep_{epoch+1}.pt"))





if __name__ == "__main__":

    train()





