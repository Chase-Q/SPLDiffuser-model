import os

import torch

import torch.nn.functional as F

from torch import optim

from torch.utils.data import DataLoader

from tqdm import tqdm

import matplotlib.pyplot as plt



from jiadataset_latent import FluidPhysicsDatasetLatent

from pmiaa import PMIAA

from physics_losses import gradient_loss, ns_physics_loss



BASE_DIR =

FEATURE_FILES = {

    "/": os.path.join(BASE_DIR, ""),



}

NPY_DIR = os.path.join(BASE_DIR, "")

STATS_PATH = os.path.join(BASE_DIR, "")



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE =

LR =

EPOCHS =



CHECKPOINT_DIR = ""

VIS_DIR = ""



# AE loss weights. Start with accurate reconstruction; physics is warm-started.






def save_vis(model, gt, epoch, mean_tensor, std_tensor):

    model.eval()

    with torch.no_grad():

        pred, z = model(gt[:1])

        gt_phys = gt[:1] * std_tensor + mean_tensor

        pr_phys = pred * std_tensor + mean_tensor



    names = ["u", "v", "p"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 9))

    for c in range(3):

        g = gt_phys[0, c].cpu().numpy()

        p = pr_phys[0, c].cpu().numpy()

        e = p - g

        vmin = min(g.min(), p.min())

        vmax = max(g.max(), p.max())

        im0 = axes[c, 0].imshow(g, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)

        im1 = axes[c, 1].imshow(p, origin="lower", cmap="jet", vmin=vmin, vmax=vmax)

        em = max(abs(e.min()), abs(e.max()), 1e-12)

        im2 = axes[c, 2].imshow(e, origin="lower", cmap="bwr", vmin=-em, vmax=em)

        axes[c, 0].set_title(f"GT {names[c]}")

        axes[c, 1].set_title(f"AE recon {names[c]}")

        axes[c, 2].set_title(f"Error {names[c]}")

        for j, im in enumerate((im0, im1, im2)):

            axes[c, j].axis("off")

            fig.colorbar(im, ax=axes[c, j], fraction=0.046, pad=0.04)

    fig.suptitle(f"PMIAA epoch {epoch} | latent shape={tuple(z.shape[1:])}")

    plt.tight_layout()

    plt.savefig(os.path.join(VIS_DIR, f"pmiaa_ep_{epoch}.png"), dpi=180, bbox_inches="tight")

    plt.close(fig)

    model.train()





def train():

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    os.makedirs(VIS_DIR, exist_ok=True)



    dataset = FluidPhysicsDatasetLatent(FEATURE_FILES, NPY_DIR, STATS_PATH)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)



    mean_tensor = torch.tensor(dataset.mean_img, dtype=torch.float32, device=DEVICE).view(1, 5, 1, 1)

    std_tensor = torch.tensor(dataset.std_img, dtype=torch.float32, device=DEVICE).view(1, 5, 1, 1)



    model = PMIAA(latent_ch=LATENT_CH).to(DEVICE)

    print(f"PMIAA latent shape: {(LATENT_CH, LATENT_H, LATENT_W)}")

    print(f"Compression ratio: {COMPRESSION_RATIO:.2f}x")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)



    fixed_cond, fixed_gt = next(iter(loader))

    fixed_gt = fixed_gt[:1].to(DEVICE)



    for epoch in range(1, EPOCHS + 1):

        pbar = tqdm(loader, desc=f"PMIAA AE epoch {epoch}")

        for _, x in pbar:

            x = x.to(DEVICE, non_blocking=True)

            x_hat, z = model(x)

            if tuple(z.shape[1:]) != (LATENT_CH, LATENT_H, LATENT_W):

                raise RuntimeError(

                    f"Expected latent {(LATENT_CH, LATENT_H, LATENT_W)}, "

                    f"got {tuple(z.shape[1:])}"

                )



            loss_rec = F.l1_loss(x_hat, x)

            loss_grad = gradient_loss(x_hat, x)

            loss_cont, loss_mom = ns_physics_loss(

                x_hat, mean_tensor, std_tensor, DX, DY, NU

            )

            phys_w = 0.0 if epoch <= PHYS_WARMUP else LAMBDA_PHYS

            loss = (

                LAMBDA_REC * loss_rec

                + LAMBDA_GRAD * loss_grad

                + phys_w * (loss_cont + loss_mom)

            )



            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()



            pbar.set_postfix(

                Rec=f"{loss_rec.item():.4f}",

                Grad=f"{loss_grad.item():.4f}",

                Phys=f"{(phys_w*(loss_cont+loss_mom)).item():.3e}",

                Z=str(tuple(z.shape[1:])),

            )



        if epoch == 1 or epoch % 20 == 0:

            save_vis(model, fixed_gt, epoch, mean_tensor, std_tensor)

        if epoch % 20 == 0:

            torch.save(

                {

                    "model": model.state_dict(),

                    "latent_ch": LATENT_CH,

                    "latent_shape": (LATENT_CH, LATENT_H, LATENT_W),

                    "compression_ratio": COMPRESSION_RATIO,

                    "epoch": epoch,

                },

                os.path.join(CHECKPOINT_DIR, f"pmiaa_ep_{epoch}.pt"),

            )



    torch.save(

        {"model": model.state_dict(), "latent_ch": LATENT_CH, "latent_shape": (LATENT_CH, LATENT_H, LATENT_W), "compression_ratio": COMPRESSION_RATIO, "epoch": EPOCHS},

        os.path.join(CHECKPOINT_DIR, "pmiaa_final.pt"),

    )





if __name__ == "__main__":

    train()





