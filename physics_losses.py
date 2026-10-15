import torch

import torch.nn.functional as F





def gradient_loss(pred, target):

    """L1 mismatch of first spatial derivatives in normalized space."""

    pred_dy, pred_dx = torch.gradient(pred, dim=(2, 3))

    true_dy, true_dx = torch.gradient(target, dim=(2, 3))

    return F.l1_loss(pred_dx, true_dx) + F.l1_loss(pred_dy, true_dy)





def ns_physics_loss(x_norm, mean_tensor, std_tensor, dx, dy, nu):

    """Continuity + 2-D incompressible NS residual on decoded 5-channel fields.

    x_norm channels: [u, v, p, u_t, v_t]
    mean_tensor/std_tensor: shape [1,5,1,1]
    """

    x = x_norm * std_tensor + mean_tensor

    u, v, p, u_t, v_t = (

        x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], x[:, 4:5]

    )



    du_dy, du_dx = torch.gradient(u, spacing=(dy, dx), dim=(2, 3))

    dv_dy, dv_dx = torch.gradient(v, spacing=(dy, dx), dim=(2, 3))

    dp_dy, dp_dx = torch.gradient(p, spacing=(dy, dx), dim=(2, 3))



    u_yy = torch.gradient(du_dy, spacing=(dy, dx), dim=(2, 3))[0]

    u_xx = torch.gradient(du_dx, spacing=(dy, dx), dim=(2, 3))[1]

    v_yy = torch.gradient(dv_dy, spacing=(dy, dx), dim=(2, 3))[0]

    v_xx = torch.gradient(dv_dx, spacing=(dy, dx), dim=(2, 3))[1]



    div = du_dx + dv_dy

    loss_cont = (div ** 2).mean()



    mom_u = u_t + u * du_dx + v * du_dy + dp_dx - nu * (u_xx + u_yy)

    mom_v = v_t + u * dv_dx + v * dv_dy + dp_dy - nu * (v_xx + v_yy)

    loss_mom = (mom_u ** 2).mean() + (mom_v ** 2).mean()

    return loss_cont, loss_mom

