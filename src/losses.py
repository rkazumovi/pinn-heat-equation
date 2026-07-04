# src/losses.py
# Defines the three loss functions for the PINN
#
# Total loss = physics_loss + boundary_loss + initial_loss
#
# Physics loss:  enforces du/dt = alpha * d²u/dx²
# Boundary loss: enforces u(0,t) = 0 and u(1,t) = 0
# Initial loss:  enforces u(x,0) = sin(πx)

import torch
from src.model import PINN, device

ALPHA = 0.01  # thermal diffusivity constant


def physics_loss(model, x, t):
    """
    Computes PDE residual: du/dt - alpha * d²u/dx²
    Should be zero everywhere inside the domain.

    We use PyTorch autograd to compute exact derivatives
    of the neural network output — same as symbolic differentiation
    but automatic.
    """
    # Enable gradient tracking for x and t
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    u = model(x, t)

    # First derivative: du/dt
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        create_graph=True  # needed to differentiate again
    )[0]

    # First derivative: du/dx
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    # Second derivative: d²u/dx²
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    # PDE residual: du/dt - alpha * d²u/dx² = 0
    residual = u_t - ALPHA * u_xx

    # Mean squared residual — should approach zero during training
    return torch.mean(residual ** 2)


def boundary_loss(model, x_left, x_right, t):
    """
    Enforces Dirichlet boundary conditions:
    u(0, t) = 0  and  u(1, t) = 0
    """
    u_left  = model(x_left,  t)
    u_right = model(x_right, t)

    loss_left  = torch.mean(u_left  ** 2)
    loss_right = torch.mean(u_right ** 2)

    return loss_left + loss_right


def initial_loss(model, x, t, u_exact):
    """
    Enforces initial condition:
    u(x, 0) = sin(πx)
    """
    u_pred = model(x, t)
    return torch.mean((u_pred - u_exact) ** 2)


def total_loss(model, x_col, t_col,
               x_left, x_right, t_bc,
               x_ic, t_ic, u_ic):
    """
    Combines all three losses into one scalar.
    Each loss weighted equally — can be tuned later.
    """
    loss_physics  = physics_loss(model, x_col, t_col)
    loss_boundary = boundary_loss(model, x_left, x_right, t_bc)
    loss_initial  = initial_loss(model, x_ic, t_ic, u_ic)

    loss = loss_physics + loss_boundary + loss_initial

    return loss, loss_physics, loss_boundary, loss_initial


if __name__ == "__main__":
    from src.data import (generate_collocation_points,
                          generate_boundary_points,
                          generate_initial_points)

    model = PINN().to(device)

    # Generate data
    x_col, t_col                     = generate_collocation_points(2000)
    x_left, x_right, t_bc, _, _     = generate_boundary_points(200)
    x_ic, t_ic, u_ic                 = generate_initial_points(200)

    # Compute losses before training (should be large random numbers)
    loss, lp, lb, li = total_loss(
        model,
        x_col, t_col,
        x_left, x_right, t_bc,
        x_ic, t_ic, u_ic
    )

    print(f"Losses before training (all should be large):")
    print(f"  Physics loss:  {lp.item():.6f}")
    print(f"  Boundary loss: {lb.item():.6f}")
    print(f"  Initial loss:  {li.item():.6f}")
    print(f"  Total loss:    {loss.item():.6f}")