# src/data.py
# Generates training data for the PINN
# Three types of points: collocation, boundary, initial condition

import torch
import numpy as np
from src.model import device

torch.manual_seed(42)
np.random.seed(42)


def generate_collocation_points(n=2000):
    """
    Random points inside the domain: x ∈ [0,1], t ∈ [0,1]
    These enforce the heat equation: du/dt = alpha * d²u/dx²
    """
    x = torch.rand(n, 1, dtype=torch.float32).to(device)
    t = torch.rand(n, 1, dtype=torch.float32).to(device)
    return x, t


def generate_boundary_points(n=200):
    """
    Points at x=0 and x=1 for all t ∈ [0,1]
    Boundary condition: u(0,t) = 0 and u(1,t) = 0
    """
    t = torch.rand(n, 1, dtype=torch.float32).to(device)

    # x=0 boundary
    x_left = torch.zeros(n, 1, dtype=torch.float32).to(device)
    u_left = torch.zeros(n, 1, dtype=torch.float32).to(device)

    # x=1 boundary
    x_right = torch.ones(n, 1, dtype=torch.float32).to(device)
    u_right = torch.zeros(n, 1, dtype=torch.float32).to(device)

    return x_left, x_right, t, u_left, u_right


def generate_initial_points(n=200):
    """
    Points at t=0 for all x ∈ [0,1]
    Initial condition: u(x,0) = sin(πx)
    """
    x = torch.rand(n, 1, dtype=torch.float32).to(device)
    t = torch.zeros(n, 1, dtype=torch.float32).to(device)

    # Analytical initial condition
    u = torch.sin(torch.pi * x)

    return x, t, u


def analytical_solution(x, t, alpha=0.01):
    """
    Exact solution: u(x,t) = e^(-alpha * pi² * t) * sin(πx)
    Used to verify our PINN result after training
    """
    return torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)


if __name__ == "__main__":
    # Verify all data generation works
    x_col, t_col = generate_collocation_points(2000)
    print(f"Collocation points: x shape={x_col.shape}, t shape={t_col.shape}")
    print(f"  x range: [{x_col.min():.3f}, {x_col.max():.3f}]")
    print(f"  t range: [{t_col.min():.3f}, {t_col.max():.3f}]")

    x_left, x_right, t_bc, u_left, u_right = generate_boundary_points(200)
    print(f"\nBoundary points: {x_left.shape[0]} left + {x_right.shape[0]} right")
    print(f"  u at x=0: all {u_left.unique().item():.1f} (correct)")
    print(f"  u at x=1: all {u_right.unique().item():.1f} (correct)")

    x_ic, t_ic, u_ic = generate_initial_points(200)
    print(f"\nInitial condition points: {x_ic.shape[0]} points")
    print(f"  u range: [{u_ic.min():.3f}, {u_ic.max():.3f}]")
    print(f"  (should be between 0 and 1 since u=sin(πx))")

    # Test analytical solution
    x_test = torch.tensor([[0.5]])
    t_test = torch.tensor([[0.0]])
    u_exact = analytical_solution(x_test, t_test)
    print(f"\nAnalytical solution check:")
    print(f"  u(0.5, 0.0) = {u_exact.item():.6f} (should be ~1.0 since sin(π*0.5)=1)")