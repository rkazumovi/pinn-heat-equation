# src/visualize.py
# Visualizes the PINN solution against the analytical solution
#
# Generates 3 plots:
# 1. Loss history — shows training progress
# 2. Solution comparison — PINN vs analytical at different time steps
# 3. Error map — absolute difference across full domain

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import PINN, device
from src.data import analytical_solution


def load_model():
    """Load the trained model from disk."""
    model = PINN().to(device)
    model.load_state_dict(torch.load("outputs/model.pth", map_location=device))
    model.eval()
    return model


def plot_loss_history(history):
    """Plot training loss curves."""
    epochs = range(1, len(history["total"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.semilogy(epochs, history["total"],    label="Total Loss",    linewidth=2)
    plt.semilogy(epochs, history["physics"],  label="Physics Loss",  linewidth=1.5, linestyle="--")
    plt.semilogy(epochs, history["boundary"], label="Boundary Loss", linewidth=1.5, linestyle="--")
    plt.semilogy(epochs, history["initial"],  label="Initial Loss",  linewidth=1.5, linestyle="--")

    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss History")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/loss_history.png", dpi=150)
    plt.show()
    print("  Saved: outputs/loss_history.png")


def plot_solution_comparison(model):
    """
    Compare PINN solution vs analytical solution
    at 4 different time snapshots.
    """
    x = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
    time_steps = [0.0, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, t_val in enumerate(time_steps):
        t = torch.full_like(x, t_val)

        # PINN prediction
        with torch.no_grad():
            u_pred = model(x, t).cpu().numpy()

        # Analytical solution
        u_exact = analytical_solution(x, t).cpu().numpy()
        x_np    = x.cpu().numpy()

        ax = axes[idx]
        ax.plot(x_np, u_exact, "b-",  linewidth=2.5, label="Analytical")
        ax.plot(x_np, u_pred,  "r--", linewidth=2,   label="PINN")
        ax.set_title(f"t = {t_val}")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    plt.suptitle("PINN vs Analytical Solution — 1D Heat Equation", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/solution_comparison.png", dpi=150)
    plt.show()
    print("  Saved: outputs/solution_comparison.png")


def plot_error_map(model):
    """
    Plot absolute error |PINN - analytical| across
    the full (x, t) domain as a heatmap.
    """
    x_vals = torch.linspace(0, 1, 100)
    t_vals = torch.linspace(0, 1, 100)

    X, T = torch.meshgrid(x_vals, t_vals, indexing="ij")
    x_flat = X.reshape(-1, 1).to(device)
    t_flat = T.reshape(-1, 1).to(device)

    with torch.no_grad():
        u_pred  = model(x_flat, t_flat).cpu().reshape(100, 100)

    u_exact = analytical_solution(x_flat, t_flat).cpu().reshape(100, 100)
    error   = torch.abs(u_pred - u_exact).numpy()

    plt.figure(figsize=(10, 5))
    plt.contourf(
        t_vals.numpy(),
        x_vals.numpy(),
        error,
        levels=50,
        cmap="hot_r"
    )
    plt.colorbar(label="Absolute Error |PINN - Analytical|")
    plt.xlabel("t (time)")
    plt.ylabel("x (space)")
    plt.title("Error Map — PINN vs Analytical Solution")
    plt.tight_layout()
    plt.savefig("outputs/error_map.png", dpi=150)
    plt.show()
    print("  Saved: outputs/error_map.png")


def print_accuracy(model):
    """Print max and mean error across the domain."""
    x_flat = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)
    t_flat = torch.linspace(0, 1, 200).reshape(-1, 1).to(device)

    with torch.no_grad():
        u_pred  = model(x_flat, t_flat)

    u_exact = analytical_solution(x_flat, t_flat)
    error   = torch.abs(u_pred - u_exact)

    print(f"\n  Accuracy Report:")
    print(f"  Max absolute error:  {error.max().item():.8f}")
    print(f"  Mean absolute error: {error.mean().item():.8f}")


if __name__ == "__main__":
    from src.train import train

    print("Training model...")
    model, history = train()

    print("\nGenerating plots...")
    print("=" * 55)

    plot_loss_history(history)
    plot_solution_comparison(model)
    plot_error_map(model)
    print_accuracy(model)

    print("=" * 55)
    print("  All plots saved to outputs/")