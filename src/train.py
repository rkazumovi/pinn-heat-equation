# src/train.py
# Training loop for the PINN
#
# Process:
# 1. Generate training data
# 2. Forward pass — compute predictions
# 3. Compute total loss
# 4. Backward pass — compute gradients
# 5. Update weights — optimizer step
# 6. Repeat for n epochs

import torch
import time
from src.model import PINN, device
from src.data import (generate_collocation_points,
                      generate_boundary_points,
                      generate_initial_points)
from src.losses import total_loss

# ── Hyperparameters ──
EPOCHS        = 5000   # number of training iterations
LEARNING_RATE = 1e-3   # Adam optimizer learning rate
LOG_EVERY     = 500    # print progress every N epochs


def train():
    print("=" * 55)
    print("  Physics-Informed Neural Network — Heat Equation")
    print("=" * 55)
    print(f"  Device:        {device}")
    print(f"  Epochs:        {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print("=" * 55)

    # ── Generate training data ──
    x_col, t_col                        = generate_collocation_points(2000)
    x_left, x_right, t_bc, u_left, _   = generate_boundary_points(200)
    x_ic, t_ic, u_ic                    = generate_initial_points(200)

    # ── Initialize model and optimizer ──
    model     = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ── Track loss history for plotting ──
    history = {
        "total":    [],
        "physics":  [],
        "boundary": [],
        "initial":  []
    }

    start_time = time.time()

    # ── Training loop ──
    for epoch in range(1, EPOCHS + 1):

        optimizer.zero_grad()   # clear gradients from previous step

        loss, lp, lb, li = total_loss(
            model,
            x_col, t_col,
            x_left, x_right, t_bc,
            x_ic, t_ic, u_ic
        )

        loss.backward()         # compute gradients
        optimizer.step()        # update weights

        # Record history
        history["total"].append(loss.item())
        history["physics"].append(lp.item())
        history["boundary"].append(lb.item())
        history["initial"].append(li.item())

        # Print progress
        if epoch % LOG_EVERY == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch:>5}/{EPOCHS} | "
                  f"Total: {loss.item():.6f} | "
                  f"Physics: {lp.item():.6f} | "
                  f"Boundary: {lb.item():.6f} | "
                  f"Initial: {li.item():.6f} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - start_time
    print("=" * 55)
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Final total loss: {history['total'][-1]:.6f}")
    print("=" * 55)

    # ── Save model ──
    torch.save(model.state_dict(), "outputs/model.pth")
    print("  Model saved to outputs/model.pth")

    return model, history


if __name__ == "__main__":
    model, history = train()