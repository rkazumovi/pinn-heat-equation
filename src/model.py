# src/model.py
# Neural network architecture for the PINN
# Solves the 1D Heat Equation: du/dt = alpha * d²u/dx²

import torch
import torch.nn as nn

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 64),   # input: (x, t)
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)    # output: u(x, t)
        )

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)


if __name__ == "__main__":
    model = PINN().to(device)

    print(f"Using device: {device}")
    print("\nModel Architecture:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")

    x_test = torch.tensor([[0.5]], dtype=torch.float32).to(device)
    t_test = torch.tensor([[0.1]], dtype=torch.float32).to(device)
    u_pred = model(x_test, t_test)
    print(f"\nTest input:  x=0.5, t=0.1")
    print(f"Test output: u={u_pred.item():.6f} (random before training)")