# main.py
# Entry point for the PINN Heat Equation project
#
# Runs the full pipeline:
# 1. Train the model
# 2. Visualize results
# 3. Print accuracy report

from src.train import train
from src.visualize import (
    plot_loss_history,
    plot_solution_comparison,
    plot_error_map,
    print_accuracy
)

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  PINN — 1D Heat Equation Solver")
    print("  du/dt = alpha * d²u/dx²")
    print("=" * 55 + "\n")

    # Step 1 — Train
    model, history = train()

    # Step 2 — Visualize
    print("\nGenerating plots...")
    plot_loss_history(history)
    plot_solution_comparison(model)
    plot_error_map(model)

    # Step 3 — Accuracy
    print_accuracy(model)

    print("\nDone. Check outputs/ folder for all plots.")