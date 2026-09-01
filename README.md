# 🔥 Physics-Informed Neural Network — 1D Heat Equation

A neural network that learns to solve the 1D heat equation by respecting the underlying physics — without any labeled solution data. Built with PyTorch from scratch.

📂 **Repository:** [github.com/rkazumovi/pinn-heat-equation](https://github.com/rkazumovi/pinn-heat-equation)

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?logo=scipy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)

---

## The Problem

The 1D heat equation describes how temperature distributes over time:

$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

Where:
- $u(x, t)$ = temperature at position $x$ and time $t$
- $\alpha = 0.01$ = thermal diffusivity constant
- $x \in [0, 1]$, $t \in [0, 1]$

**Boundary conditions:** $u(0, t) = 0$, $u(1, t) = 0$
**Initial condition:** $u(x, 0) = \sin(\pi x)$
**Analytical solution:** $u(x, t) = e^{-\alpha \pi^2 t} \sin(\pi x)$

## What is a PINN?

A Physics-Informed Neural Network (PINN) is a neural network trained to satisfy a partial differential equation. Instead of learning from labeled data, it minimizes a loss function composed of three terms:

- **Physics loss** — penalizes violation of the PDE at random collocation points, computed via the exact PDE residual $\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2}$
- **Boundary loss** — enforces the Dirichlet boundary conditions at $x=0$ and $x=1$
- **Initial loss** — enforces the initial temperature profile $u(x,0) = \sin(\pi x)$

The key insight: PyTorch autograd computes exact derivatives of the network output with respect to its inputs, enabling automatic differentiation of the PDE residual — no finite-difference approximation needed.

## Architecture

- 4 fully connected layers (2 → 64 → 64 → 64 → 1)
- Tanh activations (smooth and infinitely differentiable — required for computing second-order PDE derivatives)
- 8,577 trainable parameters
- Trained with the Adam optimizer (learning rate 1e-3), 5,000 epochs
- Trained on 2,000 collocation points, 200 boundary points, and 200 initial-condition points

## Results

| Metric | Value |
|---|---|
| Final total loss | 0.000009 |
| Max absolute error | 0.002254 |
| Mean absolute error | 0.000846 |
| Training time | ~3 minutes (CPU) |

### Loss History
![Loss History](outputs/loss_history.png)

### PINN vs Analytical Solution
![Solution Comparison](outputs/solution_comparison.png)

### Error Map
![Error Map](outputs/error_map.png)

## 🛠️ Built With

**Language**
- Python 3

**Core Libraries**
- [PyTorch](https://pytorch.org/) — model definition, autograd-based differentiation, and training
- [NumPy](https://numpy.org/) — numerical operations and data generation
- [SciPy](https://scipy.org/) — scientific computing utilities
- [Matplotlib](https://matplotlib.org/) — loss curves, solution comparison, and error map visualizations

**Concepts**
- Physics-Informed Machine Learning
- Deep Learning
- Partial Differential Equations
- Numerical Analysis
- Automatic Differentiation
- Scientific Computing

## Project Structure

```
pinn-heat-equation/
├── src/
│   ├── model.py        # Neural network architecture
│   ├── data.py         # Training data generation (collocation, boundary, initial points)
│   ├── losses.py       # Physics, boundary, initial losses
│   ├── train.py        # Training loop
│   └── visualize.py    # Plots and accuracy report
├── outputs/            # Generated plots and saved model weights
├── main.py             # Full pipeline entry point
└── requirements.txt    # Dependencies
```

## Run It Yourself

```bash
git clone https://github.com/rkazumovi/pinn-heat-equation.git
cd pinn-heat-equation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

## Key Concepts

**Why not just use a numerical solver?**
Traditional methods like finite difference or finite element require a discrete mesh. PINNs are mesh-free, naturally handle irregular domains, and can incorporate noisy observational data alongside physics constraints.

**Why Tanh over ReLU?**
PINNs require computing second-order derivatives of the network output. ReLU's second derivative is zero almost everywhere, making physics loss computation impossible. Tanh is smooth and infinitely differentiable.

**Mathematical foundation:**
The physics loss minimizes the PDE residual, computed via exact automatic differentiation through PyTorch autograd rather than numerical approximation.
