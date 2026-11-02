__Course__ = "CSE 574 - Intro To Machine Learning"
__author__ = "Viral Koshti"
__email__ = "viraljag@buffalo.edu"
__UBPersonNumber__ = "50761354"

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
N = 300
HIDDEN = 24
EPOCHS = 2200
START_LR = 0.008
MIN_LR = 0.00004


def forward(x, w1, b1, w2, b2):
    hidden = np.tanh(w1 * x + b1)
    return float(np.dot(w2, hidden) + b2), hidden


def mse(x, target, w1, b1, w2, b2):
    hidden = np.tanh(x[:, None] * w1 + b1)
    prediction = hidden @ w2 + b2
    return float(np.mean((target - prediction) ** 2))


def main():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 1, N)
    noise = rng.uniform(-0.1, 0.1, N)
    target = np.sin(20 * x) + 3 * x + noise
    folder = Path(__file__).resolve().parent

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, target, s=15, alpha=0.7)
    ax.set(xlabel="x", ylabel="d", title="Original noisy training samples")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "Project6_Data.png", dpi=170)
    plt.close(fig)

    # Each hidden tanh neuron starts with a different transition location.
    # All 73 weights (24+24+24+1) are trained by online BP.
    centers = np.linspace(0.02, 0.98, HIDDEN)
    w1 = rng.choice([-1.0, 1.0], size=HIDDEN) * rng.uniform(10.0, 20.0, HIDDEN)
    b1 = -w1 * centers + rng.normal(0, 0.15, HIDDEN)
    w2 = rng.normal(0, 0.12, HIDDEN)
    b2 = 1.5

    lr = START_LR
    history = [mse(x, target, w1, b1, w2, b2)]
    reductions = 0
    rejected = 0
    for epoch in range(1, EPOCHS + 1):
        old = (w1.copy(), b1.copy(), w2.copy(), b2)
        # An update occurs immediately after EACH individual data point.
        for idx in rng.permutation(N):
            prediction, h = forward(x[idx], w1, b1, w2, b2)
            error = prediction - target[idx]
            # Gradients for sample loss 0.5*(target - prediction)^2.
            delta_hidden = error * w2 * (1.0 - h * h)
            w2 -= lr * error * h
            b2 -= lr * error
            w1 -= lr * delta_hidden * x[idx]
            b1 -= lr * delta_hidden

        current = mse(x, target, w1, b1, w2, b2)
        # Roll back an overshooting epoch and retry next epoch at a lower rate.
        if not np.isfinite(current) or current > history[-1] * (1 + 1e-9):
            w1, b1, w2, b2 = old
            rejected += 1
            if lr > MIN_LR:
                lr = max(MIN_LR, lr * 0.9)
                reductions += 1
            current = history[-1]
        history.append(current)
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d} | MSE {current:.6f} | learning rate {lr:.7f}")

    grid = np.linspace(0, 1, 600)
    fitted = np.tanh(grid[:, None] * w1 + b1) @ w2 + b2
    true_curve = np.sin(20 * grid) + 3 * grid

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(history)), history)
    ax.set(
        xlabel="Epoch",
        ylabel="Training MSE",
        title="Online backpropagation: MSE by epoch",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "Project6_MSE.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(x, target, s=13, alpha=0.45, label="Noisy observations")
    ax.plot(grid, fitted, linewidth=2.2, label="Neural network fit")
    ax.plot(
        grid,
        true_curve,
        linestyle="--",
        linewidth=1,
        alpha=0.8,
        label="Noise-free target (reference)",
    )
    ax.set(
        xlabel="x",
        ylabel="d / prediction",
        title="Fitted network curve and training data",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "Project6_Fit.png", dpi=170)
    plt.close(fig)

    print(f"Initial MSE: {history[0]:.8f}")
    print(f"Final MSE: {history[-1]:.8f}")
    print(f"Initial LR: {START_LR}; Final LR: {lr:.8f}")
    print(f"LR reductions: {reductions}; rejected epochs: {rejected}")


if __name__ == "__main__":
    main()
