"""Small offline teaching figures, separate from the model's learned positions."""

import math

import torch


def sinusoidal_positions(length, width):
    if min(length, width) < 1:
        raise ValueError("Positive dimensions required")
    values = torch.zeros(length, width)
    position = torch.arange(length, dtype=torch.float32)[:, None]
    scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
    values[:, 0::2] = torch.sin(position * scale)
    values[:, 1::2] = torch.cos(position * scale[: width // 2])
    return values


def plot_mechanics(output):
    import matplotlib

    from .model import causal_mask
    from .workflow import new_output

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = new_output(output)
    positions = sinusoidal_positions(100, 32)
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].imshow(positions.numpy(), aspect="auto", cmap="RdBu")
    axes[0].set_title("Sinusoidal teaching example")
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Position")
    axes[1].plot((positions @ positions[20]).numpy())
    axes[1].axvline(20, color="gray", linestyle="--")
    axes[1].set_title("Similarity is oscillatory")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Dot product with position 20")
    axes[2].imshow(causal_mask(8).numpy(), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("White = blocked future keys")
    axes[2].set_xlabel("Key position")
    axes[2].set_ylabel("Query position")
    figure.tight_layout()
    figure.savefig(output / "mechanics.png", dpi=140)
    plt.close(figure)
    return {
        "output": str(output / "mechanics.png"),
        "scope": "offline_teaching_only_no_speed_benchmark",
    }
