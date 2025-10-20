import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import List


def plot_layer_spike_rates(spike_stats: dict):
    layers = list(spike_stats.keys())
    rates = list(spike_stats.values())
    plt.figure(figsize=(10,4))
    plt.bar(layers, rates, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Average spike rate")
    plt.title("Layer-wise average spike activity")
    plt.ylim(0, 1)
    plt.show()

def plot_spike_raster_batch(spikes: torch.Tensor, title: str = "Spike Raster (Batch)", figsize=(15, 5), subtitles: List[str] = ['Sample']):
    """
    Plot raster plots for batched spiking activity tensors.

    Args:
        spikes (torch.Tensor): Binary tensor of shape (batch_size, num_neurons, timesteps)
                               where 1 = spike, 0 = no spike.
        title (str, optional): Main title for the figure.
        figsize (tuple, optional): Overall figure size.

    Example:
        spikes = torch.randint(0, 2, (3, 1024, 8))
        plot_spike_raster_batch(spikes)
    """
    if spikes.ndim != 3:
        raise ValueError(f"Expected 3D tensor [batch, neurons, timesteps], got {spikes.shape}")

    if len(subtitles) != spikes.shape[0]:
        raise ValueError(f"You should provide a subtitle per spike train")
    batch_size, num_neurons, timesteps = spikes.shape

    fig, axes = plt.subplots(1, batch_size, figsize=figsize, sharex=True, sharey=True)
    if batch_size == 1:
        axes = [axes]

    for b in range(batch_size):
        ax = axes[b]
        neuron_ids, time_ids = torch.nonzero(spikes[b], as_tuple=True)
        ax.scatter(time_ids.cpu(), neuron_ids.cpu(), s=2, color='black')
        ax.set_title(subtitles[b])
        ax.set_xlabel("Time step")
        if b == 0:
            ax.set_ylabel("Neuron index")
        ax.set_xlim(-0.5, timesteps - 0.5)
        ax.set_ylim(-0.5, num_neurons - 0.5)
        ax.invert_yaxis()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def van_rossum_distance_matrix(
    spikes: torch.Tensor,
    tau: float = 5.0,
    show_heatmap: bool = True,
    figsize: tuple = (6, 5),
    cmap: str = "viridis"
) -> torch.Tensor:
    """
    Compute pairwise Van Rossum distances between all spike trains in a batch
    and optionally display a heatmap.

    Args:
        spikes (torch.Tensor): Binary spike tensor of shape (batch, neurons, timesteps)
        tau (float): Time constant for exponential decaying trace. Small value means spike trains must align perfectly to be similar.
        show_heatmap (bool): Whether to show a distance heatmap
        figsize (tuple): Figure size for the heatmap
        cmap (str): Colormap

    Returns:
        torch.Tensor: Symmetric (batch, batch) Van Rossum distance matrix
    """
    if spikes.ndim != 3:
        raise ValueError(f"Expected shape (batch, neurons, timesteps), got {spikes.shape}")

    batch, neurons, timesteps = spikes.shape
    device = spikes.device

    # Create exponential decay kernel
    t = torch.arange(timesteps, device=device, dtype=torch.float32)
    kernel = torch.exp(-t / tau)
    kernel = kernel / kernel.sum()  # normalize

    # Repeat the kernel for each neuron
    kernel = kernel.view(1, 1, -1).repeat(neurons, 1, 1)  # [neurons, 1, timesteps]

    # Perform grouped convolution (each neuron convolved independently)
    conv = F.conv1d(
        spikes,
        kernel,
        padding=timesteps - 1,
        groups=neurons
    )[..., :timesteps]  # [batch, neurons, timesteps]

    # Flatten neuron & time dims
    flat = conv.reshape(batch, -1)

    # Compute pairwise distances
    norm = (flat ** 2).sum(dim=1, keepdim=True) / tau
    dist_sq = norm + norm.T - 2 * (flat @ flat.T) / tau
    dist_sq = torch.clamp(dist_sq, min=0.0)
    dist = torch.sqrt(dist_sq)

    # Visualization
    if show_heatmap:
        plt.figure(figsize=figsize)
        plt.imshow(dist.cpu(), cmap=cmap, interpolation="nearest")
        plt.colorbar(label="Van Rossum Distance")
        plt.title(f"Van Rossum Distance Matrix (τ={tau})")
        plt.xlabel("Sample index")
        plt.ylabel("Sample index")
        plt.tight_layout()
        plt.show()

    return dist

