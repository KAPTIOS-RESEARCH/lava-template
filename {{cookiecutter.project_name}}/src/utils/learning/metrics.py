import torch
from lava.lib.dl.slayer.classifier import Rate
import numpy as np

def van_rossum_distance(a, b, tau=5.0):
    """
    a, b: 1D binary arrays (0/1)
    tau: time constant in same units as bin index
    returns scalar van Rossum distance
    """
    a = np.asarray(a).astype(float)
    b = np.asarray(b).astype(float)
    T = len(a)
    t = np.arange(0, T)
    kernel = np.exp(-t / tau)
    fa = np.convolve(a, kernel, mode='full')[:T]
    fb = np.convolve(b, kernel, mode='full')[:T]
    return np.sqrt(np.sum((fa - fb) ** 2))

def spike_rate_accuracy(output_spikes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy depending on firing rate
    
    Parameters
    ----------
    output_spikes : torch.Tensor
        Output spike trains of shape (B, C, T).
        B: batch size, C: number of classes, T: timesteps.
    labels : torch.Tensor
        Ground-truth class indices of shape (B,).

    Returns
    -------
    torch.Tensor
        Scalar tensor with mean classification accuracy (0–100).
    """
    preds = Rate.predict(output_spikes)
    correct_predictions = (preds == labels)
    num_correct = correct_predictions.sum().item()
    total_samples = labels.size(0)
    return num_correct / total_samples * 100

def spike_count_accuracy(output_spikes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy depending on spike count
    
    Parameters
    ----------
    output_spikes : torch.Tensor
        Output spike trains of shape (B, C, T).
        B: batch size, C: number of classes, T: timesteps.
    labels : torch.Tensor
        Ground-truth class indices of shape (B,).

    Returns
    -------
    torch.Tensor
        Scalar tensor with mean classification accuracy (0–1).
    """
    # Sum spikes across time → (B, C)
    spike_counts = output_spikes.sum(dim=-1)
    
    # Predicted class = neuron with most total spikes
    predicted_classes = spike_counts.argmax(dim=1)
    
    # Compare to true labels
    accuracy = (predicted_classes == labels).float().mean()
    return accuracy


def spike_ttfs_accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute classification accuracy from spike trains (B, C, T), resolving ties by 
    choosing the neuron with the earliest first spike among those with max spike count.

    Parameters
    ----------
    output : torch.Tensor
        Predicted spike trains of shape (B, C, T)
    labels : torch.Tensor
        True class indices of shape (B,)

    Returns
    -------
    torch.Tensor
        Scalar tensor with mean classification accuracy (0-1)
    """
    B, C, T = output.shape

    # Step 1: spike counts per neuron
    spike_counts = output.sum(dim=-1)  # (B, C)

    # Step 2: find candidates with maximum spike count
    max_count = spike_counts.max(dim=1, keepdim=True).values  # (B,1)
    candidates = (spike_counts == max_count)  # (B, C), bool

    # Step 3: compute first spike time for each neuron
    spike_mask = (output > 0).float()
    first_spike_time = spike_mask.cumsum(dim=-1).gt(0).float().argmax(dim=-1)  # (B, C)
    # neurons with no spikes → assign max time
    first_spike_time[spike_mask.sum(dim=-1) == 0] = T

    # Step 4: among candidates, pick neuron with earliest first spike
    candidate_times = first_spike_time.clone()
    candidate_times[~candidates] = T  # ignore non-candidates
    predicted_class = candidate_times.argmin(dim=1)  # (B,)

    # Step 5: compute accuracy
    accuracy = (predicted_class == labels).float().mean()
    return accuracy