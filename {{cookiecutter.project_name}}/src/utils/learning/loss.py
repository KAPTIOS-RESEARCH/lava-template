import torch
from lava.lib.dl.slayer.loss import SpikeRate

class SpikeRateRegularizationLoss(torch.nn.Module):
    """Penalty if firing rate is bellow threshold"""
    def __init__(
        self,
        lambda_reg: float = 0.01,
        true_rate: float = 0.8,
        false_rate: float = 0.125,
        target_regularization_rate: float = 0.2,
        reduction: str = "sum"
    ):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.true_rate = true_rate
        self.false_rate = false_rate
        self.target_regularization_rate = target_regularization_rate
        self.reduction = reduction

        self.rate_loss = SpikeRate(true_rate, false_rate, reduction=reduction)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        rate_err = self.rate_loss(x, y)

        avg_rate = x.float().mean()

        under_rate_penalty = torch.clamp(self.target_regularization_rate - avg_rate, min=0.0)

        regularization_err = under_rate_penalty ** 2

        total_loss = rate_err + self.lambda_reg * regularization_err
        return total_loss




def target_to_spike_train(labels, n_classes, T_max, t_target=2):
    """
    Convert class labels into a one-hot temporal spike train.

    This function transforms a 1D tensor of integer class labels of shape ``(B,)``
    into a 3D spike train tensor of shape ``(B, C, T_max)``, where:
    - ``B`` is the batch size,
    - ``C`` is the number of classes,
    - ``T_max`` is the number of timesteps.

    Each target class emits a single spike (value = 1.0) at the specified time step
    ``t_target`` along the temporal dimension. If ``t_target`` exceeds ``T_max - 1``,
    the spike is placed at the final timestep instead. All other elements remain zero.

    Parameters
    ----------
    labels : torch.Tensor
        Tensor of shape ``(B,)`` containing integer class indices for each sample.
    n_classes : int
        Total number of output classes (i.e., the size of the second dimension).
    T_max : int
        Total number of timesteps in the spike train.
    t_target : int, optional (default=2)
        The timestep index at which the correct class should emit a spike.

    Returns
    -------
    torch.Tensor
        A spike train tensor of shape ``(B, C, T_max)``, where each sample's true class
        has a single spike at the specified ``t_target``.

    Examples
    --------
    >>> labels = torch.tensor([1, 3, 0])
    >>> spike_train = target_to_spike_train(labels, n_classes=5, T_max=8, t_target=2)
    >>> spike_train[0, 1, 2]
    """
    B, C = labels.size(0), n_classes
    desired = torch.zeros((B, C, T_max), device=labels.device)

    for i in range(B):
        correct_class = labels[i]
        if t_target < T_max:
            desired[i, correct_class, t_target] = 1.0
        else:
            desired[i, correct_class, T_max - 1] = 1.0

    return desired