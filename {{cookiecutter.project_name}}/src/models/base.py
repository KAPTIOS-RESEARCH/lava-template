import torch, logging
from typing import List
from src.utils.config import get_lava_block

class SpikeMonitorWrapper(torch.nn.Module):
    """
    Wrap a Lava layer to monitor spike activity.
    Records spike rate per forward pass.
    """
    def __init__(self, layer, layer_name="layer"):
        super().__init__()
        self.layer = layer
        self.layer_name = layer_name
        self.spike_rates = []

    def forward(self, x):
        out = self.layer(x)
        spike_rate = out.float().mean().item()
        self.spike_rates.append(spike_rate)
        return out

class BaseLavaModel(torch.nn.Module):

    def __init__(self):
        super(BaseLavaModel, self).__init__()
        self.avg_spike_stats = {}
        self.detail_spike_stats = {}
    
    def init_sequential_dense_layers(self, neuron_params, features_list, lava_block_str, weight_norm: bool = True, delay: bool = False):
        lava_block = get_lava_block(lava_block_str)
        return torch.nn.Sequential(
            *[lava_block.Dense(neuron_params, in_f, out_f, weight_norm=weight_norm, delay=delay)
            for in_f, out_f in zip(features_list[:-1], features_list[1:])]
        )
    
    def init_sequential_conv1d_layer(self, neuron_params, features_list, lava_block_str, weight_norm: bool = True, delay: bool = False):
        layers = []
        lava_block = get_lava_block(lava_block_str)
        for in_ch, out_ch in zip(features_list[:-1], features_list[1:]):
            conv = lava_block.Conv(
                neuron_params=neuron_params,
                in_features=in_ch,
                out_features=out_ch,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                weight_norm=weight_norm,
                delay=delay
            )
            layers.append(conv)
        return torch.nn.Sequential(*layers)
        
    def get_layers_spike_stats(self, layers: List[SpikeMonitorWrapper]):
        stats = {}
        for layer in layers:
            stats[layer.layer_name] = sum(layer.spike_rates) / len(layer.spike_rates)
        return stats
    
    def set_monitored_layers(self, layers, prefix="layer"):
        monitored_layers = []
        for i, layer in enumerate(layers):
            monitored_layers.append(SpikeMonitorWrapper(layer, f"{prefix}_{i}"))
        return torch.nn.Sequential(*monitored_layers)


    def print_model_size(self):
        param_size = sum(p.numel() for p in self.parameters() if p.requires_grad) * 4  # 4 bytes per float32
        buffer_size = sum(p.numel() for p in self.buffers()) * 4  # 4 bytes per float32
        size_all_mb = (param_size + buffer_size) / (1024 ** 2)
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"Total Params: {n_parameters:,}")
        logging.info(f"Model Params Size: {size_all_mb:.2f} MB")
        return n_parameters, size_all_mb