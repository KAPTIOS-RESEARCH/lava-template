import torch, h5py
import lava.lib.dl.slayer as slayer
from .base import BaseLavaModel
from typing import List

DEFAULT_NEURON = {
    'threshold': 1.0,
    'current_decay': 0.1,
    'voltage_decay': 0.1,
    'requires_grad': True,
}

class MNIST_SNN(BaseLavaModel):
    def __init__(self, 
        neuron_params: dict = DEFAULT_NEURON, 
        features_list: List[int] = [784, 128, 64, 10],
        lava_block_str: str = 'cuba'
    ):
        super(MNIST_SNN, self).__init__()
    
        self.layers = self.set_monitored_layers(
            self.init_sequential_dense_layers(
                neuron_params, 
                features_list, 
                lava_block_str, 
                True, 
                False), 
            'FC')

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x