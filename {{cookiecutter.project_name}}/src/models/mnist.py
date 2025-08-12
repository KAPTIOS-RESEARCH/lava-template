import torch, h5py
import lava.lib.dl.slayer as slayer

class MNIST_SNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        neuron_params = {
            'threshold': 1.0,
            'current_decay': 0.1,
            'voltage_decay': 0.1,
            'requires_grad': True,
        }
        
        self.layers = torch.nn.ModuleList([
            slayer.block.cuba.Dense(
                neuron_params, 784, 128, weight_norm=False, delay=False),
            slayer.block.cuba.Dense(
                neuron_params, 128, 64, weight_norm=False, delay=False),
            slayer.block.cuba.Dense(
                neuron_params, 64, 10, weight_norm=False, delay=False)
        ])

    def export(self, filename):
        """
        Export the network to hdf5 format.
        """
        h = h5py.File(filename, "w")
        layer = h.create_group("layer")
        for i, block in enumerate(self.layers):
            block.export_hdf5(layer.create_group(f'{i}'))
            
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x