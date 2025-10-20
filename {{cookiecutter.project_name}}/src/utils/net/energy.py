def flops_to_energy(flops, energy_per_flop_pJ=4.6):
    """
    Convert FLOPs to energy (mJ), GFLOPs, and Watt-hour (mWh).

    Returns
    -------
    energy_mJ : float
        Energy in milliJoules
    gflops : float
        GFLOPs
    energy_mWh : float
        Energy in milliWatt-hours
    """
    energy_J = flops * energy_per_flop_pJ * 1e-12  # Joules
    gflops = flops / 1e9                           # GFLOPs
    energy_mJ = energy_J * 1e3                     # mJ
    energy_mWh = energy_J / 3600 * 1e6                   # Wh
    return energy_mJ, gflops, energy_mWh

def compute_flops_snn(conv_layers, dense_layers, input_shape, T, spike_activity=1.0):
    """
    Estimate FLOPs for a SLAYER SNN.
    
    Parameters
    ----------
    conv_layers : list of tuples
        Each tuple: (C_in, C_out, kernel_H, kernel_W)
    dense_layers : list of tuples
        Each tuple: (C_in, C_out)
    input_shape : tuple
        (B, C_in, N_points)
    T : int
        Number of timesteps
    spike_activity : float
        Average spike probability (0..1). Default 1.0 (dense)
    
    Returns
    -------
    total_flops : float
        Estimated number of FLOPs
    """
    B, _, N = input_shape
    total_flops = 0
    
    # Conv FLOPs
    for C_in, C_out, kH, kW in conv_layers:
        # Each output channel multiplies input channels * kernel size * N_points * timesteps
        flops_per_sample = C_out * C_in * kH * kW * N * T
        total_flops += B * flops_per_sample * spike_activity  # multiply by batch and spike activity
    
    # Dense FLOPs
    for C_in, C_out in dense_layers:
        flops_per_sample = C_in * C_out * T
        total_flops += B * flops_per_sample * spike_activity
    
    return total_flops


if __name__ == "__main__":
    conv_layers = [
        (3, 64, 1, 1),
        (64, 128, 1, 1),
        (128, 256, 1, 1)
    ]

    dense_layers = [
        (256, 128),
        (128, 10)
    ]

    B = 16
    N_points = 1024
    T = 8
    input_shape = (B, 3, N_points)
    spike_activity = 0.1

    flops = compute_flops_snn(conv_layers, dense_layers, input_shape, T, spike_activity)
    energy_J, gflops, energy_Wh = flops_to_energy(flops)
    print(f"Energy: {energy_J:.6f} mJ | {energy_Wh:.9f} μWh, GFLOPs: {gflops:.3f}")